"""
training/train.py
-----------------
Training loop for the CRN speech enhancement model.
"""

import argparse
import os
import sys
import time
import yaml
import logging
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.crn import CRN
from beamforming.mvdr import MVDRBeamformer
from training.losses import CombinedLoss
from data.dataset import build_dataloader
from data.fast_dataset import build_fast_dataloader
from evaluation.metrics import evaluate_batch
from utils.audio import STFT

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_scheduler(optimizer, n_warmup_epochs, n_total_epochs, steps_per_epoch):
    warmup_steps = n_warmup_epochs * steps_per_epoch
    total_steps  = n_total_epochs  * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        import math
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(state, path):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt.get("epoch", 0), ckpt.get("best_pesq", 0.0), ckpt.get("global_step", 0)


def train(cfg, resume_path=None):
    torch.manual_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    audio_cfg = cfg["audio"]
    stft = STFT(
        n_fft=audio_cfg["n_fft"],
        hop_length=audio_cfg["hop_length"],
        win_length=audio_cfg["win_length"],
    )  # kept on CPU

    m_cfg = cfg["model"]
    model = CRN(
        in_channels=m_cfg["in_channels"],
        encoder_channels=m_cfg["encoder_channels"],
        encoder_kernel=tuple(m_cfg["encoder_kernel"]),
        lstm_hidden=m_cfg["lstm_hidden"],
        lstm_layers=m_cfg["lstm_layers"],
        n_freq_bins=audio_cfg["n_fft"] // 2 + 1,
    ).to(device)

    beamformer = MVDRBeamformer(
        n_mics=audio_cfg["n_mics"],
        n_fft=audio_cfg["n_fft"],
    ).to(device)

    logger.info(f"CRN parameters: {model.count_parameters():,}")

    l_cfg = cfg["loss"]
    criterion = CombinedLoss(
        sisnr_weight=l_cfg["sisnr_weight"],
        mag_weight=l_cfg["mag_weight"],
        phase_weight=l_cfg["phase_weight"],
    )

    t_cfg = cfg["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=t_cfg["learning_rate"])
    scaler    = GradScaler("cuda", enabled=t_cfg["mixed_precision"] and device.type == "cuda")

    d_cfg = cfg["data"]
    train_pre = d_cfg["train_manifest"].replace(".json", "_precomputed.json")
    val_pre   = d_cfg["val_manifest"].replace(".json", "_precomputed.json")

    if Path(train_pre).exists():
        logger.info("Using precomputed dataset (fast I/O mode)")
        train_loader = build_fast_dataloader(train_pre, batch_size=t_cfg["batch_size"],
                                             shuffle=True, num_workers=t_cfg["num_workers"])
        val_loader   = build_fast_dataloader(val_pre,   batch_size=t_cfg["batch_size"],
                                             shuffle=False, num_workers=t_cfg["num_workers"])
    else:
        logger.info("Precomputed dataset not found — using on-the-fly loading")
        train_loader = build_dataloader(
            d_cfg["train_manifest"], batch_size=t_cfg["batch_size"],
            num_workers=t_cfg["num_workers"], shuffle=True,
            sample_rate=audio_cfg["sample_rate"], duration=d_cfg["max_duration"],
            snr_range=tuple(d_cfg["snr_range"]), n_mics=audio_cfg["n_mics"],
            rir_prob=d_cfg["rir_prob"], augment=True,
        )
        val_loader = build_dataloader(
            d_cfg["val_manifest"], batch_size=t_cfg["batch_size"],
            num_workers=t_cfg["num_workers"], shuffle=False,
            sample_rate=audio_cfg["sample_rate"], duration=d_cfg["max_duration"],
            snr_range=tuple(d_cfg["snr_range"]), n_mics=audio_cfg["n_mics"],
            rir_prob=0.0, augment=False,
        )

    scheduler = build_scheduler(
        optimizer,
        n_warmup_epochs=t_cfg["warmup_epochs"],
        n_total_epochs=t_cfg["num_epochs"],
        steps_per_epoch=len(train_loader),
    )

    start_epoch = 0
    best_pesq   = 0.0
    global_step = 0

    if resume_path and Path(resume_path).exists():
        start_epoch, best_pesq, global_step = load_checkpoint(
            resume_path, model, optimizer, scheduler
        )
        logger.info(f"Resumed from {resume_path} (epoch {start_epoch}, best PESQ {best_pesq:.3f})")

    log_cfg  = cfg["logging"]
    ckpt_cfg = cfg["checkpoint"]
    ckpt_dir = Path(ckpt_cfg["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=log_cfg["log_dir"])

    for epoch in range(start_epoch, t_cfg["num_epochs"]):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{t_cfg['num_epochs']}", leave=False)
        for batch_idx, (noisy_mc, clean, snr) in enumerate(pbar):
            noisy_mc = noisy_mc.to(device)
            clean    = clean.to(device)

            with autocast(device.type if device.type == "cuda" else "cpu",
                          enabled=t_cfg["mixed_precision"] and device.type == "cuda"):
                B, M, T_wav = noisy_mc.shape
                ref_mic = noisy_mc[:, 0, :]
                bf_real, bf_imag = stft(ref_mic)

                enhanced_real, enhanced_imag = model.enhance(bf_real, bf_imag)
                enhanced_real = torch.nan_to_num(enhanced_real, nan=0.0, posinf=0.0, neginf=0.0)
                enhanced_imag = torch.nan_to_num(enhanced_imag, nan=0.0, posinf=0.0, neginf=0.0)

                enhanced_wav = stft.inverse(enhanced_real, enhanced_imag, length=T_wav)
                clean_real, clean_imag = stft(clean)

                min_t = min(enhanced_wav.shape[-1], clean.shape[-1])
                loss, loss_dict = criterion(
                    enhanced_wav[:, :min_t], clean[:, :min_t],
                    enhanced_real, enhanced_imag,
                    clean_real,    clean_imag,
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), t_cfg["gradient_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % log_cfg["log_every"] == 0:
                for k, v in loss_dict.items():
                    writer.add_scalar(f"train/{k}", v, global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        if (epoch + 1) % log_cfg["eval_every"] == 0:
            val_metrics = validate(model, stft, val_loader, device)
            for k, v in val_metrics.items():
                writer.add_scalar(f"val/{k}", v, global_step)

            pesq = val_metrics.get("pesq", 0.0)
            logger.info(
                f"Epoch {epoch+1:3d} | loss={epoch_loss/len(train_loader):.4f} "
                f"| PESQ={pesq:.3f} | STOI={val_metrics.get('stoi', 0):.3f} "
                f"| {time.time()-t0:.0f}s"
            )

            if pesq > best_pesq:
                best_pesq = pesq
                save_checkpoint(
                    {"model": model.state_dict(), "epoch": epoch + 1,
                     "best_pesq": best_pesq, "global_step": global_step,
                     "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict()},
                    str(ckpt_dir / "best.pt"),
                )
                logger.info(f"  ✓ New best PESQ={best_pesq:.3f} → saved best.pt")

        save_checkpoint(
            {"model": model.state_dict(), "epoch": epoch + 1,
             "best_pesq": best_pesq, "global_step": global_step,
             "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict()},
            str(ckpt_dir / "last.pt"),
        )

    writer.close()
    logger.info(f"\n✓ Training complete. Best PESQ: {best_pesq:.3f}")


@torch.no_grad()
def validate(model, stft, loader, device):
    model.eval()
    all_pesq, all_stoi, all_sisnr = [], [], []

    for noisy_mc, clean, _ in tqdm(loader, desc="Validation", leave=False):
        noisy_mc = noisy_mc.to(device)
        clean    = clean.to(device)

        B, M, T_wav = noisy_mc.shape
        ref_mic = noisy_mc[:, 0, :]
        bf_real, bf_imag = stft(ref_mic)

        enhanced_real, enhanced_imag = model.enhance(bf_real, bf_imag)
        enhanced_real = torch.nan_to_num(enhanced_real, nan=0.0, posinf=0.0, neginf=0.0)
        enhanced_imag = torch.nan_to_num(enhanced_imag, nan=0.0, posinf=0.0, neginf=0.0)
        enhanced_wav  = stft.inverse(enhanced_real, enhanced_imag, length=T_wav)

        enh_np   = enhanced_wav.cpu().numpy()
        clean_np = clean.cpu().numpy()

        nan_frac = float(np.isnan(enh_np).mean())
        if nan_frac > 0:
            logger.warning(f"NaN in enhanced output: {nan_frac:.1%} of values")
            enh_np = np.nan_to_num(enh_np, nan=0.0)

        metrics = evaluate_batch(enh_np, clean_np, sample_rate=16000)
        all_pesq.extend(metrics["pesq"])
        all_stoi.extend(metrics["stoi"])
        all_sisnr.extend(metrics["sisnr"])

    return {
        "pesq":  float(np.nanmean(all_pesq)),
        "stoi":  float(np.nanmean(all_stoi)),
        "sisnr": float(np.nanmean(all_sisnr)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg, resume_path=args.resume)