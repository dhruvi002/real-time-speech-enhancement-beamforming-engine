"""
data/simulate_rir.py
--------------------
Room Impulse Response (RIR) simulation using the image-source method.
Uses pyroomacoustics (equivalent to MATLAB's RIR generator / ISM).

Simulates a variety of room geometries and RT60 values (0.2–0.8s)
to train and evaluate robustness across reverberation conditions.

Usage:
    python data/simulate_rir.py --output_dir data/rirs --n_rooms 500
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pyroomacoustics as pra

sys.path.insert(0, str(Path(__file__).parent.parent))
from beamforming.mvdr import build_ula_positions


RT60_RANGE = (0.2, 0.8)

ROOM_DIM_RANGE = {
    "x": (4.0, 10.0),
    "y": (3.0, 8.0),
    "z": (2.5, 3.5),
}

ARRAY_HEIGHT    = 1.2
SOURCE_DIST_RANGE = (0.5, 3.0)
SOURCE_DOA_RANGE  = (30, 150)


def sample_room_params(rng):
    dims = [
        rng.uniform(*ROOM_DIM_RANGE["x"]),
        rng.uniform(*ROOM_DIM_RANGE["y"]),
        rng.uniform(*ROOM_DIM_RANGE["z"]),
    ]
    rt60 = rng.uniform(*RT60_RANGE)
    return {"dims": dims, "rt60": rt60}


def simulate_rir(room_dims, rt60, mic_positions, source_position, sample_rate=16000):
    """
    Simulate per-mic RIRs using image-source method.
    Returns (n_mics, rir_len) float32 array.
    """
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dims)
    e_absorption = float(np.clip(e_absorption, 0.01, 0.99))

    room = pra.ShoeBox(
        room_dims,
        fs=sample_rate,
        materials=pra.Material(e_absorption),
        max_order=max_order,
        ray_tracing=False,
        air_absorption=True,
    )

    room.add_microphone(mic_positions.T)          # (3, n_mics)

    # pyroomacoustics >= 0.8 requires a signal on the source before simulate().
    # A unit impulse is the correct signal for RIR extraction.
    impulse = np.zeros(sample_rate)
    impulse[0] = 1.0
    room.add_source(source_position, signal=impulse)

    room.simulate()

    n_mics = mic_positions.shape[0]
    rir_arrays = [room.rir[m][0] for m in range(n_mics)]

    if any(r is None or len(r) == 0 for r in rir_arrays):
        raise ValueError("One or more mics returned an empty RIR")

    max_len = max(len(r) for r in rir_arrays)
    rir = np.zeros((n_mics, max_len), dtype=np.float32)
    for m, r in enumerate(rir_arrays):
        rir[m, :len(r)] = r.astype(np.float32)

    return rir


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed=args.seed)
    mic_positions = build_ula_positions(n_mics=args.n_mics, spacing=0.05)

    metadata = []

    for i in tqdm(range(args.n_rooms), desc="Simulating RIRs"):
        params = sample_room_params(rng)
        dims   = params["dims"]
        rt60   = params["rt60"]

        array_center = np.array([dims[0] / 2, dims[1] / 2, ARRAY_HEIGHT])
        mic_pos_abs  = mic_positions + array_center

        doa   = rng.uniform(*SOURCE_DOA_RANGE)
        dist  = rng.uniform(*SOURCE_DIST_RANGE)
        theta = np.deg2rad(doa)
        src_offset  = np.array([dist * np.cos(theta), dist * np.sin(theta), 0.0])
        source_pos  = array_center + src_offset

        margin = 0.1
        source_pos = np.clip(
            source_pos,
            [margin, margin, margin],
            [dims[0] - margin, dims[1] - margin, dims[2] - margin],
        )

        try:
            rir = simulate_rir(
                room_dims=dims,
                rt60=rt60,
                mic_positions=mic_pos_abs,
                source_position=source_pos,
                sample_rate=args.sample_rate,
            )
            fname = f"room_{i:04d}.npy"
            np.save(output_dir / fname, rir)
            metadata.append({
                "file":         fname,
                "room_dims":    dims,
                "rt60":         rt60,
                "doa_deg":      doa,
                "source_dist_m": dist,
            })
        except Exception as e:
            tqdm.write(f"  [WARN] Room {i} failed: {e}")

    with open(output_dir / "rir_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Simulated {len(metadata)}/{args.n_rooms} RIRs → {output_dir}")
    print(f"  RT60 range: {RT60_RANGE[0]:.1f}–{RT60_RANGE[1]:.1f}s")
    print(f"  Mic array:  {args.n_mics}-element ULA, 5cm spacing")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",  default="data/rirs")
    parser.add_argument("--n_rooms",     type=int, default=500)
    parser.add_argument("--n_mics",      type=int, default=4)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()
    main(args)