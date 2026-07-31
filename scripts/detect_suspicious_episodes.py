#!/usr/bin/env python3
# Copyright (c) 2026 Dexteleop Intelligence
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

"""
Scan a LeRobot dataset for suspicious episodes and write a JSON report.

Two independent checks flag an episode; either one is enough:

  Category 1 (jump): frame-to-frame jumps in the 14 arm joint-position
  columns of `action` or `observation.state` (indices 0-6 left arm, 8-14
  right arm; gripper columns 7 and 15 are excluded). `action` is the
  commanded target and is allowed a larger single-frame step (teleop step
  commands are legitimate); `observation.state` is the encoder-measured
  arm pose and cannot physically teleport, so it gets a tighter threshold.

  Category 2 (length): episode frame count outside the
  [--length-low-quantile, --length-high-quantile] range across the
  dataset's episodes (default: 10th/90th percentile), which tends to catch
  episodes that were cut short or left running too long.

Usage example:
    python scripts/detect_suspicious_episodes.py --dataset /path/to/dataset
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from delete_wrong_episodes import find_parquet_path, load_jsonl  # noqa: E402

DEFAULT_CHUNK_SIZE = 1000

CATEGORY_JUMP = 1
CATEGORY_LENGTH = 2

# Joint-position columns only: left arm 0-6, right arm 8-14 (7 and 15 are
# the grippers and are excluded per the jump-detection spec).
ARM_COLS = list(range(0, 7)) + list(range(8, 15))

DEFAULT_ACTION_THRESH = 0.25
DEFAULT_STATE_THRESH = 0.15
DEFAULT_LENGTH_LOW_QUANTILE = 0.1
DEFAULT_LENGTH_HIGH_QUANTILE = 0.9


def max_delta_with_location(arr: np.ndarray, cols: list[int]) -> tuple[float, int, int]:
    """Max abs frame-to-frame delta over `cols`, plus (frame, col) it occurs at.

    `frame` is the index of the later of the two frames in the diff (i.e. the
    frame where the jump lands). Returns (0.0, -1, -1) if there are fewer
    than 2 rows.
    """
    if arr.shape[0] < 2:
        return 0.0, -1, -1
    deltas = np.abs(np.diff(arr[:, cols], axis=0))
    flat_idx = int(np.argmax(deltas))
    row, col = np.unravel_index(flat_idx, deltas.shape)
    return float(deltas[row, col]), int(row) + 1, int(cols[col])


def check_jump(
    dataset: Path, episode_index: int, chunk_size: int,
    action_thresh: float, state_thresh: float,
) -> dict:
    """Return jump-check metrics for one episode; `flagged` is True on either rule firing."""
    path = find_parquet_path(dataset, episode_index, chunk_size)
    if path is None:
        return {"error": "parquet not found", "flagged": False}

    df = pd.read_parquet(path, columns=["action", "observation.state"])
    action_arr = np.stack(df["action"].to_numpy())
    state_arr = np.stack(df["observation.state"].to_numpy())

    action_max, action_frame, action_col = max_delta_with_location(action_arr, ARM_COLS)
    state_max, state_frame, state_col = max_delta_with_location(state_arr, ARM_COLS)

    flagged = action_max > action_thresh or state_max > state_thresh
    return {
        "flagged": flagged,
        "action_max_delta": action_max,
        "action_max_delta_frame": action_frame,
        "action_max_delta_col": action_col,
        "state_max_delta": state_max,
        "state_max_delta_frame": state_frame,
        "state_max_delta_col": state_col,
    }


def build_report(
    dataset: Path,
    action_thresh: float,
    state_thresh: float,
    length_low_quantile: float,
    length_high_quantile: float,
) -> dict:
    meta_dir = dataset / "meta"
    info_path = meta_dir / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"info.json not found: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    chunk_size = info.get("chunks_size", DEFAULT_CHUNK_SIZE)

    episodes = load_jsonl(meta_dir / "episodes.jsonl")
    if not episodes:
        raise RuntimeError("episodes.jsonl is empty or does not exist.")
    episodes = sorted(episodes, key=lambda e: e["episode_index"])

    lengths = np.array([ep.get("length", 0) for ep in episodes], dtype=float)
    length_low = float(np.quantile(lengths, length_low_quantile))
    length_high = float(np.quantile(lengths, length_high_quantile))

    episode_reports = {}
    for ep in episodes:
        idx = ep["episode_index"]
        length = ep.get("length", 0)
        categories = []

        jump = check_jump(dataset, idx, chunk_size, action_thresh, state_thresh)
        if jump["flagged"]:
            categories.append(CATEGORY_JUMP)

        length_flagged = length < length_low or length > length_high
        if length_flagged:
            categories.append(CATEGORY_LENGTH)

        if categories:
            episode_reports[str(idx)] = {
                "episode_index": idx,
                "categories": categories,
                "length": length,
                **{k: v for k, v in jump.items() if k != "flagged"},
            }

    return {
        "dataset": str(dataset),
        "thresholds": {
            "action_thresh": action_thresh,
            "state_thresh": state_thresh,
            "length_low_quantile": length_low_quantile,
            "length_high_quantile": length_high_quantile,
        },
        "total_episodes": len(episodes),
        "length_low_bound": length_low,
        "length_high_bound": length_high,
        "suspicious_episodes": episode_reports,
    }


def print_summary(report: dict) -> None:
    suspicious = report["suspicious_episodes"]
    total = report["total_episodes"]
    n_jump = sum(1 for r in suspicious.values() if CATEGORY_JUMP in r["categories"])
    n_length = sum(1 for r in suspicious.values() if CATEGORY_LENGTH in r["categories"])

    print(f"Dataset: {report['dataset']}")
    print(f"Total episodes: {total}")
    print(
        f"Length bounds (quantiles {report['thresholds']['length_low_quantile']}/"
        f"{report['thresholds']['length_high_quantile']}): "
        f"{report['length_low_bound']:.1f} .. {report['length_high_bound']:.1f} frames"
    )
    print(f"Suspicious episodes: {len(suspicious)} (category 1/jump: {n_jump}, category 2/length: {n_length})")

    if not suspicious:
        return

    print(f"{'episode':>7} {'cats':>6} {'length':>8} {'action_max':>11} {'state_max':>10}")
    for idx in sorted(suspicious, key=int):
        r = suspicious[idx]
        cats = ",".join(str(c) for c in r["categories"])
        action_max = r.get("action_max_delta")
        state_max = r.get("state_max_delta")
        action_str = f"{action_max:.3f}" if action_max is not None else "n/a"
        state_str = f"{state_max:.3f}" if state_max is not None else "n/a"
        print(f"{idx:>7} {cats:>6} {r['length']:>8} {action_str:>11} {state_str:>10}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect suspicious episodes (arm jump anomalies and length outliers) in a LeRobot dataset."
    )
    parser.add_argument("--dataset", required=True, type=Path, help="Dataset root directory.")
    parser.add_argument("--action-thresh", type=float, default=DEFAULT_ACTION_THRESH,
                         help=f"Max allowed per-frame |delta action| on arm joint columns, rad (default: {DEFAULT_ACTION_THRESH}).")
    parser.add_argument("--state-thresh", type=float, default=DEFAULT_STATE_THRESH,
                         help=f"Max allowed per-frame |delta state| on arm joint columns, rad (default: {DEFAULT_STATE_THRESH}).")
    parser.add_argument("--length-low-quantile", type=float, default=DEFAULT_LENGTH_LOW_QUANTILE,
                         help=f"Lower episode-length quantile fence (default: {DEFAULT_LENGTH_LOW_QUANTILE}).")
    parser.add_argument("--length-high-quantile", type=float, default=DEFAULT_LENGTH_HIGH_QUANTILE,
                         help=f"Upper episode-length quantile fence (default: {DEFAULT_LENGTH_HIGH_QUANTILE}).")
    parser.add_argument("--output", type=Path, default=None,
                         help="Report output path (default: <dataset>/meta/suspicious_episodes.json).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset}")

    report = build_report(
        dataset,
        args.action_thresh,
        args.state_thresh,
        args.length_low_quantile,
        args.length_high_quantile,
    )

    output_path = args.output or (dataset / "meta" / "suspicious_episodes.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print_summary(report)
    print(f"\nReport written to: {output_path}")


if __name__ == "__main__":
    main()
