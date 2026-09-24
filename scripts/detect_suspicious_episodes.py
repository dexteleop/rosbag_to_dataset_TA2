#!/usr/bin/env python3
# Copyright (c) 2026 Dexteleop Intelligence
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

"""
Scan a LeRobot dataset for episodes unfit to train on and write a JSON report.

Two independent indicators over the 14 arm joints. Either one fires and the episode is
rejected; the verdict is binary and no frames are trimmed.

  command   peak single-frame step in `action` above that joint's reject line
  tracking  |action - observation.state| above that joint's reject line for TRACK_HOLD_S

Report the two rates separately: a command fault is an upstream teleop/IK problem, a
tracking fault is an arm to repair. `tracking_pass_rate` in the summary is 1 - tracking rate.

Thresholds are the p99 of the distribution of each episode's maximum, calibrated on 18.3M
frames across 18 datasets at 45 fps (2026-09), per joint and shared between the arms. The
command line is a rate and scales by 45/fps; the tracking line is a magnitude and does not.

Reference corpus, 10,830 episodes: command 6.34%, tracking 3.54%, rejected 8.08%,
pass 91.92%, tracking pass 96.46%.

Episodes outside [--length-low-quantile, --length-high-quantile] are listed with category 2
but do not affect the verdict.

Usage:
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

CATEGORY_COMMAND = 1
CATEGORY_LENGTH = 2
CATEGORY_TRACKING = 3

# Left arm 0-6, right arm 8-14. Columns 7 and 15 are the force-commanded grippers, where a
# position jump or tracking error is meaningless.
ARM_COLS = list(range(0, 7)) + list(range(8, 15))
# Column -> joint number 1..7, so both arms share one threshold per joint.
COL_TO_JOINT = {c: (c % 8) + 1 for c in ARM_COLS}

CALIBRATION_FPS = 45.0

# Largest tolerated single-frame command step, rad per frame at CALIBRATION_FPS.
CMD_REJECT = {
    1: 0.1849,
    2: 0.2598,
    3: 0.2974,
    4: 0.2378,
    5: 0.3776,
    6: 0.3359,
    7: 0.3054,
}
# Largest tolerated sustained tracking error, rad. Not a rate, so not scaled by fps.
TRACK_REJECT = {
    1: 0.4127,
    2: 0.4401,
    3: 0.5058,
    4: 0.5660,
    5: 0.8101,
    6: 0.8172,
    7: 0.6185,
}
# In seconds, so it transfers to any frame rate. 0.09 s is 4 frames at 45 fps.
DEFAULT_TRACK_HOLD_S = 0.09

DEFAULT_LENGTH_LOW_QUANTILE = 0.1
DEFAULT_LENGTH_HIGH_QUANTILE = 0.9


def hold_frames(track_hold_s: float, fps: float) -> int:
    """The hold in frames. Rounded, not truncated: at 30 fps 0.09 s is 2.7 frames, and
    truncating would silently make the check 33% more sensitive there than at 45 fps."""
    return max(2, round(track_hold_s * fps))


def longest_run(mask: np.ndarray) -> tuple[int, int]:
    """(length, start) of the longest stretch of True. (0, -1) if there is none."""
    if not mask.any():
        return 0, -1
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    starts, ends = edges[0::2], edges[1::2]
    k = int(np.argmax(ends - starts))
    return int(ends[k] - starts[k]), int(starts[k])


def check_episode(dataset: Path, episode_index: int, chunk_size: int, fps: float,
                  track_hold_s: float) -> dict:
    """Score one episode against both indicators.

    Field names follow the original report schema: `action_max_delta` is the peak
    single-frame command step, which is exactly what indicator 1 tests. `state_max_delta`
    is kept for display only -- no indicator reads it.
    """
    path = find_parquet_path(dataset, episode_index, chunk_size)
    if path is None:
        return {"error": "parquet not found", "cmd_fail": False, "track_fail": False,
                "categories": []}
    try:
        df = pd.read_parquet(path, columns=["action", "observation.state"])
        action = np.stack(df["action"].to_numpy())
        state = np.stack(df["observation.state"].to_numpy())
    except Exception as exc:  # noqa: BLE001 - a broken file must not read as a pass
        return {"error": f"{type(exc).__name__}: {exc}", "cmd_fail": False,
                "track_fail": False, "categories": []}
    n = len(action)

    scale = CALIBRATION_FPS / fps if fps else 1.0  # a step is a rate; the error is not
    hold = hold_frames(track_hold_s, fps)

    cmd_fail = track_fail = False
    categories: set[int] = set()
    evaluated = 0
    best_cmd: dict = {}
    best_state: dict = {}
    best_track: dict = {}

    for col in ARM_COLS:
        joint = COL_TO_JOINT[col]
        cmd = action[:, col].astype(float)
        sta = state[:, col].astype(float)
        if not np.any(cmd) or not np.any(sta):
            continue  # channel not recorded in this episode
        evaluated += 1

        # --- indicator 1: peak single-frame command step -----------------------------
        limit = CMD_REJECT[joint] * scale
        if n >= 2:
            steps = np.abs(np.diff(cmd))
            f = int(np.argmax(steps))
            v = float(steps[f])
            if v / limit > best_cmd.get("ratio", -1.0):
                best_cmd = {"value": v, "frame": f + 1, "col": col, "joint": joint,
                            "thresh": limit, "ratio": v / limit}
            if v > limit:
                cmd_fail = True
                categories.add(CATEGORY_COMMAND)

            ssteps = np.abs(np.diff(sta))
            sf = int(np.argmax(ssteps))
            if float(ssteps[sf]) > best_state.get("value", -1.0):
                best_state = {"value": float(ssteps[sf]), "frame": sf + 1, "col": col}

        # --- indicator 2: tracking error held above the line -------------------------
        level = TRACK_REJECT[joint]
        err = np.abs(cmd - sta)
        run, start = longest_run(err > level)
        peak = float(err.max())
        trips = run >= hold
        # a joint that actually trips outranks one that merely came close, so the reported
        # run_s always belongs to the joint responsible for the verdict
        key = (trips, peak / level)
        if key > (best_track.get("trips", False), best_track.get("ratio", -1.0)):
            best_track = {"value": peak, "frame": start if run else int(np.argmax(err)),
                          "col": col, "joint": joint, "thresh": level,
                          "ratio": peak / level, "run_frames": run,
                          "run_s": run / fps if fps else 0.0, "trips": trips}
        if trips:
            track_fail = True
            categories.add(CATEGORY_TRACKING)

    if not evaluated:
        return {"error": "no arm joint channel recorded", "cmd_fail": False,
                "track_fail": False, "categories": []}

    return {
        "cmd_fail": cmd_fail,
        "track_fail": track_fail,
        "categories": sorted(categories),
        "track_hold_frames": hold,
        "joints_evaluated": evaluated,
        # original field names, same meaning
        "action_max_delta": best_cmd.get("value"),
        "action_max_delta_frame": best_cmd.get("frame"),
        "action_max_delta_col": best_cmd.get("col"),
        "action_max_delta_thresh": best_cmd.get("thresh"),
        "action_max_delta_ratio": best_cmd.get("ratio"),
        "state_max_delta": best_state.get("value"),
        "state_max_delta_frame": best_state.get("frame"),
        "state_max_delta_col": best_state.get("col"),
        # the tracking indicator, in the same flat style
        "track_max_err": best_track.get("value"),
        "track_max_err_frame": best_track.get("frame"),
        "track_max_err_col": best_track.get("col"),
        "track_max_err_thresh": best_track.get("thresh"),
        "track_max_err_ratio": best_track.get("ratio"),
        "track_run_frames": best_track.get("run_frames"),
        "track_run_s": best_track.get("run_s"),
    }


def build_report(dataset: Path, track_hold_s: float, length_low_quantile: float,
                 length_high_quantile: float) -> dict:
    meta_dir = dataset / "meta"
    info_path = meta_dir / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"info.json not found: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    chunk_size = info.get("chunks_size", DEFAULT_CHUNK_SIZE)
    fps = float(info.get("fps") or CALIBRATION_FPS)

    episodes = load_jsonl(meta_dir / "episodes.jsonl")
    if not episodes:
        raise RuntimeError("episodes.jsonl is empty or does not exist.")
    episodes = sorted(episodes, key=lambda e: e["episode_index"])

    lengths = np.array([ep.get("length", 0) for ep in episodes], dtype=float)
    length_low = float(np.quantile(lengths, length_low_quantile))
    length_high = float(np.quantile(lengths, length_high_quantile))

    episode_reports = {}
    n_cmd = n_track = n_reject = n_error = 0
    reject_indices: list[int] = []
    for ep in episodes:
        idx = ep["episode_index"]
        length = ep.get("length", 0)

        result = check_episode(dataset, idx, chunk_size, fps, track_hold_s)
        categories = list(result["categories"])
        if length < length_low or length > length_high:
            categories.append(CATEGORY_LENGTH)

        if result.get("error"):
            n_error += 1
            episode_reports[str(idx)] = {
                "episode_index": idx, "categories": categories, "length": length,
                "verdict": "error", "reason": result["error"],
                "cmd_fail": False, "track_fail": False,
            }
            continue

        n_cmd += result["cmd_fail"]
        n_track += result["track_fail"]
        rejected = result["cmd_fail"] or result["track_fail"]
        n_reject += rejected
        if rejected:
            reject_indices.append(idx)

        if not categories:
            continue
        reason = ("command+tracking" if result["cmd_fail"] and result["track_fail"]
                  else "command" if result["cmd_fail"]
                  else "tracking" if result["track_fail"] else "")
        episode_reports[str(idx)] = {
            "episode_index": idx,
            "categories": sorted(categories),
            "length": length,
            "verdict": "reject" if rejected else "pass",
            "reason": reason,
            **{k: v for k, v in result.items() if k != "categories"},
        }

    total = len(episodes)
    # rates are over the episodes actually scored; an unreadable one is neither pass nor fail
    scored = total - n_error
    return {
        "dataset": str(dataset),
        "fps": fps,
        "thresholds": {
            # per joint now; the arms share one line, so the key is the joint number 1..7
            "cmd_jump": {str(j): v * (CALIBRATION_FPS / fps if fps else 1.0)
                         for j, v in CMD_REJECT.items()},
            "track_err": {str(j): v for j, v in TRACK_REJECT.items()},
            "track_hold_s": track_hold_s,
            "track_hold_frames": hold_frames(track_hold_s, fps),
            "length_low_quantile": length_low_quantile,
            "length_high_quantile": length_high_quantile,
            "calibrated_fps": CALIBRATION_FPS,
            "source": "18.3M frames / 18 datasets, 2026-09; episode maximum, p99",
        },
        "total_episodes": total,
        "scored_episodes": scored,
        "length_low_bound": length_low,
        "length_high_bound": length_high,
        # everything listed, including length-only outliers that passed both indicators
        "suspicious_episodes": episode_reports,
        # the subset to delete: exactly the episodes with verdict == "reject"
        "reject_episodes": reject_indices,
        "summary": {
            "reject": n_reject,
            "pass": scored - n_reject,
            "error": n_error,
            "command_fail": n_cmd,
            "tracking_fail": n_track,
            "pass_rate": round(100.0 * (scored - n_reject) / scored, 2) if scored else 0.0,
            # the tracking-side rate on its own; the two indicators have different owners
            "tracking_pass_rate": round(100.0 * (scored - n_track) / scored, 2) if scored else 0.0,
        },
    }


def print_summary(report: dict) -> None:
    suspicious = report["suspicious_episodes"]
    total = report["total_episodes"]
    s = report["summary"]
    thr = report["thresholds"]
    n_length = sum(1 for r in suspicious.values() if CATEGORY_LENGTH in r["categories"])

    print(f"Dataset: {report['dataset']}  ({report['fps']:g} fps)")
    print(f"Total episodes: {total}")
    print(
        f"Length bounds (quantiles {thr['length_low_quantile']}/"
        f"{thr['length_high_quantile']}): "
        f"{report['length_low_bound']:.1f} .. {report['length_high_bound']:.1f} frames"
    )
    print(f"Tracking hold: {thr['track_hold_s']}s = {thr['track_hold_frames']} frames")
    print(f"Listed: {len(suspicious)}  "
          f"(1/command: {s['command_fail']}, 2/length: {n_length}, "
          f"3/tracking: {s['tracking_fail']})")
    scored = report.get("scored_episodes", total)
    print(f"Verdict over {scored} scored: "
          f"reject {s['reject']} ({100 * s['reject'] / scored:.1f}%)  "
          f"pass {s['pass']} ({s['pass_rate']:.1f}%)  "
          f"tracking pass {s['tracking_pass_rate']:.1f}%")
    if s.get("error"):
        print(f"UNSCORED: {s['error']} episode(s) could not be read -- "
              f"excluded from the rates above, listed with verdict=error")

    if not suspicious:
        return

    print(f"\n{'episode':>7} {'verdict':>7} {'cats':>6} {'length':>7} "
          f"{'cmd_step':>9} {'cmd_x':>6} {'track_err':>9} {'run_s':>6} {'trk_x':>6}  reason")
    for idx in sorted(suspicious, key=int):
        r = suspicious[idx]
        cats = ",".join(str(c) for c in r["categories"])

        def num(key: str, fmt: str) -> str:
            v = r.get(key)
            return format(v, fmt) if v is not None else "n/a"

        print(f"{idx:>7} {r.get('verdict', '-'):>7} {cats:>6} {r['length']:>7} "
              f"{num('action_max_delta', '.3f'):>9} {num('action_max_delta_ratio', '.2f'):>6} "
              f"{num('track_max_err', '.3f'):>9} {num('track_run_s', '.2f'):>6} "
              f"{num('track_max_err_ratio', '.2f'):>6}  {r.get('reason', '')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect episodes unfit to train on (command faults, tracking "
                    "failures, length outliers) in a LeRobot dataset, using the "
                    "per-joint calibrated reject lines.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, type=Path, help="Dataset root directory.")
    parser.add_argument("--track-hold-s", type=float, default=DEFAULT_TRACK_HOLD_S,
                        help=f"How long a tracking excursion must last, in seconds "
                             f"(default {DEFAULT_TRACK_HOLD_S} = 4 frames at 45 fps).")
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

    report = build_report(dataset, args.track_hold_s,
                          args.length_low_quantile, args.length_high_quantile)

    output_path = args.output or (dataset / "meta" / "suspicious_episodes.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print_summary(report)
    print(f"\nReport written to: {output_path}")


if __name__ == "__main__":
    main()
