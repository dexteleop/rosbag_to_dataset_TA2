#!/usr/bin/env python3
# Copyright (c) 2026 Dexteleop Intelligence
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

"""
Review a suspicious_episodes.json report (from detect_suspicious_episodes.py),
confirm with the operator, and delete the confirmed episodes.

Deletion reuses delete_wrong_episodes.remove_episodes, so parquet/video moves
and meta/*.jsonl reindexing stay in one place.

Usage example:
    python scripts/confirm_and_delete_episodes.py --dataset /path/to/dataset

    # Non-interactive (for scripted pipelines): delete every reported episode
    python scripts/confirm_and_delete_episodes.py --dataset /path/to/dataset --yes

    # Only act on tracking failures
    python scripts/confirm_and_delete_episodes.py --dataset /path/to/dataset --category 3

    # Also drop anything shorter than 300 frames or longer than 6000
    python scripts/confirm_and_delete_episodes.py --dataset /path/to/dataset \\
      --min-length 300 --max-length 6000

    # Web/pipeline: delete an explicit episode list (skips report-based selection)
    python scripts/confirm_and_delete_episodes.py \\
      --dataset /path/to/dataset --episodes-file approved.json --yes
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from delete_wrong_episodes import load_jsonl, remove_episodes  # noqa: E402

CATEGORY_LABELS = {1: "command", 2: "length", 3: "tracking"}
CATEGORY_LENGTH = 2
# the categories that make an episode unfit; 2 (length) is informational only
UNFIT_CATEGORIES = [1, 3]


def load_report(report_path: Path) -> dict:
    if not report_path.exists():
        raise FileNotFoundError(
            f"Report not found: {report_path}. Run detect_suspicious_episodes.py first."
        )
    return json.loads(report_path.read_text(encoding="utf-8"))


def load_episodes_file(path: Path) -> list[int]:
    """Load approved episode indices from JSON.

    Accepted shapes:
      {"episodes": [21, 77, 215]}
      [21, 77, 215]
    """
    if not path.exists():
        raise FileNotFoundError(f"Episodes file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        episodes = data.get("episodes")
        if episodes is None:
            raise ValueError(f"{path} must contain an 'episodes' list")
    elif isinstance(data, list):
        episodes = data
    else:
        raise ValueError(f"Unsupported episodes-file format in {path}")
    indices = sorted({int(x) for x in episodes})
    if not indices:
        raise ValueError(f"No episode indices found in {path}")
    return indices


def select_candidates(report: dict, categories: list[int]) -> list[dict]:
    candidates = [
        r for r in report["suspicious_episodes"].values()
        if any(c in categories for c in r["categories"])
    ]
    return sorted(candidates, key=lambda r: r["episode_index"])


def select_by_length(dataset: Path, report: dict, min_length: int | None,
                     max_length: int | None) -> list[dict]:
    """Episodes outside [min_length, max_length] frames.

    Read from meta/episodes.jsonl rather than the report, because the report only lists
    episodes that tripped something -- an episode inside the report's own length quantiles
    is absent from it and would otherwise be unreachable.
    """
    if min_length is None and max_length is None:
        return []
    reported = report.get("suspicious_episodes", {})
    out = []
    for ep in load_jsonl(dataset / "meta" / "episodes.jsonl"):
        n = ep.get("length", 0)
        if (min_length is not None and n < min_length) or \
           (max_length is not None and n > max_length):
            idx = ep["episode_index"]
            hit = reported.get(str(idx))
            if hit is None:
                hit = {"episode_index": idx, "length": n, "categories": [CATEGORY_LENGTH],
                       "verdict": "pass", "reason": "length"}
            out.append(hit)
    return out


def merge_candidates(*groups: list[dict]) -> tuple[list[dict], list[dict]]:
    """Union by episode_index. Returns (deletable, skipped).

    An episode the detector could not read carries verdict "error": its parquet is missing
    or unreadable, so it was never scored and deleting it would reindex the dataset around
    a file that is not there.
    """
    seen: dict[int, dict] = {}
    for group in groups:
        for r in group:
            seen.setdefault(r["episode_index"], r)
    rows = sorted(seen.values(), key=lambda r: r["episode_index"])
    return ([r for r in rows if r.get("verdict") != "error"],
            [r for r in rows if r.get("verdict") == "error"])


def print_candidates(candidates: list[dict]) -> None:
    print(f"{'episode':>7} {'verdict':>7} {'cats':>17} {'length':>8} "
          f"{'cmd_x':>7} {'trk_x':>7}  reason")
    def num(r: dict, key: str) -> str:
        v = r.get(key)
        return f"{v:.2f}" if v is not None else "-"

    for r in candidates:
        cats = ",".join(CATEGORY_LABELS.get(c, str(c)) for c in r["categories"])
        print(f"{r['episode_index']:>7} {r.get('verdict', '-'):>7} {cats:>17} "
              f"{r['length']:>8} {num(r, 'action_max_delta_ratio'):>7} "
              f"{num(r, 'track_max_err_ratio'):>7}  {r.get('reason', '')}")


def confirm(prompt: str) -> bool:
    answer = input(f"{prompt} [y/N]: ").strip().lower()
    return answer in ("y", "yes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Confirm and delete suspicious episodes reported by detect_suspicious_episodes.py."
    )
    parser.add_argument("--dataset", required=True, type=Path, help="Dataset root directory.")
    parser.add_argument("--report", type=Path, default=None,
                         help="Path to suspicious_episodes.json (default: <dataset>/meta/suspicious_episodes.json).")
    parser.add_argument("--category", type=int, nargs="+", choices=[1, 2, 3],
                         default=UNFIT_CATEGORIES,
                         help="Which categories to act on: 1=command, 2=length, 3=tracking "
                              "(default: 1 3, the two that make an episode unfit; 2 is "
                              "reported but does not affect the verdict).")
    parser.add_argument("--min-length", type=int, default=None,
                         help="Also select episodes shorter than this many frames.")
    parser.add_argument("--max-length", type=int, default=None,
                         help="Also select episodes longer than this many frames.")
    parser.add_argument("--episodes-file", type=Path, default=None,
                         help="JSON file with explicit episode indices to delete "
                              '(e.g. {"episodes": [21, 77]}). Skips report-based selection.')
    parser.add_argument("--yes", action="store_true",
                         help="Skip the confirmation prompt and delete every selected episode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset}")

    if args.episodes_file is not None:
        episode_indices = load_episodes_file(args.episodes_file)
        print(f"Dataset: {dataset}")
        print(f"{len(episode_indices)} episode(s) from {args.episodes_file}: {episode_indices}")
        if not args.yes and not confirm(f"\nDelete these {len(episode_indices)} episode(s)?"):
            print("Aborted: no episodes deleted.")
            return
        remove_episodes(dataset, episode_indices)
        return

    report_path = args.report or (dataset / "meta" / "suspicious_episodes.json")
    report = load_report(report_path)

    candidates, unreadable = merge_candidates(
        select_candidates(report, args.category),
        select_by_length(dataset, report, args.min_length, args.max_length),
    )
    bounds = ""
    if args.min_length is not None or args.max_length is not None:
        bounds = f", length outside [{args.min_length}, {args.max_length}]"
    if not candidates:
        print(f"No episodes selected (categories {args.category}{bounds}) in {report_path}.")
        return

    print(f"Dataset: {dataset}")
    print(f"{len(candidates)} episode(s) selected (categories {args.category}{bounds}):")
    print_candidates(candidates)

    if unreadable:
        idx = ", ".join(str(r["episode_index"]) for r in unreadable)
        print(f"\nSkipped {len(unreadable)} unreadable episode(s): {idx}")

    passing = [r for r in candidates if r.get("verdict") == "pass"]
    if passing:
        print(f"\nNote: {len(passing)} of these passed both indicators and are only listed "
              f"for episode length.")

    if not args.yes and not confirm(f"\nDelete these {len(candidates)} episode(s)?"):
        print("Aborted: no episodes deleted.")
        return

    episode_indices = sorted({r["episode_index"] for r in candidates})
    remove_episodes(dataset, episode_indices)


if __name__ == "__main__":
    main()
