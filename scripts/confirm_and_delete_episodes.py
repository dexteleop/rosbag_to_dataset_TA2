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

    # Only act on category-1 (jump) episodes, leave category-2 (length) alone
    python scripts/confirm_and_delete_episodes.py --dataset /path/to/dataset --category 1
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from delete_wrong_episodes import remove_episodes  # noqa: E402

CATEGORY_LABELS = {1: "jump", 2: "length"}


def load_report(report_path: Path) -> dict:
    if not report_path.exists():
        raise FileNotFoundError(
            f"Report not found: {report_path}. Run detect_suspicious_episodes.py first."
        )
    return json.loads(report_path.read_text(encoding="utf-8"))


def select_candidates(report: dict, categories: list[int]) -> list[dict]:
    candidates = [
        r for r in report["suspicious_episodes"].values()
        if any(c in categories for c in r["categories"])
    ]
    return sorted(candidates, key=lambda r: r["episode_index"])


def print_candidates(candidates: list[dict]) -> None:
    print(f"{'episode':>7} {'cats':>8} {'length':>8} {'action_max':>11} {'state_max':>10}")
    for r in candidates:
        cats = ",".join(CATEGORY_LABELS.get(c, str(c)) for c in r["categories"])
        action_max = r.get("action_max_delta")
        state_max = r.get("state_max_delta")
        action_str = f"{action_max:.3f}" if action_max is not None else "n/a"
        state_str = f"{state_max:.3f}" if state_max is not None else "n/a"
        print(f"{r['episode_index']:>7} {cats:>8} {r['length']:>8} {action_str:>11} {state_str:>10}")


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
    parser.add_argument("--category", type=int, nargs="+", choices=[1, 2], default=[1, 2],
                         help="Which categories to act on: 1=jump, 2=length (default: both).")
    parser.add_argument("--yes", action="store_true",
                         help="Skip the confirmation prompt and delete every selected episode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset}")

    report_path = args.report or (dataset / "meta" / "suspicious_episodes.json")
    report = load_report(report_path)

    candidates = select_candidates(report, args.category)
    if not candidates:
        print(f"No suspicious episodes found for categories {args.category} in {report_path}.")
        return

    print(f"Dataset: {dataset}")
    print(f"{len(candidates)} suspicious episode(s) selected (categories {args.category}):")
    print_candidates(candidates)

    if not args.yes and not confirm(f"\nDelete these {len(candidates)} episode(s)?"):
        print("Aborted: no episodes deleted.")
        return

    episode_indices = sorted({r["episode_index"] for r in candidates})
    remove_episodes(dataset, episode_indices)


if __name__ == "__main__":
    main()
