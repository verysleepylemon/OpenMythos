"""Offline demo for Clue Quest.

Usage:
    python -m scripts.clue_quest_demo --html-file examples/clue_quest_sample.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.clue_quest import (  # noqa: E402
    decode_clue,
    estimate_creator_payout,
    extract_public_clues,
    validate_target,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a safe Clue Quest decode demo.")
    parser.add_argument("--html-file", type=Path, required=True)
    parser.add_argument("--url", default="https://example.com/challenge")
    parser.add_argument("--allow-host", action="append", default=["example.com"])
    parser.add_argument("--consent", action="store_true")
    parser.add_argument("--max-loops", type=int, default=4)
    args = parser.parse_args()

    host = validate_target(
        url=args.url,
        allow_hosts=args.allow_host,
        consent_confirmed=args.consent,
    )
    html_text = args.html_file.read_text(encoding="utf-8")

    clues = extract_public_clues(html_text)
    print(f"[clue_quest] host={host} clues_found={len(clues)}")

    for idx, clue in enumerate(clues, start=1):
        report = decode_clue(clue, max_loops=args.max_loops)
        print(f"\n[{idx}] original: {clue}")
        for step in report.steps:
            print(
                f"  - {step.method:6s} conf={step.confidence:.2f} "
                f"=> {step.output_text}"
            )
        print(f"  final: {report.final_text}")
        print(f"  stop:  {report.stopped_reason}")

    payout = estimate_creator_payout(
        active_users=1000,
        paid_conversion_rate=0.06,
        monthly_arppu_usd=12.0,
        creator_revenue_share=0.30,
    )
    print(f"\n[clue_quest] example creator payout estimate: ${payout:.2f}/month")


if __name__ == "__main__":
    main()
