"""Tests for the safe Clue Quest feature.

This suite validates a legal, puzzle-first decoder flow and basic monetization
math helpers. It intentionally avoids any network calls.
"""

from __future__ import annotations

import base64
import codecs
import sys
from pathlib import Path

# Ensure imports work when pytest is run from repo root or tests/.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.clue_quest import (  # noqa: E402
    decode_clue,
    estimate_creator_payout,
    extract_public_clues,
    validate_target,
)


def test_decode_clue_base64_roundtrip():
    target = "flag{public_puzzle_win}"
    encoded = base64.b64encode(target.encode("utf-8")).decode("ascii")

    report = decode_clue(encoded, max_loops=4)

    assert report.final_text == target
    assert len(report.steps) >= 1


def test_decode_clue_rot13_roundtrip():
    target = "clue:the_treasure_is_in_plain_sight"
    encoded = codecs.encode(target, "rot_13")

    report = decode_clue(encoded, max_loops=4)

    assert report.final_text == target


def test_extract_public_clues_from_html():
    html = """
    <html>
      <head>
        <meta name="clue" content="U29sdmUgbWU=" />
      </head>
      <body>
        <!-- clue:uryyb_jbeyq -->
        <div data-clue="66756e5f66616374"></div>
      </body>
    </html>
    """

    clues = extract_public_clues(html)

    assert "U29sdmUgbWU=" in clues
    assert "uryyb_jbeyq" in clues
    assert "66756e5f66616374" in clues


def test_validate_target_requires_consent():
    try:
        validate_target(
            url="https://example.com/challenge",
            allow_hosts=["example.com"],
            consent_confirmed=False,
        )
        assert False, "Expected ValueError when consent is not confirmed"
    except ValueError as exc:
        assert "consent" in str(exc).lower()


def test_estimate_creator_payout_math():
    # 1,000 users, 6% paid, $12 ARPPU, creator share 30% => $216
    payout = estimate_creator_payout(
        active_users=1000,
        paid_conversion_rate=0.06,
        monthly_arppu_usd=12.0,
        creator_revenue_share=0.30,
    )

    assert payout == 216.0
