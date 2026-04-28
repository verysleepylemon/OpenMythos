"""Clue Quest: a safe, puzzle-first clue decoder.

This module is intentionally designed for legal challenge content only:
- public puzzle pages
- CTF-style challenges
- websites where explicit owner consent is confirmed

It does not perform crawling or bypass controls. You pass HTML content directly
(or from your own fetch pipeline), and this module extracts and decodes clues.
"""

from __future__ import annotations

import base64
import binascii
import codecs
import html
import re
from dataclasses import dataclass
from typing import Callable, Sequence
from urllib.parse import unquote, urlparse

_SIGNAL_MARKERS = (
    "clue:",
    "flag{",
    "hint",
    "treasure",
    "token",
    "key",
)


@dataclass(frozen=True)
class DecodeStep:
    """One decoding transform applied during iterative decoding."""

    method: str
    input_text: str
    output_text: str
    confidence: float


@dataclass(frozen=True)
class DecodeReport:
    """Final decode result after looped transform attempts."""

    original_text: str
    final_text: str
    steps: list[DecodeStep]
    stopped_reason: str


def _printable_ratio(text: str) -> float:
    if not text:
        return 0.0
    printable = sum(1 for ch in text if ch.isprintable() or ch in "\n\r\t")
    return printable / len(text)


def _language_score(text: str) -> float:
    if not text:
        return 0.0
    letters = sum(1 for ch in text if ch.isalpha())
    separators = sum(1 for ch in text if ch in " _:-{}[]()/")
    return (letters + separators) / max(1, len(text))


def _has_signal(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _SIGNAL_MARKERS)


def _try_base64_decode(text: str) -> tuple[str, float] | None:
    compact = "".join(text.split())
    if not compact:
        return None
    if re.fullmatch(r"[A-Za-z0-9+/=]+", compact) is None:
        return None

    compact += "=" * ((4 - len(compact) % 4) % 4)
    try:
        decoded = base64.b64decode(compact, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return None

    ratio = _printable_ratio(decoded)
    if ratio < 0.85 or decoded == text:
        return None
    return decoded, min(1.0, 0.65 + 0.35 * ratio)


def _try_hex_decode(text: str) -> tuple[str, float] | None:
    compact = text.strip().lower()
    if compact.startswith("0x"):
        compact = compact[2:]
    if re.fullmatch(r"(?:[0-9a-f]{2})+", compact) is None:
        return None

    try:
        decoded = bytes.fromhex(compact).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None

    ratio = _printable_ratio(decoded)
    if ratio < 0.85 or decoded == text:
        return None
    return decoded, min(1.0, 0.6 + 0.4 * ratio)


def _try_rot13_decode(text: str) -> tuple[str, float] | None:
    decoded = codecs.decode(text, "rot_13")
    if decoded == text:
        return None

    before_signal = _has_signal(text)
    after_signal = _has_signal(decoded)
    if after_signal and not before_signal:
        return decoded, 0.9

    delta = _language_score(decoded) - _language_score(text)
    if delta < 0.08:
        return None
    return decoded, min(0.85, 0.55 + delta)


def _try_url_decode(text: str) -> tuple[str, float] | None:
    decoded = unquote(text)
    if decoded == text:
        return None

    ratio = _printable_ratio(decoded)
    if ratio < 0.85:
        return None
    return decoded, min(0.8, 0.5 + 0.3 * ratio)


Decoder = Callable[[str], tuple[str, float] | None]
_DECODERS: tuple[tuple[str, Decoder], ...] = (
    ("base64", _try_base64_decode),
    ("hex", _try_hex_decode),
    ("url", _try_url_decode),
    ("rot13", _try_rot13_decode),
)


def decode_clue(clue_text: str, max_loops: int = 4) -> DecodeReport:
    """Iteratively decode an encoded clue with a recurrent-loop style process."""

    if max_loops < 1:
        raise ValueError("max_loops must be >= 1")

    current = clue_text.strip()
    steps: list[DecodeStep] = []

    for _ in range(max_loops):
        best_name = ""
        best_output = ""
        best_confidence = -1.0

        for name, decoder in _DECODERS:
            result = decoder(current)
            if result is None:
                continue
            output, confidence = result
            if output == current:
                continue
            if confidence > best_confidence:
                best_name = name
                best_output = output
                best_confidence = confidence

        if best_confidence < 0.0:
            return DecodeReport(
                original_text=clue_text,
                final_text=current,
                steps=steps,
                stopped_reason="no_high_confidence_transform",
            )

        steps.append(
            DecodeStep(
                method=best_name,
                input_text=current,
                output_text=best_output,
                confidence=round(best_confidence, 3),
            )
        )
        current = best_output

        if _has_signal(current):
            return DecodeReport(
                original_text=clue_text,
                final_text=current,
                steps=steps,
                stopped_reason="decoded_signal_found",
            )

    return DecodeReport(
        original_text=clue_text,
        final_text=current,
        steps=steps,
        stopped_reason="max_loops_reached",
    )


def extract_public_clues(html_text: str) -> list[str]:
    """Extract potential clue strings from common public HTML containers."""

    patterns = (
        r"<meta[^>]+name=[\"']clue[\"'][^>]+content=[\"']([^\"']+)[\"']",
        r"data-clue=[\"']([^\"']+)[\"']",
        r"<!--\s*clue:\s*(.*?)\s*-->",
    )

    seen: set[str] = set()
    clues: list[str] = []

    for pattern in patterns:
        for match in re.findall(pattern, html_text, flags=re.IGNORECASE | re.DOTALL):
            candidate = html.unescape(match).strip()
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            clues.append(candidate)

    return clues


def validate_target(
    url: str,
    allow_hosts: Sequence[str],
    consent_confirmed: bool,
) -> str:
    """Validate that the target is explicitly allowed and consented.

    Returns the normalized hostname when valid.
    """

    if not consent_confirmed:
        raise ValueError("Explicit consent is required before decoding website clues.")

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Target URL must use http or https.")

    host = (parsed.hostname or "").lower().strip()
    if not host:
        raise ValueError("Target URL must include a hostname.")

    normalized_allow = [h.lower().strip() for h in allow_hosts if h.strip()]
    if not normalized_allow:
        raise ValueError("allow_hosts cannot be empty.")

    def is_allowed(candidate: str) -> bool:
        return any(
            candidate == allowed or candidate.endswith("." + allowed)
            for allowed in normalized_allow
        )

    if not is_allowed(host):
        raise ValueError(
            f"Target host '{host}' is not in allow_hosts: {', '.join(normalized_allow)}"
        )

    return host


def estimate_creator_payout(
    active_users: int,
    paid_conversion_rate: float,
    monthly_arppu_usd: float,
    creator_revenue_share: float,
) -> float:
    """Estimate monthly creator payout for a clue-quest marketplace."""

    if active_users < 0:
        raise ValueError("active_users must be >= 0")
    if not 0.0 <= paid_conversion_rate <= 1.0:
        raise ValueError("paid_conversion_rate must be in [0, 1]")
    if monthly_arppu_usd < 0.0:
        raise ValueError("monthly_arppu_usd must be >= 0")
    if not 0.0 <= creator_revenue_share <= 1.0:
        raise ValueError("creator_revenue_share must be in [0, 1]")

    gross = active_users * paid_conversion_rate * monthly_arppu_usd
    payout = gross * creator_revenue_share
    return round(float(payout), 2)
