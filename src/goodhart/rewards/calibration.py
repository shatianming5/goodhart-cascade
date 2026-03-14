"""Calibration penalty reward for verl GRPO training."""

from __future__ import annotations

import re


def compute_calibration_penalty(confidence: float, outcome: float) -> float:
    """Returns -|confidence - outcome|, range [-1, 0]. 0 = perfect calibration."""
    return -abs(confidence - outcome)


def extract_confidence(text: str, default: float = 0.5) -> float:
    """Extract a confidence value (0.0-1.0) from model text output.

    Looks for patterns like:
    - "confidence: 0.85"
    - "Confidence: 85%"
    - "I am 90% confident"
    - "0.75"
    """
    if not text:
        return default

    # Pattern: "confidence: 0.85" or "confidence: 85%"
    match = re.search(r"confidence[:\s]+(\d+(?:\.\d+)?)\s*%?", text, re.IGNORECASE)
    if match:
        val = float(match.group(1))
        if val > 1.0:
            val /= 100.0
        return max(0.0, min(1.0, val))

    # Pattern: "X% confident"
    match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*confident", text, re.IGNORECASE)
    if match:
        val = float(match.group(1)) / 100.0
        return max(0.0, min(1.0, val))

    # Pattern: standalone decimal 0.0-1.0 on its own line or after key phrases
    match = re.search(r"(?:^|\n)\s*(0\.\d+|1\.0|0|1)\s*(?:\n|$)", text)
    if match:
        val = float(match.group(1))
        return max(0.0, min(1.0, val))

    return default
