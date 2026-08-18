"""The single place the leakage boundary is defined.

Flags are recorded at line crossings, so the lap that *ends* at a caution onset
was partly run after the flag flew: its late sectors already contain the
slowdown. A model that reads that lap will appear to predict the caution it is
actually observing. Every feature therefore reads only observations published at
or before ``completed_before(t)``.

``lag_s`` above zero buys extra margin at the cost of freshness. Zero is correct
for features built from sector entries, which are observed as they happen; a
full lap of lag is appropriate for features built from completed lap times.
"""

from __future__ import annotations

TYPICAL_LAP_S = 110.0


def completed_before(t: float, lag_s: float = 0.0) -> float:
    """Latest observation time a feature evaluated at ``t`` may use."""
    return t - lag_s
