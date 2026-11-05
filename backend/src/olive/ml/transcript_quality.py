"""Heuristic detection of degenerate ASR output (repetition loops / hallucination).

Local ASR engines (Cohere Transcribe, whisper-family) can emit *degenerate*
transcripts on silence, music, or difficult audio: a single phrase repeated
dozens of times, one token looped, or a short fragment echoed until the window
ends. These currently flow downstream as if they were valid speech — they get
summarized and analyzed like any real meeting.

This module scores a transcript's **text** for that degeneracy so the pipeline
can FLAG a suspect transcript for human review. It never decides a transcript is
worthless on its own: the signals are advisory, the result is non-destructive,
and the raw transcript is always kept. All signals are language-agnostic and
need no model — they look only at the shape of the text (how much of it is the
same thing over and over).

Thresholds are deliberately conservative: the goal is to catch egregious loops
(the failure mode that actually happens), not to second-guess genuinely
repetitive but valid meetings. A transcript is flagged only when a signal is far
into degenerate territory, and short transcripts are never flagged (too little
text to judge).
"""

from __future__ import annotations

import re
import zlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

# A transcript shorter than this many words is not judged: brief clips (a
# one-line voice memo, a quick "got it, thanks") are legitimately low-diversity
# and would false-positive against every ratio below.
_MIN_WORDS = 40

# Cost bound: the expensive signals (n-gram Counter, compression) are computed
# over at most this many leading tokens. A repetition loop shows up within the
# first tens of thousands of tokens, so a prefix is a faithful sample; this caps
# a 5-hour cap recording (hundreds of thousands of tokens) to well under a second
# instead of several. ``word_count`` still reflects the FULL transcript.
_MAX_ANALYZE_TOKENS = 100_000

# Egregious-loop thresholds. Each is well past what normal speech produces;
# crossing ANY one flags the transcript. Documented values, not tuned to a
# corpus — raise them if a real meeting ever trips one.
_COMPRESSION_RATIO_MAX = 8.0  # normal prose ~2-3; a looped phrase compresses far harder
_UNIQUE_WORD_RATIO_MIN = 0.10  # <10% distinct words = almost no vocabulary
_CONSECUTIVE_REPEAT_MAX = 20  # same token 20+ times back to back
_PHRASE_COVERAGE_MAX = 0.60  # one repeated phrase covering 60%+ of the text
_PHRASE_MIN_REPEATS = 5  # ...and repeating at least this many times

_WORD_RE = re.compile(r"\w+", re.UNICODE)


@dataclass
class TranscriptQuality:
    """Advisory degeneracy assessment of a transcript. ``suspect`` is a flag for
    review, never a verdict to discard — callers keep the transcript regardless."""

    suspect: bool
    word_count: int
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suspect": self.suspect,
            "wordCount": self.word_count,
            "reasons": list(self.reasons),
            "metrics": dict(self.metrics),
        }


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _compression_ratio(text: str) -> float:
    raw = text.encode("utf-8", "ignore")
    if not raw:
        return 1.0
    comp = zlib.compress(raw, 9)
    return len(raw) / max(1, len(comp))


def _max_consecutive_repeat(tokens: list[str]) -> int:
    best = run = 0
    prev: str | None = None
    for tok in tokens:
        if tok == prev:
            run += 1
        else:
            run = 1
            prev = tok
        best = max(best, run)
    return best


def _dominant_phrase_coverage(tokens: list[str]) -> tuple[float, int, int]:
    """Largest share of the token stream taken up by one repeated contiguous phrase.

    For phrase lengths 1..8, count the most frequent n-gram, treat its total
    token span (``count * n``) as coverage, and keep the best. Returns
    ``(coverage_fraction, phrase_len, repeats)``. This catches both single-token
    loops and multi-word phrase loops (e.g. "thank you for watching" x40).
    """
    total = len(tokens)
    if total == 0:
        return 0.0, 0, 0
    best_cov = 0.0
    best_n = 0
    best_reps = 0
    for n in range(1, 9):
        if total < n * _PHRASE_MIN_REPEATS:
            break
        counts = Counter(
            tuple(tokens[i : i + n]) for i in range(0, total - n + 1)
        )
        phrase, reps = counts.most_common(1)[0]
        if reps < 2:
            continue
        coverage = min(1.0, (reps * n) / total)
        if coverage > best_cov:
            best_cov = coverage
            best_n = n
            best_reps = reps
    return best_cov, best_n, best_reps


def assess_transcript_quality(text: str, *, min_words: int = _MIN_WORDS) -> TranscriptQuality:
    """Score ``text`` for ASR degeneracy. Advisory only — never discards anything.

    Returns a :class:`TranscriptQuality`; ``suspect`` is True when the text is
    long enough to judge AND a degeneracy signal is far into loop territory.
    """
    tokens = _tokens(text or "")
    word_count = len(tokens)
    metrics: dict[str, Any] = {}
    reasons: list[str] = []

    if word_count < min_words:
        metrics["note"] = f"below min_words={min_words}; not assessed"
        return TranscriptQuality(False, word_count, reasons, metrics)

    # Bound cost on very long transcripts: score a leading sample. A loop is
    # present in the sample, and the sample's ratios are representative.
    if word_count > _MAX_ANALYZE_TOKENS:
        tokens = tokens[:_MAX_ANALYZE_TOKENS]
        metrics["analyzedTokens"] = _MAX_ANALYZE_TOKENS
        comp_text = " ".join(tokens)
    else:
        comp_text = text

    sample_n = len(tokens)
    comp = _compression_ratio(comp_text)
    unique_ratio = len(set(tokens)) / sample_n
    consec = _max_consecutive_repeat(tokens)
    coverage, phrase_n, phrase_reps = _dominant_phrase_coverage(tokens)

    metrics.update(
        {
            "compressionRatio": round(comp, 2),
            "uniqueWordRatio": round(unique_ratio, 3),
            "maxConsecutiveRepeat": consec,
            "dominantPhraseCoverage": round(coverage, 3),
            "dominantPhraseLen": phrase_n,
            "dominantPhraseRepeats": phrase_reps,
        }
    )

    if comp >= _COMPRESSION_RATIO_MAX:
        reasons.append(f"compression_ratio {comp:.1f} >= {_COMPRESSION_RATIO_MAX}")
    if unique_ratio <= _UNIQUE_WORD_RATIO_MIN:
        reasons.append(f"unique_word_ratio {unique_ratio:.3f} <= {_UNIQUE_WORD_RATIO_MIN}")
    if consec >= _CONSECUTIVE_REPEAT_MAX:
        reasons.append(f"consecutive_repeat {consec} >= {_CONSECUTIVE_REPEAT_MAX}")
    if coverage >= _PHRASE_COVERAGE_MAX and phrase_reps >= _PHRASE_MIN_REPEATS:
        reasons.append(
            f"repeated_phrase (len {phrase_n}, x{phrase_reps}) covers "
            f"{coverage:.0%} >= {_PHRASE_COVERAGE_MAX:.0%}"
        )

    return TranscriptQuality(bool(reasons), word_count, reasons, metrics)
