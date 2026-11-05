"""Tests for the ASR degeneracy / hallucination-loop detector.

The detector is advisory: it must flag egregious repetition loops (the real ASR
failure mode) while leaving normal, varied speech — even short or mildly
repetitive speech — unflagged.
"""

from olive.ml.transcript_quality import assess_transcript_quality

# A normal, varied meeting excerpt well over the min-words floor.
NORMAL = (
    "Okay so let's start with the quarterly numbers. Revenue came in at about "
    "two point three million which is up twelve percent over last quarter. The "
    "biggest driver was the enterprise segment, though we also saw healthy growth "
    "in self-serve. Sarah, can you walk us through the churn picture? Sure, churn "
    "ticked down to four percent, mostly because the onboarding changes landed. We "
    "still have work to do on the mid-market cohort, but the trend is encouraging. "
    "Great, and what about hiring for next quarter? We have three open roles on "
    "engineering and two on design that we want to fill before the launch."
)


def test_normal_transcript_not_suspect():
    q = assess_transcript_quality(NORMAL)
    assert q.suspect is False
    assert q.reasons == []
    assert q.word_count > 40


def test_empty_not_suspect():
    q = assess_transcript_quality("")
    assert q.suspect is False
    assert q.word_count == 0


def test_short_repetitive_not_suspect():
    # Short clips are legitimately low-diversity; never flag them.
    q = assess_transcript_quality("thanks thanks thanks okay bye")
    assert q.suspect is False


def test_repeated_phrase_loop_is_suspect():
    text = ("thank you for watching this video " * 40).strip()
    q = assess_transcript_quality(text)
    assert q.suspect is True
    assert any("repeated_phrase" in r or "compression_ratio" in r for r in q.reasons)


def test_single_token_loop_is_suspect():
    text = "yeah " * 80
    q = assess_transcript_quality(text)
    assert q.suspect is True
    assert q.metrics["maxConsecutiveRepeat"] >= 20


def test_low_vocabulary_loop_is_suspect():
    # Two words alternating for a long stretch: high compression, tiny vocab.
    text = "okay so " * 60
    q = assess_transcript_quality(text)
    assert q.suspect is True


def test_metrics_present_for_assessed_text():
    q = assess_transcript_quality(NORMAL)
    for key in (
        "compressionRatio",
        "uniqueWordRatio",
        "maxConsecutiveRepeat",
        "dominantPhraseCoverage",
    ):
        assert key in q.metrics


def test_to_dict_roundtrip():
    q = assess_transcript_quality("yeah " * 80)
    d = q.to_dict()
    assert d["suspect"] is True
    assert isinstance(d["reasons"], list)
    assert "metrics" in d and "wordCount" in d


def test_long_loop_is_capped_and_still_flagged():
    # A 5-hour-style looped transcript: must stay flagged and bound the work to
    # the analyze cap (word_count reflects the full length).
    text = "spam " * 150_000
    q = assess_transcript_quality(text)
    assert q.suspect is True
    assert q.word_count == 150_000
    assert q.metrics["analyzedTokens"] == 100_000


def test_mildly_repetitive_real_speech_not_flagged():
    # Repetition of common function words is normal; must not trip the guard.
    text = (
        "I think that the thing is that we need to think about what the user "
        "actually wants here, and I think the point is that the point of the "
        "feature is to make it easier for the user to find the thing they need. "
        "So I think we should think about the flow and the steps the user takes "
        "and whether the steps make sense for the user in that context overall."
    )
    q = assess_transcript_quality(text)
    assert q.suspect is False, q.reasons
