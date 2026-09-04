"""Convert Lhotse references and Whisper outputs to meeteval SegLST."""

from typing import Callable, List

from meeteval.io.seglst import SegLST

def _word_field(word, name: str):
    return word[name] if isinstance(word, dict) else getattr(word, name)


def words_to_segments(words: List[object], session_id: str, speaker: str,
                      gap_threshold: float = 1.0) -> List[dict]:
    """Split timestamped word objects/dicts at gaps larger than the threshold."""
    segments = []
    cur = []
    for w in words:
        if cur and _word_field(w, "start") - _word_field(cur[-1], "end") > gap_threshold:
            segments.append(cur)
            cur = []
        cur.append(w)
    if cur:
        segments.append(cur)
    return [
        {
            "session_id": session_id,
            "speaker": speaker,
            "start_time": float(_word_field(seg[0], "start")),
            "end_time": float(_word_field(seg[-1], "end")),
            "words": " ".join(str(_word_field(w, "word")) for w in seg),
        }
        for seg in segments
    ]


def whisper_segments_to_segments(
    whisper_segments: List[dict],
    session_id: str,
    speaker: str,
    decode: Callable,
) -> List[dict]:
    """Convert timestamp segments returned by HF Whisper ``generate``."""

    output = []
    for segment in whisper_segments:
        text = decode(segment["tokens"], skip_special_tokens=True).strip()
        if not text:
            continue
        output.append(
            {
                "session_id": session_id,
                "speaker": speaker,
                "start_time": float(segment["start"]),
                "end_time": float(segment["end"]),
                "words": text,
            }
        )
    return output


def reference_segments(cut, session_id: str, text_norm: Callable[[str], str]) -> List[dict]:
    """Reference SegLST segments from cut supervisions (cut-relative times)."""
    segs = []
    for s in cut.supervisions:
        words = text_norm(s.text or "")
        if not words:
            continue
        segs.append({
            "session_id": session_id,
            "speaker": s.speaker,
            "start_time": max(0.0, float(s.start)),
            "end_time": float(s.end),
            "words": words,
        })
    return segs


def normalize_hyp_segments(segments: List[dict], text_norm: Callable[[str], str]) -> List[dict]:
    out = []
    for seg in segments:
        words = text_norm(seg["words"])
        if not words:
            continue
        out.append({**seg, "words": words})
    return out


def to_seglst(segments: List[dict]) -> SegLST:
    return SegLST.new(segments)


def dummy_segment(session_id: str) -> dict:
    return {"session_id": session_id, "speaker": "dummy", "start_time": 0.0, "end_time": 0.1, "words": "dummy"}
