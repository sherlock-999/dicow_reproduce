"""meeteval scoring: per-session tcpWER (collar=5) + cpWER, micro/macro aggregation.

Compact port of TS-ASR-Whisper src/utils/wer.py (calc_session_tcp_wer /
calc_session_cp_wer); tcorc/orc omitted for v1. Hyp and ref must be normalized
with the same text normalizer BEFORE calling (as in DiCoW's evaluation.py).
"""

from typing import Dict, List

import meeteval

from .seglst import dummy_segment, to_seglst


def score_session(ref_segments: List[dict], hyp_segments: List[dict], collar: int = 5) -> Dict:
    if not ref_segments:
        raise ValueError("ref_segments must contain at least one reference segment")
    session_id = ref_segments[0]["session_id"]
    ref = to_seglst(ref_segments)
    hyp = to_seglst(hyp_segments if hyp_segments else [dummy_segment(session_id)])

    tcp = meeteval.wer.tcpwer(reference=ref, hypothesis=hyp, collar=collar)[session_id]
    cp = meeteval.wer.cpwer(reference=ref, hypothesis=hyp)[session_id]
    return {
        "session_id": session_id,
        "tcp_wer": tcp.error_rate,
        "tcp_errors": tcp.errors,
        "tcp_length": tcp.length,
        "tcp_insertions": tcp.insertions,
        "tcp_deletions": tcp.deletions,
        "tcp_substitutions": tcp.substitutions,
        "cp_wer": cp.error_rate,
        "cp_errors": cp.errors,
        "cp_length": cp.length,
    }


def aggregate(session_results: List[Dict], group_key: str = None) -> Dict:
    if not session_results:
        return {}
    out = {"n_sessions": len(session_results)}
    for prefix in ("tcp", "cp"):
        errors = sum(r[f"{prefix}_errors"] for r in session_results)
        length = sum(r[f"{prefix}_length"] for r in session_results)
        rates = [r[f"{prefix}_wer"] for r in session_results if r[f"{prefix}_wer"] is not None]
        out[f"{prefix}_wer_micro"] = errors / max(length, 1)
        out[f"{prefix}_wer_macro"] = sum(rates) / max(len(rates), 1)
    if group_key:
        # per-group blocks (e.g. per-language) + macro of group micros. Groups
        # may be scored at different token granularities (word vs char for
        # ja/ko/th) — never read the flat micro across groups in that case.
        groups: Dict[str, list] = {}
        for r in session_results:
            groups.setdefault(str(r.get(group_key) or "?"), []).append(r)
        out[f"by_{group_key}"] = {g: aggregate(rs) for g, rs in sorted(groups.items())}
        micros = [v["cp_wer_micro"] for v in out[f"by_{group_key}"].values()]
        out[f"cp_wer_macro_over_{group_key}"] = sum(micros) / len(micros)
    return out
