"""Enrollment-data helpers."""


def _subtract_interval(base, blockers):
    pieces = [base]
    for b0, b1 in blockers:
        nxt = []
        for a0, a1 in pieces:
            if b1 <= a0 or b0 >= a1:
                nxt.append((a0, a1))
                continue
            if a0 < b0:
                nxt.append((a0, min(a1, b0)))
            if b1 < a1:
                nxt.append((max(a0, b1), a1))
        pieces = nxt
    return pieces


def solo_spans_by_speaker(cut, min_duration: float = 1.0):
    intervals = {}
    for sup in cut.supervisions:
        spk = sup.speaker
        intervals.setdefault(spk, []).append((float(sup.start), float(sup.start + sup.duration)))
    out = {}
    for spk, spans in intervals.items():
        blockers = [iv for other, other_spans in intervals.items() if other != spk for iv in other_spans]
        solo = []
        for span in spans:
            for a0, a1 in _subtract_interval(span, blockers):
                dur = a1 - a0
                if dur >= min_duration:
                    solo.append([round(a0, 3), round(dur, 3)])
        if solo:
            out[spk] = solo
    return out
