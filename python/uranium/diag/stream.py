import numpy as np


def lane_groups(stream, lanes: int, g: int, t: int, pad: int, max_select: int, states: list, limit_batches: int = 0):
    groups = -(-lanes // g)
    nb = 0
    while (b := stream.next()) is not None:
        if limit_batches and nb >= limit_batches:
            break
        steps = b.lane_steps()
        for gi in range(groups):
            s, e = gi * g, min((gi + 1) * g, lanes)
            if all(steps[l] == 0 for l in range(s, e)):
                states[gi] = None
                continue
            yield nb, b, gi, s, steps, b.group(s, e, t, pad, max_select)
        nb += 1


def real_rows(hg: dict, steps, start: int, t: int) -> np.ndarray:
    m = np.zeros(hg["b"] * t, bool)
    for li in range(hg["b"]):
        m[li * t:li * t + steps[start + li]] = True
    return m
