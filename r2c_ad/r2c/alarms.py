import numpy as np


def ranks(sorted_ref, values):
    left = np.searchsorted(sorted_ref, values, side="left")
    right = np.searchsorted(sorted_ref, values, side="right")
    return (0.5 * (left + right) + 0.5) / (len(sorted_ref) + 1)


def components(ref, raw):
    u = np.column_stack([ranks(ref[:, j], raw[:, j]) for j in range(raw.shape[1])])
    return np.column_stack([u.max(1), u.mean(1)])


def reference(raw):
    raw = raw[np.isfinite(raw).all(1)]
    if len(raw) < 20:
        raise ValueError("A referência precisa de pelo menos 20 instantes observados após o aquecimento.")
    ref = np.sort(raw, axis=0)
    return ref, np.sort(components(ref, raw), axis=0)


def fuse(ref, combined, raw):
    parts = components(ref, raw)
    score = np.maximum(ranks(combined[:, 0], parts[:, 0]), ranks(combined[:, 1], parts[:, 1]))
    score[~np.isfinite(raw).all(1)] = np.nan
    return score


def decisions(score, high):
    # One parameter indexes the entire policy, including lower and cumulative thresholds.
    active = np.zeros(len(score), dtype=bool)
    previous = False
    accumulator = 0.
    low = high - 0.3 * (1 - high)
    sequential_limit = -np.log10(max(1e-8, 1 - high)) if np.isfinite(high) else np.inf
    for i, value in enumerate(score):
        if not np.isfinite(value) or not np.isfinite(high):
            previous, accumulator = False, 0.
            continue
        moderate = value >= low
        accumulator = max(0., accumulator + value - 0.8)
        active[i] = value >= high or (moderate and previous) or accumulator >= sequential_limit
        previous = moderate
        if active[i]:
            accumulator = 0.
    return active


def starts(active):
    return active & ~np.r_[False, active[:-1]] if len(active) else active.copy()


def calibrate(score, budget):
    exposure = np.isfinite(score).sum()
    if exposure < 20:
        raise ValueError("Calibração requer pelo menos 20 instantes observados.")
    for high in np.linspace(0.90, 0.9999, 101):
        active = decisions(score, high)
        events = int(starts(active).sum())
        if 1000 * events / exposure <= budget:
            return float(high), events
    return float("inf"), 0
