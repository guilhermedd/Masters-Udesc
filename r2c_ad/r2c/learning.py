import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def train_phase(model, data, epochs, device, robust=True):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=0.002)
    for _ in range(epochs):
        for batch in DataLoader(data, batch_size=64, shuffle=True):
            x, m, y, valid = (v.to(device) for v in batch)
            h = model.history(x, m)
            # Source dropout also occurs in training, as in the graph ablations.
            edges = model.edges & (torch.rand(len(x), 1, y.size(1), device=device) > 0.2)
            cond = model.predict(h, y, valid, edges)
            pred = model.forecast(h).squeeze(-1)
            loss = F.smooth_l1_loss(cond, y, reduction="none") + F.smooth_l1_loss(pred, y, reduction="none")
            observed = valid.any(1)
            if not observed.any():
                continue
            per_window = ((loss * valid).sum(1) / valid.sum(1).clamp_min(1))[observed]
            with torch.no_grad():
                med = per_window.median()
                spread = (per_window - med).abs().median()
                weight = ((med + 2 * spread + 1e-5) / (per_window + 1e-5)).clamp(0.15, 1)
            objective = (per_window * weight).sum() / weight.sum() if robust else per_window.mean()
            opt.zero_grad()
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()


@torch.no_grad()
def learn_graph(model, blocks, device, top_k, min_gain):
    model.eval()
    d = len(model.edges)
    all_gains = []
    for block in blocks:
        sums, counts = np.zeros((d, d)), np.zeros(d)
        for batch in DataLoader(block, batch_size=64):
            x, m, y, valid = (v.to(device) for v in batch)
            h = model.history(x, m)
            def error(edges):
                output = model.predict(h, y, valid, edges)
                return F.smooth_l1_loss(output, y, reduction="none")
            full = error(model.edges)
            counts += valid.sum(0).cpu().numpy()
            for j in range(d):
                edges = model.edges.clone()
                edges[:, j] = False
                sums[:, j] += ((error(edges) - full) * valid).sum(0).cpu().numpy()
        all_gains.append(np.divide(sums, counts[:, None], out=np.full_like(sums, np.nan),
                                   where=counts[:, None] > 0))
    gains = np.stack(all_gains)
    median = np.nanmedian(gains, axis=0)
    mad = np.nanmedian(abs(gains - median), axis=0)
    confidence = np.maximum(0, median) / (mad + min_gain)
    eligible = ((gains > min_gain).sum(0) > len(blocks) / 2) & ~np.eye(d, dtype=bool)
    selected = np.zeros_like(eligible)
    for i in range(d):
        candidates = np.flatnonzero(eligible[i])
        ordered = candidates[np.argsort(-confidence[i, candidates], kind="stable")]
        selected[i, ordered[:top_k]] = True
    model.edges.copy_(torch.from_numpy(selected))
    return median, mad, confidence, selected


@torch.no_grad()
def scores(model, data, device):
    model.eval()
    result = []
    for batch in DataLoader(data, batch_size=128):
        x, m, y, valid = (v.to(device) for v in batch)
        h = model.history(x, m)
        cond = model.predict(h, y, valid, model.edges)
        predictions = (cond, model.forecast(h).squeeze(-1))
        # Mean of up to three largest observed absolute residuals per channel group.
        cols = []
        for pred in predictions:
            errors = abs(pred - y).masked_fill(~valid, float("nan"))
            largest = errors.nan_to_num(nan=-1).topk(min(3, y.size(1)), dim=1).values
            cols.append(largest.masked_fill(largest < 0, float("nan")).nanmean(1))
        result.append(torch.stack(cols, 1).cpu().numpy())
    return np.concatenate(result)
