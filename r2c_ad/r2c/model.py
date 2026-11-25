import math
import torch
from torch import nn


class RelationalDetector(nn.Module):
    def __init__(self, channels, length, patch, width, layers):
        super().__init__()
        self.patch = patch
        self.embed = nn.Linear(2 * patch, width)
        self.position = nn.Parameter(torch.randn(length // patch, width) * 0.02)
        self.sensor = nn.Parameter(torch.randn(channels, width) * 0.02)
        layer = nn.TransformerEncoderLayer(width, 4, 2 * width, 0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.forecast = nn.Linear(width, 1)
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width + 2, width)
        self.readout = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, 1))
        self.register_buffer("edges", ~torch.eye(channels, dtype=torch.bool))

    def history(self, x, mask):
        batch, length, channels = x.shape
        def patches(v):
            return v.transpose(1, 2).reshape(batch, channels, length // self.patch, self.patch)
        tokens = self.embed(torch.cat((patches(x), patches(mask.float())), -1))
        tokens = tokens + self.position + self.sensor[:, None, :]
        # Every token ends before the prediction time.
        return self.encoder(tokens.flatten(0, 1))[:, -1].reshape(batch, channels, -1)

    def predict(self, h, current, valid, edges):
        # Keys/queries use the past. Only source messages see current measurements.
        logits = self.query(h) @ self.key(h).transpose(1, 2) / math.sqrt(h.size(-1))
        attention = logits.masked_fill(~edges, -1e4).softmax(-1) * edges
        messages = self.value(torch.cat((h, current.unsqueeze(-1), valid.unsqueeze(-1)), -1))
        context = attention @ messages
        return self.readout(torch.cat((h, context), -1)).squeeze(-1)
