import csv
import numpy as np
import torch
from torch.utils.data import Dataset


def read_series(path, expected=None):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        names = next(reader, [])
        if len(names) < 2 or len(set(names)) != len(names) or any(not s.strip() for s in names):
            raise ValueError("Informe pelo menos dois sensores com nomes exclusivos no cabeçalho.")
        if expected is not None and names != expected:
            raise ValueError("Os nomes e a ordem dos sensores devem coincidir com o treino.")
        rows = []
        for line, row in enumerate(reader, 2):
            if len(row) != len(names):
                raise ValueError(f"Número de colunas inválido na linha {line}.")
            rows.append([np.nan if v.strip().lower() in {"", "na", "nan"} else float(v) for v in row])
    x = np.asarray(rows, dtype=np.float32).reshape(-1, len(names))
    if np.isinf(x).any():
        raise ValueError("Use campos vazios para ausências e valores finitos para medições.")
    return names, x


def scaler(x):
    if np.any(np.isnan(x).all(0)):
        raise ValueError("Há sensor sem observações no trecho de ajuste.")
    center = np.nanmedian(x, axis=0)
    mad = 1.4826 * np.nanmedian(np.abs(x - center), axis=0)
    return center, np.where(mad > 1e-6, mad, 1.0)


class Windows(Dataset):
    def __init__(self, x, length, center, scale):
        if len(x) <= length:
            raise ValueError(f"Cada segmento deve conter mais de {length} linhas.")
        scaled = (x - center) / scale
        self.values = torch.from_numpy(np.nan_to_num(scaled, nan=0.0))
        self.valid = torch.from_numpy(np.isfinite(scaled))
        self.length = length

    def __len__(self):
        return len(self.values) - self.length

    def __getitem__(self, index):
        end = index + self.length
        return (self.values[index:end], self.valid[index:end],
                self.values[end], self.valid[end])
