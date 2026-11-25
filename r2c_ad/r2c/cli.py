import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .alarms import calibrate, decisions, fuse, reference, starts
from .data import Windows, read_series, scaler
from .learning import learn_graph, scores, train_phase
from .model import RelationalDetector


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def fit(a):
    names, x = read_series(a.csv)
    torch.manual_seed(a.seed)
    if (a.length % a.patch or a.width % 4 or a.blocks < 3 or a.budget < 0
            or not np.isfinite(a.budget)):
        raise ValueError("Janela deve ser múltipla do patch, largura múltipla de 4, blocos >= 3 e orçamento finito >= 0.")
    train, graph, rank, cal = np.split(x, [int(len(x)*p) for p in (.50, .75, .875)])
    center, scale = scaler(train)
    def windows(segment):
        return Windows(segment, a.length, center, scale)
    data = windows(train)
    blocks = [windows(block) for block in np.array_split(graph, a.blocks)]
    rank_data, cal_data = windows(rank), windows(cal)
    settings = dict(channels=len(names), length=a.length, patch=a.patch, width=a.width, layers=a.layers)
    model = RelationalDetector(**settings).to(a.device)
    if a.graph == "none":
        model.edges.zero_()
    train_phase(model, data, a.epochs, a.device)
    audit = []
    if a.graph == "consensus":
        med, mad, confidence, chosen = learn_graph(model, blocks, a.device, a.neighbors, 1e-4)
        audit = [(names[j], names[i], med[i,j], mad[i,j], confidence[i,j], int(chosen[i,j]))
                 for i in range(len(names)) for j in range(len(names)) if i != j]
    train_phase(model, data, a.epochs, a.device)
    ref, combined = reference(scores(model, rank_data, a.device))
    cal_score = fuse(ref, combined, scores(model, cal_data, a.device))
    high, events = calibrate(cal_score, a.budget)
    out = Path(a.model)
    out.mkdir(parents=True, exist_ok=True)
    config = dict(settings=settings, names=names, high=high if np.isfinite(high) else None,
                  budget=a.budget, graph=a.graph, seed=a.seed, epochs_per_phase=a.epochs,
                  blocks=a.blocks, neighbors=a.neighbors, calibration_events=events,
                  calibration_exposure=int(np.isfinite(cal_score).sum()), edges=int(model.edges.sum()))
    torch.save(model.state_dict(), out / "weights.pt")
    np.savez(out / "reference.npz", center=center, scale=scale, ref=ref, combined=combined)
    write_csv(out / "edges.csv", ["origem","destino","ganho_mediano","mad","confianca","selecionada"], audit)
    (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(config, ensure_ascii=False))
    if high == float("inf"):
        print("Nenhuma política ativa satisfez o orçamento na calibração. Alarmes desativados.")


def detect(a):
    folder = Path(a.model)
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    _, x = read_series(a.csv, config["names"])
    model = RelationalDetector(**config["settings"]).to(a.device)
    model.load_state_dict(torch.load(folder / "weights.pt", map_location=a.device, weights_only=True))
    length = config["settings"]["length"]
    with np.load(folder / "reference.npz") as stats:
        data = Windows(x, length, stats["center"], stats["scale"])
        raw = scores(model, data, a.device)
        score = fuse(stats["ref"], stats["combined"], raw)
    active = decisions(score, config["high"] if config["high"] is not None else float("inf"))
    onset = starts(active)
    rows = [(i, "", "", "", 0, 0) for i in range(length)]
    rows.extend((i+length, *raw[i], score[i], int(active[i]), int(onset[i])) for i in range(len(raw)))
    write_csv(a.out, ["indice","condicional","temporal","fusao","ativo","inicio_alerta"], rows)
    print(f"{len(x)} instantes, {onset.sum()} eventos de alerta. Resultado em {a.out}")


def evaluate(a):
    with open(a.result, encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with open(a.labels, encoding="utf-8-sig", newline="") as stream:
        labels = np.array([int(row["anomalia"]) for row in csv.DictReader(stream)])
    if len(rows) != len(labels) or not np.isin(labels, [0,1]).all():
        raise ValueError("Rótulos devem ser binários e cobrir as mesmas linhas que o resultado.")
    truth = labels.astype(bool)
    usable = np.array([np.isfinite(float(r["fusao"] or "nan")) for r in rows], dtype=bool)
    onset = np.array([int(r["inicio_alerta"]) == 1 for r in rows], dtype=bool)
    delays = []
    start = np.flatnonzero(starts(truth))
    ends = np.flatnonzero(starts(truth[::-1])[::-1])
    eligible = 0
    for lo, hi in zip(start, ends):
        if not usable[lo:hi+1].any():
            continue
        eligible += 1
        alerts = np.flatnonzero(onset[lo:hi+1])
        if alerts.size:
            delays.append(int(alerts[0]))
    normal = int((usable & ~truth).sum())
    false_events = int((onset & ~truth & usable).sum())
    print(json.dumps(dict(eventos_avaliaveis=eligible, eventos_sem_cobertura=len(start)-eligible,
                          eventos_detectados=len(delays), recall_evento=len(delays)/eligible if eligible else None,
                          falsos_alarmes=false_events, exposicao_normal=normal,
                          far_por_1000=1000*false_events/normal if normal else None,
                          atraso_mediano_amostras=float(np.median(delays)) if delays else None), indent=2))


def demo(a):
    folder = Path(a.out)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    for split, n in [("treino",4200), ("teste",1600)]:
        u = rng.normal(size=n)
        for t in range(1,n):
            u[t] = 0.75 * u[t-1] + 0.66 * u[t]
        v = u + rng.normal(0, 0.12, n)
        w = np.r_[0, u[:-1]] + rng.normal(0, 0.18, n)
        intervals = [(210,235),(810,830)] if split == "treino" else [(300,335),(740,780),(1260,1290)]
        label = np.zeros(n, int)
        for lo, hi in intervals:
            v[lo:hi] *= -1
            label[lo:hi] = 1
        write_csv(folder / f"{split}.csv", ["fonte","copia","atrasado"], zip(u,v,w))
        if split == "teste":
            write_csv(folder / "rotulos.csv", ["anomalia"], zip(label))
    print(f"Dados gerados em {folder}")


def positive(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("Informe um inteiro positivo.")
    return value


def main():
    parser = argparse.ArgumentParser(description="R²C-AD • protótipo de pesquisa")
    commands = parser.add_subparsers(required=True)
    d = commands.add_parser("demo")
    d.add_argument("--out", required=True)
    d.add_argument("--seed", type=int, default=42)
    d.set_defaults(action=demo)
    f = commands.add_parser("treinar")
    for flag, default in [("length",32),("patch",4),("width",32),("layers",1),
                          ("epochs",4),("blocks",5),("neighbors",3)]:
        f.add_argument("--"+flag, type=positive, default=default)
    f.add_argument("--budget", type=float, default=5)
    f.add_argument("--seed", type=int, default=42)
    f.add_argument("--graph", choices=["consensus","attention","none"], default="consensus")
    f.set_defaults(action=fit)
    p = commands.add_parser("detectar")
    for command in (f,p):
        command.add_argument("--csv", required=True)
        command.add_argument("--model", required=True)
        command.add_argument("--device", default="cpu")
    p.add_argument("--out", required=True)
    p.set_defaults(action=detect)
    e = commands.add_parser("avaliar")
    e.add_argument("--result", required=True)
    e.add_argument("--labels", required=True)
    e.set_defaults(action=evaluate)
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    args.action(args)
