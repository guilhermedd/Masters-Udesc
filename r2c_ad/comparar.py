"""Compara R²C-AD, Isolation Forest, PCA e desvio robusto por sensor."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from r2c.alarms import calibrate, decisions, fuse, reference, starts
from r2c.data import read_series, scaler


def r2c(*args, capture=False):
    """Usa o mesmo treino, detector e avaliador que a execução individual."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parent), env.get("PYTHONPATH", "")])
    run = subprocess.run([sys.executable, "-m", "r2c", *map(str, args)],
                         check=True, env=env, stdout=subprocess.PIPE if capture else None,
                         encoding="utf-8")
    return json.loads(run.stdout) if capture else None


def classic_scores(fit, segments, length, seed):
    center, scale = scaler(fit)

    def transform(x):
        return np.nan_to_num((x - center) / scale, nan=0.)

    # Desvio robusto usa as medições atuais. IF e PCA também recebem um lag
    # para poder explorar dependências como as presentes no exemplo sintético.
    def features(x):
        z = transform(x)
        return np.column_stack((z[length:], z[length-1:-1]))

    design = features(fit)[np.isfinite(fit[length:]).any(1)]
    if len(design) < 2:
        raise ValueError("São necessárias duas linhas observadas após o aquecimento do ajuste.")
    forest = IsolationForest(n_estimators=100, random_state=seed, n_jobs=1).fit(design)
    pca = PCA(svd_solver="full").fit(design)
    # Reter 90% da variância, reservando ao menos uma direção para o resíduo.
    variance = np.cumsum(np.nan_to_num(pca.explained_variance_ratio_))
    k = min(np.searchsorted(variance, .90) + 1, max(1, len(variance)-1))
    basis = pca.components_[:k]
    scores = {name: [] for name in ("Isolation Forest", "PCA", "MAD por sensor")}
    for segment in segments:
        values = features(segment)
        centered = values - pca.mean_
        residual = centered - (centered @ basis.T) @ basis
        raw = (-forest.score_samples(values), np.mean(residual**2, axis=1),
               np.max(abs(transform(segment[length:])), axis=1))
        for name, series in zip(scores, raw):
            series[~np.isfinite(segment[length:]).any(1)] = np.nan
            scores[name].append(series[:, None])
    return scores


def write_predictions(path, score, length, high):
    active = decisions(score, high)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["indice", "fusao", "ativo", "inicio_alerta"])
        writer.writerows((i, "", 0, 0) for i in range(length))
        for i, (s, a, onset) in enumerate(zip(score, active, starts(active)), length):
            writer.writerow([i, s, int(a), int(onset)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--treino", required=True, type=Path)
    p.add_argument("--teste", required=True, type=Path)
    p.add_argument("--rotulos", required=True, type=Path)
    p.add_argument("--out", type=Path, default=Path("comparacao"))
    p.add_argument("--budget", type=float, default=5.)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--length", type=int, default=32)
    p.add_argument("--epochs", type=int, default=4)
    a = p.parse_args()
    names, train = read_series(a.treino)
    _, test = read_series(a.teste, names)
    if len(test) <= a.length:
        p.error("Teste insuficiente para a janela.")
    fit, _, rank, cal = np.split(train, [int(len(train)*v) for v in (.50, .75, .875)])
    a.out.mkdir(exist_ok=True, parents=True)
    model = a.out / "modelo"
    path = a.out / "r2c.csv"
    r2c("treinar", "--csv", a.treino, "--model", model, "--seed", a.seed,
        "--length", a.length, "--epochs", a.epochs, "--budget", a.budget)
    r2c("detectar", "--csv", a.teste, "--model", model, "--out", path)
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    predictions = [("R2C-AD", path, config["calibration_events"],
                    config["calibration_exposure"], config["high"] is not None)]
    classic = classic_scores(fit, [rank, cal, test], a.length, a.seed)
    for name, (reference_raw, cal_raw, test_raw) in classic.items():
        ref, combined = reference(reference_raw)
        cal_scores = fuse(ref, combined, cal_raw)
        high, count = calibrate(cal_scores, a.budget)
        path = a.out / (name.lower().replace(" ", "_") + ".csv")
        write_predictions(path, fuse(ref, combined, test_raw), a.length, high)
        predictions.append((name, path, count, int(np.isfinite(cal_scores).sum()), np.isfinite(high)))

    # Os rótulos entram somente depois de fixar todas as previsões.
    result = []
    for name, path, count, exposure, enabled in predictions:
        metrics = r2c("avaliar", "--result", path, "--labels", a.rotulos, capture=True)
        far = metrics["far_por_1000"]
        result.append(dict(metodo=name, seed=a.seed, orcamento=a.budget, politica_ativa=bool(enabled),
                           taxa_alertas_cal_por_1000=1000*count/exposure, **metrics,
                           dentro_orcamento_teste=None if far is None else far <= a.budget))
    summary = a.out / "resumo.csv"
    with summary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=result[0].keys())
        writer.writeheader()
        writer.writerows(result)
    def fmt(x):
        return "n/d" if x is None else f"{x:.2f}"
    print(f"\n{'Método':18} {'Recall':>7} {'Falsos':>7} {'FAR/1000':>9} {'Atraso':>7} {'Na meta':>8}")
    for row in result:
        within = {True: "sim", False: "não", None: "n/d"}[row["dentro_orcamento_teste"]]
        print(f"{row['metodo']:18} {fmt(row['recall_evento']):>7} {row['falsos_alarmes']:7} "
              f"{fmt(row['far_por_1000']):>9} {fmt(row['atraso_mediano_amostras']):>7} {within:>8}")
    print(f"\nResumo completo em {summary}")


if __name__ == "__main__":
    main()
