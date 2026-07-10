import csv
import os
import subprocess
import sys
from pathlib import Path

if Path("/mnt/second_drive/seisbench").is_dir():
    os.environ.setdefault("SEISBENCH_CACHE_ROOT", "/mnt/second_drive/seisbench")

import torch
import yaml
import pytorch_lightning as pl

sys.path.insert(0, str(Path(__file__).parent / "benchmark"))
from models import find_picovar_repo, PicovarLit

ROOT = Path(__file__).parent
BENCH = ROOT / "benchmark"
MODELS_DIR = find_picovar_repo() / "models"

DATASETS = {
    "ethz": "ETHZ",
    "geofon": "GEOFON",
    "instance": "InstanceCountsCombined",
    "iquique": "Iquique",
    "lendb": "LenDB",
    "neic": "NEIC",
    "scedc": "SCEDC",
    "stead": "STEAD",
}

MODEL_ARGS = {"seed": 0}


def run(cmd):
    print("+", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], cwd=BENCH, check=True)


def ensure_checkpoint(key, data_name):
    pt = MODELS_DIR / f"recovar_{key}_seisbench_benchmark.pt"
    if not pt.exists():
        return False
    version_dir = ROOT / "weights" / f"{key}_picovar" / "version_0"
    ckpt_path = version_dir / "checkpoints" / "epoch=0-step=1.ckpt"
    if ckpt_path.exists() and ckpt_path.stat().st_mtime >= pt.stat().st_mtime:
        return True

    lit = PicovarLit(**MODEL_ARGS)
    lit.model.load_representation_state_dict(str(pt), map_location="cpu")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    config = {"model": "Picovar", "data": data_name, "model_args": MODEL_ARGS}
    with open(version_dir / "hparams.yaml", "w") as f:
        yaml.safe_dump(config, f)
    with open(version_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["val_loss", "epoch", "step"])
        writer.writerow([0.0, 0, 0])
    torch.save(
        {
            "state_dict": lit.state_dict(),
            "hyper_parameters": dict(lit.hparams),
            "pytorch-lightning_version": pl.__version__,
            "epoch": 0,
            "global_step": 1,
        },
        ckpt_path,
    )
    print(f"{key}: wrapped {pt} -> {ckpt_path}")
    return True


PUBLISHED_TARGETS_URL = "https://dcache-demo.desy.de:2443/Helmholtz/HelmholtzAI/SeisBench/auxiliary/pick-benchmark/targets"


def ensure_targets(key, data_name):
    targets_dir = ROOT / "targets" / key
    if targets_dir.exists():
        return
    tmp_dir = ROOT / "targets" / f"{key}.download"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fetched = []
    for task_file in ("task1.csv", "task23.csv"):
        result = subprocess.run(
            ["curl", "-sf", "--max-time", "300",
             f"{PUBLISHED_TARGETS_URL}/{key}/{task_file}",
             "-o", str(tmp_dir / task_file)],
        )
        if result.returncode == 0:
            fetched.append(task_file)
    if len(fetched) == 2:
        tmp_dir.rename(targets_dir)
        print(f"{key}: using published targets from the benchmark paper")
        return
    for f in tmp_dir.iterdir():
        f.unlink()
    tmp_dir.rmdir()
    print(f"{key}: published targets not reachable, GENERATING targets locally "
          f"(results not row-identical to the paper tables)")
    run([sys.executable, "generate_eval_targets.py", data_name, targets_dir, "--sampling_rate", "100"])


def ensure_eval(key):
    pred_dir = ROOT / "pred" / f"{key}_picovar" / "version_0"
    expected = [pred_dir / f"{s}_task{t}.csv" for s in ("dev", "test") for t in ("1", "23")]
    if all(p.exists() for p in expected):
        print(f"{key}: predictions exist, skipping eval")
        return
    run([sys.executable, "eval.py", ROOT / "weights" / f"{key}_picovar", ROOT / "targets" / key])


if __name__ == "__main__":
    done = []
    for key, data_name in DATASETS.items():
        if not ensure_checkpoint(key, data_name):
            print(f"{key}: no trained model at {MODELS_DIR / f'recovar_{key}_seisbench_benchmark.pt'}, skipping")
            continue
        ensure_targets(key, data_name)
        ensure_eval(key)
        done.append(key)

    if done:
        run([sys.executable, "collect_results.py", ROOT / "pred", ROOT / "results_picovar.csv"])
        print("results:", ROOT / "results_picovar.csv")
    print("evaluated:", ", ".join(done) if done else "nothing")
