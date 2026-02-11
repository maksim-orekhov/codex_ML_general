#!/usr/bin/env python3
"""Train random-forest regressors for solubility and melting point from SMILES.

Datasets expected in repository root:
- curated-solubility-dataset.csv (AqSolDB curated dataset; SMILES in col 5, logS in col 6)
- Bradley_dataset (tab-separated; headers include smiles and mpC)
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

try:
    import joblib
except Exception:  # pragma: no cover - fallback for older sklearn envs
    from sklearn.externals import joblib  # type: ignore


@dataclass
class Dataset:
    name: str
    smiles: list[str]
    targets: np.ndarray


def smiles_to_morgan_bits(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray | None:
    """Convert a SMILES string into a Morgan fingerprint bit vector."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    bit_vector = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    # RDKit fills arr in-place with 0/1 values.
    from rdkit.DataStructs import ConvertToNumpyArray

    ConvertToNumpyArray(bit_vector, arr)
    return arr


def build_feature_matrix(smiles_list: Sequence[str], radius: int, n_bits: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate feature matrix from valid SMILES and track valid indices."""
    rows: list[np.ndarray] = []
    valid_idx: list[int] = []

    for idx, smiles in enumerate(smiles_list):
        fp = smiles_to_morgan_bits(smiles, radius=radius, n_bits=n_bits)
        if fp is None:
            continue
        rows.append(fp)
        valid_idx.append(idx)

    if not rows:
        raise ValueError("No valid SMILES could be parsed into fingerprints.")

    return np.vstack(rows), np.array(valid_idx, dtype=np.int64)


def load_solubility_dataset(path: pathlib.Path) -> Dataset:
    """Load AqSolDB curated solubility dataset from CSV without a header row."""
    smiles: list[str] = []
    values: list[float] = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 6:
                continue
            try:
                values.append(float(row[5]))
            except ValueError:
                continue
            smiles.append(row[4])

    if not smiles:
        raise ValueError(f"No valid records found in {path}")

    return Dataset(name="solubility_logS", smiles=smiles, targets=np.asarray(values, dtype=np.float32))


def load_bradley_dataset(path: pathlib.Path) -> Dataset:
    """Load Bradley melting point dataset (TSV with header)."""
    smiles: list[str] = []
    values: list[float] = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            s = (row.get("smiles") or "").strip()
            y = (row.get("mpC") or "").strip()
            if not s or not y:
                continue
            try:
                values.append(float(y))
            except ValueError:
                continue
            smiles.append(s)

    if not smiles:
        raise ValueError(f"No valid records found in {path}")

    return Dataset(name="melting_point_C", smiles=smiles, targets=np.asarray(values, dtype=np.float32))


def train_random_forest(
    dataset: Dataset,
    *,
    radius: int,
    n_bits: int,
    test_size: float,
    random_state: int,
    n_estimators: int,
) -> tuple[RandomForestRegressor, dict[str, float], int]:
    """Train/evaluate a RF model and return model, metrics, and dropped-count."""
    X, valid_idx = build_feature_matrix(dataset.smiles, radius=radius, n_bits=n_bits)
    y = dataset.targets[valid_idx]
    dropped = len(dataset.smiles) - len(valid_idx)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "r2": float(r2_score(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mae": float(mean_absolute_error(y_test, preds)),
        "n_total": float(len(dataset.smiles)),
        "n_valid": float(len(valid_idx)),
        "n_train": float(len(y_train)),
        "n_test": float(len(y_test)),
        "n_dropped_invalid_smiles": float(dropped),
    }

    return model, metrics, dropped


def save_model_bundle(
    out_dir: pathlib.Path,
    model_name: str,
    model: RandomForestRegressor,
    metrics: dict[str, float],
    *,
    radius: int,
    n_bits: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / f"{model_name}_rf.joblib"
    metrics_path = out_dir / f"{model_name}_metrics.json"

    payload = {
        "model": model,
        "feature_config": {"fingerprint": "morgan", "radius": radius, "n_bits": n_bits},
    }
    joblib.dump(payload, model_path)

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def print_report(name: str, metrics: dict[str, float]) -> None:
    print(f"\n{name}")
    print("-" * len(name))
    print(f"R2   : {metrics['r2']:.4f}")
    print(f"RMSE : {metrics['rmse']:.4f}")
    print(f"MAE  : {metrics['mae']:.4f}")
    print(
        "Records: "
        f"total={int(metrics['n_total'])}, "
        f"valid={int(metrics['n_valid'])}, "
        f"dropped_invalid_smiles={int(metrics['n_dropped_invalid_smiles'])}, "
        f"train={int(metrics['n_train'])}, test={int(metrics['n_test'])}"
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solubility-file", type=pathlib.Path, default=pathlib.Path("curated-solubility-dataset.csv"))
    parser.add_argument("--bradley-file", type=pathlib.Path, default=pathlib.Path("Bradley_dataset"))
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("models"))
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    solubility_data = load_solubility_dataset(args.solubility_file)
    bradley_data = load_bradley_dataset(args.bradley_file)

    solubility_model, solubility_metrics, _ = train_random_forest(
        solubility_data,
        radius=args.radius,
        n_bits=args.n_bits,
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
    )

    melting_model, melting_metrics, _ = train_random_forest(
        bradley_data,
        radius=args.radius,
        n_bits=args.n_bits,
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
    )

    save_model_bundle(
        args.out_dir,
        "solubility",
        solubility_model,
        solubility_metrics,
        radius=args.radius,
        n_bits=args.n_bits,
    )
    save_model_bundle(
        args.out_dir,
        "melting_point",
        melting_model,
        melting_metrics,
        radius=args.radius,
        n_bits=args.n_bits,
    )

    print_report("Solubility model (logS)", solubility_metrics)
    print_report("Melting-point model (°C)", melting_metrics)
    print(f"\nSaved model bundles and metrics in: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
