#!/usr/bin/env python3
"""Predict solubility and melting point for molecules from a file of IDs + SMILES.

Expected input format:
- first column: molecule ID
- second column: SMILES

The parser supports CSV/TSV (with or without header). Output includes:
id, SMILES, predicted_solubility, predicted_mp
"""

from __future__ import annotations

import argparse
import csv
import pathlib
from typing import Iterable

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

try:
    import joblib
except Exception:  # pragma: no cover
    from sklearn.externals import joblib  # type: ignore


def smiles_to_morgan_bits(smiles: str, radius: int, n_bits: int) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    bit_vector = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    from rdkit.DataStructs import ConvertToNumpyArray

    ConvertToNumpyArray(bit_vector, arr)
    return arr


def load_model_bundle(model_path: pathlib.Path):
    payload = joblib.load(model_path)
    if not isinstance(payload, dict) or "model" not in payload or "feature_config" not in payload:
        raise ValueError(f"Unexpected model payload format in: {model_path}")

    cfg = payload["feature_config"]
    radius = int(cfg.get("radius", 2))
    n_bits = int(cfg.get("n_bits", 2048))
    return payload["model"], radius, n_bits


def _is_header(first_cell: str) -> bool:
    token = first_cell.strip().lower()
    return token in {"id", "molecule_id", "mol_id", "key"}


def read_input_rows(path: pathlib.Path) -> list[tuple[str, str]]:
    content = path.read_text(encoding="utf-8").splitlines()
    if not content:
        return []

    sample = "\n".join(content[:10])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t; ")
        delimiter = dialect.delimiter
    except Exception:
        delimiter = ","

    rows: list[tuple[str, str]] = []

    if delimiter == " ":
        for line in content:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            mol_id, smiles = parts[0], parts[1]
            if not rows and _is_header(mol_id):
                continue
            rows.append((mol_id, smiles))
        return rows

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        for row in reader:
            if len(row) < 2:
                continue
            mol_id = row[0].strip()
            smiles = row[1].strip()
            if not rows and _is_header(mol_id):
                continue
            if not mol_id or not smiles:
                continue
            rows.append((mol_id, smiles))

    return rows


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=pathlib.Path, required=True, help="File with first col=id, second col=SMILES")
    parser.add_argument("--output-file", type=pathlib.Path, required=True, help="Output file for predictions")
    parser.add_argument("--solubility-model", type=pathlib.Path, default=pathlib.Path("models/solubility_rf.joblib"))
    parser.add_argument("--mp-model", type=pathlib.Path, default=pathlib.Path("models/melting_point_rf.joblib"))
    parser.add_argument("--delimiter", default="\t", help="Output delimiter (default: tab)")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    sol_model, sol_radius, sol_bits = load_model_bundle(args.solubility_model)
    mp_model, mp_radius, mp_bits = load_model_bundle(args.mp_model)

    if (sol_radius, sol_bits) != (mp_radius, mp_bits):
        raise ValueError(
            "Feature configs differ between models; expected matching radius/n_bits. "
            f"solubility=({sol_radius},{sol_bits}), mp=({mp_radius},{mp_bits})"
        )

    rows = read_input_rows(args.input_file)
    if not rows:
        raise ValueError(f"No valid (id, SMILES) records found in: {args.input_file}")

    valid_features: list[np.ndarray] = []
    valid_indices: list[int] = []

    for i, (_, smiles) in enumerate(rows):
        fp = smiles_to_morgan_bits(smiles, radius=sol_radius, n_bits=sol_bits)
        if fp is None:
            continue
        valid_indices.append(i)
        valid_features.append(fp)

    sol_preds = np.full((len(rows),), np.nan, dtype=np.float32)
    mp_preds = np.full((len(rows),), np.nan, dtype=np.float32)

    if valid_features:
        X = np.vstack(valid_features)
        sol_valid = sol_model.predict(X)
        mp_valid = mp_model.predict(X)

        for j, row_idx in enumerate(valid_indices):
            sol_preds[row_idx] = float(sol_valid[j])
            mp_preds[row_idx] = float(mp_valid[j])

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=args.delimiter)
        writer.writerow(["id", "SMILES", "predicted_solubility", "predicted_mp"])
        for i, (mol_id, smiles) in enumerate(rows):
            writer.writerow([mol_id, smiles, f"{sol_preds[i]:.6f}", f"{mp_preds[i]:.6f}"])

    print(f"Processed molecules: {len(rows)}")
    print(f"Valid SMILES: {len(valid_indices)}")
    print(f"Invalid SMILES: {len(rows) - len(valid_indices)} (predictions set to NaN)")
    print(f"Output written to: {args.output_file.resolve()}")


if __name__ == "__main__":
    main()
