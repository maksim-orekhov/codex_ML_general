## Molecular Property Regression (RDKit + scikit-learn)

This repository includes:

- `train_molecule_regressors.py` to train **two random forest regressors** from SMILES-based Morgan fingerprints:
  - Solubility prediction (`logS`) from `curated-solubility-dataset.csv`
  - Melting point prediction (`mpC`) from `Bradley_dataset`
- `predict_molecule_properties.py` to load already-trained models and predict properties for new molecules.

### 1) Train models

`train_molecule_regressors.py`:

1. Parses each dataset and extracts `(SMILES, target)` pairs.
2. Builds Morgan fingerprints with RDKit (radius=2, 2048 bits by default).
3. Trains separate `RandomForestRegressor` models for each endpoint.
4. Evaluates each model on a train/test split.
5. Saves model artifacts and metrics under `models/` by default.

Run:

```bash
python train_molecule_regressors.py
```

Optional arguments:

```bash
python train_molecule_regressors.py \
  --solubility-file curated-solubility-dataset.csv \
  --bradley-file Bradley_dataset \
  --out-dir models \
  --radius 2 \
  --n-bits 2048 \
  --n-estimators 600 \
  --test-size 0.2 \
  --random-state 42
```

### 2) Predict with pre-trained models

Use this when you already have trained model files and do not want to retrain every run.

Default model paths expected:

- `models/solubility_rf.joblib`
- `models/melting_point_rf.joblib`

Input file format:

- first column: molecule ID
- second column: SMILES

(works with CSV/TSV, with or without header)

Run:

```bash
python predict_molecule_properties.py \
  --input-file my_molecules.tsv \
  --output-file predictions.tsv
```

Custom model paths:

```bash
python predict_molecule_properties.py \
  --input-file my_molecules.tsv \
  --output-file predictions.tsv \
  --solubility-model models/solubility_rf.joblib \
  --mp-model models/melting_point_rf.joblib
```

Output columns:

- `id`
- `SMILES`
- `predicted_solubility`
- `predicted_mp`

Invalid SMILES are kept in output with `NaN` predictions.

### Requirements

You need a Python environment with:

- `rdkit`
- `scikit-learn`
- `numpy`
- `joblib`
