# Seoul Bike Sharing Demand – MLOps semester project

Hourly bike rental demand prediction (regression) on the
[Seoul Bike Sharing Demand](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand)
dataset (UCI, CC BY 4.0, DOI: 10.24432/C5F62R), randomly sampled to 5,000 rows.

Author: Józsa Máté Szabolcs (RAPC9Y) – Óbuda University, MLSecOps course

> Work in progress – the full documentation (architecture, API, Docker, monitoring) will follow.

## Dataset

| File                         | Description                                            | Versioned with |
| ---------------------------- | ------------------------------------------------------ | -------------- |
| `data/raw/SeoulBikeData.csv` | original UCI file, 8,760 rows, cp1252 encoded          | DVC            |
| `data/seoul_bike_5000.csv`   | random sample of 5,000 rows (`random_state=42`), UTF-8 | DVC            |

The sampling is documented and reproducible in
[`notebooks/01_dataset_sampling.ipynb`](notebooks/01_dataset_sampling.ipynb).

Target variable: `Rented Bike Count` (`rented_bike_count`) – regression.

## Local development setup (Windows, PowerShell)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .    # installs the bike_demand package from src/ (editable)
pre-commit install
dvc pull            # restores the datasets from the DVC remote
```

## Data pipeline

The cleaning step is a DVC pipeline stage (`dvc.yaml`), so the processed dataset is reproducible:

```powershell
dvc repro           # runs: python -m bike_demand.data <sample> <processed>
dvc push
```

| Step       | Module                                    | What it does                                                                                        |
| ---------- | ----------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Load       | `bike_demand.data.load_csv`               | reads the CSV (UTF-8, or cp1252 for the original UCI file)                                          |
| Clean      | `bike_demand.data.clean`                  | snake_case columns, schema validation, date parsing, type coercion, drops invalid / duplicated rows |
| Features   | `bike_demand.features.add_date_features`  | month, day of week, weekend flag from the date                                                      |
| Preprocess | `bike_demand.features.build_preprocessor` | standard scaling (numeric) + one-hot encoding (categorical)                                         |

Output: `data/processed/seoul_bike_processed.csv` (5,000 rows, 14 columns).

## Quality checks

| Check                            | Tool                    | Command        | Where it runs automatically     |
| -------------------------------- | ----------------------- | -------------- | ------------------------------- |
| PEP8 (line length 120)           | flake8                  | `flake8`       | pre-commit hook, GitHub Actions |
| Commit message format            | conventional-pre-commit | –              | commit-msg hook                 |
| Type checking (standard, `src/`) | pyright                 | `pyright`      | GitHub Actions                  |
| Tests + coverage (min. 95 %)     | pytest, coverage        | `pytest --cov` | GitHub Actions                  |

All local hooks, including the manual ones:

```powershell
pre-commit run --all-files
pre-commit run --hook-stage manual --all-files
```

## Commit message convention

[Conventional Commits](https://www.conventionalcommits.org/): `<type>(<scope>): <description>`,
e.g. `feat(api): add predict endpoint`. Types used: `feat`, `fix`, `test`, `docs`, `refactor`,
`chore`, `ci`, `build`, `style`. Work is done on `feature/*` branches and merged into `main`
via pull requests only (protected branch).
