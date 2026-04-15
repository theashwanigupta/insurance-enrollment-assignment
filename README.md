# Insurance Enrollment Prediction (ML Take-Home)

This project predicts whether an employee will enroll in a voluntary insurance product using census-style employee data.

## Project Files

- `employee_data.csv` — source dataset (provided)
- `insurance_enrollment_prediction.ipynb` — end-to-end notebook (EDA, preprocessing, modeling, evaluation)
- `insurance_enrollment_prediction/insurance_enrollment_prediction.md` — assignment report with observations, model rationale, and outcomes
- `requirements.txt` — Python dependencies
- `train.txt` — Model Training
- `inference.txt` — Predection Inferencing

## Setup

Go to the folder where you want your python project/environment to be setup and execute below commands

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

1. Start Jupyter:

```bash
jupyter notebook
```

2. Open `insurance_enrollment_prediction.ipynb` and run all cells.

## Train and Inference Scripts

You can also run training/inference directly using Python files derived from the notebook.

### 1) Train model and save artifacts

```bash
python train.py --data employee_data.csv --model-dir model
```

This creates:
- `model/enrollment_model.joblib`
- `model/metadata.json`

### 2) Run inference on a new record

```bash
python inference.py --model-dir model --record '{"employee_id":1001,"age":37,"salary":72000,"tenure_years":4,"gender":"Male","marital_status":"Married","employment_type":"Full-time","region":"North","has_dependents":"Yes"}'
```

You can also pass a JSON file containing one record or a list of records:

```bash
python inference.py --model-dir model --record-file new_records.json
```
