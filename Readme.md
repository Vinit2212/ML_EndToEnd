# Student Exam Performance Predictor (End-to-End ML)

Predict a student's math score from demographic and study-related inputs using a trained regression model served via a Flask web app. The repository includes data processing, model training, inference, and helpful utilities.

## Overview
- Web app built with Flask to submit features and get a predicted math score.
- End-to-end training pipeline: ingestion → transformation → model selection → persistence.
- Pretrained artifacts included: `artifacts/model.pkl` and `artifacts/preprocessor.pkl` for immediate inference.
- Supports multiple regressors with grid search (RandomForest, GradientBoosting, LinearRegression, KNN, XGBoost, CatBoost, AdaBoost).

## Tech Stack
- Python 3.11
- Flask, scikit-learn, pandas, numpy
- XGBoost, CatBoost, dill

## Project Structure
```
.
├── application.py                 # Flask app entry
├── templates/
│   ├── index.html                 # Landing page
│   └── home.html                  # Prediction form
├── src/
│   ├── components/
│   │   ├── data_ingestion.py      # Read CSV, split train/test
│   │   ├── data_transformation.py # Pipelines: impute, OHE, scale
│   │   └── model_trainer.py       # Model selection + save best model
│   ├── pipeline/
│   │   └── predict_pipeline.py    # Load artifacts and predict
│   ├── exception.py               # Custom exception with context
│   ├── logger.py                  # Timestamped file logging
│   └── utils.py                   # save/load objects, evaluate models
├── artifacts/
│   ├── model.pkl                  # Trained model (best model)
│   ├── preprocessor.pkl           # Fitted ColumnTransformer
│   └── *.csv                      # Raw/train/test CSVs (if generated)
├── requirements.txt               # Python dependencies
└── Readme.md                      # This file
```

## Setup
Prerequisites:
- Python 3.11
- pip

Create a virtual environment and install dependencies:
```bash
python3.11 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The repo includes pretrained artifacts, so you can run the app and predict immediately without retraining.

## Run the Application
Start the Flask app directly with Python:
```bash
python application.py
```

By default it listens on `http://127.0.0.1:5001`.

Routes:
- `/` – Landing page
- `/predictdata` – GET: prediction form, POST: submit form and get prediction
- `/health` – Health check endpoint returning `ok`

Use the form at `http://127.0.0.1:5001/predictdata` and fill in:
- `gender` (male/female)
- `race_ethnicity` (group A–E)
- `parental_level_of_education`
- `lunch` (standard/free/reduced)
- `test_preparation_course` (none/completed)
- `reading_score` (0–100)
- `writing_score` (0–100)

## Programmatic Prediction (Python)
```python
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

row = CustomData(
    gender="female",
    race_ethnicity="group B",
    parental_level_of_education="bachelor's degree",
    lunch="standard",
    test_preparation_course="completed",
    reading_score=85,
    writing_score=88,
)

df = row.get_data_as_data_frame()
pred = PredictPipeline().predict(df)
print("Predicted math score:", float(pred[0]))
```

## Retrain the Model
Data file: `notebook/data/stud.csv` (not tracked here by default).

Pipeline steps live under `src/components/`. A minimal retraining flow:
```python
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

ing = DataIngestion()
train_csv, test_csv = ing.initiate_data_ingestion()

transform = DataTransformation()
train_arr, test_arr, _ = transform.initiate_data_transformation(train_csv, test_csv)

trainer = ModelTrainer()
r2 = trainer.initiate_model_trainer(train_arr, test_arr)
print("R2 on test:", r2)
```

Important:
- `src/components/data_ingestion.py` currently reads from an absolute CSV path. Update it to a relative path like `notebook/data/stud.csv` before running on a different machine.
- New artifacts will be written under `artifacts/` and used by the app automatically.

## Logging
Logs are written to `logs/<timestamp>/<timestamp>.log` with a timestamped directory and file name.

## Known Issues / TODOs
- `data_ingestion.py` uses an absolute dataset path; make this configurable or relative for portability.
- `templates/home.html` has the reading/writing input labels swapped; update labels for clarity.

## License
No license specified. Add one if you intend to distribute.
