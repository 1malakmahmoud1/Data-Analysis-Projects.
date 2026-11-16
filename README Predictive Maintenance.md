# Predictive Maintenance — Machine Failure Analysis & Classification

**Python · Pandas · Seaborn · Plotly · Scikit-learn · Machine Learning · Dash**

An exploratory analysis and binary machine-failure classification project using industrial-style operating data. The notebook investigates **machine types, operating conditions, sensor measurements, and recorded failure modes**, then builds and compares machine-learning models that classify whether an observation has a recorded failure.

> **Dataset:** [UCI Machine Learning Repository — AI4I 2020 Predictive Maintenance](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset). **AI4I 2020 is a synthetic benchmark dataset**, not live factory telemetry. This is an independent learning/portfolio project, not a deployed industrial monitoring system.

## Project objectives

- Explore data quality and identify patterns associated with machine failures.
- Compare machine-failure rates across product/machine types (`L`, `M`, and `H`).
- Study **air temperature, process temperature, rotational speed, torque, and tool wear**.
- Inspect recorded failure mode flags: **TWF, HDF, PWF, OSF, and RNF**.
- Train, evaluate, and compare multiple classification models.
- Tune a Random Forest model and assess its performance on held-out test data.
- Demonstrate a prediction for a new observation and save the trained model.

## Analysis and modeling workflow

| Stage | Implemented in the notebook |
| --- | --- |
| Data inspection | Data types, descriptive statistics, missing values, duplicates, and unique identifiers |
| EDA | Class distribution, failure rates by type, sensor distributions, correlations, and failure-mode counts |
| Exploratory feature engineering | Temperature difference, speed–torque interaction, and tool-wear interaction features |
| Train/test split | **80% train / 20% test**, stratified by the failure label, with `random_state=42` |
| Preprocessing | `StandardScaler` for numeric features and `OneHotEncoder` for machine type inside Scikit-learn pipelines |
| Baseline models | **Logistic Regression, Decision Tree, Random Forest** |
| Evaluation | Accuracy, precision, recall, F1 score, classification reports, and confusion matrices |
| Hyperparameter search | `RandomizedSearchCV` with **5-fold stratified cross-validation** for Random Forest |
| Model interpretation | Random Forest feature-importance visualization |
| Example inference | Prediction for a sample observation using the selected tuned Random Forest |
| Saving | Export the fitted model as `predictive_maintenance_model.pkl` with `joblib` |

### How target leakage is avoided

The dataset includes `TWF`, `HDF`, `PWF`, `OSF`, and `RNF`, which describe failure modes. These columns are **excluded from model inputs**, together with the target and identifier columns. The fitted classifier instead uses machine type and original sensor/operating variables. The notebook explores additional engineered features separately; those engineered columns are **not used by the classifier pipeline**.

### What the prediction means

The target `Machine failure` is a **binary recorded-failure label**, not remaining useful life or a time-to-failure forecast. Model outputs and estimated probabilities should **not** be treated as calibrated real-world risk scores or maintenance instructions.

## Interactive dashboard

The notebook includes a Plotly/ipywidgets-based dashboard section. An **optional Dash + Plotly replacement** adds browser-style interactive filters with a distinctive **Limestone / Sage / Rust / Ochre / Charcoal** palette.

In the Dash version:

- **Filters:** machine type, selected sensor, and tool-wear range.
- **Live KPIs:** observations, number of failures, failure rate, and average tool wear.
- **Visualizations:** machine-failure distribution, failure rates by machine type, sensor vs. tool-wear scatter plot, sensor histogram, recorded failure-mode flags, and model comparison.
- **Model comparison:** F1 and recall on the **original held-out test set**, independent of dashboard filters.

**Note:** A Dash app does not run inside GitHub's static notebook preview. To publish this as an interactive feature, replace the existing dashboard cell with the Dash code and run it locally.

## Dataset and installation

1. Download the [AI4I 2020 Predictive Maintenance dataset from UCI](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset).
2. Save the CSV as **`ai4i2020.csv`** in the same directory as the notebook.
3. Install the required packages:

   ```bash
   python -m pip install jupyter pandas numpy matplotlib seaborn plotly scikit-learn joblib missingno ipywidgets dash
   ```

4. Open the notebook:

   ```bash
   jupyter notebook "Predictive Maintenance Analysis(mini project)_Fixed.ipynb"
   ```

5. Run the cells **in order**. Model tuning can be more computationally intensive than the EDA cells. After adding the Dash replacement, run its cell at the end. The supplied interactive version uses port **8051**.

## Project structure

```text
Predictive_Maintenance/
├── README.md
├── Predictive Maintenance Analysis(mini project)_Fixed.ipynb
├── ai4i2020.csv                       # Download separately
└── predictive_maintenance_model.pkl  # Generated when the notebook is run
```

> The data file and trained-model artifact are not bundled in this README download.

## Interpretation and limitations

- **Synthetic data:** relationships may not match those in a particular factory.
- **Class imbalance:** failures are less frequent than normal observations, making recall, precision, and F1 important alongside accuracy.
- **No temporal validation:** the random train/test split is appropriate for a benchmark demonstration, but a real deployment would also need chronological testing.
- **Not causal:** correlations between sensor readings and failures do not prove their causes.
- **No downtime/repair-cost data:** operational value or cost savings cannot be calculated from this dataset alone.
- **No probability calibration or deployment monitoring:** scores require further validation before use in safety-critical operations.

## Data credit

**UCI Machine Learning Repository — AI4I 2020 Predictive Maintenance Dataset**. [Dataset page](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset).
