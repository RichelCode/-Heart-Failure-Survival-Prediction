# Heart Failure Survival Prediction — End-to-End Machine Learning Project

## Project Overview
This project applies a comprehensive machine learning pipeline to predict survival outcomes for heart failure patients using clinical, demographic, and biochemical data sourced from the UCI Machine Learning Repository.

The goal is to develop an interpretable, high-performing model to help identify patients at the highest risk of death, enabling early intervention and improving patient outcomes.

## Dataset
- Source: [UCI Heart Failure Clinical Records Dataset](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records)
- Observations: 299 patients
- Features: 13 clinical, demographic, and biochemical attributes

### Key Variables
| Variable                    | Description |
|----------------|----------------|
| age                          | Patient age in years |
| anaemia                  | Low red blood cell count (1 = Yes, 0 = No) |
| creatinine_phosphokinase  | CPK enzyme level (mcg/L) |
| diabetes                | Diabetes diagnosis (1 = Yes, 0 = No) |
| ejection_fraction     | Blood leaving heart per contraction (%) |
| high_blood_pressure  | Hypertension diagnosis (1 = Yes, 0 = No) |
| platelets                | Platelet count (kiloplatelets/mL) |
| serum_creatinine      | Serum creatinine level (mg/dL) |
| serum_sodium         | Serum sodium level (mEq/L) |
| sex                         | Gender (1 = Male, 0 = Female) |
| smoking                  | Smoking history (1 = Yes, 0 = No) |
| time                     | Follow-up period in days |
| DEATH_EVENT      | Target: Death during follow-up (1 = Yes, 0 = No) |

---

## Project Workflow
This project follows a structured end-to-end machine learning pipeline, including:

### 1. Exploratory Data Analysis (EDA)
- Distribution of clinical attributes
- Survival rates across demographic and clinical groups
- Bivariate and multivariate analysis
- Identification of outliers and data quality checks

### 2. Feature Engineering
- Log Transformation applied to skewed biochemical markers (CPK, platelets, creatinine)
- Clinical Binning applied to:
    - Age (grouped into meaningful age brackets)
    - Ejection Fraction (categorized into severe, moderate, and normal)
    - Follow-up Time (grouped into short, medium, and long)

### 3. Addressing Class Imbalance
- Applied SMOTE (Synthetic Minority Oversampling Technique) to the training set to ensure sufficient representation of patients who died.

### 4. Model Training & Evaluation
The following classification models were trained and evaluated across four distinct feature sets (Original, Binned, Log Transformed, Log + Binned):
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest

All models were:
- Tuned using GridSearchCV for optimal hyperparameters.
- Evaluated using the original imbalanced test set to simulate real-world conditions.
- Performance was compared using:
    - Confusion Matrix
    - Classification Report (Precision, Recall, F1-score)
    - ROC AUC (Primary metric)

---

## Final Model Selection

| Feature Set                | Logistic Regression | KNN   | SVM   | Decision Tree | Random Forest |
|-------------------|---------------------|------|------|----------------|----------------|
| Original                | 0.7356 | 0.7375 | 0.7375 | 0.6624 | 0.8023 |
| Binned                   | 0.8665 | 0.8710 | 0.7843 | 0.8196 | 0.8485 |
| Log Transformed    | 0.7227 | 0.7728 | 0.7548 | 0.5822 | 0.7933 |
| Log + Binned         | 0.8460 | 0.7715 | 0.8370 | 0.8293 | 0.8890 |

Best Model: Random Forest trained on Log + Binned Features (ROC AUC = 0.8890)

This model offered:
- Strong predictive performance.
- Robust handling of class imbalance through SMOTE.
- Interpretability via feature importance scores.

---

## Key Insights
- Feature Engineering Matters: Combining domain expertise (clinical bins) with statistical transformations (log) produced the best results across all models.
- SMOTE Improves Minority Class Learning: Applying SMOTE gave all models better exposure to patients who died, improving sensitivity to high-risk cases.
- Ensemble Models Excel: Random Forest outperformed all other models, confirming that ensembles can better capture the complex interactions between clinical and biochemical factors driving survival outcomes.

---

## Technologies Used
- Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn, Imbalanced-Learn)
- Jupyter Notebook

---

## Visual Highlights
This project includes a rich set of visualizations, such as:
- Distribution plots for each clinical variable
- Survival rates across demographic and clinical subgroups
- Correlation heatmaps (colorblind friendly)
- Performance comparison bar charts for all models and feature sets

---

## How to Run
1. Clone the repository.
2. Install required libraries:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the Jupyter Notebook to see the complete analysis and modeling pipeline.

---

## Lessons Learned
- Importance of domain knowledge in healthcare machine learning.
- How feature engineering directly impacts model performance.
- Effective handling of imbalanced clinical datasets.
- End-to-end project documentation for portfolio-ready projects.

---

## Future Work
- Explore advanced models like XGBoost or LightGBM.
- Apply SHAP values for deeper interpretability.
- Investigate more sophisticated sampling techniques like SMOTE-ENN or SMOTE-Tomek.
- Package into a Streamlit app for clinicians to use interactively.

---

## Final Note
This project showcases both technical machine learning skills and the ability to apply domain knowledge in a real-world healthcare context — a must-have for any aspiring data scientist in healthtech.

---

## Credits
Data Source: [UCI Machine Learning Repository - Heart Failure Clinical Records](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records)

---

## Repository Structure (Optional Example)

```
heart-failure-survival-prediction/
├── Heart_Survival_Prediction_Analysis.ipynb  # Full notebook
├── README.md
├── requirements.txt
├── data/
│   └── heart_failure_clinical_records.csv
└── images/
    └── Key EDA and Model Performance Plots
```

