# Breast Cancer Diagnosis using Machine Learning

Mini Project | Machine Learning | TY Div B | Academic Year 2026-27 (Semester-I)

**Name:** Lazina Shaikh
**PRN:** RBT24CS136
**Roll No:** 64
**College:** Rajarshi Shahu College of Engineering, Tathawade, Pune

---

## About the Project

This project uses machine learning to classify breast tumours as **benign (0)** or **malignant (1)** using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset. Seven classification algorithms are trained, compared, cross-validated and tuned. Recall is given special importance because missing a cancer case is more dangerous than a false alarm.

## Dataset

- **Name:** Wisconsin Diagnostic Breast Cancer (WDBC), file `data.csv`
- **Samples:** 569 (357 benign, 212 malignant)
- **Features:** 30 numeric features computed from cell nuclei (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension, each as mean, standard error and worst value)
- **Target:** `diagnosis` (M = 1, B = 0)
- **Missing values / duplicates:** none

## Methodology

1. Data loading and cleaning
2. Exploratory data analysis (class counts, histograms, box plots, correlation heatmap, pair plot)
3. Stratified 80:20 train/test split and StandardScaler
4. PCA visualisation
5. Feature selection: correlation filter (above 0.90), then SelectKBest with ANOVA F-test (top 8 features)
6. Training of 7 models
7. Evaluation using accuracy, precision, recall, F1-score, AUC, confusion matrix and ROC curve
8. Stratified 5-fold cross-validation
9. Hyperparameter tuning with GridSearchCV (scoring = recall)
10. Model comparison, saving the best model and sample prediction

## Models Used

| Level | Model |
|---|---|
| Basic | Logistic Regression, K-Nearest Neighbours, Decision Tree |
| Intermediate | Support Vector Machine, Random Forest |
| Advanced | AdaBoost, Gradient Boosting |

## Results (Tuned Models, Test Set)

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.9649** | **0.9750** | **0.9286** | **0.9512** | **0.9947** |
| SVM | 0.9649 | 0.9750 | 0.9286 | 0.9512 | 0.9950 |
| Random Forest | 0.9474 | 0.9737 | 0.8810 | 0.9250 | 0.9848 |
| Gradient Boosting | 0.9474 | 0.9737 | 0.8810 | 0.9250 | 0.9838 |
| AdaBoost | 0.9386 | 0.9487 | 0.8810 | 0.9136 | 0.9924 |
| KNN | 0.9298 | 0.9474 | 0.8571 | 0.9000 | 0.9889 |
| Decision Tree | 0.9298 | 0.9250 | 0.8810 | 0.9024 | 0.9196 |

**Best model:** Logistic Regression (C = 10). It detected 39 of 42 malignant cases, missed 3, and gave 1 false alarm. SVM matched its scores.

**Most important features:** radius_mean, concavity_mean, concavity_worst.

## Files in this Repository

| File | Description |
|---|---|
| `ml_mini_project_.ipynb` | Main notebook with all code and outputs |
| `data.csv` | Dataset |
| `ML_Mini_Project_Report_Lazina_Shaikh.docx` | Full project report |

## How to Run

1. Open `ml_mini_project_.ipynb` in Google Colab or Jupyter Notebook.
2. Upload `data.csv` when the notebook asks for it (or place it in the same folder).
3. Run all cells (Runtime > Run all).

**Libraries required:** numpy, pandas, matplotlib, seaborn, scikit-learn, joblib

To install them, run: `pip install numpy pandas matplotlib seaborn scikit-learn joblib`

## Conclusion

All seven models performed well (accuracy between 92.98% and 96.49%). The tuned Logistic Regression model gave the best result with 96.49% accuracy and 92.86% recall. Larger and more irregularly shaped cell nuclei (radius and concavity) are the strongest signs of malignancy.
