# Student Performance Classification Using K-Nearest Neighbours

A machine learning project that predicts whether a student is likely to **Pass or Fail** based on demographic, family, social, school, and study-related characteristics using **K-Nearest Neighbours (KNN)**.

This project follows an end-to-end data science workflow including data provenance, data auditing, exploratory data analysis, leakage-safe preprocessing, baseline modeling, KNN development and tuning, model comparison, evaluation, error analysis, and reproducibility.

---

## 📌 Project Overview

Student academic performance can be influenced by several factors such as study time, previous failures, absences, family background, school-related factors, and social conditions.

The goal of this project is to build a classification system that predicts a student's final outcome:

* **Pass (1)** — Final grade `G3 >= 10`
* **Fail (0)** — Final grade `G3 < 10`

The main machine learning algorithm used in this project is **K-Nearest Neighbours (KNN)**.

Additional models are used for comparison to determine whether KNN provides a useful predictive performance.

---

## 🎯 Objectives

The main objectives of this project are:

1. Understand and audit the student performance dataset.
2. Perform exploratory data analysis (EDA).
3. Create a meaningful Pass/Fail classification target.
4. Split the dataset into training and testing sets without data leakage.
5. Build a leakage-safe preprocessing pipeline.
6. Establish a simple baseline model.
7. Develop and tune a KNN classifier.
8. Compare KNN with other classification models.
9. Evaluate the models using appropriate classification metrics.
10. Analyze classification errors and limitations.
11. Produce reproducible results and visualizations.

---

## 📊 Dataset

### Dataset Name

**Student Performance Dataset**

### Source

The dataset is obtained from the **UCI Machine Learning Repository**.

Dataset page:

https://archive.ics.uci.edu/dataset/320/student%2Bperformance

### Dataset Information

The dataset contains information about students from two Portuguese secondary schools.

It includes information related to:

* Student demographics
* Family background
* School information
* Study habits
* Social activities
* Previous academic performance
* Absences
* Final grades

For this project, the **Mathematics (`student-mat.csv`)** dataset is used.

### License

The UCI Student Performance dataset is available under the **CC BY 4.0** license.

### Dataset Citation

Cortez, Paulo. (2008). Student Performance [Dataset]. UCI Machine Learning Repository.

DOI:

https://doi.org/10.24432/C5TG7T

---

## 🎯 Target Variable

The original dataset contains the final grade:

```text
G3
```

The final grade ranges from `0` to `20`.

For classification, we create a new target variable called `pass`:

```python
df["pass"] = (df["G3"] >= 10).astype(int)
```

Therefore:

| G3      | Class    |
| ------- | -------- |
| `< 10`  | Fail (0) |
| `>= 10` | Pass (1) |

---

## ⚠️ Feature Selection and Leakage Consideration

The features `G1`, `G2`, and `G3` are excluded from the final feature set.

```python
X = df.drop(columns=["G1", "G2", "G3", "pass"])
y = df["pass"]
```

The reason for excluding `G1` and `G2` is that they are strongly related to the final grade `G3`.

Including these variables would make the classification problem much easier but would be less useful for an early student-risk prediction scenario.

Therefore, the project focuses on other available student characteristics instead of directly using previous/final grades.

---

## 🔄 Project Workflow

The project follows the following workflow:

```text
Problem Formulation
        ↓
Data Provenance
        ↓
Data Audit
        ↓
Exploratory Data Analysis
        ↓
Train/Test Split
        ↓
Preprocessing Pipeline
        ↓
Baseline Model
        ↓
KNN Development & Tuning
        ↓
Comparison Models
        ↓
Final Evaluation
        ↓
Error Analysis
        ↓
Interpretation & Limitations
        ↓
Conclusion
```

---

# 🧪 Methodology

## 1. Problem Formulation

The problem is formulated as a **binary classification problem**.

### Input

Student demographic, family, social, school, and study-related characteristics.

### Output

```text
0 → Fail
1 → Pass
```

### Main Model

**K-Nearest Neighbours (KNN)**

---

## 2. Data Audit

The dataset is checked for:

* Dataset dimensions
* Missing values
* Duplicate records
* Data types
* Numerical ranges
* Categorical values
* Invalid values
* Target distribution
* Descriptive statistics

Examples of checks include:

```python
df.shape
df.info()
df.describe()
df.isnull().sum()
df.duplicated().sum()
```

---

## 3. Exploratory Data Analysis

Several visualizations are used to understand the dataset.

The analysis includes:

* Pass/Fail distribution
* Final grade distribution
* Absences vs performance
* Study time vs performance
* Previous failures vs performance
* Feature relationships
* Correlation analysis

The visualizations help identify patterns and potential relationships before model development.

---

# ✂️ Train/Test Split

The dataset is divided into:

* **80% Training Data**
* **20% Testing Data**

A stratified split is used so that the class distribution is maintained.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)
```

The test set remains untouched until final evaluation.

---

# ⚙️ Preprocessing

Different preprocessing techniques are applied to numerical and categorical features.

### Numerical Features

```text
Missing-value imputation
        ↓
StandardScaler
```

### Categorical Features

```text
Missing-value imputation
        ↓
OneHotEncoder
```

These transformations are combined using `ColumnTransformer`.

The preprocessing steps are included inside a machine learning `Pipeline`.

This helps prevent **data leakage** because preprocessing parameters are learned only from the training data.

---

# 🤖 Models

## 1. Dummy Classifier

A simple baseline model is used to establish a minimum performance reference.

```python
DummyClassifier(strategy="most_frequent")
```

The machine learning models should be compared against this baseline.

---

## 2. Logistic Regression

Logistic Regression is used as a simple linear classification benchmark.

It provides a useful comparison against the distance-based KNN model.

---

## 3. K-Nearest Neighbours ⭐

KNN is the main model of this project.

KNN classifies a student based on the classes of nearby students in the feature space.

The model is tuned using cross-validation.

### Hyperparameters considered

* `n_neighbors`
* `weights`
* `metric`

Example values:

```text
n_neighbors:
3, 5, 7, 9, 11, 13, 15, 17, 19

weights:
uniform
distance

metric:
euclidean
manhattan
```

---

## 4. Random Forest

Random Forest is used as a nonlinear comparison model.

It combines multiple decision trees to make predictions and provides a useful comparison against KNN.

---

# 🔍 KNN Scaling Experiment

Because KNN is a distance-based algorithm, feature scaling is particularly important.

Without scaling, features with larger numerical ranges can have a greater influence on distance calculations.

The project therefore investigates the effect of scaling on KNN performance.

The comparison considers:

```text
KNN without scaling
        vs
KNN with scaling
```

This demonstrates why preprocessing decisions are important for distance-based machine learning algorithms.

---

# 🔧 Hyperparameter Tuning

KNN hyperparameters are selected using cross-validation on the training data.

A 5-fold stratified cross-validation strategy is used:

```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

The final test set is not used during hyperparameter selection.

---

# 📏 Evaluation Metrics

The models are evaluated using several classification metrics.

### Accuracy

Measures the percentage of correctly classified students.

### Precision

Measures how many students predicted as a particular class actually belong to that class.

### Recall

Measures how many students from a particular actual class were correctly identified.

### F1-Score

The harmonic mean of precision and recall.

### ROC-AUC

Measures the model's ability to distinguish between the two classes across classification thresholds.

---

## 📊 Confusion Matrix

A confusion matrix is used to analyze classification errors.

```text
                    Predicted
                 Fail       Pass

Actual Fail       TN         FP

Actual Pass       FN         TP
```

Where:

* **TN** = True Negative
* **TP** = True Positive
* **FP** = False Positive
* **FN** = False Negative

False positives and false negatives are discussed because different types of classification errors can have different consequences in a student-support scenario.

---

# 📈 Visualizations

The project generates several visualizations, including:

* Target distribution
* Feature distributions
* Correlation heatmap
* Study time vs performance
* Absences vs performance
* Confusion matrix
* ROC curve
* Precision-Recall curve
* Model performance comparison
* KNN hyperparameter performance

All generated figures are stored in:

```text
results/figures/
```

---

# 📊 Model Comparison

The final models are compared using the same test set and consistent evaluation metrics.

Example structure:

| Model               | Accuracy | Precision | Recall | F1 | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -: | ------: |
| Dummy Classifier    |        — |         — |      — |  — |       — |
| Logistic Regression |        — |         — |      — |  — |       — |
| KNN                 |        — |         — |      — |  — |       — |
| Random Forest       |        — |         — |      — |  — |       — |

The actual values in the final report are generated from the executed notebook.

The final model is selected based on evidence from:

* F1-score
* Recall
* Precision
* ROC-AUC
* Confusion matrix
* Robustness
* Interpretability
* Computational cost

---

# ❌ Error Analysis

The project investigates incorrectly classified students.

The analysis includes:

* False positives
* False negatives
* Misclassified student records
* Class imbalance
* Possible feature limitations
* Generalization concerns

This helps explain not only **how accurate the model is**, but also **where and why it makes mistakes**.

---

# ⚠️ Limitations

Some important limitations of this project are:

1. The dataset comes from Portuguese secondary schools and may not represent students from other countries or educational systems.
2. The dataset contains a relatively limited number of students.
3. Student performance can be influenced by factors that are not included in the dataset.
4. Classification into Pass/Fail simplifies a student's actual academic performance.
5. KNN can become computationally expensive as the dataset grows.
6. KNN performance can be sensitive to feature scaling and the choice of distance metric.
7. The model should not be used as the sole basis for important decisions about students.

---

# 🔐 Ethical Considerations

Student data can contain sensitive information about individuals and their personal circumstances.

Therefore:

* The dataset source and license are documented.
* The project uses a publicly available research dataset.
* The model should be used for educational analysis rather than making high-stakes decisions.
* Predictions should not be treated as definitive judgments about a student's ability.
* Model limitations and potential biases should be considered before real-world deployment.

---

# ♻️ Reproducibility

A fixed random seed is used throughout the project:

```python
RANDOM_STATE = 42
```

The project also documents the Python packages required to run the notebook.

Recommended environment:

```text
Python 3.x
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
jupyter
```

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

# 📁 Project Structure

```text
Student_Performance_Classification/
│
├── data/
│   └── raw/
│       └── student-mat.csv
│
├── notebooks/
│   └── Student_Performance_Classification.ipynb
│
├── results/
│   ├── figures/
│   ├── metrics/
│   └── model/
│
├── requirements.txt
├── README.md
└── Report.pdf
```

---

# ▶️ How to Run the Project

## Option 1: Google Colab

1. Upload or clone the project into Google Drive.
2. Open:

```text
notebooks/Student_Performance_Classification.ipynb
```

3. Mount Google Drive if required.
4. Update the project path.
5. Run the notebook cells from top to bottom.

---

## Option 2: Local Jupyter Notebook

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Move into the project directory:

```bash
cd Student_Performance_Classification
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Start Jupyter:

```bash
jupyter notebook
```

Open:

```text
notebooks/Student_Performance_Classification.ipynb
```

Run all cells from top to bottom.

---

# 💾 Output Files

The project generates:

```text
results/
│
├── figures/
│   ├── target_distribution.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   └── model_comparison.png
│
├── metrics/
│   └── model_comparison.csv
│
└── model/
    └── knn_student_performance_model.joblib
```

The exact filenames may vary depending on the final implementation.

---

# 👥 Team Contributions

The project is divided among three team members according to the end-to-end data science workflow.

## Member 1 — Data & Exploratory Analysis

Responsibilities:

* Problem formulation
* Dataset collection and provenance
* Data dictionary
* Data auditing
* Missing-value and duplicate checks
* Exploratory data analysis
* Initial visualizations

### Main workflow sections

```text
DW-1
DW-2
DW-3
DW-4
```

---

## Member 2 — Preprocessing & KNN

Responsibilities:

* Target variable creation
* Feature selection
* Train/test split
* Preprocessing pipeline
* Feature scaling
* Baseline model
* KNN implementation
* KNN hyperparameter tuning
* Cross-validation

### Main workflow sections

```text
DW-5
DW-6
DW-7
DW-8
```

---

## Member 3 — Evaluation & Model Comparison

Responsibilities:

* Logistic Regression comparison model
* Random Forest comparison model
* Final model evaluation
* Confusion matrix
* ROC/PR curves
* Model comparison
* Error analysis
* Limitations
* Final interpretation and conclusion

### Main workflow sections

```text
DW-8
DW-9
DW-10
```

---

# 📚 Course Requirements Covered

This project addresses the required data workflow:

| Requirement | Implementation                 |
| ----------- | ------------------------------ |
| DW-1        | Problem formulation            |
| DW-2        | Dataset provenance             |
| DW-3        | Data audit                     |
| DW-4        | Exploratory data analysis      |
| DW-5        | Train/test split               |
| DW-6        | Leakage-safe preprocessing     |
| DW-7        | Baseline model                 |
| DW-8        | Model development and tuning   |
| DW-9        | Final test evaluation          |
| DW-10       | Error analysis and limitations |

The project also includes:

* Reproducibility
* Visualization
* Model comparison
* Classification metrics
* Confusion matrix
* Error analysis
* Responsible-use discussion
* Computational documentation

---

# 📖 References

1. Cortez, P. (2008). **Student Performance**. UCI Machine Learning Repository.

   https://archive.ics.uci.edu/dataset/320/student%2Bperformance

2. Scikit-learn Documentation.

   https://scikit-learn.org/

3. Python Documentation.

   https://docs.python.org/3/

---

# ⭐ Acknowledgement

This project was developed as part of the **Programming in Python** course at **American International University-Bangladesh (AIUB)**.

The project focuses on applying fundamental data science and machine learning concepts in a reproducible and defensible workflow rather than focusing only on achieving the highest possible accuracy.

---

## 📌 Project Status

**Status:** Completed / In Progress

**Main Algorithm:** K-Nearest Neighbours (KNN)

**Problem Type:** Binary Classification

**Dataset:** UCI Student Performance

**Programming Language:** Python

**Environment:** Google Colab / Jupyter Notebook
