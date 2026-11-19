# Data-Science-Tool-Mastery-Project
A Python portfolio project showcasing NumPy, pandas, seaborn, SciPy and scikit-learn through hands-on exercises: vectorization, data wrangling, dashboards, statistical tests, model tuning and PCA.
Data Science Tool Mastery Project

A portfolio project demonstrating practical proficiency with the core Python data science stack: NumPy, pandas, Matplotlib/seaborn, SciPy and scikit-learn. One script walks through each tool with a focused, working exercise and produces figures and a metrics file you can show as evidence of your skills.

Objectives
Master industry-standard tools relevant to an AI and machine learning specialization.
Demonstrate proficiency through practical exercises.
Create portfolio pieces showcasing those skills.
Dataset

The Breast Cancer Wisconsin (Diagnostic) dataset, bundled with scikit-learn (569 samples, 30 numeric features, binary diagnosis). It needs no download, so the project runs fully offline. A synthetic time series and synthetic noisy curve data are generated for the time-series and curve-fitting exercises.

Project Structure
data_science_tool_mastery/
├── tool_mastery.py     # main script, one function per tool
├── requirements.txt    # dependencies
├── README.md           # this file
└── output/             # created when you run the script
    ├── 01_dashboard.png
    ├── 02_scipy_analysis.png
    ├── 03_sklearn_evaluation.png
    ├── model_comparison.csv
    └── results.json
Setup

Python 3.9 or later is recommended.

bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
How to Run
bash
python tool_mastery.py                              # run all sections
python tool_mastery.py --sections numpy sklearn     # run selected sections
python tool_mastery.py --output results             # custom output folder

Available sections: numpy, pandas, visualization, scipy, sklearn.

What Each Section Demonstrates
Tool	Skills demonstrated
NumPy	Vectorization versus Python loops (timed), broadcasting to standardize columns, solving linear systems, SVD low-rank approximation
pandas	Method chaining, binning with qcut, groupby with named aggregation, pivot tables, merges, time-series resampling and rolling windows
Matplotlib / seaborn	A six-panel dashboard: histogram with KDE, box plot, scatter plot, correlation heatmap, violin plot, time series with rolling mean
SciPy	Welch t-test, Mann-Whitney U test, Cohen's d effect size, BFGS optimization of the Rosenbrock function, nonlinear curve_fit
scikit-learn	Leakage-safe Pipelines, stratified 5-fold cross-validation of four models, GridSearchCV tuning, confusion matrix, ROC curves, feature importance, PCA
Sample Results

Figures from one run (timings vary by machine; model metrics are reproducible because seeds are fixed).

NumPy

Vectorized sum of squares over 1 million values ran roughly 150x to 170x faster than a Python loop, with identical results.
A rank-20 SVD approximation of a 100 x 60 random matrix had a relative error of about 0.30.

pandas

Malignancy rate rose sharply with tumour size: 2.1% in the smallest quartile versus 95.8% in the largest.

SciPy

Mean radius differs strongly between diagnoses (Welch t-test p < 1e-60, Cohen's d = 2.05).
BFGS recovered the Rosenbrock minimum at (1, 1), and curve_fit recovered the decay parameters (a, b, c) = (2.98, 1.285, 0.499) against true values (3.0, 1.3, 0.5).

scikit-learn (5-fold CV on the training set)

Model	CV accuracy	CV ROC-AUC
Logistic Regression	0.978	0.996
SVM (RBF)	0.969	0.996
Gradient Boosting	0.952	0.992
Random Forest	0.963	0.989
A tuned Random Forest (max_depth=5, n_estimators=300) scored 95.6% accuracy and 0.993 ROC-AUC on the held-out test set.
The simple, scaled Logistic Regression matched or beat the tree ensembles in cross-validation, a useful reminder that simple baselines can be very strong on clean tabular data.
Two PCA components explain 63.2% of the variance, and the classes separate clearly in that projection.
Key Practices Shown
No data leakage: scaling happens inside pipelines, so each cross-validation fold is scaled using only its own training data.
Reproducibility: fixed random seeds throughout.
Fair evaluation: stratified splits, cross-validation for model selection, and a held-out test set used only once at the end.
Statistical rigor: parametric and non-parametric tests plus an effect size, not just p-values.
Extending the Project
Add a deep learning section (PyTorch or TensorFlow) for a neural network baseline.
Add gradient boosting libraries such as XGBoost or LightGBM.
Add model explainability with SHAP.
Track experiments with MLflow and package the best model with joblib.
Swap in a real-world dataset from your own domain and reuse the same workflow.
Tools Used

Python, NumPy, pandas, Matplotlib, seaborn, SciPy, scikit-learn.
