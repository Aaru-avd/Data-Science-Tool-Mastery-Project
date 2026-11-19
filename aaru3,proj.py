"""
Data Science Tool Mastery Project
---------------------------------
A portfolio script that demonstrates practical proficiency with the core
industry-standard Python data science stack:

    1. NumPy         - vectorization, broadcasting, linear algebra
    2. pandas        - cleaning, grouping, pivoting, merging, time series
    3. Matplotlib /
       seaborn       - multi-panel visualization dashboard
    4. SciPy         - hypothesis testing, optimization, curve fitting
    5. scikit-learn  - pipelines, cross-validation, tuning, evaluation, PCA

It uses scikit-learn's bundled Breast Cancer Wisconsin dataset, so it runs
fully offline.

Usage:
    python tool_mastery.py                         # run everything
    python tool_mastery.py --sections numpy sklearn
    python tool_mastery.py --output results
"""

import argparse
import json
import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")  # save figures without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import optimize, stats
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="deep")

SEED = 42
RESULTS = {}  # collected metrics, written to the final report


def banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def save_fig(fig, name, outdir):
    path = os.path.join(outdir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved figure: {path}")


# ----------------------------------------------------------------------
# 1. NUMPY
# ----------------------------------------------------------------------
def numpy_demo():
    banner("1. NUMPY: vectorization, broadcasting, linear algebra")
    rng = np.random.default_rng(SEED)

    # Vectorization vs Python loop
    data = rng.random(1_000_000)

    start = time.perf_counter()
    loop_total = 0.0
    for v in data:
        loop_total += v * v
    loop_time = time.perf_counter() - start

    start = time.perf_counter()
    vec_total = np.dot(data, data)
    vec_time = time.perf_counter() - start

    speedup = loop_time / max(vec_time, 1e-9)
    print(f"Sum of squares (1M values): loop {loop_time:.4f}s, NumPy {vec_time:.6f}s")
    print(f"Results match: {np.isclose(loop_total, vec_total)} | speedup ~{speedup:,.0f}x")

    # Broadcasting: standardize each column of a matrix without loops
    matrix = rng.normal(loc=[10, 50, 200], scale=[1, 5, 20], size=(1000, 3))
    standardized = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)
    print("Column means after standardizing:", np.round(standardized.mean(axis=0), 6))
    print("Column stds after standardizing: ", np.round(standardized.std(axis=0), 6))

    # Linear algebra: solve a linear system and verify
    A = rng.random((50, 50)) + 50 * np.eye(50)
    b = rng.random(50)
    x = np.linalg.solve(A, b)
    print(f"Linear system solved correctly: {np.allclose(A @ x, b)}")

    # SVD: low-rank approximation
    M = rng.random((100, 60))
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    k = 20
    M_k = (U[:, :k] * S[:k]) @ Vt[:k]
    rel_err = np.linalg.norm(M - M_k) / np.linalg.norm(M)
    print(f"Rank-{k} SVD approximation relative error: {rel_err:.3f}")

    RESULTS["numpy"] = {
        "vectorization_speedup_x": round(float(speedup), 1),
        "linear_solve_correct": bool(np.allclose(A @ x, b)),
        "svd_rank20_relative_error": round(float(rel_err), 4),
    }


# ----------------------------------------------------------------------
# 2. PANDAS
# ----------------------------------------------------------------------
def pandas_demo():
    banner("2. PANDAS: loading, grouping, pivoting, merging, time series")
    raw = load_breast_cancer(as_frame=True)
    df = raw.frame.copy()
    df["diagnosis"] = df["target"].map({0: "malignant", 1: "benign"})
    print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Missing values: {int(df.isnull().sum().sum())}")

    # Method chaining: feature + filter + sort
    top = (
        df.assign(area_to_perimeter=lambda d: d["mean area"] / d["mean perimeter"])
        .query("`mean radius` > 15")
        .sort_values("area_to_perimeter", ascending=False)
        .loc[:, ["mean radius", "mean area", "area_to_perimeter", "diagnosis"]]
        .head(5)
    )
    print("\nTop 5 large tumours by area-to-perimeter ratio:")
    print(top.round(2).to_string(index=False))

    # Binning + groupby + aggregation
    df["size_group"] = pd.qcut(
        df["mean radius"], q=4, labels=["Small", "Medium", "Large", "X-Large"]
    )
    grouped = df.groupby("size_group", observed=True).agg(
        patients=("target", "size"),
        malignant_rate=("target", lambda s: 1 - s.mean()),
        avg_area=("mean area", "mean"),
    )
    print("\nMalignancy rate by tumour size group:")
    print(grouped.round(3).to_string())

    # Pivot table
    pivot = pd.pivot_table(
        df, values="mean texture", index="size_group", columns="diagnosis",
        aggfunc="mean", observed=True,
    )
    print("\nMean texture, size group x diagnosis:")
    print(pivot.round(2).to_string())

    # Merge with a lookup table
    lookup = pd.DataFrame({
        "size_group": ["Small", "Medium", "Large", "X-Large"],
        "follow_up": ["Annual", "6 months", "3 months", "Immediate"],
    })
    merged = df.merge(lookup, on="size_group", how="left")
    print("\nFollow-up plan counts after merge:")
    print(merged["follow_up"].value_counts().to_string())

    # Time series: synthetic daily readings, resample and rolling mean
    rng = np.random.default_rng(SEED)
    days = pd.date_range("2024-01-01", periods=365, freq="D")
    trend = np.linspace(100, 130, 365)
    seasonal = 8 * np.sin(np.arange(365) * 2 * np.pi / 30)
    ts = pd.Series(trend + seasonal + rng.normal(0, 3, 365), index=days, name="reading")
    monthly = ts.resample("MS").mean()
    rolling = ts.rolling(window=7).mean()
    print(f"\nTime series: {len(ts)} daily points -> {len(monthly)} monthly means")
    print(f"First monthly mean {monthly.iloc[0]:.1f}, last {monthly.iloc[-1]:.1f}")

    RESULTS["pandas"] = {
        "rows": int(df.shape[0]),
        "malignancy_rate_small": round(float(grouped.loc["Small", "malignant_rate"]), 3),
        "malignancy_rate_xlarge": round(float(grouped.loc["X-Large", "malignant_rate"]), 3),
        "monthly_points": int(len(monthly)),
    }
    return df, ts, rolling


# ----------------------------------------------------------------------
# 3. MATPLOTLIB / SEABORN
# ----------------------------------------------------------------------
def viz_demo(df, ts, rolling, outdir):
    banner("3. MATPLOTLIB / SEABORN: visualization dashboard")
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    sns.histplot(data=df, x="mean radius", hue="diagnosis", kde=True,
                 ax=axes[0, 0], palette={"malignant": "#c0392b", "benign": "#27ae60"})
    axes[0, 0].set_title("Mean Radius Distribution")

    sns.boxplot(data=df, x="diagnosis", y="mean area", ax=axes[0, 1],
                palette={"malignant": "#c0392b", "benign": "#27ae60"})
    axes[0, 1].set_title("Mean Area by Diagnosis")

    sns.scatterplot(data=df, x="mean radius", y="mean texture", hue="diagnosis",
                    alpha=0.7, ax=axes[0, 2],
                    palette={"malignant": "#c0392b", "benign": "#27ae60"})
    axes[0, 2].set_title("Radius vs Texture")

    corr = df.drop(columns=["diagnosis", "size_group"]).corr()["target"].abs()
    top_features = corr.sort_values(ascending=False).index[1:9]
    sns.heatmap(df[top_features].corr(), annot=True, fmt=".2f", cmap="coolwarm",
                ax=axes[1, 0], cbar=False, annot_kws={"size": 7})
    axes[1, 0].set_title("Correlation of Top 8 Features")
    axes[1, 0].tick_params(axis="both", labelsize=7)

    sns.violinplot(data=df, x="size_group", y="mean concavity", ax=axes[1, 1])
    axes[1, 1].set_title("Concavity by Tumour Size Group")

    axes[1, 2].plot(ts.index, ts.values, alpha=0.4, label="Daily")
    axes[1, 2].plot(rolling.index, rolling.values, color="black", label="7-day mean")
    axes[1, 2].set_title("Time Series with Rolling Mean")
    axes[1, 2].legend()
    axes[1, 2].tick_params(axis="x", rotation=30)

    fig.suptitle("Data Science Tool Mastery: Visualization Dashboard", fontsize=16)
    save_fig(fig, "01_dashboard.png", outdir)
    RESULTS["visualization"] = {"panels": 6, "figure": "01_dashboard.png"}


# ----------------------------------------------------------------------
# 4. SCIPY
# ----------------------------------------------------------------------
def scipy_demo(df, outdir):
    banner("4. SCIPY: statistics, optimization, curve fitting")

    # Hypothesis tests
    mal = df.loc[df["diagnosis"] == "malignant", "mean radius"]
    ben = df.loc[df["diagnosis"] == "benign", "mean radius"]
    t, p_t = stats.ttest_ind(mal, ben, equal_var=False)
    u, p_u = stats.mannwhitneyu(mal, ben)
    pooled = np.sqrt((mal.var(ddof=1) + ben.var(ddof=1)) / 2)
    cohens_d = (mal.mean() - ben.mean()) / pooled
    print(f"Welch t-test:   t = {t:.2f}, p = {p_t:.3g}")
    print(f"Mann-Whitney U: U = {u:.0f}, p = {p_u:.3g}")
    print(f"Cohen's d (effect size): {cohens_d:.2f}")

    # Optimization: Rosenbrock function
    def rosenbrock(v):
        return (1 - v[0]) ** 2 + 100 * (v[1] - v[0] ** 2) ** 2

    res = optimize.minimize(rosenbrock, x0=[-1.5, 2.0], method="BFGS")
    print(f"Rosenbrock minimum found at {np.round(res.x, 4)} (true: [1, 1])")

    # Curve fitting: exponential decay with noise
    rng = np.random.default_rng(SEED)
    x = np.linspace(0, 5, 80)
    true_a, true_b, true_c = 3.0, 1.3, 0.5
    y = true_a * np.exp(-true_b * x) + true_c + rng.normal(0, 0.05, x.size)

    def decay(x, a, b, c):
        return a * np.exp(-b * x) + c

    popt, _ = optimize.curve_fit(decay, x, y)
    print(f"Curve fit parameters a, b, c = {np.round(popt, 3)} (true: {[true_a, true_b, true_c]})")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].scatter(x, y, s=12, alpha=0.6, label="Noisy data")
    ax[0].plot(x, decay(x, *popt), "r-", lw=2, label="Fitted curve")
    ax[0].set_title("SciPy curve_fit: Exponential Decay")
    ax[0].legend()
    sns.kdeplot(mal, fill=True, label="Malignant", ax=ax[1], color="#c0392b")
    sns.kdeplot(ben, fill=True, label="Benign", ax=ax[1], color="#27ae60")
    ax[1].set_title(f"Mean Radius by Diagnosis (Cohen's d = {cohens_d:.2f})")
    ax[1].legend()
    save_fig(fig, "02_scipy_analysis.png", outdir)

    RESULTS["scipy"] = {
        "welch_t_p_value": float(p_t),
        "mann_whitney_p_value": float(p_u),
        "cohens_d": round(float(cohens_d), 2),
        "rosenbrock_solution": [round(float(v), 4) for v in res.x],
        "curve_fit_params": [round(float(v), 3) for v in popt],
    }


# ----------------------------------------------------------------------
# 5. SCIKIT-LEARN
# ----------------------------------------------------------------------
def sklearn_demo(outdir):
    banner("5. SCIKIT-LEARN: pipelines, CV, tuning, evaluation, PCA")
    data = load_breast_cancer()
    X, y = data.data, data.target
    names = data.feature_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Compare models with cross-validation inside pipelines (no data leakage)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "SVM (RBF)": SVC(probability=True, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for name, model in models.items():
        pipe = Pipeline([("scale", StandardScaler()), ("model", model)])
        scores = cross_validate(pipe, X_train, y_train, cv=cv,
                                scoring=["accuracy", "roc_auc"], n_jobs=-1)
        rows.append({
            "model": name,
            "cv_accuracy": scores["test_accuracy"].mean(),
            "cv_accuracy_std": scores["test_accuracy"].std(),
            "cv_roc_auc": scores["test_roc_auc"].mean(),
        })
    cv_table = pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False)
    print("\n5-fold cross-validation results:")
    print(cv_table.round(4).to_string(index=False))

    # Hyperparameter tuning with GridSearchCV
    pipe = Pipeline([("scale", StandardScaler()),
                     ("model", RandomForestClassifier(random_state=SEED))])
    grid = GridSearchCV(
        pipe,
        param_grid={
            "model__n_estimators": [100, 300],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_leaf": [1, 3],
        },
        cv=cv, scoring="roc_auc", n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    print(f"\nBest Random Forest params: {grid.best_params_}")
    print(f"Best CV ROC-AUC: {grid.best_score_:.4f}")

    # Final evaluation on the held-out test set
    best = grid.best_estimator_
    pred = best.predict(X_test)
    proba = best.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, proba)
    print(f"\nTest ROC-AUC: {test_auc:.4f}")
    print(classification_report(y_test, pred, target_names=data.target_names))
    test_acc = float((pred == y_test).mean())

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(StandardScaler().fit_transform(X))
    var = pca.explained_variance_ratio_
    print(f"PCA: 2 components explain {var.sum() * 100:.1f}% of variance")

    # Figures
    fig, ax = plt.subplots(2, 2, figsize=(13, 11))
    ConfusionMatrixDisplay.from_estimator(best, X_test, y_test, ax=ax[0, 0],
                                          display_labels=data.target_names, cmap="Blues")
    ax[0, 0].set_title("Confusion Matrix (Test Set)")

    for name, model in models.items():
        p = Pipeline([("scale", StandardScaler()), ("model", model)]).fit(X_train, y_train)
        RocCurveDisplay.from_estimator(p, X_test, y_test, ax=ax[0, 1], name=name)
    ax[0, 1].plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax[0, 1].set_title("ROC Curves (Test Set)")

    importances = pd.Series(best.named_steps["model"].feature_importances_, index=names)
    importances.nlargest(10).sort_values().plot(kind="barh", ax=ax[1, 0], color="teal")
    ax[1, 0].set_title("Top 10 Feature Importances (Tuned Random Forest)")

    scatter = ax[1, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="coolwarm", alpha=0.7, s=20)
    ax[1, 1].set_xlabel(f"PC1 ({var[0] * 100:.1f}%)")
    ax[1, 1].set_ylabel(f"PC2 ({var[1] * 100:.1f}%)")
    ax[1, 1].set_title("PCA Projection")
    handles, _ = scatter.legend_elements()
    ax[1, 1].legend(handles, list(data.target_names), title="Class")
    save_fig(fig, "03_sklearn_evaluation.png", outdir)

    cv_table.to_csv(os.path.join(outdir, "model_comparison.csv"), index=False)

    RESULTS["sklearn"] = {
        "best_cv_model": cv_table.iloc[0]["model"],
        "best_cv_roc_auc": round(float(cv_table.iloc[0]["cv_roc_auc"]), 4),
        "tuned_rf_params": {k: (None if v is None else int(v)) for k, v in grid.best_params_.items()},
        "test_accuracy": round(test_acc, 4),
        "test_roc_auc": round(float(test_auc), 4),
        "pca_2_component_variance_pct": round(float(var.sum() * 100), 1),
    }


# ----------------------------------------------------------------------
# FINAL REPORT
# ----------------------------------------------------------------------
def write_report(outdir):
    banner("SKILLS SUMMARY")
    skills = {
        "numpy": "Vectorization, broadcasting, linear solve, SVD",
        "pandas": "Method chaining, groupby/agg, pivot, merge, time series",
        "visualization": "Multi-panel dashboards with Matplotlib and seaborn",
        "scipy": "Hypothesis tests, effect size, optimization, curve fitting",
        "sklearn": "Pipelines, CV, GridSearch, ROC/confusion matrix, PCA",
    }
    for key, desc in skills.items():
        status = "completed" if key in RESULTS else "skipped"
        print(f"  [{status:>9}] {key:<14} {desc}")

    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump(RESULTS, f, indent=2)
    print(f"\nMetrics saved to: {os.path.join(outdir, 'results.json')}")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Data Science Tool Mastery Project")
    parser.add_argument("--sections", nargs="+",
                        choices=["numpy", "pandas", "visualization", "scipy", "sklearn"],
                        default=["numpy", "pandas", "visualization", "scipy", "sklearn"],
                        help="Which sections to run (default: all)")
    parser.add_argument("--output", default="output", help="Output folder (default: output)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    run = set(args.sections)

    df = ts = rolling = None
    if "numpy" in run:
        numpy_demo()
    if run & {"pandas", "visualization", "scipy"}:
        df, ts, rolling = pandas_demo()
        if "pandas" not in run:
            RESULTS.pop("pandas", None)
    if "visualization" in run:
        viz_demo(df, ts, rolling, args.output)
    if "scipy" in run:
        scipy_demo(df, args.output)
    if "sklearn" in run:
        sklearn_demo(args.output)

    write_report(args.output)
    print("\nAll selected sections finished.")


if __name__ == "__main__":
    main()
