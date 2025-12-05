import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_squared_error,
    make_scorer,
)

from sklearn.decomposition import PCA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet


def run_models_auto(
    df,
    y_col,
    X_cols,
    test_size=0.3,
    random_state=1,
    max_classes_for_classification=10,
    cv=5,
):

    if isinstance(y_col, str):
        y = df[y_col]
    else:
        y = pd.Series(y_col)

    if isinstance(X_cols, (list, tuple, pd.Index)):
        X = df[list(X_cols)]
    elif isinstance(X_cols, pd.DataFrame):
        X = X_cols
    else:
        raise ValueError("X_cols must be a list.")

    is_numeric = np.issubdtype(y.dtype, np.number)
    value_counts = y.value_counts()

    if (not is_numeric) or (
        value_counts.nunique() > 0
        and len(value_counts) <= max_classes_for_classification
        and value_counts.min() >= cv
    ):
        task = "classification"
    else:
        task = "regression"

    print(f"\nDetected task type: {task.upper()}")

    results = []

    if task == "classification":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        acc_scorer = make_scorer(accuracy_score)

        models_and_grids = {
            "LDA": (
                Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", LinearDiscriminantAnalysis())
                ]),
                {
                    "clf__solver": ["svd", "lsqr"],
                },
            ),

            "QDA": (
                Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", QuadraticDiscriminantAnalysis())
                ]),
                {
                    "clf__reg_param": [0.0, 0.001, 0.01, 0.1, 0.5],
                },
            ),

            "Naive Bayes": (
                GaussianNB(),
                {
                    "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6],
                },
            ),

            "Decision Tree": (
                DecisionTreeClassifier(random_state=random_state),
                {
                    "max_depth": [None, 3, 5, 10],
                    "min_samples_split": [2, 5, 10],
                },
            ),

            "Random Forest": (
                RandomForestClassifier(random_state=random_state),
                {
                    "n_estimators": [100, 300],
                    "max_depth": [None, 5, 10],
                    "min_samples_split": [2, 5],
                },
            ),

            "KNN": (
                Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", KNeighborsClassifier())
                ]),
                {
                    "clf__n_neighbors": [3, 5, 7, 11],
                    "clf__weights": ["uniform", "distance"],
                },
            ),

            "Linear SVM": (
                Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", SVC(kernel="linear", probability=True, random_state=random_state))
                ]),
                {
                    "clf__C": [0.1, 1, 10, 100],
                },
            ),

            "RBF SVM": (
                Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", SVC(kernel="rbf", probability=True, random_state=random_state))
                ]),
                {
                    "clf__C": [0.1, 1, 10, 100],
                    "clf__gamma": ["scale", 0.01, 0.1, 1],
                },
            ),
        }

        for name, (estimator, param_grid) in models_and_grids.items():
            print(f"\nRunning {name}...")

            grid = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                scoring=acc_scorer,
                cv=cv,
                n_jobs=-1,
            )

            grid.fit(X_train, y_train)

            best_cv_acc = grid.best_score_
            best_cv_error = 1 - best_cv_acc

            y_pred_test = grid.best_estimator_.predict(X_test)
            test_acc = accuracy_score(y_test, y_pred_test)
            test_error = 1 - test_acc

            print(f"  Best params: {grid.best_params_}")
            print(f"  CV accuracy:   {best_cv_acc:.4f} (error {best_cv_error:.4f})")
            print(f"  Test accuracy: {test_acc:.4f} (error {test_error:.4f})")

            results.append({
                "model": name,
                "best_params": grid.best_params_,
                "cv_accuracy": best_cv_acc,
                "cv_error": best_cv_error,
                "test_accuracy": test_acc,
                "test_error": test_error,
                "best_estimator": grid.best_estimator_,
            })

        best_result = min(results, key=lambda r: r["test_error"])
        best_model = best_result["best_estimator"]

        y_best_pred = best_model.predict(X_test)
        best_conf_mat = confusion_matrix(y_test, y_best_pred)

        alpha_like = None
        for key, val in best_result["best_params"].items():
            if any(k in key.lower() for k in ["alpha", "c", "reg_param", "gamma"]):
                alpha_like = (key, val)
                break

        print("\nModel Selection Summary: Classification")
        print(f"For this data, {best_result['model']} is best to use,")
        print(f"because it has the lowest test error rate = {best_result['test_error']:.4f} "
              f"(accuracy = {best_result['test_accuracy']:.4f}).")

        if alpha_like is not None:
            print(f"Key tuned hyperparameter (your 'alpha') is {alpha_like[0]} = {alpha_like[1]}.")

        print("\nBest model hyperparameters:")
        print(best_result["best_params"])

        print("\nConfusion matrix for the best model on the test set:")
        print(best_conf_mat)

        results_df = pd.DataFrame([
            {
                "model": r["model"],
                "cv_accuracy": r["cv_accuracy"],
                "cv_error": r["cv_error"],
                "test_accuracy": r["test_accuracy"],
                "test_error": r["test_error"],
                "best_params": r["best_params"],
            }
            for r in results
        ]).sort_values(by="test_error")

        extra = best_conf_mat

    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

        models_and_grids = {
            "Linear Regression": (
                Pipeline([
                    ("scaler", StandardScaler()),
                    ("reg", LinearRegression())
                ]),
                {
                    "reg__fit_intercept": [True, False],
                },
            ),

            "Ridge": (
                Pipeline([
                    ("scaler", StandardScaler()),
                    ("reg", Ridge(random_state=random_state))
                ]),
                {
                    "reg__alpha": [0.01, 0.1, 1, 10, 100],
                },
            ),

            "Lasso": (
                Pipeline([
                    ("scaler", StandardScaler()),
                    ("reg", Lasso(random_state=random_state, max_iter=10000))
                ]),
                {
                    "reg__alpha": [0.0001, 0.001, 0.01, 0.1, 1],
                },
            ),

            "Elastic Net": (
                Pipeline([
                    ("scaler", StandardScaler()),
                    ("reg", ElasticNet(random_state=random_state, max_iter=10000))
                ]),
                {
                    "reg__alpha": [0.0001, 0.001, 0.01, 0.1, 1],
                    "reg__l1_ratio": [0.1, 0.5, 0.9],
                },
            ),

            "Decision Tree Regressor": (
                DecisionTreeRegressor(random_state=random_state),
                {
                    "max_depth": [None, 3, 5, 10],
                    "min_samples_split": [2, 5, 10],
                },
            ),

            "Random Forest Regressor": (
                RandomForestRegressor(random_state=random_state),
                {
                    "n_estimators": [100, 300],
                    "max_depth": [None, 5, 10],
                    "min_samples_split": [2, 5],
                },
            ),

            "KNN Regressor": (
                Pipeline([
                    ("scaler", StandardScaler()),
                    ("reg", KNeighborsRegressor())
                ]),
                {
                    "reg__n_neighbors": [3, 5, 7, 11],
                    "reg__weights": ["uniform", "distance"],
                },
            ),

            "Linear SVR": (
                Pipeline([
                    ("scaler", StandardScaler()),
                    ("reg", SVR(kernel="linear"))
                ]),
                {
                    "reg__C": [0.1, 1, 10, 100],
                },
            ),

            "RBF SVR": (
                Pipeline([
                    ("scaler", StandardScaler()),
                    ("reg", SVR(kernel="rbf"))
                ]),
                {
                    "reg__C": [0.1, 1, 10, 100],
                    "reg__gamma": ["scale", 0.01, 0.1, 1],
                },
            ),

            "PCR (PCA + Linear Regression)": (
                Pipeline([
                    ("scaler", StandardScaler()),
                    ("pca", PCA()),
                    ("reg", LinearRegression())
                ]),
                {
                    # tune number of principal components
                    "pca__n_components": [2, 5, 10, 15, 20],
                    "reg__fit_intercept": [True, False],
                },
            ),
        }

        for name, (estimator, param_grid) in models_and_grids.items():
            print(f"\nRunning {name}...")

            grid = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                scoring=mse_scorer,  # negative MSE
                cv=cv,
                n_jobs=-1,
            )

            grid.fit(X_train, y_train)

            best_cv_mse = -grid.best_score_
            y_pred_test = grid.best_estimator_.predict(X_test)
            test_mse = mean_squared_error(y_test, y_pred_test)

            print(f"  Best params: {grid.best_params_}")
            print(f"  CV MSE:   {best_cv_mse:.4f}")
            print(f"  Test MSE: {test_mse:.4f}")

            results.append({
                "model": name,
                "best_params": grid.best_params_,
                "cv_mse": best_cv_mse,
                "test_mse": test_mse,
                "best_estimator": grid.best_estimator_,
            })

        best_result = min(results, key=lambda r: r["test_mse"])
        best_model = best_result["best_estimator"]

        alpha_like = None
        for key, val in best_result["best_params"].items():
            if "alpha" in key.lower():
                alpha_like = (key, val)
                break

        print("\nModel Selection Summary: Regression")
        print(f"For this data, {best_result['model']} is best to use,")
        print(f"because it has the lowest test MSE = {best_result['test_mse']:.4f}.")

        if alpha_like is not None:
            print(f"The best alpha-like parameter is {alpha_like[0]} = {alpha_like[1]}.")

        print("\nBest model hyperparameters:")
        print(best_result["best_params"])

        results_df = pd.DataFrame([
            {
                "model": r["model"],
                "cv_mse": r["cv_mse"],
                "test_mse": r["test_mse"],
                "best_params": r["best_params"],
            }
            for r in results
        ]).sort_values(by="test_mse")

        extra = None

    return task, results_df, best_model, extra


#USAGE

file = pd.read_csv("Data-Default.csv")

file["student"] = (file["student"] == "Yes").astype(int)
file["default"] = (file["default"] == "Yes").astype(int)

X = file[["balance", "income"]]
y = file["default"]

df = file
X_cols = X
y_col = y

task, results_df, best_model, extra = run_models_auto(
    df=df,
    y_col=y_col,
    X_cols=X_cols,
    test_size=0.3,
    random_state=1,
)

print("\nDetected task:", task)
print(results_df)

if task == "classification":
    print("Confusion matrix for best model:\n", extra)