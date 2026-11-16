import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ALPHAS = np.logspace(-2, 4, 60)


def build_models():
    return {
        "Linear": LinearRegression(),
        "Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)),
        "Lasso": make_pipeline(StandardScaler(), LassoCV(alphas=ALPHAS, cv=3, max_iter=100000)),
        "ElasticNet": make_pipeline(StandardScaler(), ElasticNetCV(alphas=ALPHAS, l1_ratio=[0.2, 0.5, 0.8],
                                                                   cv=3, max_iter=100000)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "lab2_1_DS.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.file)
    target = df.columns[-1]
    X, y = df[["experience"]], df[target]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

    fitted, rows = {}, []
    for name, model in build_models().items():
        model.fit(X_tr, y_tr)
        fitted[name] = model
        pred = model.predict(X_te)
        rows.append({
            "Модель": name,
            "MAE": mean_absolute_error(y_te, pred),
            "R2": r2_score(y_te, pred),
            "MAPE, %": 100 * mean_absolute_percentage_error(y_te, pred),
        })
    table = pd.DataFrame(rows).set_index("Модель")
    print(table.round(3).to_string())
    print(f"\nЛучшая модель по MAE: {table['MAE'].idxmin()}")
    print(f"Лучшая модель по R2: {table['R2'].idxmax()}")
    print(f"Лучшая модель по MAPE: {table['MAPE, %'].idxmin()}")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(X["experience"], y, color="tab:blue", label="данные")
    xs = pd.DataFrame({"experience": np.linspace(X["experience"].min(), X["experience"].max(), 100)})
    styles = {"Linear": ("tab:red", "-", 3), "Ridge": ("tab:green", "--", 2),
              "Lasso": ("tab:orange", "-.", 2), "ElasticNet": ("tab:purple", ":", 2.5)}
    for name, model in fitted.items():
        color, ls, lw = styles[name]
        ax.plot(xs["experience"], model.predict(xs), color=color, ls=ls, lw=lw, label=name)
    ax.set(xlabel="experience", ylabel=target, title="Линейная регрессия")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.savefig("lab2_1_plot.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()