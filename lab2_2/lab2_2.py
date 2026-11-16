import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

DEGREES = [1, 2, 3, 5, 9]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "lab2_2_DS.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.file)
    xn, yn = df.columns[:2]
    X, y = df[[xn]], df[yn]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

    xs = pd.DataFrame({xn: np.linspace(X[xn].min(), X[xn].max(), 300)})
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.scatter(X[xn], y, color="tab:blue", label="данные")
    colors = {1: "gray", 2: "tab:red", 3: "tab:green", 5: "tab:orange", 9: "tab:purple"}

    rows = []
    for d in DEGREES:
        model = make_pipeline(PolynomialFeatures(d, include_bias=False), StandardScaler(), LinearRegression())
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        rows.append({
            "Степень": d,
            "MSE": mean_squared_error(y_te, pred),
            "R2": r2_score(y_te, pred),
            "MAPE, %": 100 * mean_absolute_percentage_error(y_te, pred),
        })
        label = "линейная (степень 1)" if d == 1 else f"полином степени {d}"
        ax.plot(xs[xn], model.predict(xs), color=colors[d], lw=2.2, ls="--" if d == 1 else "-", label=label)

    table = pd.DataFrame(rows).set_index("Степень")
    print(table.round(3).to_string())
    poly = table.drop(index=1)
    print(f"\nЛучшая степень по MSE: {poly['MSE'].idxmin()}")
    print(f"Лучшая степень по R2: {poly['R2'].idxmax()}")
    print(f"Лучшая степень по MAPE: {poly['MAPE, %'].idxmin()}")

    ax.set(xlabel=xn, ylabel=yn, title="Полиномиальная регрессия")
    ax.set_ylim(y.min() - 10, y.max() + 10)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.savefig("lab2_1_plot.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()

