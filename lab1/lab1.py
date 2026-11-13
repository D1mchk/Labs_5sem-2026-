import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


def load_data(path):
    df = pd.read_csv(path)
    df = df.select_dtypes(include="number").dropna()
    if df.shape[1] < 2:
        raise SystemExit("В файле должно быть минимум два числовых столбца")
    return df


def choose_columns(df, x_name, y_name):
    cols = list(df.columns)
    if x_name is None or y_name is None:
        if len(cols) == 2:
            x_name, y_name = cols
        else:
            print("Доступные столбцы:", ", ".join(f"{i}: {c}" for i, c in enumerate(cols)))
            x_name = cols[int(input("Номер столбца для X: "))]
            y_name = cols[int(input("Номер столбца для Y: "))]
    for name in (x_name, y_name):
        if name not in cols:
            raise SystemExit(f"Нет столбца '{name}'. Доступны: {cols}")
    return x_name, y_name


def print_statistics(df):
    print("\nСтатистика по столбцам:")
    print(f"{'столбец':<28}{'кол-во':>8}{'min':>14}{'max':>14}{'среднее':>16}")
    for c in df.columns:
        v = df[c].to_numpy(dtype=float)
        print(f"{c:<28}{len(v):>8}{v.min():>14.4f}{v.max():>14.4f}{v.mean():>16.4f}")


def least_squares(x, y):
    """y = b0 + b1 * x.  b1 = sum((x-mx)(y-my)) / sum((x-mx)^2),  b0 = my - b1*mx"""
    mx, my = x.mean(), y.mean()
    b1 = np.sum((x - mx) * (y - my)) / np.sum((x - mx) ** 2)
    b0 = my - b1 * mx
    return b0, b1


def fig_points(x, y, xn, yn):
    fig, ax = plt.subplots(figsize=(8, 5.5), num="Рис. 1 - исходные точки")
    ax.scatter(x, y, color="tab:blue", label="данные")
    ax.set(xlabel=xn, ylabel=yn, title="Исходные точки")
    ax.grid(alpha=0.3)
    ax.legend()
    return fig


def fig_line(x, y, xn, yn, b0, b1):
    fig, ax = plt.subplots(figsize=(8, 5.5), num="Рис. 2 - регрессионная прямая")
    ax.scatter(x, y, color="tab:blue", label="данные")
    xs = np.array([x.min(), x.max()])
    ax.plot(xs, b0 + b1 * xs, color="tab:red", lw=2, label=f"y = {b0:.2f} + {b1:.2f}·x")
    ax.set(xlabel=xn, ylabel=yn, title="Регрессионная прямая (МНК)")
    ax.grid(alpha=0.3)
    ax.legend()
    return fig


def fig_squares(x, y, xn, yn, b0, b1):
    fig, ax = plt.subplots(figsize=(9, 6), num="Рис. 3 - квадраты ошибок")
    y_hat = b0 + b1 * x
    err = y - y_hat

    xr, yr = x.max() - x.min(), y.max() - y.min()
    ax.set_xlim(x.min() - 0.05 * xr, x.max() + 0.12 * xr)
    ax.set_ylim(y.min() - 0.08 * yr, y.max() + 0.08 * yr)
    ax.set(xlabel=xn, ylabel=yn, title="Квадраты ошибок (остатков)")
    fig.tight_layout()
    fig.canvas.draw()

    p0 = ax.transData.transform((0, 0))
    p1 = ax.transData.transform((1, 1))
    px_per_x, px_per_y = p1[0] - p0[0], p1[1] - p0[1]

    for xi, yi, yh, e in zip(x, y, y_hat, err):
        side_x = abs(e) * px_per_y / px_per_x
        ax.add_patch(Rectangle((xi, min(yi, yh)), side_x, abs(e), facecolor="tab:orange",
                               edgecolor="darkorange", alpha=0.35, hatch="///", lw=0.8))

    ax.scatter(x, y, color="tab:blue", zorder=3, label="данные")
    xs = np.array([x.min(), x.max()])
    ax.plot(xs, b0 + b1 * xs, color="tab:red", lw=2, zorder=3, label="регрессионная прямая")
    ax.add_patch(Rectangle((0, 0), 0, 0, facecolor="tab:orange", edgecolor="darkorange",
                           alpha=0.35, hatch="///", label="квадраты ошибок"))
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    return fig


def main():
    ap = argparse.ArgumentParser(description="ЛР1: линейная регрессия методом наименьших квадратов")
    ap.add_argument("file", help="CSV-файл с данными")
    ap.add_argument("--x", help="название столбца для X")
    ap.add_argument("--y", help="название столбца для Y")
    ap.add_argument("--save", action="store_true", help="сохранить рисунки в PNG")
    args = ap.parse_args()

    df = load_data(args.file)
    print_statistics(df)
    xn, yn = choose_columns(df, args.x, args.y)
    x = df[xn].to_numpy(dtype=float)
    y = df[yn].to_numpy(dtype=float)

    b0, b1 = least_squares(x, y)
    err = y - (b0 + b1 * x)
    print(f"\nРегрессионная прямая: {yn} = {b0:.4f} + {b1:.4f} * {xn}")
    print(f"Сумма квадратов ошибок SSE = {np.sum(err**2):.2f},  MSE = {np.mean(err**2):.2f}")

    figs = [fig_points(x, y, xn, yn), fig_line(x, y, xn, yn, b0, b1), fig_squares(x, y, xn, yn, b0, b1)]
    if args.save:
        for i, f in enumerate(figs, 1):
            f.savefig(f"lab1_fig{i}.png", dpi=130, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
