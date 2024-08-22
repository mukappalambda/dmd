import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dmd import DMD

plt.style.use("ggplot")


def read_temperature(only_returns_temp: bool = True) -> pd.DataFrame:
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
    df_raw = pd.read_csv(url)

    if only_returns_temp:
        return df_raw[["Temp"]]

    return df_raw


def fit_dmd(ts_length: int) -> DMD:
    dmd = DMD()
    dmd.fit(df, ts_length=ts_length)
    return dmd


def make_pred_from_dmd(dmd: DMD, ts_length: int, df: pd.DataFrame, t: int):
    pred_x = np.arange(t, t + ts_length)
    pred_y = dmd.predict_future(t)
    return pred_x, pred_y


def main_plot(df: pd.DataFrame, fname: str):
    ts_length = 1500
    dmd = fit_dmd(ts_length)
    pred_x0, pred_y0 = make_pred_from_dmd(dmd, ts_length, df, 0)
    pred_x1, pred_y1 = make_pred_from_dmd(dmd, ts_length, df, df.shape[0])

    figsize = (24, 10)
    plt.clf()
    plt.figure(figsize=figsize)

    plt.plot(df, "r", label="raw data")
    plt.plot(pred_x0, pred_y0, "b", label="fitted data")
    plt.plot(pred_x1, pred_y1, "g", label="future prediction")
    plt.legend()

    plt.savefig(fname)


if __name__ == "__main__":
    df = read_temperature()
    fname = "temperature-data.png"
    main_plot(df=df, fname=fname)
