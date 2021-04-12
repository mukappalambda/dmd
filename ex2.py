import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dmd import DMD

plt.style.use("ggplot")

if __name__ == "__main__":
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
    df = pd.read_csv(url)

    figsize = (24, 10)
    ts_length = 1500

    dmd = DMD()
    dmd.fit(df[["Temp"]], ts_length=ts_length)

    fname = "temperature-data.png"
    plt.clf()
    plt.figure(figsize=figsize)
    plt.plot(df["Temp"], "r", label="raw data")

    t = 0
    pred_x = np.arange(t, t+ts_length)
    pred_y = dmd.predict_future(t)
    plt.plot(pred_x, pred_y, "b", label="fitted data")

    t = df.shape[0]
    pred_x = np.arange(t, t+ts_length)
    pred_y = dmd.predict_future(t)
    plt.plot(pred_x, pred_y, "c", label="future prediction")
    plt.legend()
    plt.savefig(fname)
