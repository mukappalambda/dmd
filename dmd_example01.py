import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dmd import DMD

plt.style.use("ggplot")

if __name__ == "__main__":
    x = np.arange(1000)
    y = np.sin(0.02*x) + 5e-2 * np.random.randn(np.prod(x.shape))

    figsize = (24, 10)
    ts_length = 200

    dmd = DMD()
    dmd.fit(pd.DataFrame(y), ts_length=ts_length)

    fname = "synthetic-data.png"
    plt.clf()
    plt.figure(figsize=figsize)
    plt.plot(y, "r", label="raw data")

    t = 0
    pred_x = np.arange(t, t+ts_length)
    pred_y = dmd.predict_future(t)
    plt.plot(pred_x, pred_y, "b", label="fitted data")

    t = y.shape[0]
    pred_x = np.arange(t, t+ts_length)
    pred_y = dmd.predict_future(t)
    plt.plot(pred_x, pred_y, "c", label="future prediction")
    plt.legend()
    plt.savefig(fname)
