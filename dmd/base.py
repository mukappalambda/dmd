import numpy as np
import pandas as pd


class DMD:
    """
    Examples
    --------
    >>> dmd = DMD()
    >>> dmd.fit(df, ts_length=10)
    >>> dmd.predict_future(t=11)
    """

    def get_dmd_pair(self, df: pd.DataFrame, ts_length: int):
        y = df.to_numpy().flatten()
        data = [
            y[start : start + ts_length] for start in range(0, y.shape[0] - ts_length)
        ]
        tensor = np.array(data).T
        self.x1 = tensor[:, :-1]
        self.x2 = tensor[:, 1:]
        self.x0 = self.x1[:, 0]

    def fit(self, df: pd.DataFrame, ts_length: int):
        self.get_dmd_pair(df=df, ts_length=ts_length)
        self.svd_x1()
        self.get_atilde()
        self.get_eig_atilde()
        self.get_dmd()

    def svd_x1(self, thresh=0.7):
        u, s, vt = np.linalg.svd(self.x1)
        q = np.cumsum(s) / np.sum(s)
        mask = q > thresh
        r = np.where(mask)[0][0]

        self.u = u[:, :r]
        self.s = s[:r]
        self.vt = vt[:r, :]

    def get_atilde(self):
        self.atilde = self.u.T @ self.x2 @ self.vt.T @ np.diag(1 / self.s)

    def get_eig_atilde(self):
        lamb, w = np.linalg.eig(self.atilde)
        self.lamb = lamb
        self.w = w

    def get_dmd(self):
        phi = self.x2 @ self.vt.T @ np.diag(1 / self.s) @ self.w
        self.phi = phi

    def predict_future(self, t: int):
        pseudophix0 = np.linalg.pinv(self.phi) @ self.x0.reshape(-1, 1)
        atphi = self.phi @ np.diag(self.lamb**t)
        xt = (atphi @ pseudophix0).reshape(
            -1,
        )

        return np.real(xt)
