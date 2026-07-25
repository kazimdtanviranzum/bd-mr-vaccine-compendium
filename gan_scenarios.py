"""
gan_scenarios.py
================
Generative, uncertainty-aware demand modelling (NEW contribution 4.8).

A compact CONDITIONAL GAN, implemented in pure NumPy with manual backpropagation
and an Adam optimiser (no torch dependency, so it runs anywhere), learns the
distribution of *seasonal demand shapes* (12-month normalised profiles)
conditioned on division. Sampling the generator produces an ensemble of
plausible annual MR demand trajectories that carry forecast uncertainty into the
two-stage stochastic optimisation (4.9) and the stress test.

Real training data: the historical monthly demand shares (2020-2024) for each
division and dose from the synthetic panel. Generated output: N_SCENARIOS
trajectories per division scaled to the planning-year demand level.

Outputs
-------
outputs/tables/gan_scenarios.csv         : scenario x division x month demand
outputs/tables/gan_training_log.csv      : D/G loss per epoch
outputs/tables/gan_ensemble_bands.csv    : p10/p50/p90 seasonal shares
"""

import numpy as np
import pandas as pd

import config as C

RNG = np.random.default_rng(C.SEED)


# --------------------------------------------------------------------------- #
# Real seasonal-shape training set
# --------------------------------------------------------------------------- #
def build_training_shapes():
    """
    For each division x dose x year, extract the 12-month demand profile and
    normalise it to a share vector (sums to 1). Returns X (n,12) and one-hot
    conditions Cnd (n, n_div).
    """
    df = pd.read_csv(C.PANEL_CSV)
    df = df[df.year.between(C.HIST_START_YEAR, C.HIST_END_YEAR)]
    div_idx = {d: i for i, d in enumerate(C.DIVISIONS)}
    X, Cnd = [], []
    for (div, dose, yr), g in df.groupby(["division", "dose", "year"]):
        g = g.sort_values("month")
        v = g["doses_baseline"].values
        if len(v) != 12 or v.sum() <= 0:
            continue
        X.append(v / v.sum())
        oh = np.zeros(len(C.DIVISIONS))
        oh[div_idx[div]] = 1.0
        Cnd.append(oh)
    return np.array(X, dtype=np.float64), np.array(Cnd, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Layers / activations
# --------------------------------------------------------------------------- #
def relu(x):
    return np.maximum(0, x)


def drelu(x):
    return (x > 0).astype(x.dtype)


def lrelu(x, a=0.2):
    return np.where(x > 0, x, a * x)


def dlrelu(x, a=0.2):
    return np.where(x > 0, 1.0, a)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def softmax(x):
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class Adam:
    def __init__(self, params, lr):
        self.lr, self.b1, self.b2, self.eps = lr, 0.9, 0.999, 1e-8
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


# --------------------------------------------------------------------------- #
# Conditional GAN
# --------------------------------------------------------------------------- #
class ConditionalGAN:
    def __init__(self, n_cond, latent=C.GAN_LATENT, hidden=C.GAN_HIDDEN, out=12):
        s = 0.1
        self.out = out
        # Generator: [z, cond] -> hidden -> out (softmax)
        self.G = {
            "W1": RNG.normal(0, s, (latent + n_cond, hidden)),
            "b1": np.zeros(hidden),
            "W2": RNG.normal(0, s, (hidden, out)),
            "b2": np.zeros(out),
        }
        # Discriminator: [x, cond] -> hidden -> 1 (sigmoid)
        self.D = {
            "W1": RNG.normal(0, s, (out + n_cond, hidden)),
            "b1": np.zeros(hidden),
            "W2": RNG.normal(0, s, (hidden, 1)),
            "b2": np.zeros(1),
        }
        self.latent = latent
        self.n_cond = n_cond
        self.optG = Adam(self.G, C.GAN_LR)
        self.optD = Adam(self.D, C.GAN_LR)

    def gen_forward(self, z, cond):
        inp = np.hstack([z, cond])
        h_pre = inp @ self.G["W1"] + self.G["b1"]
        h = relu(h_pre)
        o_pre = h @ self.G["W2"] + self.G["b2"]
        o = softmax(o_pre)
        cache = (inp, h_pre, h, o_pre, o)
        return o, cache

    def disc_forward(self, x, cond):
        inp = np.hstack([x, cond])
        h_pre = inp @ self.D["W1"] + self.D["b1"]
        h = lrelu(h_pre)
        logit = h @ self.D["W2"] + self.D["b2"]
        p = sigmoid(logit)
        cache = (inp, h_pre, h, logit, p)
        return p, cache

    def _disc_grads(self, cache, dlogit):
        inp, h_pre, h, logit, p = cache
        gW2 = h.T @ dlogit
        gb2 = dlogit.sum(0)
        dh = dlogit @ self.D["W2"].T
        dhpre = dh * dlrelu(h_pre)
        gW1 = inp.T @ dhpre
        gb1 = dhpre.sum(0)
        dinp = dhpre @ self.D["W1"].T
        return {"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}, dinp

    def train(self, X, Cnd, epochs=C.GAN_EPOCHS, batch=C.GAN_BATCH):
        n = len(X)
        log = []
        for ep in range(epochs):
            idx = RNG.integers(0, n, size=batch)
            xr, cr = X[idx], Cnd[idx]
            # ---- Discriminator update ----
            z = RNG.normal(0, 1, (batch, self.latent))
            xf, _ = self.gen_forward(z, cr)
            pr, cache_r = self.disc_forward(xr, cr)
            pf, cache_f = self.disc_forward(xf, cr)
            # BCE: real->1, fake->0. dL/dlogit = (p - target)/n
            dlr = (pr - 1.0) / batch
            dlf = (pf - 0.0) / batch
            gD_r, _ = self._disc_grads(cache_r, dlr)
            gD_f, _ = self._disc_grads(cache_f, dlf)
            gD = {k: gD_r[k] + gD_f[k] for k in self.D}
            self.optD.step(self.D, gD)
            d_loss = float(-np.mean(np.log(pr + 1e-9) + np.log(1 - pf + 1e-9)))
            # ---- Generator update (maximise log D(G)) ----
            z = RNG.normal(0, 1, (batch, self.latent))
            xf, gcache = self.gen_forward(z, cr)
            pf, dcache = self.disc_forward(xf, cr)
            dlogit = (pf - 1.0) / batch  # want D to output 1
            _, dinp = self._disc_grads(dcache, dlogit)
            dx = dinp[:, : self.out]  # grad wrt generated shares
            # backprop through generator softmax
            inp, h_pre, h, o_pre, o = gcache
            # softmax jacobian-vector product
            dot = (dx * o).sum(1, keepdims=True)
            do_pre = o * (dx - dot)
            gW2 = h.T @ do_pre
            gb2 = do_pre.sum(0)
            dh = do_pre @ self.G["W2"].T
            dhpre = dh * drelu(h_pre)
            gW1 = inp.T @ dhpre
            gb1 = dhpre.sum(0)
            gG = {"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}
            self.optG.step(self.G, gG)
            g_loss = float(-np.mean(np.log(pf + 1e-9)))
            log.append((ep, d_loss, g_loss))
        return pd.DataFrame(log, columns=["epoch", "d_loss", "g_loss"])

    def sample(self, cond_onehot, n):
        cond = np.tile(cond_onehot, (n, 1))
        z = RNG.normal(0, 1, (n, self.latent))
        shares, _ = self.gen_forward(z, cond)
        return shares


def main():
    X, Cnd = build_training_shapes()
    gan = ConditionalGAN(n_cond=len(C.DIVISIONS))
    log = gan.train(X, Cnd)
    log.to_csv(f"{C.TAB_DIR}/gan_training_log.csv", index=False)

    # Planning-year demand level per division (baseline total, first plan year).
    demand = pd.read_csv(f"{C.TAB_DIR}/forecast_demand_2026_2028.csv")
    annual = (demand[demand.year == C.PLAN_YEARS[0]]
              .groupby("division")["doses"].sum())

    div_idx = {d: i for i, d in enumerate(C.DIVISIONS)}
    rows, band_rows = [], []
    for div in C.DIVISIONS:
        oh = np.zeros(len(C.DIVISIONS))
        oh[div_idx[div]] = 1.0
        shares = gan.sample(oh, C.N_SCENARIOS)          # (S,12)
        level = float(annual.get(div, 0.0))
        p10, p50, p90 = np.percentile(shares, [10, 50, 90], axis=0)
        for mo in range(12):
            band_rows.append(dict(division=div, month=mo + 1,
                                  p10=p10[mo], p50=p50[mo], p90=p90[mo]))
        for s in range(C.N_SCENARIOS):
            for mo in range(12):
                rows.append(dict(scenario=s, division=div, month=mo + 1,
                                 demand=shares[s, mo] * level))
    scen = pd.DataFrame(rows)
    scen.to_csv(f"{C.TAB_DIR}/gan_scenarios.csv", index=False)
    pd.DataFrame(band_rows).to_csv(f"{C.TAB_DIR}/gan_ensemble_bands.csv", index=False)

    print("GAN trained. epochs:", len(log),
          "| final D_loss=%.3f G_loss=%.3f" % (log.d_loss.iloc[-1], log.g_loss.iloc[-1]))
    print("Real training shapes:", X.shape, "| scenarios generated:", len(scen))
    # sanity: mean coefficient of variation of generated monthly demand
    cv = (scen.groupby(["division", "month"])["demand"].std()
          / scen.groupby(["division", "month"])["demand"].mean()).mean()
    print("Mean across-scenario CV of monthly demand: %.3f" % cv)
    return scen


if __name__ == "__main__":
    main()
