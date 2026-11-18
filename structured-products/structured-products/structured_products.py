"""
structured_products.py
======================
Monte Carlo pricer for equity structured products.

Products:
  - Barrier reverse convertible: fixed coupon + short down-and-in put
  - Capital-protected note: zero-coupon bond + participation in the upside (a call)
  - Athena autocall (accumulated coupon paid at call) and Phoenix (conditional
    coupon with memory)

Method, in 4 steps:
  1. simulate paths of the underlying in the risk-neutral world
     (the underlying earns r - q on average),
  2. compute the product's payoff on each path,
  3. discount at the risk-free rate,
  4. average: that is the price.

The fair coupon is the one for which the value of the cash flows = 100 - structurer's margin.

Usage:
    python structured_products.py
    python structured_products.py --sigma 0.25 --margin 0.02 --plot
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace

import numpy as np
from scipy.stats import norm

NOMINAL = 100.0


# =============================================================================
# MARKET AND SIMULATION
# =============================================================================


@dataclass(frozen=True)
class Market:
    """Market assumptions: flat volatility, constant rate and dividend yield."""

    r: float = 0.025      # risk-free rate
    q: float = 0.03       # dividend yield
    sigma: float = 0.20   # volatility
    spot: float = 1.0     # current level / initial level of the product (1.0 = on the strike date)


def normals(n_paths, n_steps, seed=0):
    """Gaussian draws. We reuse the same ones for every price, which keeps the Greeks stable."""
    return np.random.default_rng(seed).standard_normal((n_paths, n_steps))


def simulate_levels(m: Market, times, z):
    """Levels S(t) / S_initial at the observation dates (geometric Brownian motion)."""
    dt = np.diff(np.r_[0.0, np.asarray(times, dtype=float)])
    incr = (m.r - m.q - 0.5 * m.sigma**2) * dt + m.sigma * np.sqrt(dt) * z
    return m.spot * np.exp(np.cumsum(incr, axis=1))


def bs_call(S, K, T, r, q, sigma):
    """European Black-Scholes call, used to check the simulation."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# =============================================================================
# PRODUCTS: each returns the cash flows (n_paths x n_dates) as % of the nominal
# =============================================================================


@dataclass
class ReverseConvertible:
    """Fixed coupon every year. At maturity: 100 if S_T >= barrier, otherwise delivery
    of shares (100 * S_T / S_0). The investor is short a down-and-in put."""

    coupon: float
    barrier: float = 0.70
    maturity: int = 3

    @property
    def times(self):
        return np.arange(1, self.maturity + 1, dtype=float)

    def cashflows(self, L):
        cf = np.full(L.shape, NOMINAL * self.coupon)
        cf[:, -1] += np.where(L[:, -1] >= self.barrier, NOMINAL, NOMINAL * L[:, -1])
        return cf, np.full(L.shape[0], -1)

    def loss_mask(self, L, called_at):
        return L[:, -1] < self.barrier


@dataclass
class CapitalProtectedNote:
    """Zero-coupon bond + participation in the upside: 100 * (protection + part * max(S_T/S_0 - 1, 0))."""

    participation: float
    protection: float = 1.0
    maturity: int = 5

    @property
    def times(self):
        return np.arange(1, self.maturity + 1, dtype=float)

    def cashflows(self, L):
        cf = np.zeros(L.shape)
        cf[:, -1] = NOMINAL * (self.protection + self.participation * np.maximum(L[:, -1] - 1.0, 0.0))
        return cf, np.full(L.shape[0], -1)

    def loss_mask(self, L, called_at):
        return np.full(L.shape[0], self.protection < 1.0) & (L[:, -1] < 1.0)


@dataclass
class Autocall:
    """
    Autocall with annual observation.
    - Call: if S_i >= autocall barrier, 100 is repaid and the product stops.
    - Athena (coupon_barrier=None): the coupon accumulates and is paid at the call (100 * (1 + c * i)).
    - Phoenix (coupon_barrier given): coupon c paid each year where S_i >= coupon barrier,
      with memory of missed coupons if memory=True. This coupon is a digital option.
    - Maturity without a call: 100 if S_N >= protection barrier, otherwise 100 * S_N / S_0.
    """

    coupon: float
    maturity: int = 5
    autocall_barrier: float = 1.0
    protection_barrier: float = 0.60
    coupon_barrier: float | None = None
    memory: bool = True

    @property
    def times(self):
        return np.arange(1, self.maturity + 1, dtype=float)

    def cashflows(self, L):
        n, N = L.shape
        cf = np.zeros((n, N))
        alive = np.ones(n, dtype=bool)        # product not yet called
        called_at = np.full(n, -1)            # index of the call date (-1 = never)
        missed = np.zeros(n)                  # missed coupons (Phoenix memory)
        phoenix = self.coupon_barrier is not None

        for i in range(N):
            if phoenix:
                pays = alive & (L[:, i] >= self.coupon_barrier)
                mult = 1.0 + missed if self.memory else np.ones(n)
                cf[pays, i] += NOMINAL * self.coupon * mult[pays]
                missed = np.where(pays, 0.0, np.where(alive, missed + 1.0, missed))
            hit = alive & (L[:, i] >= self.autocall_barrier)
            if phoenix:
                cf[hit, i] += NOMINAL
            else:
                cf[hit, i] += NOMINAL * (1.0 + self.coupon * (i + 1))
            called_at[hit] = i
            alive &= ~hit

        final = np.where(L[:, -1] >= self.protection_barrier, NOMINAL, NOMINAL * L[:, -1])
        cf[alive, -1] += final[alive]
        return cf, called_at

    def loss_mask(self, L, called_at):
        return (called_at == -1) & (L[:, -1] < self.protection_barrier)


# =============================================================================
# PRICING, FAIR COUPON, GREEKS, STATISTICS
# =============================================================================


@dataclass
class Result:
    pv: float          # present value of the cash flows (% of nominal)
    se: float          # standard error of the simulation
    levels: np.ndarray
    called_at: np.ndarray


def price(product, m: Market, z) -> Result:
    L = simulate_levels(m, product.times, z)
    cf, called_at = product.cashflows(L)
    pv = (cf * np.exp(-m.r * product.times)).sum(axis=1)
    return Result(float(pv.mean()), float(pv.std(ddof=1) / np.sqrt(len(pv))), L, called_at)


def fair_coupon(product, m: Market, z, margin=0.015):
    """
    Annual coupon for which the value of the cash flows = 100 - margin.
    The cash flows are linear in the coupon: value(c) = value(0) + c * slope.
    Two pricings are enough: a zero coupon, then a 100% coupon to get the slope.
    Returns 0 if, even with no coupon, the value already exceeds the target.
    """
    target = NOMINAL * (1.0 - margin)
    v0 = price(replace(product, coupon=0.0), m, z).pv
    v1 = price(replace(product, coupon=1.0), m, z).pv
    return max(0.0, (target - v0) / (v1 - v0))


def fair_participation(maturity, m: Market, margin=0.015, protection=1.0):
    """Fair participation of a capital-protected note: we spend on the call whatever is left
    after buying the zero-coupon bond (closed form)."""
    budget = NOMINAL * (1.0 - margin) - NOMINAL * protection * np.exp(-m.r * maturity)
    call = NOMINAL * bs_call(1.0, 1.0, maturity, m.r, m.q, m.sigma)
    return budget / call


def greeks(product, m: Market, z, bump_spot=0.01, bump_vol=0.01):
    """Greeks by bump and reprice, in nominal points: we move the spot by 1% or the
    volatility by 1 point, using the same random draws, and look at the change in price."""
    p0 = price(product, m, z).pv
    up = price(product, replace(m, spot=m.spot * (1 + bump_spot)), z).pv
    dn = price(product, replace(m, spot=m.spot * (1 - bump_spot)), z).pv
    vu = price(product, replace(m, sigma=m.sigma + bump_vol), z).pv
    vd = price(product, replace(m, sigma=m.sigma - bump_vol), z).pv
    return {"delta": (up - dn) / 2, "gamma": up - 2 * p0 + dn, "vega": (vu - vd) / 2}


def autocall_probabilities(called_at, n_dates):
    """P(call at each date), then P(no call). The sum is 1."""
    p = [(called_at == i).mean() for i in range(n_dates)]
    return np.array(p + [(called_at == -1).mean()])


def risk_profile(product, res: Result):
    N = len(product.times)
    probs = autocall_probabilities(res.called_at, N)
    loss = product.loss_mask(res.levels, res.called_at)
    avg_loss = float(NOMINAL * (1.0 - res.levels[loss, -1]).mean()) if loss.any() else 0.0
    life = float((np.arange(1, N + 1) * probs[:-1]).sum() + N * probs[-1])
    return {"probs": probs, "p_loss": float(loss.mean()), "avg_loss": avg_loss, "life": life}


# =============================================================================
# CHARTS (optional)
# =============================================================================

_SURFACE, _INK, _SECONDARY, _MUTED, _GRID, _BASE = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7")
_SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua


def _style_axes(fig, ax):
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(_BASE)
    ax.grid(axis="y", color=_GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=_MUTED, length=0)
    ax.xaxis.label.set_color(_SECONDARY)
    ax.yaxis.label.set_color(_SECONDARY)


def plot_coupon_vs_vol(vols, series, path="coupon_vs_vol.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    _style_axes(fig, ax)
    x = np.array(vols) * 100
    for (name, ys), color in zip(series.items(), _SERIES):
        y = np.array(ys) * 100
        ax.plot(x, y, color=color, lw=2, marker="o", ms=7, mec=_SURFACE, mew=2, label=name)
        ax.annotate(name, (x[-1], y[-1]), xytext=(10, 0), textcoords="offset points",
                    va="center", color=_SECONDARY, fontsize=9)
    ax.set_xlim(x[0] - 1, x[-1] + 9)
    ax.set_xticks(x)
    ax.set_xlabel("underlying volatility (%)")
    ax.set_ylabel("fair coupon (% per year)")
    ax.set_title("The higher the volatility, the higher the coupon", loc="left",
                 color=_INK, fontsize=12, fontweight="bold")
    ax.legend(frameon=False, labelcolor=_SECONDARY, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_autocall_probs(probs, path="autocall_probs.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"Year {i + 1}" for i in range(len(probs) - 1)] + ["Not called"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _style_axes(fig, ax)
    vals = np.array(probs) * 100
    ax.bar(labels, vals, color=_SERIES[0], width=0.65)
    for i in (int(vals[:-1].argmax()), len(vals) - 1):
        ax.annotate(f"{vals[i]:.0f}%", (i, vals[i]), xytext=(0, 4), textcoords="offset points",
                    ha="center", color=_INK, fontsize=9)
    ax.set_ylabel("probability (%)")
    ax.set_title("When is the autocall called?", loc="left",
                 color=_INK, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =============================================================================
# DEMO
# =============================================================================


def _pct(x, d=2):
    return f"{x * 100:.{d}f}%"


def demo(m: Market, n_paths, margin, plot):
    z3, z5 = normals(n_paths, 3), normals(n_paths, 5)
    print("=" * 74)
    print(f"MARKET: r = {_pct(m.r, 1)} | dividend yield = {_pct(m.q, 1)} | vol = {_pct(m.sigma, 0)} "
          f"| structurer margin = {_pct(margin, 1)} | {n_paths:,} paths")
    print("=" * 74)

    # 1. Capital-protected note
    T = 5
    part = fair_participation(T, m, margin)
    cpn = CapitalProtectedNote(part, 1.0, T)
    res = price(cpn, m, normals(n_paths, T))
    zc = NOMINAL * np.exp(-m.r * T)
    call = NOMINAL * bs_call(1.0, 1.0, T, m.r, m.q, m.sigma)
    print("\n1. 5-YEAR CAPITAL-PROTECTED NOTE (100% guaranteed)")
    print(f"   5-year zero-coupon = {zc:.2f} | budget for the option = {NOMINAL * (1 - margin) - zc:.2f} "
          f"| 5-year ATM call = {call:.2f}")
    print(f"   Fair participation in the upside: {_pct(part, 1)}")
    print(f"   Simulation check: value = {res.pv:.3f} +/- {res.se:.3f} (target {NOMINAL * (1 - margin):.2f})")

    # 2. Reverse convertible
    rc0 = ReverseConvertible(coupon=0.0, barrier=0.70, maturity=3)
    c_rc = fair_coupon(rc0, m, z3, margin)
    rc = replace(rc0, coupon=c_rc)
    r_rc, g_rc = price(rc, m, z3), greeks(rc, m, z3)
    rp = risk_profile(rc, r_rc)
    print("\n2. 3-YEAR REVERSE CONVERTIBLE (European barrier at 70%)")
    print(f"   Fair coupon: {_pct(c_rc)} per year")
    print(f"   P(capital loss) = {_pct(rp['p_loss'], 1)} | average loss if loss = {rp['avg_loss']:.1f}%")
    print(f"   Delta = {g_rc['delta']:+.3f} | Gamma = {g_rc['gamma']:+.3f} | Vega = {g_rc['vega']:+.3f} "
          "(nominal points, per +1% spot / +1 vol pt)")

    # 3. Autocalls
    athena0 = Autocall(coupon=0.0, maturity=5)
    phoenix0 = Autocall(coupon=0.0, maturity=5, coupon_barrier=0.70, memory=True)
    probs_athena = None
    print("\n3. 5-YEAR AUTOCALLS (annual observation, call at 100%, protection at 60%)")
    for name, prod0 in (("Athena", athena0), ("Phoenix (coupon barrier 70%, memory)", phoenix0)):
        c = fair_coupon(prod0, m, z5, margin)
        prod = replace(prod0, coupon=c)
        r, g = price(prod, m, z5), greeks(prod, m, z5)
        rp = risk_profile(prod, r)
        if name == "Athena":
            probs_athena = rp["probs"]
        print(f"\n   {name}")
        print(f"   Fair coupon: {_pct(c)} per year")
        print("   P(call) by year: " + " | ".join(
            f"Year {i + 1} {_pct(p, 1)}" for i, p in enumerate(rp["probs"][:-1])) +
            f" | not called {_pct(rp['probs'][-1], 1)}")
        print(f"   Average life = {rp['life']:.2f} years | P(capital loss) = {_pct(rp['p_loss'], 1)}"
              f" | average loss if loss = {rp['avg_loss']:.1f}%")
        print(f"   Delta = {g['delta']:+.3f} | Gamma = {g['gamma']:+.3f} | Vega = {g['vega']:+.3f}")

    # 4. Sensitivities
    vols = [0.12, 0.15, 0.20, 0.25, 0.30]
    table = {"Reverse convertible": [], "Athena": [], "Phoenix": []}
    print("\n4. SENSITIVITY OF THE FAIR COUPON TO VOLATILITY")
    print(f"   {'vol':>6} {'Rev. conv.':>12} {'Athena':>10} {'Phoenix':>10}")
    for v in vols:
        mv = replace(m, sigma=v)
        row = (fair_coupon(rc0, mv, z3, margin), fair_coupon(athena0, mv, z5, margin),
               fair_coupon(phoenix0, mv, z5, margin))
        for key, c in zip(table, row):
            table[key].append(c)
        print(f"   {_pct(v, 0):>6} " + " ".join(f"{_pct(c):>10}" for c in row))

    print("\n   SENSITIVITY OF THE ATHENA COUPON TO THE PROTECTION BARRIER")
    print(f"   {'barrier':>9} {'fair coupon':>12}")
    for b in (0.50, 0.60, 0.70, 0.80):
        c = fair_coupon(replace(athena0, protection_barrier=b), m, z5, margin)
        print(f"   {_pct(b, 0):>9} {_pct(c):>12}")

    if plot:
        plot_coupon_vs_vol(vols, table)
        plot_autocall_probs(probs_athena)
        print("\nCharts saved: coupon_vs_vol.png, autocall_probs.png")


def main():
    ap = argparse.ArgumentParser(description="Structured products pricer (Monte Carlo)")
    ap.add_argument("--r", type=float, default=0.025, help="risk-free rate")
    ap.add_argument("--q", type=float, default=0.03, help="dividend yield")
    ap.add_argument("--sigma", type=float, default=0.20, help="volatility")
    ap.add_argument("--margin", type=float, default=0.015, help="structurer's margin (as a fraction of the nominal)")
    ap.add_argument("--paths", type=int, default=200_000, help="number of paths")
    ap.add_argument("--plot", action="store_true", help="save the charts as PNG files")
    args = ap.parse_args()
    demo(Market(r=args.r, q=args.q, sigma=args.sigma), args.paths, args.margin, args.plot)


if __name__ == "__main__":
    main()
