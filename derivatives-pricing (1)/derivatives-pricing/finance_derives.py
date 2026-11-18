"""
finance_derives.py
==================
Personal markets project: derivatives basics in Python (Hull textbook level).

1. Options: Black-Scholes-Merton, put-call parity, Greeks, implied volatility
2. CRR binomial tree (European and American), convergence to Black-Scholes
3. Rates: bond price, duration, convexity, effect of a change in yields
4. Forwards and futures: theoretical price, hedging a portfolio with its beta
5. Rates: Hull-White model (simple version), zero-coupon bond price, short-rate simulation

Usage:
    python finance_derives.py          # full demo
    python finance_derives.py --plot   # + PNG charts
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

# =============================================================================
# 1. OPTIONS: Black-Scholes-Merton
# =============================================================================


def _d1_d2(S, K, T, r, sigma, q):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


def black_scholes(S, K, T, r, sigma, q=0.0, kind="call"):
    """European option price. q = dividend yield (or the foreign rate for a currency)."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    if kind == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def put_from_call(call, S, K, r, T, q=0.0):
    """Put-call parity: c + K e^(-rT) = p + S e^(-qT)."""
    return call + K * np.exp(-r * T) - S * np.exp(-q * T)


def bs_greeks(S, K, T, r, sigma, q=0.0, kind="call"):
    """Greeks: delta, gamma, vega (per +1.00 of vol), theta (per year), rho (per +1.00 of rate)."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    pdf = norm.pdf(d1)
    dq, dr = np.exp(-q * T), np.exp(-r * T)
    theta_common = -S * dq * pdf * sigma / (2 * np.sqrt(T))
    if kind == "call":
        delta = dq * norm.cdf(d1)
        theta = theta_common - r * K * dr * norm.cdf(d2) + q * S * dq * norm.cdf(d1)
        rho = K * T * dr * norm.cdf(d2)
    else:
        delta = dq * (norm.cdf(d1) - 1)
        theta = theta_common + r * K * dr * norm.cdf(-d2) - q * S * dq * norm.cdf(-d1)
        rho = -K * T * dr * norm.cdf(-d2)
    return {
        "delta": delta,
        "gamma": dq * pdf / (S * sigma * np.sqrt(T)),
        "vega": S * dq * pdf * np.sqrt(T),
        "theta": theta,
        "rho": rho,
    }


def implied_vol(price, S, K, T, r, q=0.0, kind="call"):
    """The volatility to plug into Black-Scholes to recover the observed price."""
    return brentq(lambda s: black_scholes(S, K, T, r, s, q, kind) - price, 1e-4, 5.0)


# =============================================================================
# 2. BINOMIAL TREE (Cox-Ross-Rubinstein)
# =============================================================================


def binomial_tree(S, K, T, r, sigma, n=500, q=0.0, kind="call", american=False):
    """
    At each step the stock moves up by a factor u or down by a factor d = 1/u.
    We value backwards, discounting at the risk-free rate with the risk-neutral
    probability p. For an American option, at each node we compare the continuation
    value with immediate exercise.
    """
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    sign = 1.0 if kind == "call" else -1.0

    def payoff(s):
        return np.maximum(sign * (s - K), 0.0)

    j = np.arange(n + 1)  # j = number of down moves
    V = payoff(S * u ** (n - j) * d**j)  # values at maturity
    for i in range(n - 1, -1, -1):
        V = disc * (p * V[: i + 1] + (1 - p) * V[1 : i + 2])
        if american:
            jj = np.arange(i + 1)
            V = np.maximum(V, payoff(S * u ** (i - jj) * d**jj))
    return float(V[0])


# =============================================================================
# 3. RATES: bond, duration, convexity
# =============================================================================


def bond_cashflows(face, coupon_rate, maturity):
    """Annual coupons, face value repaid at the end."""
    t = np.arange(1, maturity + 1)
    cf = np.full(maturity, face * coupon_rate, dtype=float)
    cf[-1] += face
    return t, cf


def bond_price(face, coupon_rate, ytm, maturity):
    t, cf = bond_cashflows(face, coupon_rate, maturity)
    return float(np.sum(cf / (1 + ytm) ** t))


def bond_metrics(face, coupon_rate, ytm, maturity):
    """Price, Macaulay duration, modified duration, convexity and DV01 (price change per basis point)."""
    t, cf = bond_cashflows(face, coupon_rate, maturity)
    pv = cf / (1 + ytm) ** t
    price = float(pv.sum())
    macaulay = float((t * pv).sum() / price)
    modified = macaulay / (1 + ytm)
    convexity = float((t * (t + 1) * cf / (1 + ytm) ** (t + 2)).sum() / price)
    return {"price": price, "macaulay": macaulay, "modified": modified,
            "convexity": convexity, "dv01": modified * price * 1e-4}


def price_change_estimates(m, dy):
    """Price change for a yield change dy: with duration alone, then with convexity added."""
    by_duration = -m["modified"] * dy * m["price"]
    return by_duration, by_duration + 0.5 * m["convexity"] * dy**2 * m["price"]


# =============================================================================
# 4. FORWARDS AND FUTURES
# =============================================================================


def forward_price(S0, r, q, T):
    """F0 = S0 e^((r - q) T): the no-arbitrage price, not a forecast."""
    return S0 * np.exp((r - q) * T)


def futures_to_sell(beta, portfolio_value, futures_price, multiplier):
    """Number of futures to sell to hedge a portfolio: N = beta * V / value of one contract."""
    return beta * portfolio_value / (futures_price * multiplier)


# =============================================================================
# 5. RATES: HULL-WHITE MODEL (simple version)
# =============================================================================


@dataclass
class ZeroCurve:
    """
    Today's zero-coupon curve, for the demo: it starts at z0 (short end) and tends
    to z_inf (long end). In practice it is built from market rates (OIS, swaps).
    """

    z0: float = 0.02
    z_inf: float = 0.04
    tau: float = 3.0

    def zero(self, T):
        T = np.asarray(T, dtype=float)
        return self.z_inf + (self.z0 - self.z_inf) * np.exp(-T / self.tau)

    def discount(self, T):
        """Price of a zero-coupon bond maturing at T: P(0, T) = e^(-z(T) T)."""
        T = np.asarray(T, dtype=float)
        return np.exp(-self.zero(T) * T)

    def forward(self, T):
        """Instantaneous forward rate f(0, T) = derivative of T * z(T)."""
        T = np.asarray(T, dtype=float)
        c = self.z0 - self.z_inf
        return self.z_inf + c * (1 - T / self.tau) * np.exp(-T / self.tau)

    def forward_slope(self, T):
        """Derivative of the forward rate with respect to T (needed to compute theta)."""
        T = np.asarray(T, dtype=float)
        c = self.z0 - self.z_inf
        return -(c / self.tau) * (2 - T / self.tau) * np.exp(-T / self.tau)


@dataclass
class HullWhite:
    """
    Short-rate model: dr = (theta(t) - a r) dt + sigma dW
      - a: speed of mean reversion
      - sigma: volatility of the short rate
      - theta(t): chosen so that the model exactly reprices today's yield curve
    """

    curve: ZeroCurve
    a: float = 0.10
    sigma: float = 0.01

    def theta(self, t):
        a, s = self.a, self.sigma
        return (self.curve.forward_slope(t) + a * self.curve.forward(t)
                + s**2 / (2 * a) * (1 - np.exp(-2 * a * t)))

    def bond_price(self, t, T, r_t):
        """Price at t of a zero-coupon bond maturing at T when the short rate is r_t (closed form)."""
        a, s = self.a, self.sigma
        B = (1 - np.exp(-a * (T - t))) / a
        A = (self.curve.discount(T) / self.curve.discount(t)
             * np.exp(B * self.curve.forward(t) - s**2 / (4 * a) * (1 - np.exp(-2 * a * t)) * B**2))
        return A * np.exp(-B * r_t)

    def simulate(self, T, n_steps, n_paths, seed=0):
        """
        Simulates the short rate step by step: r(t+dt) = r(t) + (theta - a r) dt + sigma sqrt(dt) Z.
        Returns the dates, the paths of r and the discount factor exp(-sum of r dt)
        at each date (excluding date 0).
        """
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        t = np.arange(n_steps + 1) * dt
        theta = self.theta(t)
        z = rng.standard_normal((n_paths, n_steps))
        r = np.empty((n_paths, n_steps + 1))
        r[:, 0] = self.curve.zero(0.0)
        for i in range(n_steps):
            r[:, i + 1] = r[:, i] + (theta[i] - self.a * r[:, i]) * dt + self.sigma * np.sqrt(dt) * z[:, i]
        disc = np.exp(-np.cumsum(r[:, :-1], axis=1) * dt)
        return t, r, disc


# =============================================================================
# CHARTS (optional)
# =============================================================================

_SURFACE, _INK, _SECONDARY, _MUTED, _GRID, _BASE = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7")
_BLUE, _ORANGE, _AQUA = "#2a78d6", "#eb6834", "#1baf7a"


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


def plot_convergence(ns, errors, path="convergence_binomial.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    _style_axes(fig, ax)
    ax.semilogx(ns, errors, color=_BLUE, lw=2, marker="o", ms=7, mec=_SURFACE, mew=2)
    ax.axhline(0, color=_BASE, lw=0.8)
    ax.set_xlabel("number of tree steps")
    ax.set_ylabel("tree price - Black-Scholes price")
    ax.set_title("The binomial tree converges to Black-Scholes", loc="left",
                 color=_INK, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_bond(face=100.0, coupon=0.04, ytm0=0.04, maturity=5, path="duration_convexity.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = bond_metrics(face, coupon, ytm0, maturity)
    ys = np.linspace(0.005, 0.10, 60)
    exact = np.array([bond_price(face, coupon, y, maturity) for y in ys])
    dy = ys - ytm0
    by_duration = m["price"] * (1 - m["modified"] * dy)
    by_both = by_duration + 0.5 * m["convexity"] * dy**2 * m["price"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    _style_axes(fig, ax)
    x = ys * 100
    ax.plot(x, exact, color=_BLUE, lw=2, label="Exact price")
    ax.plot(x, by_duration, color=_ORANGE, lw=2, ls="--", label="Duration only")
    ax.plot(x, by_both, color=_AQUA, lw=2, ls=":", label="Duration + convexity")
    for y, name, off in ((by_both[-1], "Duration + convexity", 7), (exact[-1], "Exact price", -7),
                         (by_duration[-1], "Duration only", 0)):
        ax.annotate(name, (x[-1], y), xytext=(8, off), textcoords="offset points",
                    va="center", color=_SECONDARY, fontsize=9)
    ax.set_xlim(x[0], x[-1] + 3.5)
    ax.set_xticks([2, 4, 6, 8, 10])
    ax.set_xlabel("yield to maturity (%)")
    ax.set_ylabel("bond price")
    ax.set_title("5-year bond, 4% coupon: convexity corrects duration", loc="left",
                 color=_INK, fontsize=12, fontweight="bold")
    ax.legend(frameon=False, labelcolor=_SECONDARY, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_hull_white(t, r, curve, path="hull_white_paths.png", n_shown=25):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    _style_axes(fig, ax)
    for i in range(n_shown):
        ax.plot(t, r[i] * 100, color=_BLUE, lw=0.8, alpha=0.35,
                label="Simulated paths" if i == 0 else None)
    ax.plot(t, r.mean(axis=0) * 100, color=_ORANGE, lw=2, label="Simulated mean")
    ax.plot(t, curve.forward(t) * 100, color=_AQUA, lw=2, ls="--", label="Forward rate of the curve")
    ax.set_xlabel("time (years)")
    ax.set_ylabel("short rate (%)")
    ax.set_title("Hull-White: the simulated short rate follows the yield curve", loc="left",
                 color=_INK, fontsize=12, fontweight="bold")
    ax.legend(frameon=False, labelcolor=_SECONDARY, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =============================================================================
# DEMO
# =============================================================================


def demo_options():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.03, 0.20
    print("=" * 70)
    print("1. OPTIONS: BLACK-SCHOLES  (S=100, K=100, T=1 year, r=3%, sigma=20%)")
    print("=" * 70)
    call = black_scholes(S, K, T, r, sigma, kind="call")
    put = black_scholes(S, K, T, r, sigma, kind="put")
    print(f"Call = {call:.4f} | Put = {put:.4f}")
    print(f"Put-call parity: c + K e^(-rT) - p - S = {call + K * np.exp(-r * T) - put - S:.2e}")
    g = bs_greeks(S, K, T, r, sigma, kind="call")
    print("Call Greeks: " + " | ".join(f"{k} = {v:.4f}" for k, v in g.items()))
    iv = implied_vol(call, S, K, T, r)
    print(f"Implied volatility recovered from the call price: {iv:.4%}")

    print("\nCheck against Hull's examples:")
    p = put_from_call(call=3.0, S=31.0, K=30.0, r=0.10, T=0.25)
    print(f"  Parity: stock 31, K 30, r 10%, 3 months, call 3 -> put = {p:.2f} (expected 1.26)")
    v = bs_greeks(49.0, 50.0, 20 / 52, 0.05, 0.20)["vega"]
    print(f"  Vega: S 49, K 50, r 5%, sigma 20%, 20 weeks -> vega = {v:.1f} "
          f"(expected 12.1: +1 vol point = +{v / 100:.2f} EUR)")


def demo_binomial(plot):
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.03, 0.20
    call = black_scholes(S, K, T, r, sigma)
    print("\n" + "=" * 70)
    print("2. CRR BINOMIAL TREE")
    print("=" * 70)
    ns = [10, 25, 50, 100, 250, 500, 1000]
    prices = [binomial_tree(S, K, T, r, sigma, n=n) for n in ns]
    print(f"Convergence to Black-Scholes ({call:.4f}):")
    print(f"{'n':>6} {'price':>10} {'error':>10}")
    for n, p in zip(ns, prices):
        print(f"{n:>6} {p:>10.4f} {p - call:>+10.4f}")
    eu = binomial_tree(S, K, T, r, sigma, kind="put")
    am = binomial_tree(S, K, T, r, sigma, kind="put", american=True)
    print(f"\nEuropean put = {eu:.4f} | American put = {am:.4f} (early-exercise premium = {am - eu:.4f})")
    ca = binomial_tree(S, K, T, r, sigma, american=True)
    ce = binomial_tree(S, K, T, r, sigma)
    print(f"European call = {ce:.4f} | American call = {ca:.4f} "
          "(no dividend: early exercise of a call is never optimal)")
    if plot:
        plot_convergence(ns, np.array(prices) - call)


def demo_bond(plot):
    face, coupon, ytm, maturity = 100.0, 0.04, 0.04, 5
    m = bond_metrics(face, coupon, ytm, maturity)
    print("\n" + "=" * 70)
    print("3. RATES: 5-YEAR BOND, 4% COUPON, 4% YIELD")
    print("=" * 70)
    print(f"Price = {m['price']:.4f} | Macaulay duration = {m['macaulay']:.3f} years | "
          f"modified duration = {m['modified']:.3f} | convexity = {m['convexity']:.2f} | "
          f"DV01 = {m['dv01']:.4f}")
    print("\nPrice change when yields move:")
    print(f"{'move':>10} {'exact':>10} {'duration':>10} {'dur.+conv.':>12}   (as % of price)")
    for dy in (-0.02, -0.01, 0.01, 0.02):
        exact = bond_price(face, coupon, ytm + dy, maturity) - m["price"]
        d1, d2 = price_change_estimates(m, dy)
        print(f"{dy:>+10.0%} {exact / m['price']:>+10.2%} {d1 / m['price']:>+10.2%} {d2 / m['price']:>+12.2%}")
    if plot:
        plot_bond()


def demo_forwards():
    print("\n" + "=" * 70)
    print("4. FORWARDS, FUTURES AND HEDGING")
    print("=" * 70)
    f = forward_price(8000.0, 0.04, 0.02, 1.0)
    print(f"Index at 8000, r = 4%, dividend yield = 2%, 1 year: theoretical forward price = {f:.1f}")
    print(f"  If the futures trades at 8400: too expensive by {8400 - f:.1f} -> sell the futures and buy the stocks")
    n = futures_to_sell(beta=1.5, portfolio_value=5_050_000, futures_price=5050, multiplier=10)
    print(f"Portfolio of EUR 5.05M, beta 1.5, futures at 5050 points (EUR 10/point): "
          f"{n:.0f} futures to sell")


def demo_hull_white(plot):
    curve = ZeroCurve()
    hw = HullWhite(curve, a=0.10, sigma=0.01)
    print("\n" + "=" * 70)
    print("5. RATES: HULL-WHITE (simple version)")
    print("=" * 70)
    print(f"Today's zero-coupon curve: 1y {curve.zero(1.0):.2%} | 5y {curve.zero(5.0):.2%} "
          f"| 10y {curve.zero(10.0):.2%}")
    print(f"Parameters: a = {hw.a} (mean reversion) | sigma = {hw.sigma:.2%}")
    p_formula = hw.bond_price(0.0, 5.0, curve.zero(0.0))
    print(f"5-year zero-coupon bond: curve = {curve.discount(5.0):.5f} | model formula = {p_formula:.5f}")
    t, r, disc = hw.simulate(T=5.0, n_steps=260, n_paths=20000)
    final = disc[:, -1]
    print(f"Simulation (20,000 paths): 5-year zero-coupon price = {final.mean():.5f} "
          f"+/- {final.std(ddof=1) / np.sqrt(len(final)):.5f} -> the model reprices the curve")
    print("Simulated short rate:")
    for yr in (1, 3, 5):
        col = r[:, 52 * yr]
        print(f"  in {yr} year(s): mean {col.mean():.2%} | std dev {col.std():.2%} "
              f"| share of negative rates {(col < 0).mean():.1%}")
    if plot:
        plot_hull_white(t, r, curve)


def main():
    ap = argparse.ArgumentParser(description="Derivatives basics: Black-Scholes, binomial tree, rates, forwards, Hull-White")
    ap.add_argument("--plot", action="store_true", help="save the charts as PNG files")
    args = ap.parse_args()
    demo_options()
    demo_binomial(args.plot)
    demo_bond(args.plot)
    demo_forwards()
    demo_hull_white(args.plot)
    if args.plot:
        print("\nCharts saved: convergence_binomial.png, duration_convexity.png, "
              "hull_white_paths.png")


if __name__ == "__main__":
    main()
