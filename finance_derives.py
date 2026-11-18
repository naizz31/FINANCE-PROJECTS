"""
finance_derives.py
==================
Projet perso de finance de marche : pricing de derives en Python.

1. Option equity : Black-Scholes (+ grecques) et arbre binomial CRR
   (europeen / americain).
2. Taux, calibration : estimation de (a, sigma) sur des taux courts historiques
   par maximum de vraisemblance sur grille, puis detection de regimes de crise
   (melange gaussien) avec backtest hors echantillon.
3. Taux, pricing : modele de Hull-White a un facteur cale sur une courbe de taux,
   option sur zero-coupon en formule fermee et par Monte Carlo, stress de la
   volatilite calme / crise.

Usage :
    python finance_derives.py                   # demo sur donnees synthetiques
    python finance_derives.py --csv taux.csv    # calibration sur tes donnees
    python finance_derives.py --plot            # sauvegarde les graphiques
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import norm

# =============================================================================
# 1. OPTION EQUITY : Black-Scholes et arbre binomial
# =============================================================================


def _d1_d2(S, K, T, r, sigma, q):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


def black_scholes(S, K, T, r, sigma, q=0.0, kind="call"):
    """Prix Black-Scholes-Merton d'une option europeenne (dividende continu q)."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    if kind == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def bs_greeks(S, K, T, r, sigma, q=0.0, kind="call"):
    """Grecques analytiques (vega par unite de vol, theta par an, rho par unite de taux)."""
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


def binomial_tree(S, K, T, r, sigma, n=500, q=0.0, kind="call", american=False):
    """Arbre binomial Cox-Ross-Rubinstein (induction arriere vectorisee)."""
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)  # probabilite risque-neutre
    disc = np.exp(-r * dt)
    sign = 1.0 if kind == "call" else -1.0

    def payoff(s):
        return np.maximum(sign * (s - K), 0.0)

    j = np.arange(n + 1)  # j = nombre de baisses
    V = payoff(S * u ** (n - j) * d**j)
    for i in range(n - 1, -1, -1):
        V = disc * (p * V[: i + 1] + (1 - p) * V[1 : i + 2])
        if american:  # exercice anticipe a chaque noeud
            jj = np.arange(i + 1)
            V = np.maximum(V, payoff(S * u ** (i - jj) * d**jj))
    return float(V[0])


# =============================================================================
# 2. COURBE DES TAUX (Nelson-Siegel) ET MODELE DE HULL-WHITE
# =============================================================================


@dataclass
class NelsonSiegel:
    """Courbe zero-coupon continue : z(T) = b0 + b1*g(T) + b2*(g(T) - exp(-T/tau))."""

    b0: float
    b1: float
    b2: float
    tau: float

    def zero(self, T):
        T = np.maximum(np.asarray(T, dtype=float), 1e-8)
        x = T / self.tau
        g = (1 - np.exp(-x)) / x
        return self.b0 + self.b1 * g + self.b2 * (g - np.exp(-x))

    def discount(self, T):
        """Facteur d'actualisation P(0, T)."""
        T = np.asarray(T, dtype=float)
        return np.exp(-self.zero(T) * T)

    def forward(self, T):
        """Taux forward instantane f(0, T) = d(T * z(T)) / dT."""
        x = np.asarray(T, dtype=float) / self.tau
        return self.b0 + self.b1 * np.exp(-x) + self.b2 * x * np.exp(-x)


def fit_nelson_siegel(maturities, zero_rates):
    """Cale Nelson-Siegel sur des taux zero-coupon de marche (OIS, swaps...)."""
    m, z = np.asarray(maturities, float), np.asarray(zero_rates, float)

    def err(p):
        return NelsonSiegel(*p).zero(m) - z

    sol = least_squares(
        err, x0=[z[-1], z[0] - z[-1], 0.0, 2.0],
        bounds=([-1, -1, -1, 0.1], [1, 1, 1, 30]),
    )
    return NelsonSiegel(*sol.x)


@dataclass
class HullWhite:
    """
    Hull-White 1 facteur : dr = (theta(t) - a*r) dt + sigma dW.
    theta(t) est choisi pour reproduire exactement la courbe initiale.
    On simule r(t) = x(t) + alpha(t), avec dx = -a*x dt + sigma dW et x(0) = 0.
    """

    curve: NelsonSiegel
    a: float
    sigma: float

    def B(self, t, T):
        return (1 - np.exp(-self.a * (T - t))) / self.a

    def alpha(self, t):
        t = np.asarray(t, dtype=float)
        return (self.curve.forward(t)
                + self.sigma**2 / (2 * self.a**2) * (1 - np.exp(-self.a * t)) ** 2)

    def bond_price(self, t, T, r_t):
        """Prix P(t, T) du zero-coupon sachant le taux court r(t)."""
        a, s = self.a, self.sigma
        B = self.B(t, T)
        A = (self.curve.discount(T) / self.curve.discount(t)
             * np.exp(B * self.curve.forward(t)
                      - s**2 / (4 * a) * (1 - np.exp(-2 * a * t)) * B**2))
        return A * np.exp(-B * r_t)

    def zcb_option(self, T, S, K, kind="call"):
        """Option d'echeance T sur un zero-coupon d'echeance S (formule fermee)."""
        a, s = self.a, self.sigma
        sig_p = s * np.sqrt((1 - np.exp(-2 * a * T)) / (2 * a)) * self.B(T, S)
        P_T, P_S = self.curve.discount(T), self.curve.discount(S)
        h = np.log(P_S / (P_T * K)) / sig_p + 0.5 * sig_p
        if kind == "call":
            return P_S * norm.cdf(h) - K * P_T * norm.cdf(h - sig_p)
        return K * P_T * norm.cdf(-h + sig_p) - P_S * norm.cdf(-h)

    def simulate(self, T, n_steps, n_paths, seed=0):
        """Trajectoires de r (schema exact pour x) et facteurs d'actualisation cumules."""
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        t = np.linspace(0.0, T, n_steps + 1)
        decay = np.exp(-self.a * dt)
        std = self.sigma * np.sqrt((1 - np.exp(-2 * self.a * dt)) / (2 * self.a))
        z = rng.standard_normal((n_paths // 2, n_steps))
        z = np.vstack([z, -z])  # variables antithetiques
        x = np.zeros((z.shape[0], n_steps + 1))
        for i in range(n_steps):
            x[:, i + 1] = x[:, i] * decay + std * z[:, i]
        r = x + self.alpha(t)
        integral = np.cumsum(0.5 * (r[:, 1:] + r[:, :-1]) * dt, axis=1)  # trapezes
        disc = np.exp(-np.hstack([np.zeros((r.shape[0], 1)), integral]))
        return t, r, disc

    def zcb_option_mc(self, T, S, K, kind="call", n_paths=20000, n_steps=100, seed=0):
        """Prix Monte Carlo et erreur standard (prudente : paires antithetiques ignorees)."""
        _, r, disc = self.simulate(T, n_steps, n_paths, seed)
        P_TS = self.bond_price(T, S, r[:, -1])
        pay = np.maximum(P_TS - K, 0.0) if kind == "call" else np.maximum(K - P_TS, 0.0)
        pv = disc[:, -1] * pay
        return float(pv.mean()), float(pv.std(ddof=1) / np.sqrt(len(pv)))


# =============================================================================
# 3. CALIBRATION SUR TAUX HISTORIQUES + REGIMES DE CRISE
# =============================================================================


def var_to_sigma(var, a, dt):
    """Variance des residus a 1 pas -> volatilite annualisee sigma du modele OU."""
    return float(np.sqrt(var * 2 * a / (1 - np.exp(-2 * a * dt))))


def ou_ols(rates, dt):
    """Estimation OU par regression r(t+1) = c + phi * r(t). Renvoie (a, b, sigma)."""
    x, y = rates[:-1], rates[1:]
    phi, _ = np.polyfit(x, y, 1)
    a = max(-np.log(np.clip(phi, 1e-6, 1 - 1e-12)) / dt, 1e-3)
    phi = np.exp(-a * dt)
    b = float(rates.mean())
    resid = y - (b + (x - b) * phi)
    return float(a), b, var_to_sigma(np.mean(resid**2), a, dt)


def ou_loglik(rates, dt, a, sigma, b):
    """Log-vraisemblance exacte d'un processus OU (transition gaussienne)."""
    x, y = rates[:-1], rates[1:]
    mean = b + (x - b) * np.exp(-a * dt)
    var = sigma**2 * (1 - np.exp(-2 * a * dt)) / (2 * a)
    return -0.5 * np.sum(np.log(2 * np.pi * var) + (y - mean) ** 2 / var)


def calibrate_grid(rates, dt, a_grid, sigma_grid):
    """Recherche sur grille du couple (a, sigma) qui maximise la vraisemblance."""
    b = rates.mean()
    ll = np.array([[ou_loglik(rates, dt, a, s, b) for s in sigma_grid] for a in a_grid])
    i, j = np.unravel_index(ll.argmax(), ll.shape)
    return float(a_grid[i]), float(sigma_grid[j]), ll


def regime_backtest(rates, dt, window=20, train_frac=0.7, seed=0):
    """
    Detection de regimes de volatilite par melange gaussien (2 composantes) sur la
    volatilite glissante des variations de taux. Tout est estime sur la premiere
    partie de l'echantillon, puis la vraisemblance predictive est comparee hors
    echantillon : sigma unique vs sigma dependant du regime observe la veille.
    """
    from sklearn.mixture import GaussianMixture

    d = np.diff(rates)
    vol = pd.Series(d).rolling(window).std().bfill().to_numpy()
    feat = np.log(vol + 1e-12).reshape(-1, 1)
    n_tr = int(train_frac * len(d))

    a, b, _ = ou_ols(rates[: n_tr + 1], dt)
    resid = rates[1:] - (b + (rates[:-1] - b) * np.exp(-a * dt))  # erreur a 1 pas

    gmm = GaussianMixture(n_components=2, random_state=seed).fit(feat[:n_tr])
    crisis_state = int(np.argmax(gmm.means_.ravel()))
    is_crisis = gmm.predict(feat) == crisis_state
    prev_crisis = np.r_[False, is_crisis[:-1]]  # regime connu la veille : pas de look-ahead

    tr, te = slice(0, n_tr), slice(n_tr, None)
    var_all = np.mean(resid[tr] ** 2)
    m_cr, m_ca = prev_crisis[tr], ~prev_crisis[tr]
    var_crisis = np.mean(resid[tr][m_cr] ** 2) if m_cr.any() else var_all
    var_calm = np.mean(resid[tr][m_ca] ** 2) if m_ca.any() else var_all

    def loglik(e, var):
        return float(-0.5 * np.sum(np.log(2 * np.pi * var) + e**2 / var))

    var_regime = np.where(prev_crisis, var_crisis, var_calm)
    return {
        "a": a, "b": b, "n_train": n_tr, "n_test": len(d) - n_tr,
        "sigma_all": var_to_sigma(var_all, a, dt),
        "sigma_calm": var_to_sigma(var_calm, a, dt),
        "sigma_crisis": var_to_sigma(var_crisis, a, dt),
        "ll_single": loglik(resid[te], var_all),
        "ll_regime": loglik(resid[te], var_regime[te]),
        "crisis_share_train": float(prev_crisis[tr].mean()),
        "crisis_share_test": float(prev_crisis[te].mean()),
        "is_crisis": is_crisis,
    }


def synthetic_rates(n=3000, dt=1 / 252, a=0.15, b=0.02,
                    sig_calm=0.005, sig_crisis=0.02, seed=1):
    """Taux court simule (OU) avec deux episodes de crise : un dans l'echantillon
    d'estimation, un dans l'echantillon de test."""
    rng = np.random.default_rng(seed)
    sigma = np.full(n, sig_calm)
    sigma[int(0.25 * n): int(0.33 * n)] = sig_crisis
    sigma[int(0.80 * n): int(0.88 * n)] = sig_crisis
    decay = np.exp(-a * dt)
    std = np.sqrt((1 - np.exp(-2 * a * dt)) / (2 * a))
    r = np.empty(n + 1)
    r[0] = b
    for i in range(n):
        r[i + 1] = b + (r[i] - b) * decay + sigma[i] * std * rng.standard_normal()
    return r


def load_rates_csv(path, rate_col="rate", date_col="date"):
    """Charge un CSV (date, taux en %) et renvoie les taux en decimal, tries par date."""
    df = pd.read_csv(path, parse_dates=[date_col]).sort_values(date_col)
    return df[rate_col].astype(float).dropna().to_numpy() / 100.0


# =============================================================================
# DEMOS
# =============================================================================


def demo_equity(plot=False):
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.03, 0.20
    print("=" * 70)
    print("1. OPTION EQUITY  (S=100, K=100, T=1 an, r=3 %, sigma=20 %)")
    print("=" * 70)
    call = black_scholes(S, K, T, r, sigma, kind="call")
    put = black_scholes(S, K, T, r, sigma, kind="put")
    print(f"Black-Scholes : call = {call:.4f} | put = {put:.4f}")
    print(f"Parite call-put (ecart) : {call - put - (S - K * np.exp(-r * T)):.2e}")
    g = bs_greeks(S, K, T, r, sigma, kind="call")
    print("Grecques du call : " + " | ".join(f"{k} = {v:.4f}" for k, v in g.items()))

    ns = [10, 25, 50, 100, 250, 500, 1000]
    prices = [binomial_tree(S, K, T, r, sigma, n=n) for n in ns]
    print("\nConvergence de l'arbre CRR vers Black-Scholes (call europeen) :")
    print(f"{'n':>6} {'prix':>10} {'ecart':>10}")
    for n, p in zip(ns, prices):
        print(f"{n:>6} {p:>10.4f} {p - call:>+10.4f}")

    eu = binomial_tree(S, K, T, r, sigma, n=500, kind="put")
    am = binomial_tree(S, K, T, r, sigma, n=500, kind="put", american=True)
    print(f"\nPut europeen = {eu:.4f} | put americain = {am:.4f} "
          f"(prime d'exercice anticipe = {am - eu:.4f})")

    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4))
        plt.semilogx(ns, np.array(prices) - call, "o-")
        plt.axhline(0, color="k", lw=0.8)
        plt.xlabel("nombre de pas n")
        plt.ylabel("arbre CRR - Black-Scholes")
        plt.title("Convergence de l'arbre binomial")
        plt.tight_layout()
        plt.savefig("convergence_binomial.png", dpi=150)
        plt.close()


def demo_rates_calibration(rates, dt, source, plot=False):
    print("\n" + "=" * 70)
    print(f"2. CALIBRATION DES TAUX ET REGIMES DE CRISE  [{source}]")
    print("=" * 70)
    a_ols, b_ols, s_ols = ou_ols(rates, dt)
    a_grid, s_grid, _ = calibrate_grid(
        rates, dt, np.linspace(0.01, 1.0, 80), np.linspace(0.001, 0.05, 120))
    print(f"{len(rates)} observations, pas dt = {dt:.5f} an")
    print(f"Regression OLS    : a = {a_ols:.3f} | sigma = {s_ols * 100:.2f} % | b = {b_ols * 100:.2f} %")
    print(f"Grille (max vrais.): a = {a_grid:.3f} | sigma = {s_grid * 100:.2f} %")

    res = regime_backtest(rates, dt)
    print(f"\nRegimes (melange gaussien, estimes sur les {res['n_train']} premieres observations) :")
    print(f"  sigma unique  = {res['sigma_all'] * 100:.2f} %")
    print(f"  sigma calme   = {res['sigma_calm'] * 100:.2f} %")
    print(f"  sigma crise   = {res['sigma_crisis'] * 100:.2f} %")
    print(f"  part de crise : estimation {res['crisis_share_train']:.1%} | test {res['crisis_share_test']:.1%}")
    gain = (res["ll_regime"] - res["ll_single"]) / res["n_test"]
    print(f"Hors echantillon ({res['n_test']} obs) : log-vraisemblance sigma unique = "
          f"{res['ll_single']:.1f} | avec regimes = {res['ll_regime']:.1f} "
          f"(gain {gain:+.3f} par observation)")

    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(rates * 100, lw=0.8)
        crisis = np.r_[False, res["is_crisis"]]
        ax.fill_between(np.arange(len(rates)), rates.min() * 100, rates.max() * 100,
                        where=crisis, color="tab:red", alpha=0.25, label="regime de crise detecte")
        ax.axvline(res["n_train"], ls="--", color="k", label="fin de l'echantillon d'estimation")
        ax.set_ylabel("taux court (%)")
        ax.legend()
        ax.set_title("Detection des regimes de crise")
        fig.tight_layout()
        fig.savefig("regimes_taux.png", dpi=150)
        plt.close(fig)
    return res


def demo_hull_white(sig_calm, sig_crisis):
    print("\n" + "=" * 70)
    print("3. TAUX : HULL-WHITE ET OPTION SUR ZERO-COUPON")
    print("=" * 70)
    curve = NelsonSiegel(0.035, -0.015, 0.010, 2.0)  # courbe illustrative : remplacer par OIS/swaps
    T, S = 1.0, 5.0
    K = curve.discount(S) / curve.discount(T)  # strike at-the-money forward
    hw = HullWhite(curve, a=0.10, sigma=0.01)

    _, _, disc = hw.simulate(S, n_steps=130, n_paths=20000)
    mc = disc[:, -1].mean()
    print(f"Verification du calage : P(0,5) courbe = {curve.discount(S):.5f} | Monte Carlo = {mc:.5f}")

    cf = hw.zcb_option(T, S, K, "call")
    mc_c, se = hw.zcb_option_mc(T, S, K, "call")
    print(f"\nCall 1 an sur ZC 5 ans (nominal 100, K = {K:.4f}, a = 0.10, sigma = 1.00 %) :")
    print(f"  formule fermee = {cf * 100:.4f} | Monte Carlo = {mc_c * 100:.4f} (+/- {se * 100:.4f})")

    print("\nStress de volatilite (parametres du modele de regimes) :")
    print(f"{'regime':<8} {'sigma':>8} {'call (nominal 100)':>20}")
    for name, s in [("calme", sig_calm), ("crise", sig_crisis)]:
        price = HullWhite(curve, hw.a, s).zcb_option(T, S, K, "call")
        print(f"{name:<8} {s * 100:>7.2f}% {price * 100:>20.4f}")


def main():
    ap = argparse.ArgumentParser(description="Pricing de derives : equity (BS, binomial) et taux (Hull-White)")
    ap.add_argument("--csv", help="CSV de taux courts historiques (colonnes date et rate, en %%)")
    ap.add_argument("--rate-col", default="rate")
    ap.add_argument("--date-col", default="date")
    ap.add_argument("--dt", type=float, default=1 / 252, help="pas de temps des donnees (defaut : journalier)")
    ap.add_argument("--plot", action="store_true", help="sauvegarde les graphiques en PNG")
    args = ap.parse_args()

    demo_equity(args.plot)
    if args.csv:
        rates, source = load_rates_csv(args.csv, args.rate_col, args.date_col), args.csv
    else:
        rates, source = synthetic_rates(dt=args.dt), "donnees synthetiques, 2 crises simulees"
    res = demo_rates_calibration(rates, args.dt, source, args.plot)
    demo_hull_white(res["sigma_calm"], res["sigma_crisis"])


if __name__ == "__main__":
    main()
