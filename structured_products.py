"""
structured_products.py
======================
Pricer de produits structures actions par simulation (Monte Carlo).

Produits :
  - Reverse convertible a barriere : coupon fixe + vente d'un put down-and-in
  - Note a capital protege : zero-coupon + participation a la hausse (un call)
  - Autocall Athena (coupon cumule paye au rappel) et Phoenix (coupon conditionnel
    avec memoire)

Methode, en 4 etapes :
  1. on simule des trajectoires du sous-jacent dans le monde risque-neutre
     (le sous-jacent rapporte r - q en moyenne),
  2. on calcule le payoff du produit sur chaque trajectoire,
  3. on actualise au taux sans risque,
  4. on fait la moyenne : c'est le prix.

Le coupon juste est celui pour lequel la valeur des flux = 100 - marge du structureur.

Usage :
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
# MARCHE ET SIMULATION
# =============================================================================


@dataclass(frozen=True)
class Market:
    """Hypotheses de marche : volatilite plate, taux et dividende constants."""

    r: float = 0.025      # taux sans risque
    q: float = 0.03       # rendement du dividende
    sigma: float = 0.20   # volatilite
    spot: float = 1.0     # niveau actuel / niveau initial du produit (1.0 = a la date de strike)


def normals(n_paths, n_steps, seed=0):
    """Tirages gaussiens. On garde les memes pour tous les prix, ce qui rend les grecques stables."""
    return np.random.default_rng(seed).standard_normal((n_paths, n_steps))


def simulate_levels(m: Market, times, z):
    """Niveaux S(t) / S_initial aux dates d'observation (mouvement brownien geometrique)."""
    dt = np.diff(np.r_[0.0, np.asarray(times, dtype=float)])
    incr = (m.r - m.q - 0.5 * m.sigma**2) * dt + m.sigma * np.sqrt(dt) * z
    return m.spot * np.exp(np.cumsum(incr, axis=1))


def bs_call(S, K, T, r, q, sigma):
    """Call europeen Black-Scholes, sert a verifier la simulation."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# =============================================================================
# PRODUITS : chacun renvoie les flux (n_trajectoires x n_dates) en % du nominal
# =============================================================================


@dataclass
class ReverseConvertible:
    """Coupon fixe chaque annee. A maturite : 100 si S_T >= barriere, sinon livraison
    de titres (100 * S_T / S_0). L'investisseur est vendeur d'un put down-and-in."""

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
    """Zero-coupon + participation a la hausse : 100 * (protection + part * max(S_T/S_0 - 1, 0))."""

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
    Autocall a observation annuelle.
    - Rappel : si S_i >= barriere de rappel, remboursement de 100 et arret du produit.
    - Athena (coupon_barrier=None) : le coupon est cumule et paye au rappel (100 * (1 + c * i)).
    - Phoenix (coupon_barrier donne) : coupon c paye chaque annee ou S_i >= barriere de coupon,
      avec memoire des coupons manques si memory=True. Ce coupon est une option digitale.
    - Maturite sans rappel : 100 si S_N >= barriere de protection, sinon 100 * S_N / S_0.
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
        alive = np.ones(n, dtype=bool)        # produit pas encore rappele
        called_at = np.full(n, -1)            # indice de la date de rappel (-1 = jamais)
        missed = np.zeros(n)                  # coupons manques (memoire Phoenix)
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
# PRICING, COUPON JUSTE, GRECQUES, STATISTIQUES
# =============================================================================


@dataclass
class Result:
    pv: float          # valeur actuelle des flux (% du nominal)
    se: float          # erreur standard de la simulation
    levels: np.ndarray
    called_at: np.ndarray


def price(product, m: Market, z) -> Result:
    L = simulate_levels(m, product.times, z)
    cf, called_at = product.cashflows(L)
    pv = (cf * np.exp(-m.r * product.times)).sum(axis=1)
    return Result(float(pv.mean()), float(pv.std(ddof=1) / np.sqrt(len(pv))), L, called_at)


def fair_coupon(product, m: Market, z, margin=0.015):
    """
    Coupon annuel pour lequel la valeur des flux = 100 - marge.
    Les flux sont lineaires en le coupon : valeur(c) = valeur(0) + c * pente.
    Deux calculs suffisent : coupon nul, puis coupon de 100 % pour trouver la pente.
    Renvoie 0 si, meme sans coupon, la valeur depasse deja la cible.
    """
    target = NOMINAL * (1.0 - margin)
    v0 = price(replace(product, coupon=0.0), m, z).pv
    v1 = price(replace(product, coupon=1.0), m, z).pv
    return max(0.0, (target - v0) / (v1 - v0))


def fair_participation(maturity, m: Market, margin=0.015, protection=1.0):
    """Participation juste d'une note a capital protege : on depense en call ce qui reste
    apres l'achat du zero-coupon (formule fermee)."""
    budget = NOMINAL * (1.0 - margin) - NOMINAL * protection * np.exp(-m.r * maturity)
    call = NOMINAL * bs_call(1.0, 1.0, maturity, m.r, m.q, m.sigma)
    return budget / call


def greeks(product, m: Market, z, bump_spot=0.01, bump_vol=0.01):
    """Grecques par bump and reprice, en points de nominal : on bouge le spot de 1 %
    ou la volatilite de 1 point, avec les memes tirages, et on regarde la variation du prix."""
    p0 = price(product, m, z).pv
    up = price(product, replace(m, spot=m.spot * (1 + bump_spot)), z).pv
    dn = price(product, replace(m, spot=m.spot * (1 - bump_spot)), z).pv
    vu = price(product, replace(m, sigma=m.sigma + bump_vol), z).pv
    vd = price(product, replace(m, sigma=m.sigma - bump_vol), z).pv
    return {"delta": (up - dn) / 2, "gamma": up - 2 * p0 + dn, "vega": (vu - vd) / 2}


def autocall_probabilities(called_at, n_dates):
    """P(rappel a chaque date) puis P(pas de rappel). La somme vaut 1."""
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
# GRAPHIQUES (optionnels)
# =============================================================================

_SURFACE, _INK, _SECONDARY, _MUTED, _GRID, _BASE = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7")
_SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # bleu, orange, aqua


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
    ax.set_xlabel("volatilite du sous-jacent (%)")
    ax.set_ylabel("coupon juste (% par an)")
    ax.set_title("Plus la volatilite monte, plus le coupon est eleve", loc="left",
                 color=_INK, fontsize=12, fontweight="bold")
    ax.legend(frameon=False, labelcolor=_SECONDARY, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_autocall_probs(probs, path="autocall_probas.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"An {i + 1}" for i in range(len(probs) - 1)] + ["Pas de rappel"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _style_axes(fig, ax)
    vals = np.array(probs) * 100
    ax.bar(labels, vals, color=_SERIES[0], width=0.65)
    for i in (int(vals[:-1].argmax()), len(vals) - 1):
        ax.annotate(f"{vals[i]:.0f} %", (i, vals[i]), xytext=(0, 4), textcoords="offset points",
                    ha="center", color=_INK, fontsize=9)
    ax.set_ylabel("probabilite (%)")
    ax.set_title("Quand l'autocall est-il rappele ?", loc="left",
                 color=_INK, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =============================================================================
# DEMO
# =============================================================================


def _pct(x, d=2):
    return f"{x * 100:.{d}f} %"


def demo(m: Market, n_paths, margin, plot):
    z3, z5 = normals(n_paths, 3), normals(n_paths, 5)
    print("=" * 74)
    print(f"MARCHE : r = {_pct(m.r, 1)} | dividende = {_pct(m.q, 1)} | vol = {_pct(m.sigma, 0)} "
          f"| marge structureur = {_pct(margin, 1)} | {n_paths} trajectoires")
    print("=" * 74)

    # 1. Note a capital protege
    T = 5
    part = fair_participation(T, m, margin)
    cpn = CapitalProtectedNote(part, 1.0, T)
    res = price(cpn, m, normals(n_paths, T))
    zc = NOMINAL * np.exp(-m.r * T)
    call = NOMINAL * bs_call(1.0, 1.0, T, m.r, m.q, m.sigma)
    print("\n1. NOTE A CAPITAL PROTEGE 5 ANS (100 % garanti)")
    print(f"   Zero-coupon 5 ans = {zc:.2f} | budget pour l'option = {NOMINAL * (1 - margin) - zc:.2f} "
          f"| call ATM 5 ans = {call:.2f}")
    print(f"   Participation juste a la hausse : {_pct(part, 1)}")
    print(f"   Controle par simulation : valeur = {res.pv:.3f} +/- {res.se:.3f} (cible {NOMINAL * (1 - margin):.2f})")

    # 2. Reverse convertible
    rc0 = ReverseConvertible(coupon=0.0, barrier=0.70, maturity=3)
    c_rc = fair_coupon(rc0, m, z3, margin)
    rc = replace(rc0, coupon=c_rc)
    r_rc, g_rc = price(rc, m, z3), greeks(rc, m, z3)
    rp = risk_profile(rc, r_rc)
    print("\n2. REVERSE CONVERTIBLE 3 ANS (barriere europeenne 70 %)")
    print(f"   Coupon juste : {_pct(c_rc)} par an")
    print(f"   P(perte en capital) = {_pct(rp['p_loss'], 1)} | perte moyenne si perte = {rp['avg_loss']:.1f} %")
    print(f"   Delta = {g_rc['delta']:+.3f} | Gamma = {g_rc['gamma']:+.3f} | Vega = {g_rc['vega']:+.3f} "
          "(points de nominal, par +1 % de spot / +1 pt de vol)")

    # 3. Autocalls
    athena0 = Autocall(coupon=0.0, maturity=5)
    phoenix0 = Autocall(coupon=0.0, maturity=5, coupon_barrier=0.70, memory=True)
    probs_athena = None
    print("\n3. AUTOCALLS 5 ANS (observation annuelle, rappel a 100 %, protection a 60 %)")
    for name, prod0 in (("Athena", athena0), ("Phoenix (barriere coupon 70 %, memoire)", phoenix0)):
        c = fair_coupon(prod0, m, z5, margin)
        prod = replace(prod0, coupon=c)
        r, g = price(prod, m, z5), greeks(prod, m, z5)
        rp = risk_profile(prod, r)
        if name == "Athena":
            probs_athena = rp["probs"]
        print(f"\n   {name}")
        print(f"   Coupon juste : {_pct(c)} par an")
        print("   P(rappel) par annee : " + " | ".join(
            f"An {i + 1} {_pct(p, 1)}" for i, p in enumerate(rp["probs"][:-1])) +
            f" | pas de rappel {_pct(rp['probs'][-1], 1)}")
        print(f"   Duree de vie moyenne = {rp['life']:.2f} ans | P(perte en capital) = {_pct(rp['p_loss'], 1)}"
              f" | perte moyenne si perte = {rp['avg_loss']:.1f} %")
        print(f"   Delta = {g['delta']:+.3f} | Gamma = {g['gamma']:+.3f} | Vega = {g['vega']:+.3f}")

    # 4. Sensibilites
    vols = [0.12, 0.15, 0.20, 0.25, 0.30]
    table = {"Reverse convertible": [], "Athena": [], "Phoenix": []}
    print("\n4. SENSIBILITE DU COUPON JUSTE A LA VOLATILITE")
    print(f"   {'vol':>6} {'Rev. conv.':>12} {'Athena':>10} {'Phoenix':>10}")
    for v in vols:
        mv = replace(m, sigma=v)
        row = (fair_coupon(rc0, mv, z3, margin), fair_coupon(athena0, mv, z5, margin),
               fair_coupon(phoenix0, mv, z5, margin))
        for key, c in zip(table, row):
            table[key].append(c)
        print(f"   {_pct(v, 0):>6} " + " ".join(f"{_pct(c):>10}" for c in row))

    print("\n   SENSIBILITE DU COUPON ATHENA A LA BARRIERE DE PROTECTION")
    print(f"   {'barriere':>9} {'coupon juste':>13}")
    for b in (0.50, 0.60, 0.70, 0.80):
        c = fair_coupon(replace(athena0, protection_barrier=b), m, z5, margin)
        print(f"   {_pct(b, 0):>9} {_pct(c):>13}")

    if plot:
        plot_coupon_vs_vol(vols, table)
        plot_autocall_probs(probs_athena)
        print("\nGraphiques sauvegardes : coupon_vs_vol.png, autocall_probas.png")


def main():
    ap = argparse.ArgumentParser(description="Pricer de produits structures (Monte Carlo)")
    ap.add_argument("--r", type=float, default=0.025, help="taux sans risque")
    ap.add_argument("--q", type=float, default=0.03, help="rendement du dividende")
    ap.add_argument("--sigma", type=float, default=0.20, help="volatilite")
    ap.add_argument("--margin", type=float, default=0.015, help="marge du structureur (en fraction du nominal)")
    ap.add_argument("--paths", type=int, default=200_000, help="nombre de trajectoires")
    ap.add_argument("--plot", action="store_true", help="sauvegarde les graphiques en PNG")
    args = ap.parse_args()
    demo(Market(r=args.r, q=args.q, sigma=args.sigma), args.paths, args.margin, args.plot)


if __name__ == "__main__":
    main()
