import numpy as np

from finance_derives import (
    HullWhite, ZeroCurve, binomial_tree, black_scholes, bond_metrics, bond_price, bs_greeks,
    forward_price, futures_to_sell, implied_vol, put_from_call,
)

S, K, T, R, SIG = 100.0, 100.0, 1.0, 0.03, 0.20


def test_put_call_parity():
    call = black_scholes(S, K, T, R, SIG, kind="call")
    put = black_scholes(S, K, T, R, SIG, kind="put")
    assert abs(call - put - (S - K * np.exp(-R * T))) < 1e-10


def test_parity_example_from_the_book():
    assert abs(put_from_call(3.0, 31.0, 30.0, 0.10, 0.25) - 1.26) < 0.01


def test_vega_example_from_the_book():
    assert abs(bs_greeks(49.0, 50.0, 20 / 52, 0.05, 0.20)["vega"] - 12.1) < 0.05


def test_greeks_match_finite_differences():
    h = 1e-3
    g = bs_greeks(S, K, T, R, SIG)
    delta = (black_scholes(S + h, K, T, R, SIG) - black_scholes(S - h, K, T, R, SIG)) / (2 * h)
    vega = (black_scholes(S, K, T, R, SIG + h) - black_scholes(S, K, T, R, SIG - h)) / (2 * h)
    assert abs(g["delta"] - delta) < 1e-5
    assert abs(g["vega"] - vega) < 1e-4


def test_implied_vol_round_trip():
    price = black_scholes(S, 110.0, 0.5, R, 0.27, kind="put")
    assert abs(implied_vol(price, S, 110.0, 0.5, R, kind="put") - 0.27) < 1e-8


def test_binomial_converges_to_black_scholes():
    assert abs(binomial_tree(S, K, T, R, SIG, n=1000) - black_scholes(S, K, T, R, SIG)) < 0.02


def test_american_put_worth_more_than_european():
    assert binomial_tree(S, K, T, R, SIG, kind="put", american=True) > binomial_tree(S, K, T, R, SIG, kind="put")


def test_american_call_without_dividend_equals_european():
    am = binomial_tree(S, K, T, R, SIG, n=300, american=True)
    eu = binomial_tree(S, K, T, R, SIG, n=300)
    assert abs(am - eu) < 1e-8


def test_bond_at_par_prices_to_face():
    assert abs(bond_price(100.0, 0.04, 0.04, 5) - 100.0) < 1e-9


def test_modified_duration_matches_numerical_derivative():
    h = 1e-6
    m = bond_metrics(100.0, 0.04, 0.05, 5)
    num = -(bond_price(100.0, 0.04, 0.05 + h, 5) - bond_price(100.0, 0.04, 0.05 - h, 5)) / (2 * h * m["price"])
    assert abs(num - m["modified"]) < 1e-6


def test_convexity_matches_numerical_second_derivative():
    h = 1e-4
    m = bond_metrics(100.0, 0.04, 0.05, 5)
    up, mid, dn = (bond_price(100.0, 0.04, 0.05 + e, 5) for e in (h, 0.0, -h))
    num = (up - 2 * mid + dn) / (h**2 * mid)
    assert abs(num / m["convexity"] - 1) < 1e-4


def test_forward_price_example_from_the_book():
    assert abs(forward_price(8000.0, 0.04, 0.02, 1.0) - 8161.6) < 0.1


def test_hedge_example_from_the_book():
    assert round(futures_to_sell(1.5, 5_050_000, 5050, 10)) == 150


def test_forward_rate_is_the_derivative_of_T_times_zero_rate():
    c, T0, h = ZeroCurve(), 4.0, 1e-6
    num = ((T0 + h) * c.zero(T0 + h) - (T0 - h) * c.zero(T0 - h)) / (2 * h)
    assert abs(num - c.forward(T0)) < 1e-7


def test_forward_slope_matches_numerical_derivative():
    c, T0, h = ZeroCurve(), 4.0, 1e-5
    num = (c.forward(T0 + h) - c.forward(T0 - h)) / (2 * h)
    assert abs(num - c.forward_slope(T0)) < 1e-8


def test_hull_white_formula_gives_back_the_curve_at_time_zero():
    curve = ZeroCurve()
    hw = HullWhite(curve)
    assert abs(hw.bond_price(0.0, 5.0, curve.zero(0.0)) - curve.discount(5.0)) < 1e-12


def test_hull_white_simulation_reprices_the_curve():
    curve = ZeroCurve()
    _, _, disc = HullWhite(curve).simulate(T=5.0, n_steps=260, n_paths=20000)
    assert abs(disc[:, -1].mean() / curve.discount(5.0) - 1) < 0.003
