from dataclasses import replace

import numpy as np

from structured_products import (
    NOMINAL, Autocall, CapitalProtectedNote, Market, ReverseConvertible,
    autocall_probabilities, bs_call, fair_coupon, fair_participation, greeks,
    normals, price,
)

M = Market()
N = 100_000


def test_capital_protected_note_matches_closed_form():
    cpn = CapitalProtectedNote(participation=0.7, protection=1.0, maturity=5)
    res = price(cpn, M, normals(N, 5))
    exact = NOMINAL * (np.exp(-M.r * 5) + 0.7 * bs_call(1.0, 1.0, 5, M.r, M.q, M.sigma))
    assert abs(res.pv - exact) < 4 * res.se


def test_fair_participation_reprices_to_target():
    part = fair_participation(5, M, margin=0.015)
    res = price(CapitalProtectedNote(part, 1.0, 5), M, normals(N, 5))
    assert abs(res.pv - 98.5) < 4 * res.se


def test_reverse_convertible_without_coupon_or_barrier_is_a_zero_coupon():
    rc = ReverseConvertible(coupon=0.0, barrier=0.0, maturity=3)
    assert abs(price(rc, M, normals(N, 3)).pv - NOMINAL * np.exp(-M.r * 3)) < 1e-9


def test_athena_is_called_on_first_date_when_barrier_is_zero():
    ac = Autocall(coupon=0.07, maturity=5, autocall_barrier=0.0)
    res = price(ac, M, normals(N, 5))
    assert (res.called_at == 0).all()
    assert abs(res.pv - NOMINAL * 1.07 * np.exp(-M.r)) < 1e-9


def test_price_is_linear_in_the_coupon():
    z = normals(N, 5)
    ac = Autocall(coupon=0.0, coupon_barrier=0.70)
    p0, p5, p10 = (price(replace(ac, coupon=c), M, z).pv for c in (0.0, 0.05, 0.10))
    assert abs(p5 - (p0 + p10) / 2) < 1e-8


def test_fair_coupon_reprices_to_target():
    z = normals(N, 5)
    ac = Autocall(coupon=0.0)
    c = fair_coupon(ac, M, z, margin=0.015)
    assert abs(price(replace(ac, coupon=c), M, z).pv - 98.5) < 1e-6


def test_fair_coupon_increases_with_volatility():
    z = normals(N, 5)
    ac = Autocall(coupon=0.0)
    low = fair_coupon(ac, replace(M, sigma=0.15), z)
    high = fair_coupon(ac, replace(M, sigma=0.25), z)
    assert high > low


def test_phoenix_memory_is_worth_more_than_no_memory():
    z = normals(N, 5)
    base = Autocall(coupon=0.08, coupon_barrier=0.70, memory=True)
    assert price(base, M, z).pv > price(replace(base, memory=False), M, z).pv


def test_autocall_probabilities_sum_to_one():
    res = price(Autocall(coupon=0.07), M, normals(N, 5))
    assert abs(autocall_probabilities(res.called_at, 5).sum() - 1.0) < 1e-12


def test_reverse_convertible_has_positive_delta_and_negative_vega():
    z = normals(N, 3)
    g = greeks(ReverseConvertible(coupon=0.05), M, z)
    assert g["delta"] > 0 and g["vega"] < 0
