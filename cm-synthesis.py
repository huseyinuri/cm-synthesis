from __future__ import annotations

from typing import List

import numpy as np
from numpy.polynomial import polynomial

"""
=========================
C: characteristic function
C(w) = F(w) / P(w)
=========================
S11: reflection function
S11(w) = F(w) / E(w)
=========================
S21: transfer function
S21(w) = P(w) / E(w)e
=========================
"""

order = 4
f0 = 3700
bw = 80
ftzs_w = [1.80, 1.32]


def _populate_tzs_w(order: int,
                    ftzs_w: List[float] | None = None) -> List[float]:
    tzs_w = [np.Inf]*order

    if ftzs_w is None:
        return tzs_w
    if len(ftzs_w) > order:
        raise ValueError('Number of finite tz cannot exceed the filter order')
    else:
        tzs_w[:len(ftzs_w)] = sorted(ftzs_w)
    return tzs_w


def _calc_fbw(f0: int | float, bw: int | float) -> float:
    try:
        return bw/f0
    except ZeroDivisionError:
        return 0


def _calc_polys_w(order: int, tzs: List[float]):
    temp = [1]
    for tz in tzs:
        temp = polynomial.polymul(temp, [1, -1/tz])
    p_w = temp

    init_0 = [1]
    init_1 = [-1/tzs[0], 1]
    if order + 1 < 0:
        raise ValueError("i must be >=0")
    f0 = init_0
    if order + 1 == 0:
        return f0
    f1 = init_1
    for j in range(2, order + 1):
        A = ((1-1/(tzs[j-1]**2)) / (1-1/(tzs[j-2]**2)))**0.5
        fnext_1 = polynomial.polymul(f1, [-1/tzs[j-1], 1])
        fnext_2 = polynomial.polymul(f1, [-1/tzs[j-2], 1])
        fnext_3 = polynomial.polymul(f0,
                                     polynomial.polypow([1, -1/tzs[j-2]], 2))
        temp = polynomial.polyadd(fnext_1, A*fnext_2)
        fnext = polynomial.polysub(temp, A*fnext_3)
        f0 = f1
        f1 = fnext
    f_w = f1
    return _make_monic(f_w, p_w)


def _make_monic(f, p):
    return (polynomial.polyfromroots(polynomial.polyroots(f)),
            polynomial.polyfromroots(polynomial.polyroots(p)))


def main():
    print(f'Fractional bandwidth-->{_calc_fbw(f0, bw)}')
    tzs_w = _populate_tzs_w(order, ftzs_w)
    print(f'TZs in real frequency-->{tzs_w}')
    f_w, p_w = _calc_polys_w(order, tzs_w)
    print(f'Monic F(w)-->{f_w}')
    print(f'Monic P(w)-->{p_w}')
