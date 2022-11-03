from __future__ import annotations

from math import sqrt
from typing import List
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from numpy.polynomial.polynomial import polyadd
from numpy.polynomial.polynomial import polyfromroots
from numpy.polynomial.polynomial import polymul
from numpy.polynomial.polynomial import polypow
from numpy.polynomial.polynomial import polyroots
from numpy.polynomial.polynomial import polysub
from numpy.polynomial.polynomial import polyval
"""
=========================
C: characteristic function
C(w) = F(w) / P(w)
=========================
S11: reflection function
S11(w) = F(w) / E(w)e_r
=========================
S21: transfer function
S21(w) = P(w) / E(w)e
=========================
"""

order = 4
f0 = 3700
bw = 80
rl = 22
omega = np.arange(-4, 4, 0.001)
ftzs_w = np.array([1.3217, 1.8082])


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


def _calc_eps(order: int, f_s, p_s, rl: float) -> Tuple[float, float]:

    if len(ftzs_w) < order:
        e_r = 1
        e = (1/sqrt(10**(rl/10)-1)) * abs(polyval(1j, p_s) * e_r /
                                          polyval(1j, f_s))
    else:
        pass
    return e, e_r


def _calc_Es(e, e_r, f_s, p_s):
    # Alternating singularity method is impelemented
    # for details please check Cameron's book p.188
    e_e_conj = polyadd(e_r * p_s, e * f_s)
    alt_roots = polyroots(e_e_conj)
    alt_roots.real *= np.where(alt_roots.real > 0, -1, 1)
    e_s = polyfromroots(alt_roots)
    return e_s


def _calc_FsPs(order: int, tzs: List[float]):
    temp = [1]
    for tz in tzs:
        temp = polymul(temp, [1, -1/tz])
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
        fnext_1 = polymul(f1, [-1/tzs[j-1], 1])
        fnext_2 = polymul(f1, [-1/tzs[j-2], 1])
        fnext_3 = polymul(f0, polypow([1, -1/tzs[j-2]], 2))
        temp = polyadd(fnext_1, A*fnext_2)
        fnext = polysub(temp, A*fnext_3)
        f0 = f1
        f1 = fnext
    f_w = f1
    return _make_monic(f_w, p_w)


def _make_monic(f_w, p_w):
    # multiplying by 1j maps roots from real frequncy to complex plane
    # for more details please check Cameron's book p.182
    f_s = polyfromroots(polyroots(f_w)*1j)
    if (order - len(ftzs_w)) % 2:
        p_s = polyfromroots(polyroots(p_w)*1j)
    else:
        p_s = polyfromroots(polyroots(p_w)*1j)*1j
    return f_s, p_s


def plot_S11_S21(f_s, p_s, e_s, e, e_r):
    s11 = 20 * np.log10(abs(polyval(omega*1j, f_s) /
                        (polyval(omega*1j, e_s) * e_r)))
    s21 = 20 * np.log10(abs(polyval(omega*1j, p_s) /
                        (polyval(omega*1j, e_s) * e)))

    _, ax = plt.subplots()
    ax.plot(omega, s11, c='y', label='S11')
    ax.plot(omega, s21, label='S21')
    ax.set_ylim([-50, 1])
    ax.set_xlabel(r'$\omega$')
    ax.set_ylabel(r'|S11| and |S21| (dB)')
    ax.grid(linestyle='--')
    ax.legend()

    ax2 = plt.axes([0, 0, 1, 1])
    ip = InsetPosition(ax, [0.72, 0.6, 0.25, 0.25])
    ax2.set_axes_locator(ip)
    ax2.set_xlim([0.8, 1.1])
    ax2.set_ylim([-1.6, 0.5])
    ax2.plot(omega, s21, label='S11')
    ax2.legend()
    mark_inset(ax, ax2, loc1=1, loc2=3, fc='none', ec='0.5')

    plt.show()


def main():
    fbw = _calc_fbw(f0, bw)
    tzs_w = _populate_tzs_w(order, ftzs_w)
    f_s, p_s = _calc_FsPs(order, tzs_w)
    e, e_r = _calc_eps(order, f_s, p_s, rl)
    e_s = _calc_Es(e, e_r, f_s, p_s)

    print(f'Order --> {order}')
    print(f'Fractional bandwidth --> {fbw}')
    print(f'TZs in real frequency --> {tzs_w}')
    print(f'Monic F(s) --> {f_s}')
    print(f'Monic P(s) --> {p_s}')
    print(f'Monic E(s) --> {e_s}')
    print(f'e --> {e}')
    print(f'e_r --> {e_r}')

    plot_S11_S21(f_s, p_s, e_s, e, e_r)


if __name__ == '__main__':
    main()
