from __future__ import annotations

from math import sqrt
from typing import List
from typing import Tuple
import argparse

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from functools import wraps
from .colors import FgColors, BgColors
from collections import defaultdict
from itertools import zip_longest
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
#order = 4
f0 = 3700
bw = 80
#rl = 20
omega = np.arange(-4, 4, 0.001)
#ftzs_w = np.array([1.3217, 1.8082])


def _populate_tzs(order: int,
                    ftzs: List[float] | None = None):
    tzs = np.ones(order) * np.Inf

    if ftzs is None:
        return tzs
    if len(ftzs) > order:
        raise ValueError('Number of finite tz cannot exceed the filter order')
    else:
        tzs[:len(ftzs)] = sorted(ftzs)
    return tzs


def _calc_fbw(f0: int | float, bw: int | float) -> float:
    try:
        return bw/f0
    except ZeroDivisionError:
        return 0


def _calc_eps(order: int, ftzs: List[float],  rl: float, f_s, p_s) -> Tuple[float, float]:

    if len(ftzs) < order:
        e_r = 1
        e = (1/sqrt(10**(rl/10)-1)) * abs(polyval(1j, p_s) * e_r /
                                          polyval(1j, f_s))
    else:
        pass
    return e, e_r


def _calc_Es(e, e_r, f_s, p_s):
    # Alternating singularity method is impelemented
    # For details please check Cameron's book p.188
    e_e_conj = polyadd(e_r * p_s, e * f_s)
    alt_roots = polyroots(e_e_conj)
    alt_roots.real *= np.where(alt_roots.real > 0, -1, 1)
    e_s = polyfromroots(alt_roots)
    return e_s


def _calc_FsPs(order: int, ftzs: List[float], tzs: List[float]):
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
    return _make_monic(order, ftzs, f_w, p_w)


def _make_monic(order, ftzs, f_w, p_w):
    # Multiplying by 1j maps roots from real frequncy to complex plane
    # For more details please check Cameron's book p.182
    f_s = polyfromroots(polyroots(f_w)*1j)
    if (order - len(ftzs)) % 2:
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
    ax2.set_xlim([0.3, 1])
    ax2.set_ylim([-0.05, 0.01])
    ax2.plot(omega, s21, label='S11')
    ax2.set_xticks([])
    mark_inset(ax, ax2, loc1=1, loc2=3, fc='none', ec='0.5')
    plt.savefig('plot.png')
    #plt.show()


def _setup_params(params):
    
    data_list = {k:v.tolist() if isinstance(v,np.ndarray) else [v] for k,v in params.items()}
    data_str={}
    for k,v in data_list.items():
        v_list=[]
        for i in v:
            if isinstance(i, complex) or isinstance(i, float): 
                v_list.append(f"{i:.4f}")
            else:
                v_list.append(f"{i}")
        data_str[k]=v_list
    
    return data_str



def tabulated(fg, bg):
    def dec(func):
        table = []
        headers = ['TZs', 'F(S)', 'P(s)', 'E(s)', 'N', 'RL', 'e', 'e_r']
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            result=_setup_params(result)
            result = [list(x) for x in list(zip_longest(*result.values(),fillvalue='-'))]
            print(f'\033[{fg};{bg};1m{tabulate(result,headers=headers,tablefmt="grid")}\033[0m')
            
        return wrapper
    return dec



@tabulated(FgColors.BLACK,BgColors.YELLOW)
def cli(args: argparse.Namespace):
    fbw = _calc_fbw(f0, bw)
    tzs = _populate_tzs(args.order, args.zeros)
    f_s, p_s = _calc_FsPs(args.order, args.zeros, tzs)
    e, e_r = _calc_eps(args.order, args.zeros, args.return_loss, f_s, p_s)
    e_s = _calc_Es(e, e_r, f_s, p_s)

    plot_S11_S21(f_s, p_s, e_s, e, e_r)

    params = {'tzs':tzs,'fs':f_s, 'ps':p_s, 'es':e_s,
             'n':args.order, 'rl':args.return_loss, 'e':e, 'er':e_r}

    return params

    """    print(f'Order --> {args.order}')
    print(f'Fractional bandwidth --> {fbw}')
    print(f'TZs in real frequency --> {tzs}')
    print(f'Monic F(s) --> {f_s}')
    print(f'Monic P(s) --> {p_s}')
    print(f'Monic E(s) --> {e_s}')
    print(f'e --> {e}')
    print(f'e_r --> {e_r}')

    
    """


if __name__ == '__main__':
    cli()
