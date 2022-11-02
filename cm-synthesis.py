import argparse
import numpy as np
from numpy.polynomial import polynomial
from typing import List

N=3
tzs=[3.37,np.Inf,np.Inf]


def _populate_tzs(order: int, finite_tzs: List[float] | None = None) -> List[float]:
    tzs = [np.Inf]*order

    if finite_tzs is None:
        return tzs
    if len(finite_tzs)>order:
        raise ValueError('Number of finite tz cannot exceed the filter order')
    else:
        tzs[:len(finite_tzs)] = sorted(finite_tzs)
    return tzs

def _calc_fbw(f0: int | float, bw: int | float) -> float:
    try:
        return bw/f0
    except ZeroDivisionError:
        return 0



def _calc_denum(tzs: List[float]):
    denum_coeffs=[1]
    for tz in tzs:
        denum_coeffs = polynomial.polymul(denum_coeffs,[1,-1/tz])
    return denum_coeffs


def _calc_num(order: int, tzs: List[float]):
    
    num_coeffs = order + 1
    init_0 = [1]
    init_1 = [-1/tzs[0],1]

    if num_coeffs < 0:
        raise ValueError("i must be >=0")
    p0=init_0
    if num_coeffs==0:
        return p0
    p1=init_1
   

    for j in range(2,num_coeffs):
        A=((1-1/(tzs[j-1]**2)) / (1-1/(tzs[j-2]**2)))**0.5

        pnext_1=polynomial.polymul(p1,[-1/tzs[j-1],1])
        pnext_2=polynomial.polymul(p1,[-1/tzs[j-2],1])
        pnext_3=polynomial.polymul(p0,polynomial.polypow([1, -1/tzs[j-2]],2))

        temp=polynomial.polyadd(pnext_1,A*pnext_2)
        pnext=polynomial.polysub(temp,A*pnext_3)

        p0=p1
        p1=pnext
    return p1


def main():
    Pn=_calc_num(N,tzs)
    Dn=_calc_denum(tzs)
    print(Pn)
    print(Dn)

"""
if __name__ == '__main__':
    exit(main())
"""