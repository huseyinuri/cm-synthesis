import numpy as np
import math
import numpy.polynomial.polynomial as P
from functools import reduce
from functools import cache,wraps
from tabulate import tabulate
from cmdesigner.utils import splot 

from itertools import zip_longest

to_real = lambda x : x / 1j if x != np.Inf else x

def monic_complex(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        roots, coeffs = func(self, *args, **kwargs)
        if func.__name__ == 's21_num':
            if not (self.order - self.finite_tzs_len) % 2:
                coeffs = P.polyfromroots(P.polyroots(coeffs)*1j)*1j
                return roots.tolist(),coeffs.tolist()
        coeffs = P.polyfromroots(P.polyroots(coeffs)*1j)
        return roots.tolist(),coeffs.tolist()
    return wrapper

def monic(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        roots, coeffs = func(self, *args, **kwargs)
        coeffs = P.polyfromroots(roots)
        return roots.tolist(), coeffs.tolist()
    return wrapper

class CharPoly:
    def __init__(self, order, return_loss,  finite_tzs = None) -> None:
        self.order = order
        self.return_loss = return_loss
        self.transmission_zeros = finite_tzs
        self.finite_tzs_len = 0 if finite_tzs is None else len(finite_tzs)

    def __str__(self) -> str:
        data = []
        headers = ['i', 'Ref. zero (wp_i)', 'F(s)', 'Trans. zero (wz_i)', 'P(s)', 'Trans./Ref. poles','E(s)', 'e_r', 'e']
        data.append(headers)

        indices = [str(i) for i in range(self.order + 1)]
        f_s_roots, f_s = self.s11_num
        _f_s_roots = [f"{i:.4f}" for i in f_s_roots]
        _f_s = [f"{i:.4f}" for i in f_s]
        p_s_roots, p_s = self.s21_num
        _p_s_roots = [f"{i:.4f}" for i in p_s_roots]
        _p_s = [f"{i:.4f}" for i in p_s]
        e_s_roots, e_s = self.denum
        _e_s_roots = [f"{i:.4f}" for i in e_s_roots]
        _e_s = [f"{i:.4f}" for i in e_s]
        e_r, e = self.ripple_factors
        _e_r = [f"{e_r:.6f}"]
        _e = [f"{e:.6f}"]

        for row in zip_longest(indices,_f_s_roots,_f_s,_p_s_roots,_p_s,_e_s_roots,_e_s,_e_r,_e,fillvalue='-'):
            data.append(list(row))
        return tabulate(data, headers = 'firstrow', tablefmt = 'outline')
    
    @property
    def transmission_zeros(self):
        return self._transmission_zeros

    @transmission_zeros.setter
    def transmission_zeros(self,values):
        if values is None: #empty list
            self._transmission_zeros = [np.Inf] * self.order
        else:
            if not all(isinstance(value, complex) for value in values):
                raise TypeError('All finite transmission zeros must be given as complex')
            if len(values) > self.order:
                raise ValueError('Number of prescribed zeros cannot exceed the order')
            self._transmission_zeros = sorted(values, key=abs) + [np.Inf] * (self.order - len(values))
    
    @property
    def ripple_factors(self):
        return self._compute_ripple_factors()
    
    @property
    @monic_complex
    def s21_num(self):
        return self._compute_s21_num()
    
    @property
    @monic_complex
    def s11_num(self):
        return self._compute_s11_num()
    
    @property
    @monic
    def denum(self):
        return self._compute_denum()
    
    # Alternating singularity method for real 
    # reflection zeros. 
    # Cameron's book p.188
    @cache
    def _compute_denum(self):
        e_r, e = self.ripple_factors
        _, F_s = self.s11_num
        _, P_s = self.s21_num
        # check any complex reflection zeros exist
        """
        if np.count_nonzero(np.isreal(F_s_roots)) < len(F_s_roots):
            raise NotImplementedError('Conservation of energy not implemented yet')
        """
        p_term = [i * e_r for i in list(P_s)]
        f_term = [i * e for i in list(F_s)]
        E_E_conj = P.polyadd(p_term, f_term)
        roots = P.polyroots(E_E_conj)
        roots.real *= np.where(roots.real > 0, -1, 1)
        coeffs = P.polyfromroots(roots)
        return roots, coeffs

    @cache
    def _compute_ripple_factors(self):
        _, P_s = self.s21_num
        _, F_s = self.s11_num
        if self.finite_tzs_len < self.order:
            e_r = 1
            e = (1/math.sqrt(10**(self.return_loss/10)-1)) * abs(P.polyval(1j, P_s) * e_r /
                                          P.polyval(1j, F_s))
            return e_r, e
        elif self.finite_tzs_len == self.order:
            e = (1/math.sqrt(10**(self.return_loss/10)-1)) * abs(P.polyval(1j, P_s)  /
                                          P.polyval(1j, F_s))
            e_r = e / math.sqrt((e**2)-1)
            return e_r, e
        else:
            raise ValueError('Number of prescribed zero cannot exceed the filter order')

    @cache
    def _compute_s21_num(self):
        real_zeros = list(map(to_real, self.transmission_zeros))
        temp = [[1, -1/tz] for tz in real_zeros]
        coeffs = reduce(lambda x,y : P.polymul(x, y), temp)
        roots = P.polyroots(coeffs)
        return roots, coeffs
    
    @cache
    def _compute_s11_num(self):
        real_zeros = list(map(to_real, self.transmission_zeros))
        first_zero = real_zeros[0]
        F_0 = [1]
        F_1 = [-1/first_zero, 1]

        for i in range(2, self.order + 1):
            A = ((1-1/(real_zeros[i-1] ** 2)) / 
                (1-1/(real_zeros[i-2] ** 2))) ** 0.5
            
            Fnext_1 = P.polymul(F_0, P.polypow([1, -1/real_zeros[i-2]], 2))
            temp = A * Fnext_1
            Fnext_2 = P.polymul(F_1, [-1/real_zeros[i-1], 1])
            Fnext_3 = P.polymul(F_1, [-1/real_zeros[i-2], 1])
            Fnext = P.polysub(P.polyadd(Fnext_2, A * Fnext_3), temp)
            F_0 = F_1
            F_1 = Fnext
        roots = P.polyroots(F_1)
        return roots, F_1

class AdmitPoly:
    def __init__(self,char_poly):
        self.char_poly = char_poly
        if self.char_poly.finite_tzs_len == self.char_poly.order:
            self.Kinf = self._compute_Kinf()
        else:
            self.Kinf = 0

    def _compute_Kinf(self):
        e_r, e = self.char_poly.ripple_factors
        return (e / e_r) * (e_r - 1)

    def _compute_m_n(self):
        e_r, e = self.char_poly.ripple_factors
        _, f_coeffs = self.char_poly.s11_num
        f_coeffs /= e_r
        _, e_coeffs = self.char_poly.denum
        temp = np.add(f_coeffs, e_coeffs)
        
        m_coeffs = np.zeros(temp.shape[0], dtype=np.complex128)
        n_coeffs = np.zeros(temp.shape[0], dtype=np.complex128)

        m_coeffs[::2] = temp[::2].real
        m_coeffs[1::2] = (temp[1::2].imag)*1j

        n_coeffs[::2] = (temp[::2].imag)*1j
        n_coeffs[1::2] = temp[1::2].real

        return m_coeffs,n_coeffs


    @property
    def denum(self):
        if self.char_poly.order % 2 : 
            _, coeffs = self._compute_m_n()    # denum is "n" if N is odd
            
        else:
            coeffs, _= self._compute_m_n()    # denum is "m" if N is even
            
        return coeffs/coeffs[-1].real

    @property
    def y22_num(self):
        m_coeffs, n_coeffs = self._compute_m_n()
        if self.char_poly.order % 2 :
            return m_coeffs/n_coeffs[-1].real
        else:
            return n_coeffs/m_coeffs[-1].real
            
    @property
    def y21_num(self):
        m_coeffs, n_coeffs = self._compute_m_n()
        kinf = self.Kinf
        denum = self.denum
        _, e = self.char_poly.ripple_factors
        _, P_s = self.char_poly.s21_num
        if self.char_poly.order % 2 :
            temp = (P_s/e)/(n_coeffs[-1].real)
        else:
            temp = (P_s/e)/(m_coeffs[-1].real)
        print("temp")
        print(temp)
        return temp - 1j*kinf*denum


def cli(args):
    c=CharPoly(args.order, args.return_loss, args.zeros)
    print(c)
    a=AdmitPoly(c)
    print("denum")
    print(a.denum)
    print("y22 num")
    print(a.y22_num)
    print("y21 num")
    print(a.y21_num)

    x_lims = np.arange(-4, 4, 0.001)
    splot(c,x_lims=x_lims)         

"""
if __name__ == '__main__':
    c=CharPoly(7,23,[0.9218-0.1546j, -0.9218-0.1546j, 1.2576j])
    print(c)
    x_lims = np.arange(-4, 4, 0.001)
    splot(c,x_lims=x_lims)         
"""

    