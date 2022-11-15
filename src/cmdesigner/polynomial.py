import numpy as np
import numpy.polynomial.polynomial as P
from functools import reduce
from typing import List, Tuple
from functools import cache,wraps

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


class CharPoly:
    def __init__(self, order, return_loss,  finite_tzs = None) -> None:
        self.order = order
        self.return_loss = return_loss
        self.transmission_zeros = finite_tzs
        self.finite_tzs_len = 0 if finite_tzs is None else len(finite_tzs)
    
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
            if len(values) > self.order - 1:
                raise ValueError('Number of prescribed transmission zeros cannot exceed the order')
            self._transmission_zeros = sorted(values, key=abs) + [np.Inf] * (self.order - len(values))
    
    @property
    @monic_complex
    def s21_num(self):
        return self._compute_s21_num()
        
    @cache
    def _compute_s21_num(self):
        print("...cached...")
        real_zeros = list(map(to_real, self.transmission_zeros))
        temp = [[1, -1/tz] for tz in real_zeros]
        coeffs = reduce(lambda x,y : P.polymul(x, y), temp)
        roots = P.polyroots(coeffs)
        return roots, coeffs
    
if __name__ == '__main__':

    c=CharPoly(4,2,[1.3217j, 1.8082j])
    print(f'{c.s21_num}')



    """
    def _calc_F(self):
        first_zero = self.transmission_zeros[0]
        F_0 = [1]
        F_1 = [-1/first_zero, 1]

        for i in range(2, self.order + 1):
            A = ((1-1/(self.transmission_zeros[i-1] ** 2)) / 
                (1-1/(self.transmission_zeros[i-2] ** 2))) ** 0.5
            
            Fnext_1 = P.polymul(F_0, P.polypow([1, -1/self.transmission_zeros[i-2]], 2))
            temp = A * Fnext_1
            Fnext_2 = P.polymul(F_1, [-1/self.transmission_zeros[i-1], 1])
            Fnext_3 = P.polymul(F_1, [-1/self.transmission_zeros[i-2], 1])
            Fnext = P.polysub(P.polyadd(Fnext_2, A * Fnext_3), temp)
            F_0 = F_1
            F_1 = Fnext
        F_roots = P.polyfromroots(F_1)
        F_w = P.polyfromroots(F_roots)

    """



        

    