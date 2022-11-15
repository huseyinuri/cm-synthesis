import numpy as np
import numpy.polynomial.polynomial as P
from functools import reduce

to_real = lambda x : x / 1j
to_complex = lambda x : x * 1j

class CharPoly:
    def __init__(self, order, return_loss,  finite_tzs = None) -> None:
        self.order = order
        self.return_loss = return_loss
        self.transmission_zeros = finite_tzs
        self.polynomials = []
                             
    @property
    def transmission_zeros(self):
        return self._transmission_zeros

    @transmission_zeros.setter
    def transmission_zeros(self,values):
        if values is None: #empty list
            self._transmission_zeros = [np.Inf] * self.order
        else:
            if not all(isinstance(value, complex) for value in values):
                raise TypeError('All finite transmission zeros must be given as'
                                                                      'complex')
            if len(values) > self.order - 1:
                    raise ValueError('Number of prescribed transmission zeros' 
                                    '[{}] cannot exceed the {}'.format(len(values,self.order -1)))
            
            self._transmission_zeros = sorted(list(map(to_real,values)), key=abs) + [np.Inf] * (self.order - len(values))

    def _calc_P(self):
        P_factorized = [[1, -1/tz] for tz in self.transmission_zeros]
        P_w = reduce(lambda x,y : P.polymul(x, y), P_factorized)
        P_roots = P.polyroots(P_w)
        P_w = P.polyfromroots(P_roots)
        self.polynomials.append(P_w)
        return P_roots, P_w
    
    def _calc_P(self):
        pass


if __name__ == '__main__':
    c = CharPoly(4,5,[1.3217j,1.8082j])
    x,Y = c._calc_P()
    print('hello')





    def calc_poly(self):

        pass

    def calc_roots(self):
        pass

        

    