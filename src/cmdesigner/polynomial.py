import numpy as np
import typing as t
class CharPoly:
    def __init__(self, order, return_loss, complex, finite_tzs = None) -> None:
        self.order = order
        self.return_loss = return_loss
        self.complex = complex
        self.transmission_zeros = finite_tzs
        self.results = []

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
                    raise ValueError('Number of prescribed transmission zeros cannot exceed')
            self._transmission_zeros = values + [np.Inf] * (self.order - len(values))    





    def calc_poly(self):

        pass

    def calc_roots(self):
        pass

        

    