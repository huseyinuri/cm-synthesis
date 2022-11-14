"""
           2
          / \ 
    S---1/---\3---L

        2--3      
        |  |
    S---1--4---L

        2--3      6  
        |  |     / \ 
    S---1--4---5/---\7---L 

        2--3        
        |  |      
    S---1--4--7---L 
           |  |
           5--6

    S----1----2
    |  /    / |
    L/---4/---3
"""

import numpy as np
from scipy.optimize import minimize

fmt = {
    'triplet' : {
        'top' : np.array([0,1,0,0,0,1,1,1,0,1,1,0,1,1,0]),
        'indices' : [7]
    },
    'quadruplet' : {
        'top' :  np.array([0,1,0,0,0,0,1,1,0,1,0,1,1,0,0,1,1,0,1,1,0]),
        'indices' : [9]
    } 
}

def arr2mat(arr, offset = 0):
    n = int(np.sqrt(arr.size*2))
    if (n*(n+1))//2 != arr.size:
        return None
    else:
        mat = np.zeros((n,n))
        mat[np.triu_indices(n, k=offset)] = arr
        mat = mat + mat.T - np.diag(np.diag(mat))
        return mat

class TopologyMatrix:
    def __init__(self,size):
        if size < 0:
            raise ValueError('Size must be positive')
        self._size = size
        self._top = np.zeros((self._size+1)*self._size //2)
        self._c_indices = []

    def __len__(self):
        return self._top.size
    
    def __setitem__(self, position, value):
        index = self._get_index(position)
        self._top[index] = value

    def __getitem__(self, position):
        index = self._get_index(position)
        return self._top[index]
    
    def _get_index(self, position):
        # map (row,col) of triu matrix to (k) linear index of _top array
        row, column = position
        if row > column:
            row, column = column, row
        return self._size * row - ((row - 1) * row) // 2 + (column - row)

    @property
    def top(self):
        if np.all(self._top == 0):
            raise ValueError('Topology must be set before access')
        return self._top, self._c_indices

    @top.setter
    def top(self, val):
        if not isinstance(val, str):
            raise TypeError('Expected a string')
        arr = fmt[val]['top']
        ind = fmt[val]['indices']
        if arr.size != len(self):
            raise ValueError('Size not match')
        self._top = arr[:]
        self._c_indices = ind[:]


class CMOptimizer:
    def __init__(self,optim) -> None:
        self._optim = optim
    
    def _init_x(self, top, dir_val = 0.9, cross_val = 0.1):
        x, cross_indices = top.top
        x = x.astype(np.float64)
        x[np.isin(x, 1)] = dir_val
        x[cross_indices] = cross_val
        return x[np.nonzero(x)], np.flatnonzero(x == 0)

    def optimize(self, top):
        x0, free_indices = self._init_x(top)
        return x0, free_indices

def cost_function(x,top,wz,wp,e):
    R = np.zeros((top._size, top._size))
    R[0, 0] = 1
    R[top._size - 1, top._size - 1] = 1
    W = np.eye()
    

    
if __name__ == '__main__':
    topology = TopologyMatrix(5)
    topology.top = 'triplet'
    t,c = topology.top
    print(f'{arr2mat(t)}')
    optimizer = CMOptimizer('fmin')
    x,f = optimizer.optimize(topology)
    print('hellow')
    

