import numpy as np
from numpy.polynomial import polynomial
tzs=[3.37,np.Inf,np.Inf]

prod=[1]
for tz in tzs:
    prod = polynomial.polymul(prod,[1,-1/tz])
Dn=prod

def calc_coeff(i,init_0,init_1):
    

    if i < 0:
        raise ValueError("i must be >=0")
    p0=init_0
    if i==0:
        return p0
    p1=init_1
   

    for j in range(2,i):
        A=((1-1/(tzs[j-1]**2)) / (1-1/(tzs[j-2]**2)))**0.5
        pnext_1=polynomial.polymul(p1,[-1/tzs[j-1],1])
        pnext_2=polynomial.polymul(p1,[-1/tzs[j-2],1])
        pnext_3=polynomial.polymul(p0,polynomial.polypow([1, -1/tzs[j-2]],2))


        temp=polynomial.polyadd(pnext_1,A*pnext_2)
        pnext=polynomial.polysub(temp,A*pnext_3)

        p0=p1
        p1=pnext
        
    return p1
init_0 = [1]
init_1 = [-1/tzs[0], 1]

Pn=calc_coeff(4,init_0,init_1)
print(Pn)
print(Dn)