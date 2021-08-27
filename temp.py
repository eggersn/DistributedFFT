import numpy as np

b1 = 68*(10**9)
l1 = 1.8*(10**(-5))

b2 = 25*(10**9)
l2 = 2*(10**(-5))

N = 8*(256**3)

for p in range(2, 10):
    P = p**2

    slab = N/(P*P*b1) + 2*N/(P*P*b2) + l2 
    
    pencil = N/(b1*(p**3)) + N/(b1*(p**3)) + l1 + l2 

    print(slab, pencil)
