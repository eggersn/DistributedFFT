import numpy as np
# L = 0
# W = 3**9

# P = range(1, 500)
# N = [2**k for k in range(5, 14)]

# for p in P:
#     for n in N:
#         duration_slab = 1-1/p**2
#         duration_pencil = 2*(1-1/p)
#         print("{}, {}: {} {}".format(p, n, duration_slab, duration_pencil))

for p1 in range(2, 29):
    for p2 in range(2, 29):
        if p1 * p2 <= 56:
            print("({}, {})".format(p1, p2))