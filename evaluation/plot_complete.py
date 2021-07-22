import matplotlib.pyplot as plt
import numpy as np

x_vals = [r"$128^3$", r"$128^2 \times 256$", r"$128 \times 256^2$", r"$256^3$", r"$256^2 \times 512$", r"$256 \times 512^2$", r"$512^3$", r"$512^2 \times 1024$", r"$512 \times 1024^2$"]

# pcsgs
y_vals = [
[29.2770, 59.0838, 116.0176, 301.6503, 455.4458, 906.2078, 1826.2120, 3596.7229, 7572.4816],
[36.9244, 60.1292, 116.5844, 232.8071, 453.1416, 923.1547, 1803.3436, 3616.2099, 7548.5975],
[29.8976, 59.2610, 115.1021, 228.2983, 462.4344, 904.4720, 1821.5728, 3595.4920, 7464.9903],
[30.4229, 59.3988, 117.0058, 234.8397, 459.2503, 918.3916, 1825.8398, 3703.7720, 7523.9209],
[38.4350, 75.3342, 150.6604, 298.1847, 593.0137, 1187.1930, 2578.1477, 4916.6308, 9829.3767]]

# pcsgs cuda_aware
# y_vals = [
# [29.7602, 59.0332, 146.7097, 227.9518, 449.9262, 894.1393, 1774.6199, 3596.5810, 7216.4814],
# [31.4960, 59.0856, 114.8241, 228.3552, 445.3578, 884.7643, 1783.4624, 3688.8935, 7177.6466],
# [69.7374, 79.8173, 174.9031, 350.0543, 468.3927, 907.2840, 1791.4662, 3691.7335, 7341.7471],
# [29.9609, 57.5422, 111.3434, 225.7484, 445.4755, 976.5535, 1789.1993, 3613.0918, 7427.7910],
# [40.1798, 85.6534, 152.0414, 298.1875, 593.7068, 1180.9268, 2341.9986, 4910.7784, 9684.9097],
# [41.3714, 90.1146, 168.3532, 309.0739, 605.1804, 1185.7997, 2325.0477, 4628.0387, 9434.8847]]

# krypton
# y_vals = [
# [2.6104, 2.8496, 5.3392, 9.4160, 17.7620, 34.8284, 69.7371, 136.5089, 271.5267, 590.1514175],
# [2.2298, 3.1817, 5.8016, 11.0513, 21.8313, 44.0290, 86.5286, 155.2354, 305.0805, 656.58155],
# [1.4494, 2.6333, 5.4334, 10.9788, 23.9823, 46.6907, 90.3210, 180.0794, 398.3437, 749.6851175],
# [1.6598, 3.2467, 5.5026, 11.1406, 23.4140, 47.4440, 95.7999, 185.2786, 385.9134, 768.8481],
# [2.0890, 3.5741, 6.3829, 12.4129, 23.4707, 44.7869, 97.7494, 172.8221, 351.8331, 805.8163],
# [2.3780, 3.3779, 6.2975, 11.9340, 23.2622, 46.5644, 89.4847, 176.8941, 364.2885, 759.9095]]

# krypton cuda_aware
# y_vals = [
# [0.3959, 0.6148, 1.1284, 1.9599, 3.8229, 7.6069, 15.1816, 29.7223, 58.9517, 157.5153975],
# [0.3696, 0.6517, 1.3555, 3.1947, 7.1253, 15.7236, 30.1591, 47.7335, 99.1228, 228.568125],
# [0.3953, 0.7326, 1.4746, 3.1498, 7.4521, 16.4230, 32.8572, 68.0527, 155.8264, 302.40018],
# [0.3427, 0.6831, 1.3874, 3.0574, 7.0121, 15.5847, 32.2413, 70.0283, 153.2492, 307.5333025],
# [1.0204, 1.3587, 1.9262, 3.6406, 5.2576, 8.6689, 24.2949, 34.3805, 78.1835, 261.3697],
# [0.4211, 0.6782, 1.2039, 2.2691, 4.3046, 8.4441, 16.7861, 36.8559, 88.9093, 216.5879]]

title = "Best Selection [PCSGS, cuda_aware]"
legend = ["Slab", "Slab (Opt1)", "Slab Z_Then_YX", "Slab Z_Then_YX (Opt1)", "Pencil", "Pencil (Opt1)"]

labels = []
for y in y_vals:
    label, = plt.plot(x_vals, y, "D-", zorder=3, linewidth=3, markersize=10)
    labels.append(label)


plt.title(title, fontsize=22)
plt.xlabel("Size", fontsize=20)
plt.ylabel("Time [ms]", fontsize=20)
plt.legend(labels, legend, prop={"size":16})
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(zorder=0, color="grey")
plt.yscale('log', base=10)
plt.show()
plt.close()

title = "Best Selection; Difference to Slab [PCSGS, cuda_aware]"
labels = []
for y in y_vals:
    label, = plt.plot(x_vals, [y_vals[0][i]-y[i] for i in range(0, len(y))], "D-", zorder=3, linewidth=3, markersize=10)
    labels.append(label)


plt.title(title, fontsize=22)
plt.xlabel("Size", fontsize=20)
plt.ylabel("Time [ms]", fontsize=20)
plt.legend(labels, legend, prop={"size":16})
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(zorder=0, color="grey")
plt.yscale('symlog', base=10)
plt.show()