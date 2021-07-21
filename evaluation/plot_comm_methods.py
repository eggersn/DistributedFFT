import matplotlib.pyplot as plt
import numpy as np

x_vals = [r"$128^3$", r"$128^2 \times 256$", r"$128 \times 256^2$", r"$256^3$", r"$256^2 \times 512$", r"$256 \times 512^2$", r"$512^3$", r"$512^2 \times 1024$", r"$512 \times 1024^2$"]

# pcsgs slab decomposition (z_then_yx)
y_vals = [
[29.8976, 59.2610, 115.1021, 228.2983, 462.4344, 904.4720, 1821.5728, 3595.4920, 7464.9903],
[37.0801, 65.2882, 120.2887, 224.0085, 458.5023, 913.5096, 2233.5597, 3630.1114, 10022.2591],
[29.9981, 58.5645, 136.0473, 229.6296, 464.0806, 922.4286, 1812.0803, 3649.5250, 7693.7304],
[30.3094, 58.0269, 120.3529, 235.4092, 470.7516, 1072.5999, 1815.4222, 3639.5119, 7383.6306],
[30.7418, 60.8607, 123.3428, 233.4447, 593.9605, 1015.0366, 2001.6057, 3765.4497, 7715.6044]]

# pcsgs cuda_aware
# y_vals = [
# [29.7602, 59.0332, 146.7097, 227.9518, 449.9262, 894.1393, 1774.6199, 3596.5810, 7216.4814],
# [31.4960, 59.0856, 114.8241, 228.3552, 445.3578, 884.7643, 1783.4624, 3688.8935, 7177.6466],
# [69.7374, 79.8173, 174.9031, 350.0543, 468.3927, 907.2840, 1791.4662, 3691.7335, 7341.7471],
# [29.9609, 57.5422, 111.3434, 225.7484, 445.4755, 976.5535, 1789.1993, 3613.0918, 7427.7910]]

title = "Communication Methods [PCSGS, Slab (Z_Then_YX)]"
legend = ["Peer2Peer (Sync)", "Peer2Peer (Streams)", "Peer2Peer (MPI_Type)", "All2All (Sync)", "All2All (MPI_Type)"]

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

title = "Best Selection; Difference to Slab [PCSGS]"
labels = []
for y in y_vals:
    label, = plt.plot(x_vals, [y_vals[0][i] - y[i] for i in range(0, len(y))], "D-", zorder=3, linewidth=3, markersize=10)
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