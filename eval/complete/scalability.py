# Copyright (C) 2021 Simon Egger
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from os import listdir
from os.path import isfile, join, isdir
import os
import pathlib
import re
import csv
import numpy as np
from scipy.stats import t
import matplotlib as mpl
import matplotlib.transforms as bb
# mpl.use("pgf")
import matplotlib.pyplot as plt
# plt.rcParams.update({
#   "text.usetex": True,
#   "pgf.rcfonts": False,
#   "font.size": 16
# })

markers = ["D", "X", "o", "s", "v"]
linestyles = ["solid", "dotted", "dashed", "dashdot"]

def ConvertSizesToLabels(sizes):
    labels = []
    for s in sizes:
        dims = s.split("_")
        if dims[0] == dims[1] and dims[1] == dims[2]:
            labels.append(r"${}^3$".format(dims[0]))
        elif dims[0] == dims[1]:
            labels.append(r"${}^2 \times {}$".format(dims[0], dims[2]))
        elif dims[1] == dims[2]:
            labels.append(r"${} \times {}^2$".format(dims[0], dims[2]))
        else:
            labels.append(r"${} \times {} \times {}$".format(dims[0], dims[1], dims[2]))

    return labels

def collectProcessAndMethod(cuda_aware, opt1, size, P, prefix):
    file = join(prefix, "runs_{}_{}_{}.csv".format(opt1, P, 1 if cuda_aware else 0))
    if not isfile(file):
        return -1
    best = -1
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        method_sizes = next(csv_reader)
        size_index = method_sizes.index(size)

        row = next(csv_reader)
        while row != []:
            val = float(row[size_index])
            if best == -1 or val < best:
                best = val
            row = next(csv_reader)
    return best

def collectProcess(cuda_aware, size, P, pencil_group, prefix):
    subdirs = ["pencil/exact", "slab_default", "slab_z_then_yx"]
    data_collection = {}

    for subdir in subdirs:
        for opt in range(0,2):
            new_prefix = join(prefix, subdir, "runs")
            if isdir(new_prefix):
                if subdir == "pencil/exact":
                    for g in pencil_group:
                        p = "{}_{}".format(int(P/g), g)
                        val = collectProcessAndMethod(cuda_aware, opt, size, p, new_prefix)
                        if val != -1:
                            data_collection["Pencil [{}, P_1x{}]".format("Realigned" if opt==1 else "Default", g)] = val
                else:
                    val = collectProcessAndMethod(cuda_aware, opt, size, P, new_prefix)
                    if val != -1:
                        data_collection["Slab [{}, {}]".format("2D-1D" if subdir == "slab_default" else "1D-2D", "Realigned" if opt==1 else "Default")] = val

    return data_collection

def collect(forward, cuda_aware, size, processes, pencil_group, title, prefix):
    data_collection = {}
    for P in processes:
        new_prefix = join(prefix, "forward" if forward else "inverse")
        data = collectProcess(cuda_aware, size, P, pencil_group, new_prefix)
        for k in data:
            if k not in data_collection:
                data_collection[k] = []
            data_collection[k].append(data[k])

    # get single gpu performance if available
    path = join(prefix, "forward" if forward else "inverse", "single")
    if isdir(path):
        with open(join(path, "runs.csv")) as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            sizes = next(reader)
            size_index = sizes.index(size)
            val = float(next(reader)[size_index])
            for k in data:
                data_collection[k].insert(0, val)
    
    x_vals = [1] + processes

    plt.title("Strong Scaling on {} [{}, {}{}]".format(title, ConvertSizesToLabels([size])[0], "Forward" if forward else "Inverse", ", CUDA-aware" if cuda_aware else ""))
    plt.grid(zorder=0, color="grey")
    plt.yscale('log', base=2)
    plt.xscale('log', base=2)
    plt.ylabel("Time [ms]", fontsize=24)
    plt.yticks(fontsize=22)
    plt.xticks(fontsize=22)

    print(data_collection)
    
    labels = []; legend = []
    count = 0
    for k in data_collection:
        if len(data_collection[k]) == len(x_vals):
            label, = plt.plot(x_vals, data_collection[k], zorder=3, linewidth=5, markersize=12, marker=markers[count%len(markers)], linestyle=linestyles[count%len(linestyles)])
            labels.append(label)
            legend.append(k)
            count += 1

    plt.legend(labels, legend, prop={"size":22})
    plt.show()


def main():
    collection = [
    # ["eval/benchmarks/bwunicluster/gpu8/large", [8, 16, 32, 48, 64], [4,8], "BW-GPU8"],
    ["eval/benchmarks/bwunicluster/gpu4", [4, 8, 16, 24, 32], [2], "BW-GPU4"]
    ]

    for c in collection:
        for forward in range(0,2):
            forward = 1-forward
            for cuda_aware in range(0,2):
                collect(forward==1, cuda_aware==1, "1024_1024_1024", c[1], c[2], c[3], c[0])

if __name__ == "__main__":
    main()