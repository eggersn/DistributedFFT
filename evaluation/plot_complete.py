from os import listdir
from os.path import isfile, join
import os
import pathlib
import re
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

prefix = "evaluation/benchmarks/bwunicluster/gpu8/small/forward"
title = "BwUniCluster GPU8 Forward"
P = 16

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

def collect(cuda_aware):
    subdirs = ["pencil/exact", "slab_default", "slab_z_then_yx"]

    sizes = []

    plt.title("Comparison " + title + " [P={}{}]".format(P, ", CUDA-aware" if cuda_aware else ""), fontsize=20)
    plt.grid(zorder=0, color="grey")
    plt.yscale('symlog', base=10)
    plt.ylabel("Time [ms]", fontsize=16)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14, rotation=30)

    labels = []; legend = []

    x_vals_collection = []
    values_collection = []

    count = 0
    for subdir in subdirs:
        if os.path.isdir(join(prefix, join(subdir, "runs"))):
            files = []
            offset = 0

            if subdir == subdirs[0]:
                files = [f for f in listdir(join(prefix, join(subdir, "runs"))) if int(f.split("_")[2])*int(f.split("_")[3]) == P and (f.split("_")[-1]=="1.csv")==cuda_aware]
                offset = 4
            else:
                files = [f for f in listdir(join(prefix, join(subdir, "runs"))) if int(f.split("_")[2]) == P and (f.split("_")[-1]=="1.csv")==cuda_aware]
                offset = 2

            files.sort()
            for f in files:
                file = join(prefix, subdir, "runs", f)
                with open(file) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',')
                    it_sizes = next(csv_reader)[offset:]
                    if len(it_sizes) > len(sizes):
                            sizes = it_sizes

                    x_vals = ConvertSizesToLabels(it_sizes)
                    values = [-1 for s in it_sizes]
                    row = next(csv_reader)
                    while row != []:
                        runs = [float(x) for x in row[offset:]]
                        for i in range(len(runs)):
                            if values[i] == -1 or runs[i] < values[i]:
                                values[i] = runs[i]

                        row = next(csv_reader)

                    x_vals_collection.append(x_vals)
                    values_collection.append(values)

                    label, = plt.plot(x_vals, values, zorder=3, linewidth=3, markersize=10, marker=markers[count%len(markers)], linestyle=linestyles[count%len(linestyles)])
                    count += 1
                    labels.append(label)
                    if subdir == subdirs[0]:
                        legend.append("Pencil [{}, {}x{}]".format("default" if f.split("_")[1] == "0" else "opt1", f.split("_")[2], f.split("_")[3]))
                    else:
                        legend.append("Slab [{}, {}]".format("ZY_Then_X" if subdir == subdirs[1] else "Z_Then_YX", "default" if f.split("_")[1] == "0" else "opt1"))
                    print(legend[-1], values)

    plt.legend(labels, legend, prop={"size":16})
    fig = plt.gcf()
    fig.set_size_inches(15, 8)
    plt.savefig(prefix+"/comparison_{}_{}".format(P, 1 if cuda_aware else 0), dpi=100)
    plt.close()

    plt.title("Proportions " + title + " [P={}{}]".format(P, ", CUDA-aware" if cuda_aware else ""), fontsize=20)
    plt.grid(zorder=0, color="grey")
    plt.ylabel("Proportion", fontsize=16)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14, rotation=30)

    count = 0
    for i in range(len(values_collection)):
        plt.plot(x_vals_collection[i], [values_collection[i][j]/min([values[j] for values in values_collection if j < len(values)]) for j in range(len(values_collection[i]))], zorder=3, linewidth=3, markersize=10, marker=markers[count%len(markers)], linestyle=linestyles[count%len(linestyles)])
        count += 1

    plt.legend(labels, legend, prop={"size":16})
    fig = plt.gcf()
    fig.set_size_inches(15, 8)
    plt.savefig(prefix+"/proportions_{}_{}".format(P, 1 if cuda_aware else 0), dpi=100)
    plt.close()

def main():
    for c in range(2):
        collect(c==1)
        print()

if __name__ == "__main__":
    main()


