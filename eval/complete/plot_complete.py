from os import listdir
from os.path import isfile, join
import os
import pathlib
import re
import csv
import numpy as np
from scipy.stats import t
import matplotlib as mpl
import matplotlib.pyplot as plt
import pylab

# Config
Table = False

if not Table:
    mpl.use("pgf")
    plt.rcParams.update({
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.size": 30
    })

markers = ["D", "X", "o", "s", "v", "*"]
linestyles = ["solid", "dotted"]

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

def collect(cuda_aware, P, max_labels, title, prefix):
    subdirs = ["slab_default", "slab_z_then_yx", "pencil/approx"]

    sizes = []
    reference = False

    labels = []; legend = []

    x_vals_collection = []
    values_collection = []
    max_xvals = []

    ax = plt.gca()

    count = 0
    slab_count = 0; pencil_count = 0
    
    with open("{}/results_{}.csv".format(prefix, P), mode="a" if cuda_aware else "w") as out_file:
        writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        with open("{}/proportions_{}_{}.csv".format(prefix, P, 1 if cuda_aware else 0), "w") as out_file1:
            writer1 = csv.writer(out_file1, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            with open("{}/abs_proportions_{}_{}.csv".format(prefix, P, 1 if cuda_aware else 0), "w") as out_file2:
                writer2 = csv.writer(out_file2, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([title])
                for subdir in subdirs:
                    if os.path.isdir(join(prefix, join(subdir, "runs"))):
                        files = []
                        offset = 0

                        if subdir == "pencil/approx":
                            files = [f for f in listdir(join(prefix, join(subdir, "runs"))) if int(f.split("_")[2])*int(f.split("_")[3]) == P and (f.split("_")[-1]=="1.csv")==cuda_aware]
                            files.sort(key=lambda e: int(e.split("_")[1]))
                            files.sort(key=lambda e: int(e.split("_")[3]))
                            offset = 4
                        else:
                            files = [f for f in listdir(join(prefix, join(subdir, "runs"))) if int(f.split("_")[2]) == P and (f.split("_")[-1]=="1.csv")==cuda_aware]
                            files.sort(key=lambda e: int(e.split("_")[1]))
                            offset = 2

                        for f in files:
                            file = join(prefix, subdir, "runs", f)
                            with open(file) as csv_file:
                                csv_reader = csv.reader(csv_file, delimiter=',')
                                it_sizes = next(csv_reader)[offset:]
                                if len(it_sizes) > len(sizes):
                                        sizes = it_sizes

                                x_vals = ConvertSizesToLabels(it_sizes)
                                values = [-1 for s in it_sizes]
                                best_comm = [[] for s in it_sizes]
                                row = next(csv_reader)
                                while row != []:
                                    runs = [float(x) for x in row[offset:]]
                                    for i in range(len(runs)):
                                        if i < len(values) and  (values[i] == -1 or runs[i] < values[i]):
                                            values[i] = runs[i]
                                            best_comm[i] = row[:offset]

                                    row = next(csv_reader)

                                row = next(csv_reader)
                                while row != []:
                                    row = next(csv_reader)

                                # get t student intervals
                                row = next(csv_reader)
                                student_count = 0
                                lower_interval = [0 for s in it_sizes]
                                upper_interval = [0 for s in it_sizes]
                                while row != []:
                                    try:
                                        row = next(csv_reader)
                                        indices = [i for i in range(len(best_comm)) if best_comm[i] == row[:offset]]
                                        for i in indices:
                                            if student_count % 2 == 0:
                                                lower_interval[i] = row[offset+i]
                                            else:
                                                upper_interval[i] = row[offset+i]
                                        student_count += 1
                                    except:
                                        break                            
                                
                                print(best_comm)
                                with open(join(prefix, subdir, "proportions", "proportions"+f.split("runs")[1])) as csv_file1:
                                    csv_reader1 = csv.reader(csv_file1, delimiter=',')
                                    method = next(csv_reader1)
                                    indices = [i for i in range(len(best_comm)) if best_comm[i] == method]
                                    proportions = []
                                    proportion_names = []
                                    prop_count = 0
                                    for row in csv_reader1:
                                        if len(row) == 0:
                                            try: 
                                                method = next(csv_reader1)
                                                indices = [i for i in range(len(best_comm)) if best_comm[i] == method]
                                                prop_count = 0
                                            except:
                                                break
                                        else:
                                            if prop_count == len(proportion_names):
                                                proportion_names.append(row[0])
                                            
                                            for i in indices:
                                                if prop_count == len(proportions):
                                                    proportions.append([0 for s in it_sizes])
                                                proportions[prop_count][i] = row[i+1]
                                            prop_count += 1
                                
                                if subdir == "pencil/approx":
                                    for w in [writer1, writer2]:
                                        w.writerow(["Pencil", "Default" if f.split("_")[1] == "0" else "Realigned", f.split("_")[3]])
                                        w.writerow([""] + it_sizes)
                                        w.writerow([""] + ["{}_{}-{}_{}".format(b[0], b[1], b[2], b[3]) for b in best_comm])
                                else:
                                    for w in [writer1, writer2]:
                                        w.writerow(["Slab", "2D-1D" if subdir == "slab_default" else "1D-2D", "Default" if f.split("_")[1] == "0" else "Realigned"])
                                        w.writerow([""] + it_sizes)
                                        w.writerow([""] + ["{}_{}".format(b[0], b[1]) for b in best_comm])

                                for i in range(1, len(proportions)-1):
                                    writer1.writerow([proportion_names[i]] + proportions[i])
                                    writer2.writerow([proportion_names[i]] + [float(proportions[i][j])*float(values[j]) for j in range(len(values))])
                                writer1.writerow([proportion_names[-1]] + values)
                                writer1.writerow([])
                                writer2.writerow([proportion_names[-1]] + values)
                                writer2.writerow([])

                                if len(x_vals) > len(max_xvals):
                                    max_xvals = x_vals
                                if len(values) > 2:
                                    x_vals_collection.append(x_vals)
                                    values_collection.append(values)

                                    if subdir == "pencil/approx":
                                        label, = plt.plot(x_vals, values, zorder=3, linewidth=5, markersize=15, marker=markers[count%len(markers)], linestyle=":")
                                        labels.append(label)
                                        legend.append(r"Pencil [{}, $P_2 = {}$]".format("Default" if f.split("_")[1] == "0" else "Realigned", f.split("_")[3]))
                                        pencil_count += 1
                                        writer.writerow(["Pencil", "Default" if f.split("_")[1] == "0" else "Realigned", f.split("_")[3]] + lower_interval)
                                        writer.writerow(["Pencil", "Default" if f.split("_")[1] == "0" else "Realigned", f.split("_")[3]] + values)
                                        writer.writerow(["Pencil", "Default" if f.split("_")[1] == "0" else "Realigned", f.split("_")[3]] + upper_interval)
                                    else:
                                        label, = plt.plot(x_vals, values, zorder=3, linewidth=5, markersize=15, marker=markers[count%len(markers)], linestyle="-")
                                        labels.append(label)
                                        legend.append("Slab [{}, {}]".format("2D-1D" if subdir == "slab_default" else "1D-2D", "Default" if f.split("_")[1] == "0" else "Realigned"))
                                        slab_count += 1
                                        writer.writerow(["Slab", "2D-1D" if subdir == "slab_default" else "1D-2D", "Default" if f.split("_")[1] == "0" else "Realigned"] + lower_interval)
                                        writer.writerow(["Slab", "2D-1D" if subdir == "slab_default" else "1D-2D", "Default" if f.split("_")[1] == "0" else "Realigned"] + values)
                                        writer.writerow(["Slab", "2D-1D" if subdir == "slab_default" else "1D-2D", "Default" if f.split("_")[1] == "0" else "Realigned"] + upper_interval)
                                    count += 1
                                    print(legend[-1], values)
                                    
                                else:
                                    x_vals_collection.append({})
                                    values_collection.append({})
                                    next(ax._get_lines.prop_cycler) 
                                    count += 1
                writer.writerow([])

    if os.path.isdir(join(prefix, join("single"))):
        reference = True
        for i in range(0, max_labels-count):
            next(ax._get_lines.prop_cycler) 
            count += 1
        file = join(prefix, "single/runs.csv")
        with open(file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            it_sizes = next(csv_reader)
            x_vals = ConvertSizesToLabels(it_sizes)
            if len(x_vals) > len(max_xvals):
                max_xvals = x_vals
            s_values = next(csv_reader)
            values = [float(x) for x in s_values]
            label, = plt.plot(x_vals, values, zorder=3, linewidth=5, markersize=15, marker=markers[count%len(markers)], linestyle="-.")
            labels.append(label)
            legend.append("Reference")
        x_vals_collection.append(x_vals)
        values_collection.append(values)


    plt.title("Comparison " + title + " [P={}{}]".format(P, ", CUDA-aware" if cuda_aware else ""), fontsize=24)
    plt.grid(zorder=0, color="grey")
    plt.yscale('symlog', base=10)

    if not Table:
        plt.ylabel("Time [ms]", fontsize=36)
        plt.xticks(fontsize=34, rotation=30, horizontalalignment="right")
        plt.yticks(fontsize=34)
    else:
        plt.xticks(fontsize=32)
        ax.set_xticks(max_xvals)
        ax.set_xticklabels([max_xvals[i] if i%3==0 else "" for i in range(len(max_xvals))])
        plt.yticks(fontsize=32)

    plt.ylim(bottom=0)
    pathlib.Path(prefix+"/{}".format("table_plots" if Table else "plots")).mkdir(parents=True, exist_ok=True)
    plt.legend(labels, legend, prop={"size":32 if Table else 34})
    fig = plt.gcf()
    fig.set_size_inches(13, 8)
    plt.tight_layout()
    plt.savefig(prefix+"/{}/comparison_{}_{}.png".format("table_plots" if Table else "plots",P, 1 if cuda_aware else 0), dpi=100)
    plt.title("")
    plt.tight_layout()
    plt.savefig(prefix+"/{}/legend_comparison_{}_{}.pdf".format("table_plots" if Table else "plots", P, 1 if cuda_aware else 0),)
    ax.get_legend().remove()
    plt.tight_layout()
    plt.savefig(prefix+"/{}/comparison_{}_{}.pdf".format("table_plots" if Table else "plots", P, 1 if cuda_aware else 0))
    plt.close()

    plt.title("Proportions " + title + " [P={}{}]".format(P, ", CUDA-aware" if cuda_aware else ""), fontsize=24)
    ax = plt.gca()
    plt.grid(zorder=0, color="grey")

    count = 0
    for i in range(len(values_collection)-1):
        linestyle = "-"
        if count >= slab_count:
            linestyle=":"
        if x_vals_collection[i] != {}:
            plt.plot(x_vals_collection[i], [values_collection[i][j]/min([values[j] for values in values_collection if j < len(values)]) for j in range(len(values_collection[i]))], zorder=3, linewidth=5, markersize=15, marker=markers[count%len(markers)], linestyle=linestyle)
        else:
            next(ax._get_lines.prop_cycler)     
        count += 1
    for i in range(0, 9-count):
        next(ax._get_lines.prop_cycler) 
        count += 1
    plt.plot(x_vals_collection[-1], [values_collection[-1][j]/min([values[j] for values in values_collection if j < len(values)]) for j in range(len(values_collection[-1]))], zorder=3, linewidth=5, markersize=15, marker=markers[count%len(markers)], linestyle="-.")

    if not Table:
        plt.ylabel("Relative", fontsize=36)
        plt.xticks(fontsize=34, rotation=30, horizontalalignment="right")
        plt.yticks(fontsize=34)
    else:
        plt.xticks(fontsize=32)
        ax.set_xticks(max_xvals)
        ax.set_xticklabels([max_xvals[i] if i%3==0 else "" for i in range(len(max_xvals))])
        plt.yticks(fontsize=32)

    plt.legend(labels, legend, prop={"size":(32 if Table else 25)})
    fig = plt.gcf()
    fig.set_size_inches(13, 8)
    plt.tight_layout()
    plt.savefig(prefix+"/{}/proportions_{}_{}.png".format("table_plots" if Table else "plots", P, 1 if cuda_aware else 0), dpi=100)
    plt.title("")
    plt.tight_layout()
    plt.savefig(prefix+"/{}/legend_proportions_{}_{}.pdf".format("table_plots" if Table else "plots", P, 1 if cuda_aware else 0))
    ax.get_legend().remove()
    plt.tight_layout()
    plt.savefig(prefix+"/{}/proportions_{}_{}.pdf".format("table_plots" if Table else "plots", P, 1 if cuda_aware else 0))
    plt.close()

    figlegend = pylab.figure(figsize=(12, 2.3))
    leg = figlegend.legend(labels, legend, ncol=3, handlelength=5, labelspacing=1.5, prop={"size":16}, frameon=False)
    for line in leg.get_lines():
        line.set_linewidth(3.0)
        line.set_markersize(9.0)

    figlegend.savefig(prefix+"/{}/legend_{}_{}.pdf".format("table_plots" if Table else "plots", P, 1 if cuda_aware else 0))
    plt.close()

    if reference:
        plt.title("Proportions " + title + " [P={}{}]".format(P, ", CUDA-aware" if cuda_aware else ""), fontsize=24)
        plt.grid(zorder=0, color="grey")
    
        ax = plt.gca()
        count = 0
        plt_count = 0
        for i in range(len(values_collection)-1):
            linestyle = "-"
            if count >= slab_count:
                linestyle=":"
            if values_collection[i] != {}:
                plt.plot(x_vals_collection[i], [values_collection[i][j]/min([values[j] for values in values_collection[:-1] if j < len(values)]) for j in range(len(values_collection[i]))], zorder=3, linewidth=5, markersize=15, marker=markers[count%len(markers)], linestyle=linestyle)
                plt_count += 1
            else:
                next(ax._get_lines.prop_cycler) 
            count += 1

        if plt_count > 0:
            if not Table:
                plt.ylabel("Relative", fontsize=36)
                plt.xticks(fontsize=34, rotation=30, horizontalalignment="right")
                plt.yticks(fontsize=34)
            else:
                plt.xticks(fontsize=32)
                ax.set_xticks(max_xvals)
                ax.set_xticklabels([max_xvals[i] if i%3==0 else "" for i in range(len(max_xvals))])
                plt.yticks(fontsize=32)

            plt.legend(labels, legend, prop={"size":(32 if Table else 25)})
            fig = plt.gcf()
            fig.set_size_inches(13, 8)
            plt.tight_layout()
            plt.savefig(prefix+"/{}/reduced_proportions_{}_{}.png".format("table_plots" if Table else "plots", P, 1 if cuda_aware else 0), dpi=100)
            plt.title("")
            plt.tight_layout()
            plt.savefig(prefix+"/{}/legend_reduced_proportions_{}_{}.pdf".format("table_plots" if Table else "plots", P, 1 if cuda_aware else 0))
            ax.get_legend().remove()
            plt.tight_layout()
            plt.savefig(prefix+"/{}/reduced_proportions_{}_{}.pdf".format("table_plots" if Table else "plots", P, 1 if cuda_aware else 0))
        plt.close()

def main():
    collection = [
    ["bwunicluster/gpu4/forward", "BwUniCluster GPU4 Forward", [32, 24, 16, 8, 4], 10], ["bwunicluster/gpu4/inverse", "BwUniCluster GPU4 Inverse", [32, 24, 16, 8, 4], 10],
    # ["argon/forward", "Argon Forward", [4], 6], ["argon/inverse", "Argon Inverse", [4], 6], 
    # ["pcsgs/forward", "PCSGS Forward", [4], 6], ["pcsgs/inverse", "PCSGS Inverse", [4], 6], 
    # ["krypton/forward", "Krypton Forward", [4], 6], ["krypton/inverse", "Krypton Inverse", [4], 6],
    # ["bwunicluster/gpu8/small/forward", "BwUniCluster GPU8 Forward", [8, 16], 10], ["bwunicluster/gpu8/small/inverse", "BwUniCluster GPU8 Inverse", [8, 16], 10],
    # ["bwunicluster/gpu8/large/forward", "BwUniCluster GPU8 Forward", [16, 32, 48, 64], 10], ["bwunicluster/gpu8/large/inverse", "BwUniCluster GPU8 Inverse", [16, 32, 48, 64], 10]
    ]

    for entry in collection:
        for p in entry[2]:
            for c in range(2):
                print(entry[1], p, c)
                collect(c==1, p, entry[3], entry[1], "eval/benchmarks/{}".format(entry[0]))
                print()

if __name__ == "__main__":
    main()


