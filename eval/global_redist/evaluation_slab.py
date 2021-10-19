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
import sys

# Config
Table = False
prefix = "benchmarks/bwunicluster/gpu8/large"




if not Table:
    mpl.use("pgf")
    plt.rcParams.update({
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.size": 30
    })

comm_methods = {"Peer2Peer": 0, "All2All": 1}
send_methods = [{"Sync": 0, "Streams": 1, "MPI_Type": 2}, {"Sync": 0, "MPI_Type": 2}]
markers = ["D", "X", "o", "s", "v"]
linestyles = ["solid", "dotted", "dashed", "dashdot", (0, (3, 1, 1, 1, 1, 1))]

if len(sys.argv) > 1:
    prefix = "benchmarks" + str(sys.argv[1])
prec = "double"

run_iterations = 20

def getSizes(opt, comm, snd, P, cuda_aware, subdir):
    files = [f for f in listdir(join(prefix, subdir)) if isfile(join(join(prefix, subdir), f)) and re.match("test_{}_{}_{}_\d*_\d*_\d*_{}_{}".format(opt, comm, snd, (1 if cuda_aware else 0), P), f)]
    sizes = []
    for f in files:
        sizes.append(f.split("_")[4] + "_" + f.split("_")[5] + "_" + f.split("_")[6])
    sizes.sort(key=lambda e: int(e.split("_")[0]))
    sizes.sort(key=lambda e: int(e.split("_")[1]))
    sizes.sort(key=lambda e: int(e.split("_")[2]))
    return sizes

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

def getPartitions(opt, cuda_aware, subdir):
    files = [f for f in listdir(join(prefix, subdir)) if isfile(join(join(prefix, subdir), f)) and re.match("test_{}_\d_\d_\d*_\d*_\d*_{}_\d*".format(opt, (1 if cuda_aware else 0)), f)]
    partitions = []
    for f in files:
        partitions.append(int(f.split("_")[-1].split(".")[0]))
    partitions.sort()
    partitions = list(dict.fromkeys(partitions))
    return partitions

def reduceRun(opt, comm, snd, P, cuda_aware, size, subdir):
    file = join(join(prefix, subdir), "test_{}_{}_{}_{}_{}_{}.csv".format(opt, comm, snd, size, (1 if cuda_aware else 0), P))
    
    data = {}
    sd = {}
    data_max = {}; data_min = {}

    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        iterations = 0
        for row in csv_reader:
            if len(row) > 0:
                if row[0] not in data:
                    data[row[0]] = 0
                    data_max[row[0]] = 0
                    data_min[row[0]] = 0
                data[row[0]] += sum([float(x) for x in row[1:-1]]) 
                data_max[row[0]] += max([float(x) for x in row[1:-1]])
                data_min[row[0]] += min([float(x) for x in row[1:-1]])
                if row[0] == "Run complete":
                    iterations += 1
        
        for d in data:
            data[d] /= iterations*P
            data_max[d] /= iterations 
            data_min[d] /= iterations

    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        iterations = 0
        for row in csv_reader:
            if len(row) > 0:
                if row[0] not in sd:
                    sd[row[0]] = 0
                sd[row[0]] += sum([(float(x)-data[row[0]])**2 for x in row[1:-1]]) 
                if row[0] == "Run complete":
                    iterations += 1

        for s in sd:
            sd[s] = np.sqrt(sd[s]/(iterations*P-1))
        
    return data, sd, data_max, data_min

def getFFTDuration(d, forward, seq):
    length = len(d["init"])
    if forward:
        if seq == 0:
            return [d["2D FFT Y-Z-Direction"][i] + (d["1D FFT X-Direction"][i]-d["Transpose (Unpacking)"][i]) for i in range(length)]
        elif seq == 1:
            return [d["1D FFT Z-Direction"][i] + (d["2D FFT Y-X-Direction"][i]-d["Transpose (Unpacking)"][i]) for i in range(length)]
    else:
        if seq == 0:
            return [d["1D FFT X-Direction"][i] + (d["2D FFT Y-Z-Direction"][i]-d["Transpose (Unpacking)"][i]) for i in range(length)]
        elif seq == 1:
            return [d["2D FFT Y-X-Direction"][i] + (d["1D FFT Z-Direction"][i]-d["Transpose (Unpacking)"][i]) for i in range(length)]

def getFirstFFTDuration(d, forward, seq):
    length = len(d["init"])
    if forward:
        if seq == 0:
            return [d["2D FFT Y-Z-Direction"][i] for i in range(length)]
        elif seq == 1:
            return [d["1D FFT Z-Direction"][i] for i in range(length)]
    else:
        if seq == 0:
            return [d["1D FFT X-Direction"][i] for i in range(length)]
        elif seq == 1:
            return [d["2D FFT Y-X-Direction"][i] for i in range(length)]

def getSecondFFTDuration(d, forward, seq):
    length = len(d["init"])
    if forward:
        if seq == 0:
            return [(d["1D FFT X-Direction"][i]-d["Transpose (Unpacking)"][i]) for i in range(length)]
        elif seq == 1:
            return [(d["2D FFT Y-X-Direction"][i]-d["Transpose (Unpacking)"][i]) for i in range(length)]
    else:
        if seq == 0:
            return [(d["2D FFT Y-Z-Direction"][i]-d["Transpose (Unpacking)"][i]) for i in range(length)]
        elif seq == 1:
            return [(d["1D FFT Z-Direction"][i]-d["Transpose (Unpacking)"][i]) for i in range(length)]

def getCommDuration(d, peer2peer, forward, seq):
    length = len(d["init"])
    if peer2peer:
        fft_dur = getFFTDuration(d, forward, seq)
        if forward:
            if seq == 0:
                return [d["Run complete"][i] - fft_dur[i] - (d["Transpose (First Send)"][i] - d["2D FFT Y-Z-Direction"][i]) - (d["Transpose (Unpacking)"][i] - d["Transpose (Finished Receive)"][i]) for i in range(length)]
            elif seq == 1:
                return [d["Run complete"][i] - fft_dur[i] - (d["Transpose (First Send)"][i] - d["1D FFT Z-Direction"][i]) - (d["Transpose (Unpacking)"][i] - d["Transpose (Finished Receive)"][i]) for i in range(length)]
        else:
            if seq == 0:
                return [d["Run complete"][i] - fft_dur[i] - (d["Transpose (First Send)"][i] - d["1D FFT X-Direction"][i]) - (d["Transpose (Unpacking)"][i] - d["Transpose (Finished Receive)"][i]) for i in range(length)]
            elif seq == 1:
                return [d["Run complete"][i] - fft_dur[i] - (d["Transpose (First Send)"][i] - d["2D FFT Y-X-Direction"][i]) - (d["Transpose (Unpacking)"][i] - d["Transpose (Finished Receive)"][i]) for i in range(length)]   
    else:
        return [d["Transpose (Finished All2All)"][i] - d["Transpose (Start All2All)"][i] for i in range(length)]

def getPackingDuration(d, forward, seq):
    length = len(d["init"])
    if forward:
        if seq == 0:
            return [d["Transpose (Packing)"][i] - d["2D FFT Y-Z-Direction"][i] for i in range(length)]
        elif seq == 1:
            return [d["Transpose (Packing)"][i] - d["1D FFT Z-Direction"][i] for i in range(length)]
    else:
        if seq == 0:
            return [d["Transpose (Packing)"][i] - d["1D FFT X-Direction"][i] for i in range(length)]
        elif seq == 1:
            return [d["Transpose (Packing)"][i] - d["2D FFT Y-X-Direction"][i] for i in range(length)]

def getUnpackingDuration(d, peer2peer):
    length = len(d["init"])
    if peer2peer:
        return [d["Transpose (Unpacking)"][i] - d["Transpose (First Receive)"][i] for i in range(length)]
    else:
        return [d["Transpose (Unpacking)"][i] - d["Transpose (Finished All2All)"][i] for i in range(length)]

def getPackingCommOverlap(d, peer2peer):
    length = len(d["init"])
    if peer2peer:
        return [max(d["Transpose (Packing)"][i] - d["Transpose (First Send)"][i],0) for i in range(length)]
    else:
        return [0 for i in range(length)]

def getUnpackingCommOverlap(d, peer2peer):
    length = len(d["init"])
    if peer2peer:
        return [d["Transpose (Finished Receive)"][i] - d["Transpose (First Receive)"][i] for i in range(length)]
    else:
        return [0 for i in range(length)]

def reduceTestcase(opt, comm, snd, P, cuda_aware, sizes, forward, seq, subdir):
    data = {}
    sd = {}
    data_min = {}; data_max = {}

    for size in sizes:
        run_data, run_sd, run_data_max, run_data_min = reduceRun(opt, comm, snd, P, cuda_aware, size, subdir)
        if len(data.keys()) == 0:
            for key in run_data:
                data[key] = []
                sd[key] = []
                data_max[key] = []
                data_min[key] = []
        for key in run_data:
            data[key].append(run_data[key])
            sd[key].append(run_sd[key])
            data_max[key].append(run_data_max[key])
            data_min[key].append(run_data_min[key])

    return data, sd, data_max, data_min

def getProportions(data, seq, comm, forward):
    proportions = {}
    proportions["First_FFT_Duration"] = np.array(getFirstFFTDuration(data, forward, seq))
    proportions["Second_FFT_Duration"] = np.array(getSecondFFTDuration(data, forward, seq))
    proportions["FFT_Duration"] = np.array(getFFTDuration(data, forward, seq))
    proportions["Comm_Duration"] = np.array(getCommDuration(data, comm==0, forward, seq))
    proportions["Packing_Duration"] = np.array(getPackingDuration(data, forward, seq))
    proportions["Unpacking_Duration"] = np.array(getUnpackingDuration(data, comm==0))
    proportions["Packing_Comm_Overlap"] = np.array(getPackingCommOverlap(data, comm==0))
    proportions["Unpacking_Comm_Overlap"] = np.array(getUnpackingCommOverlap(data, comm==0))
    proportions["Run"] = np.array(data["Run complete"])

    # check = proportions["FFT_Duration"] + proportions["Comm_Duration"] + proportions["Packing_Duration"] + proportions["Unpacking_Duration"]
    # check -= proportions["Packing_Comm_Overlap"] + proportions["Unpacking_Comm_Overlap"]
    # check /= proportions["Run"]

    # print(seq, comm, forward)
    # print(check)

    return proportions


def compareMethods(opt, P, cuda_aware, forward, seq, subdir):
    data_collection = {}
    variance_collection = {}
    data_max_collection = {}
    data_min_collection = {}

    pathlib.Path("eval/{}/sd".format(join(prefix, subdir))).mkdir(parents=True, exist_ok=True)
    with open('eval/{}/sd/sd_{}_{}_{}.csv'.format(join(prefix, subdir), opt, P, 1 if cuda_aware else 0), mode='w') as out_file:
        writer_sd = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for comm in comm_methods:
            for snd in send_methods[comm_methods[comm]]:
                sizes = getSizes(opt, comm_methods[comm], send_methods[comm_methods[comm]][snd], P, cuda_aware, subdir)

                data, sd, data_max, data_min = reduceTestcase(opt, comm_methods[comm], send_methods[comm_methods[comm]][snd], P, cuda_aware, sizes, forward, seq, subdir)

                if data != {}:
                    data_collection["{}-{}".format(comm, snd)] = data 
                    data_max_collection["{}-{}".format(comm, snd)] = data_max
                    data_min_collection["{}-{}".format(comm, snd)] = data_min
                    variance_collection["{}-{}".format(comm, snd)] = [sd["Run complete"][i]**2 for i in range(len(sizes))]

                    # Write sd 
                    for d in sd:
                        if sum(data[d]) != 0:
                            writer_sd.writerow([d]+sd[d])

                    writer_sd.writerow([])

    pathlib.Path("eval/{}/data".format(join(prefix, subdir))).mkdir(parents=True, exist_ok=True)
    with open('eval/{}/data/max_data{}_{}_{}.csv'.format(join(prefix, subdir), opt, P, 1 if cuda_aware else 0), mode='w') as out_file:
        writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        with open('eval/{}/data/min_data{}_{}_{}.csv'.format(join(prefix, subdir), opt, P, 1 if cuda_aware else 0), mode='w') as out_file1:
            writer1 = csv.writer(out_file1, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            for comm in comm_methods:
                    for snd in send_methods[comm_methods[comm]]:
                        if "{}-{}".format(comm, snd) in data_collection:
                            sizes = getSizes(opt, comm_methods[comm], send_methods[comm_methods[comm]][snd], P, cuda_aware, subdir)
                            writer.writerow([comm, snd])
                            writer.writerow([""] + sizes)
                            writer1.writerow([comm, snd])
                            writer1.writerow([""] + sizes)

                            data_max = data_max_collection["{}-{}".format(comm, snd)]
                            data_min = data_min_collection["{}-{}".format(comm, snd)]

                            for d in data_max:
                                if sum(data_max[d]) != 0:
                                    writer.writerow([d]+data_max[d])
                                    writer1.writerow([d]+data_min[d])

                            writer.writerow([])
                            writer1.writerow([])

    with open('eval/{}/data/data{}_{}_{}.csv'.format(join(prefix, subdir), opt, P, 1 if cuda_aware else 0), mode='w') as out_file:
        writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        pathlib.Path("eval/{}/proportions".format(join(prefix, subdir))).mkdir(parents=True, exist_ok=True)
        with open('eval/{}/proportions/proportions_{}_{}_{}.csv'.format(join(prefix, subdir), opt, P, 1 if cuda_aware else 0), mode='w') as out_file1:
            writer1 = csv.writer(out_file1, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            pathlib.Path("eval/{}/runs".format(join(prefix, subdir))).mkdir(parents=True, exist_ok=True)
            with open('eval/{}/runs/runs_{}_{}_{}.csv'.format(join(prefix, subdir), opt, P, 1 if cuda_aware else 0), mode='w') as out_file2:
                writer2 = csv.writer(out_file2, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                sizes = getSizes(opt, 0, 0, P, cuda_aware, subdir)
                sizes_id = sizes.copy()
                runs = [{} for s in sizes]
                writer2.writerow(["", ""] + sizes)

                for comm in comm_methods:
                    for snd in send_methods[comm_methods[comm]]:
                        if "{}-{}".format(comm, snd) in data_collection:
                            sizes = getSizes(opt, comm_methods[comm], send_methods[comm_methods[comm]][snd], P, cuda_aware, subdir)
                            writer.writerow([comm, snd])
                            writer.writerow([""] + sizes)
                            writer1.writerow([comm, snd])
                            writer1.writerow([""] + sizes)

                            data = data_collection["{}-{}".format(comm, snd)]
                            proportions = getProportions(data, seq, comm_methods[comm], forward)

                            for d in data:
                                if sum(data[d]) != 0:
                                    writer.writerow([d]+data[d])
                            
                            for p in proportions:
                                if p != "Run":
                                    writer1.writerow([p] + [max(proportions[p][i] / proportions["Run"][i], 0) for i in range(len(proportions[p]))])
                            writer1.writerow(["Run"] + [proportions["Run"][i] for i in range(len(proportions["Run"]))])

                            if (not cuda_aware) or (not ((forward == 1 and opt==1) or (forward==0 and opt==0))) or snd != "Streams":
                                writer2.writerow([comm, snd] + data["Run complete"])
                                for j in range(len(sizes)):
                                    i = sizes_id.index(sizes[j])
                                    runs[i]["{}-{}".format(comm, snd)] = data["Run complete"][j]  

                            writer.writerow([])
                            writer1.writerow([])

                writer2.writerow([])
                writer2.writerow(["Proportions w.r.t. Minimum"])

                diff = [{} for i in range(len(runs))]
                for i in range(len(runs)):
                    for key in runs[i]:
                        diff[i][key] = runs[i][key] / min(runs[i].values())

                for key in runs[0]:
                    writer2.writerow(key.split("-") + [diff[i][key] for i in range(len(sizes)) if key in diff[i]])

                writer2.writerow([])
                writer2.writerow(["t-Student Approximation of Mean"])

                quantile = t.ppf(0.99, run_iterations*P-1)

                fig1, ax1 = plt.subplots()
                fig2, ax2 = plt.subplots()
                fig3, ax3 = plt.subplots()
                labels = []; legend = []

                m_count = 0
                for comm in comm_methods:
                    linestyle = "-"
                    if comm == "All2All":
                        linestyle = ":"
                    for snd in send_methods[comm_methods[comm]]:
                        if "{}-{}".format(comm, snd) in variance_collection:
                            if (not cuda_aware) or (not ((forward == 1 and opt==1) or (forward==0 and opt==0))) or snd != "Streams":
                                sizes = getSizes(opt, comm_methods[comm], send_methods[comm_methods[comm]][snd], P, cuda_aware, subdir)
                                sd_arr = [0 for s in sizes_id]
                                for i in range(len(sizes)):
                                    sd_arr[sizes_id.index(sizes[i])] = np.sqrt(variance_collection["{}-{}".format(comm, snd)][i])
                                eps = [quantile*sd/np.sqrt(P*run_iterations) for sd in sd_arr]

                                writer2.writerow([comm, snd] + [runs[i]["{}-{}".format(comm, snd)]-eps[i] for i in range(len(sd_arr)) if i < len(runs) and "{}-{}".format(comm, snd) in runs[i]])
                                writer2.writerow([comm, snd] + [runs[i]["{}-{}".format(comm, snd)]+eps[i] for i in range(len(sd_arr)) if i < len(runs) and "{}-{}".format(comm, snd) in runs[i]])

                                x_vals = ConvertSizesToLabels(sizes)

                                label, = ax1.plot(x_vals, [runs[sizes_id.index(s)]["{}-{}".format(comm, snd)] for s in sizes], linestyle=linestyle, marker=markers[m_count%len(markers)], zorder=3, linewidth=5, markersize=15)
                                ax2.plot(x_vals, [diff[sizes_id.index(s)]["{}-{}".format(comm, snd)] for s in sizes], linestyle=linestyle, marker=markers[m_count%len(markers)], zorder=3, linewidth=5, markersize=15)
                                ax2.errorbar(x_vals, [diff[sizes_id.index(s)]["{}-{}".format(comm, snd)] for s in sizes], [eps[sizes_id.index(s)] / min(runs[sizes_id.index(s)].values()) for s in sizes], fmt='.k', elinewidth=3, capsize=5)
                                ax2.fill_between(x_vals, [diff[sizes_id.index(s)]["{}-{}".format(comm, snd)] - eps[sizes_id.index(s)] / min(runs[sizes_id.index(s)].values()) for s in sizes], [diff[sizes_id.index(s)]["{}-{}".format(comm, snd)] + eps[sizes_id.index(s)] / min(runs[sizes_id.index(s)].values()) for s in sizes], zorder=3, alpha=0.3)
                                if snd != "MPI_Type":
                                    ax3.plot(x_vals, [diff[sizes_id.index(s)]["{}-{}".format(comm, snd)] for s in sizes], linestyle=linestyle, marker=markers[m_count%len(markers)], zorder=3, linewidth=5, markersize=15)
                                    ax3.errorbar(x_vals, [diff[sizes_id.index(s)]["{}-{}".format(comm, snd)] for s in sizes], [eps[sizes_id.index(s)] / min(runs[sizes_id.index(s)].values()) for s in sizes], fmt='.k', elinewidth=3, capsize=5)
                                    ax3.fill_between(x_vals, [diff[sizes_id.index(s)]["{}-{}".format(comm, snd)] - eps[sizes_id.index(s)] / min(runs[sizes_id.index(s)].values()) for s in sizes], [diff[sizes_id.index(s)]["{}-{}".format(comm, snd)] + eps[sizes_id.index(s)] / min(runs[sizes_id.index(s)].values()) for s in sizes], zorder=3, alpha=0.3)
                                else:
                                    next(ax3._get_lines.prop_cycler) 
                                    next(ax3._get_patches_for_fill.prop_cycler)
                                labels.append(label)
                                legend.append("{}-{}".format(comm, snd))
                            else:
                                next(ax1._get_lines.prop_cycler) 
                                next(ax2._get_lines.prop_cycler) 
                                next(ax2._get_patches_for_fill.prop_cycler)
                                next(ax3._get_lines.prop_cycler) 
                                next(ax3._get_patches_for_fill.prop_cycler)
                        else:
                            next(ax1._get_lines.prop_cycler) 
                            next(ax2._get_lines.prop_cycler) 
                            next(ax2._get_patches_for_fill.prop_cycler)
                            next(ax3._get_lines.prop_cycler) 
                            next(ax3._get_patches_for_fill.prop_cycler)
                        m_count += 1

                # ax1.set_title(r"Slab Communication Methods [{}, {}, {}{}]".format(P, r"ZY $\rightarrow$ X" if seq==0 else r"Z $\rightarrow$ YX", "default" if opt==0 else "data realignment", ", CUDA-aware" if cuda_aware else ""), fontsize=18)
                # ax2.set_title(r"Slab Communication Methods Proportions [{}, {}, {}{}]".format(P, r"ZY $\rightarrow$ X" if seq==0 else r"Z $\rightarrow$ YX", "default" if opt==0 else "data realignment", ", CUDA-aware" if cuda_aware else ""), fontsize=18)
                # ax3.set_title(r"Slab Communication Methods Proportions [{}, {}, {}{}]".format(P, r"ZY $\rightarrow$ X" if seq==0 else r"Z $\rightarrow$ YX", "default" if opt==0 else "data realignment", ", CUDA-aware" if cuda_aware else ""), fontsize=18)
                for p in [[fig1, ax1], [fig2, ax2], [fig3, ax3]]:
                    fig = p[0]; ax = p[1]

                    ax.legend(labels, legend, prop={"size":(32 if Table else 25)})
                    ax.grid(zorder=0, color="grey")
                    if fig == fig1:
                        ax.set_yscale('symlog', base=10)
                        if not Table:
                            ax.set_ylabel("Time [ms]", fontsize=36)
                    else: 
                        if not Table:
                            ax.set_ylabel("Relative", fontsize=36)

                    if not Table:
                        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
                        ax.tick_params(axis='x', labelsize=34)
                        ax.tick_params(axis='y', labelsize=34)
                    else:
                        plt.xticks(fontsize=32)
                        ax.set_xticks(x_vals)
                        ax.set_xticklabels([x_vals[i] if i%3==0 else "" for i in range(len(x_vals))])
                        plt.yticks(fontsize=32)


                    fig.set_size_inches(13, 8)

                # ax2.set_ylim(top=2)

                path = ""
                if not Table:
                    path = 'eval/{}/plots_legend/{}_{}_{}'.format(join(prefix, subdir), opt, P, 1 if cuda_aware else 0)
                else:
                    path = 'eval/{}/plots_tables_legend/{}_{}_{}'.format(join(prefix, subdir), opt, P, 1 if cuda_aware else 0)
                pathlib.Path(path).mkdir(parents=True, exist_ok=True)
                print(path)
                fig1.savefig("{}/plot.png".format(path), dpi=100)
                fig1.savefig("{}/plot.pdf".format(path), dpi=100, bbox_inches='tight')
                fig2.savefig("{}/diff.png".format(path), dpi=100)
                fig2.savefig("{}/diff.pdf".format(path), dpi=100, bbox_inches='tight')
                fig3.savefig("{}/reduced_diff.png".format(path), dpi=100)
                fig3.savefig("{}/reduced_diff.pdf".format(path), dpi=100, bbox_inches='tight')

                if not Table:
                    path = 'eval/{}/plots/{}_{}_{}'.format(join(prefix, subdir), opt, P, 1 if cuda_aware else 0)
                else:
                    path = 'eval/{}/plots_tables/{}_{}_{}'.format(join(prefix, subdir), opt, P, 1 if cuda_aware else 0)
                pathlib.Path(path).mkdir(parents=True, exist_ok=True)
                ax1.get_legend().remove()
                fig1.savefig("{}/plot.png".format(path), dpi=100)
                fig1.savefig("{}/plot.pdf".format(path), dpi=100, bbox_inches='tight')
                ax2.get_legend().remove()
                fig2.savefig("{}/diff.png".format(path), dpi=100)
                fig2.savefig("{}/diff.pdf".format(path), dpi=100, bbox_inches='tight')
                ax3.get_legend().remove()
                fig3.savefig("{}/reduced_diff.png".format(path), dpi=100)
                fig3.savefig("{}/reduced_diff.pdf".format(path), dpi=100, bbox_inches='tight')
                plt.close()

                
def main():
    for c in range(0, 2):
        cuda_aware = True if c == 0 else False
        for forward in range(0, 2):
            for seq in range(0, 2):
                for opt in range(0, 2):
                    subdir = "{}/slab_{}".format("forward" if forward==1 else "inverse", "default" if seq==0 else "z_then_yx")
                    pathlib.Path("eval/{}".format(join(prefix, subdir))).mkdir(parents=True, exist_ok=True)
                    
                    partitions = getPartitions(opt, cuda_aware, subdir)
                    for P in partitions:
                        compareMethods(opt, P, cuda_aware, forward, seq, subdir)

if __name__ == "__main__":
    main()
