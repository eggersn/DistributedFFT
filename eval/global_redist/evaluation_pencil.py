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
import argparse


# Config
Table = False
prefix = "benchmarks/bwunicluster/gpu4"

parser = argparse.ArgumentParser(description='Slab Evaluation Script.')
parser.add_argument('--prefix', metavar="p", type=str, nargs=1, dest='p', help='Benchmark Prefix')

args = parser.parse_args()
if args.p != None:
    prefix = args.p[0]


if not Table:
    mpl.use("pgf")
    plt.rcParams.update({
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.size": 30
    })

comm_methods = {"Peer2Peer": 0, "All2All": 1}
send_methods = [{"Sync": 0, "Streams": 1, "MPI\_Type": 2}, {"Sync": 0, "MPI\_Type": 2}]
markers = ["D", "X", "o", "s", "v"]

prec = "double"

# Used to compute the bounds of the mean-interval
run_iterations = 20

def getSizes(opt, comm1, snd1, comm2, snd2, P_str, cuda_aware, subdir):
    files = [f for f in listdir(join(prefix, subdir)) if isfile(join(join(prefix, subdir), f)) and re.match("test_{}_{}_{}_{}_{}_\d*_\d*_\d*_{}_{}".format(opt, comm1, snd1, comm2, snd2, (1 if cuda_aware else 0), P_str), f)]
    sizes = []
    for f in files:
        sizes.append(f.split("_")[6] + "_" + f.split("_")[7] + "_" + f.split("_")[8])
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
            labels.append(r"${}^2\times{}$".format(dims[0], dims[2]))
        elif dims[1] == dims[2]:
            labels.append(r"${}\times{}^2$".format(dims[0], dims[2]))
        else:
            labels.append(r"${}\times{}\times{}$".format(dims[0], dims[1], dims[2]))

    return labels


def getPartitions(opt, cuda_aware, subdir):
    files = [f for f in listdir(join(prefix, subdir)) if isfile(join(join(prefix, subdir), f)) and re.match("test_{}_\d_\d_\d_\d_\d*_\d*_\d*_{}_\d*_\d*".format(opt, (1 if cuda_aware else 0)), f)]
    partitions = []
    partitions_strings = []
    for f in files:
        partitions_strings.append("{}_{}".format(f.split("_")[-2], f.split("_")[-1].split(".")[0]))
    partitions_strings.sort()
    partitions_strings = list(dict.fromkeys(partitions_strings))

    partitions = [int(p.split("_")[0])*int(p.split("_")[1]) for p in partitions_strings]

    return partitions, partitions_strings

def reduceRun(opt, comm1, snd1, comm2, snd2, P_str, P, cuda_aware, size, subdir):
    file = join(join(prefix, subdir), "test_{}_{}_{}_{}_{}_{}_{}_{}.csv".format(opt, comm1, snd1, comm2, snd2, size, (1 if cuda_aware else 0), P_str))
    
    data = {}
    sd = {}
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        iterations = 0
        for row in csv_reader:
            if len(row) > 0:
                if row[0] not in data:
                    data[row[0]] = 0
                data[row[0]] += sum([float(x) for x in row[1:-1]]) 
                if row[0] == "Run complete":
                    iterations += 1

        for d in data:
            data[d] /= iterations*P

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
        
    return data, sd

def getZFFTDuration(d, forward):
    length = len(d["Run complete"])
    if forward==1:
        return [d["1D FFT Z-Direction"][i] for i in range(length)]
    else:
        return [d["1D FFT Z-Direction"][i]-d["Second Transpose (Unpacking)"][i] for i in range(length)]

def getYFFTDuration(d, forward):
    length = len(d["Run complete"])
    return [d["1D FFT Y-Direction"][i]-d["First Transpose (Unpacking)"][i] for i in range(length)]


def getXFFTDuration(d, forward):
    length = len(d["Run complete"])
    if forward==1:
        return [d["1D FFT X-Direction"][i]-d["Second Transpose (Unpacking)"][i] for i in range(length)]
    else:
        return [d["1D FFT X-Direction"][i] for i in range(length)]


def getFirstCommDuration(d, peer2peer, forward):
    length = len(d["Run complete"])
    if peer2peer:
        yfft_dur = getYFFTDuration(d, forward)
        return [d["First Transpose (Send Complete)"][i] - yfft_dur[i] - d["First Transpose (First Send)"][i] - (d["First Transpose (Unpacking)"][i] - d["First Transpose (Finished Receive)"][i]) for i in range(length)]
    else:
        return [d["First Transpose (Finished All2All)"][i] - d["First Transpose (Start All2All)"][i] for i in range(length)]

def getSecondCommDuration(d, peer2peer, forward):
    length = len(d["Run complete"])
    if peer2peer:
        fft_dur = []
        if forward==1:
            fft_dur = getXFFTDuration(d, forward)
        else:
            fft_dur = getZFFTDuration(d, forward)
        return [d["Run complete"][i] - d["Second Transpose (First Send)"][i] - (d["Second Transpose (Unpacking)"][i] - d["Second Transpose (Finished Receive)"][i]) - fft_dur[i] for i in range(length)]
    else:
        return [d["Second Transpose (Finished All2All)"][i] - d["Second Transpose (Start All2All)"][i] for i in range(length)]

def getFirstPackingDuration(d, peer2peer, forward):
    length = len(d["Run complete"])
    if forward==1:
        if peer2peer:
            return [d["First Transpose (Packing)"][i] - d["1D FFT Z-Direction"][i] for i in range(length)]
        else:
            return [d["First Transpose (Start All2All)"][i] - d["1D FFT Z-Direction"][i] for i in range(length)]
    else:
        if peer2peer:
            return [d["First Transpose (Packing)"][i] - d["1D FFT X-Direction"][i] for i in range(length)]
        else:
            return [d["First Transpose (Start All2All)"][i] - d["1D FFT X-Direction"][i] for i in range(length)]

def getSecondPackingDuration(d, peer2peer):
    length = len(d["Run complete"])
    if peer2peer:
        return [d["Second Transpose (Packing)"][i] - d["First Transpose (Send Complete)"][i] for i in range(length)]
    else:
        return [d["Second Transpose (Start All2All)"][i] - d["First Transpose (Send Complete)"][i] for i in range(length)]


def getFirstUnpackingDuration(d, peer2peer):
    length = len(d["Run complete"])
    if peer2peer:
        return [d["First Transpose (Unpacking)"][i] - d["First Transpose (First Receive)"][i] for i in range(length)]
    else:
        return [d["First Transpose (Unpacking)"][i] - d["First Transpose (Finished All2All)"][i] for i in range(length)]

def getSecondUnpackingDuration(d, peer2peer):
    length = len(d["Run complete"])
    if peer2peer:
        return [d["Second Transpose (Unpacking)"][i] - d["Second Transpose (First Receive)"][i] for i in range(length)]
    else:
        return [d["Second Transpose (Unpacking)"][i] - d["Second Transpose (Finished All2All)"][i] for i in range(length)]

def getFirstPackingCommOverlap(d, peer2peer):
    length = len(d["Run complete"])
    if peer2peer:
        return [d["First Transpose (Packing)"][i] - d["First Transpose (First Send)"][i] for i in range(length)]
    else:
        return [0 for i in range(length)]

def getSecondPackingCommOverlap(d, peer2peer):
    length = len(d["Run complete"])
    if peer2peer:
        return [d["Second Transpose (Packing)"][i] - d["Second Transpose (First Send)"][i] for i in range(length)]
    else:
        return [0 for i in range(length)]

def getFirstUnpackingCommOverlap(d, peer2peer):
    length = len(d["Run complete"])
    if peer2peer:
        return [d["First Transpose (Finished Receive)"][i] - d["First Transpose (First Receive)"][i] for i in range(length)]
    else:
        return [0 for i in range(length)]

def getSecondUnpackingCommOverlap(d, peer2peer):
    length = len(d["Run complete"])
    if peer2peer:
        return [d["Second Transpose (Finished Receive)"][i] - d["Second Transpose (First Receive)"][i] for i in range(length)]
    else:
        return [0 for i in range(length)]

def getProportions(data, comm1, comm2, forward):
    proportions = {}

    proportions["Z_FFT_Duration"] = np.array(getZFFTDuration(data, forward))
    proportions["Y_FFT_Duration"] = np.array(getYFFTDuration(data, forward))
    proportions["X_FFT_Duration"] = np.array(getXFFTDuration(data, forward))

    proportions["First_Comm_Duration"] = np.array(getFirstCommDuration(data, (comm1==0 and forward==1) or (comm2==0 and forward==0), forward))
    proportions["First_Packing_Duration"] = np.array(getFirstPackingDuration(data, (comm1==0 and forward==1) or (comm2==0 and forward==0), forward))
    proportions["First_Unpacking_Duration"] = np.array(getFirstUnpackingDuration(data, (comm1==0 and forward==1) or (comm2==0 and forward==0)))
    proportions["First_Packing_Comm_Overlap"] = np.array(getFirstPackingCommOverlap(data, (comm1==0 and forward==1) or (comm2==0 and forward==0)))
    proportions["First_Unpacking_Comm_Overlap"] = np.array(getFirstUnpackingCommOverlap(data, (comm1==0 and forward==1) or (comm2==0 and forward==0)))
    proportions["Second_Comm_Duration"] = np.array(getSecondCommDuration(data, (comm2==0 and forward==1) or (comm1==0 and forward==0), forward))
    proportions["Second_Packing_Duration"] = np.array(getSecondPackingDuration(data, (comm2==0 and forward==1) or (comm1==0 and forward==0)))
    proportions["Second_Unpacking_Duration"] = np.array(getSecondUnpackingDuration(data, (comm2==0 and forward==1) or (comm1==0 and forward==0)))
    proportions["Second_Packing_Comm_Overlap"] = np.array(getSecondPackingCommOverlap(data, (comm2==0 and forward==1) or (comm1==0 and forward==0)))
    proportions["Second_Unpacking_Comm_Overlap"] = np.array(getSecondUnpackingCommOverlap(data, (comm2==0 and forward==1) or (comm1==0 and forward==0)))
    proportions["Run"] = np.array(data["Run complete"])

    # Verify that the proportions are calculated correctly (sum should be close to 1)
    # check = proportions["Z_FFT_Duration"] + proportions["Y_FFT_Duration"] + proportions["X_FFT_Duration"] + proportions["First_Comm_Duration"] + proportions["First_Packing_Duration"] + proportions["First_Unpacking_Duration"]
    # check -= proportions["First_Packing_Comm_Overlap"] + proportions["First_Unpacking_Comm_Overlap"]
    # check += proportions["Second_Comm_Duration"] + proportions["Second_Packing_Duration"] + proportions["Second_Unpacking_Duration"]
    # check -= proportions["Second_Packing_Comm_Overlap"] + proportions["Second_Unpacking_Comm_Overlap"]
    # check /= proportions["Run"]

    # print(opt, comm1, snd1, comm2, snd2, P_str, P, cuda_aware, sizes, forward, subdir)
    # print(check)

    return proportions

def reduceTestcase(opt, comm1, snd1, comm2, snd2, P_str, P, cuda_aware, sizes, forward, subdir):
    data = {}
    sd = {}
    for size in sizes:
        run_data, run_sd = reduceRun(opt, comm1, snd1, comm2, snd2, P_str, P, cuda_aware, size, subdir)
        if len(data.keys()) == 0:
            for key in run_data:
                data[key] = []
                sd[key] = []
        for key in run_data:
            data[key].append(run_data[key])
            sd[key].append(run_sd[key])

    return data, sd

def compareMethods(opt, P_str, P, cuda_aware, forward, subdir):
    data_collection = {}
    variance_collection = {}
    
    pathlib.Path("eval/{}/exact/sd".format(join(prefix, subdir))).mkdir(parents=True, exist_ok=True)
    with open('eval/{}/exact/sd/sd_{}_{}_{}.csv'.format(join(prefix, subdir), opt, P_str, 1 if cuda_aware else 0), mode='w') as out_file:
        writer_sd = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for comm1 in comm_methods:
            for snd1 in send_methods[comm_methods[comm1]]:
                for comm2 in comm_methods:
                    for snd2 in send_methods[comm_methods[comm2]]:
                        if (comm1 == "Peer2Peer" and snd1 == "Sync") or (comm2 == "Peer2Peer" and snd2 == "Sync"):
                            if not (cuda_aware and snd2 == "Streams" and ((forward==1 and opt==1) or (forward==0 and opt==0))) and not (cuda_aware and snd1 == "Streams" and forward==1 and opt==1):
                                sizes = getSizes(opt, comm_methods[comm1], send_methods[comm_methods[comm1]][snd1], comm_methods[comm2], send_methods[comm_methods[comm2]][snd2], P_str, cuda_aware, subdir)

                                data, sd = reduceTestcase(opt, comm_methods[comm1], send_methods[comm_methods[comm1]][snd1], comm_methods[comm2], send_methods[comm_methods[comm2]][snd2], P_str, P, cuda_aware, sizes, forward, subdir)

                                if data != {}:
                                    data_collection["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] = data 
                                    variance_collection["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] = {}
                                    for key in sd:
                                        variance_collection["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)][key] = [sd[key][i]**2 for i in range(len(sizes))]

                                    # Write sd 
                                    writer_sd.writerow([comm1, snd1, comm2, snd2])
                                    writer_sd.writerow([""] + sizes)
                                    for d in sd:
                                        if sum(data[d]) != 0:
                                            writer_sd.writerow([d]+sd[d])

                                    writer_sd.writerow([])

        # Approximate other comm / snd methods 

        first_sec = []
        second_sec = []

        for comm1 in comm_methods:
            for snd1 in send_methods[comm_methods[comm1]]:
                for comm2 in comm_methods:
                    for snd2 in send_methods[comm_methods[comm2]]:
                        if (comm1 != "Peer2Peer" or snd1 != "Sync") and (comm2 != "Peer2Peer" or snd2 != "Sync"):
                            if "Peer2Peer-Sync-{}-{}".format(comm2, snd2) in data_collection and "{}-{}-Peer2Peer-Sync".format(comm1, snd1) in data_collection:
                                if forward==1:
                                    if comm1 == "Peer2Peer":
                                        first_sec = ["1D FFT Z-Direction", "First Transpose (First Send)", "First Transpose (Packing)", "First Transpose (Start Local Transpose)", "First Transpose (Start Receive)", "First Transpose (First Receive)", "First Transpose (Finished Receive)", "First Transpose (Unpacking)", "First Transpose (Send Complete)", "1D FFT Y-Direction"]
                                    else:
                                        first_sec = ["1D FFT Z-Direction", "First Transpose (Packing)", "First Transpose (Start All2All)", "First Transpose (Finished All2All)", "First Transpose (Unpacking)", "First Transpose (Send Complete)", "1D FFT Y-Direction"]
                                    
                                    if comm2 == "Peer2Peer":
                                        second_sec = ["Second Transpose (First Send)", "Second Transpose (Packing)", "Second Transpose (Start Local Transpose)", "Second Transpose (Start Receive)", "Second Transpose (First Receive)", "Second Transpose (Finished Receive)", "Second Transpose (Unpacking)", "1D FFT X-Direction", "Run complete"]
                                    else:
                                        second_sec = ["Second Transpose (Packing)", "Second Transpose (Start All2All)", "Second Transpose (Finished All2All)", "Second Transpose (Unpacking)", "1D FFT X-Direction", "Run complete"]

                                    data_collection["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] = {}
                                    variance_collection["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] = {}
                                    for key in first_sec:
                                        data_collection["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)][key] = data_collection["{}-{}-Peer2Peer-Sync".format(comm1, snd1)][key]
                                        variance_collection["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)][key] = variance_collection["{}-{}-Peer2Peer-Sync".format(comm1, snd1)][key]
                                    for key in second_sec:
                                        length = min(len(data_collection["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)]["First Transpose (Send Complete)"]), len(data_collection["Peer2Peer-Sync-{}-{}".format(comm2, snd2)][key]))
                                        if np.array(data_collection["Peer2Peer-Sync-{}-{}".format(comm2, snd2)][key][:length]).any() != 0:
                                            data_collection["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)][key] = np.array(data_collection["Peer2Peer-Sync-{}-{}".format(comm2, snd2)][key][:length]) - np.array(data_collection["Peer2Peer-Sync-{}-{}".format(comm2, snd2)]["First Transpose (Send Complete)"][:length]) + np.array(data_collection["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)]["First Transpose (Send Complete)"][:length])
                                            variance_collection["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)][key] = np.array(variance_collection["Peer2Peer-Sync-{}-{}".format(comm2, snd2)][key][:length]) + np.array(variance_collection["Peer2Peer-Sync-{}-{}".format(comm2, snd2)]["First Transpose (Send Complete)"][:length]) + np.array(variance_collection["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)]["First Transpose (Send Complete)"][:length])

                                else:
                                    if comm2 == "Peer2Peer":
                                        first_sec = ["1D FFT X-Direction", "First Transpose (First Send)", "First Transpose (Packing)", "First Transpose (Start Local Transpose)", "First Transpose (Start Receive)", "First Transpose (First Receive)", "First Transpose (Finished Receive)", "First Transpose (Unpacking)", "First Transpose (Send Complete)", "1D FFT Y-Direction"]
                                    else:
                                        first_sec = ["1D FFT X-Direction", "First Transpose (Packing)", "First Transpose (Start All2All)", "First Transpose (Finished All2All)", "First Transpose (Unpacking)", "First Transpose (Send Complete)", "1D FFT Y-Direction"]

                                    if comm1 == "Peer2Peer":
                                        second_sec = ["Second Transpose (First Send)", "Second Transpose (Packing)", "Second Transpose (Start Local Transpose)", "Second Transpose (Start Receive)", "Second Transpose (First Receive)", "Second Transpose (Finished Receive)", "Second Transpose (Unpacking)", "1D FFT Z-Direction", "Run complete"]
                                    else:
                                        second_sec = ["Second Transpose (Packing)", "Second Transpose (Start All2All)", "Second Transpose (Finished All2All)", "Second Transpose (Unpacking)", "1D FFT Z-Direction", "Run complete"]                                

                                    data_collection["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] = {}
                                    variance_collection["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] = {}
                                    for key in first_sec:
                                        data_collection["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)][key] = data_collection["Peer2Peer-Sync-{}-{}".format(comm2, snd2)][key]
                                        variance_collection["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)][key] = variance_collection["Peer2Peer-Sync-{}-{}".format(comm2, snd2)][key]
                                    for key in second_sec:
                                        length = min(len(data_collection["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)]["First Transpose (Send Complete)"]), len(data_collection["{}-{}-Peer2Peer-Sync".format(comm1, snd1)][key]))
                                        if np.array(data_collection["{}-{}-Peer2Peer-Sync".format(comm1, snd1)][key][:length]).any() != 0:
                                            data_collection["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)][key] = np.array(data_collection["{}-{}-Peer2Peer-Sync".format(comm1, snd1)][key][:length]) - np.array(data_collection["{}-{}-Peer2Peer-Sync".format(comm1, snd1)]["First Transpose (Send Complete)"][:length]) + np.array(data_collection["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)]["First Transpose (Send Complete)"][:length])
                                            variance_collection["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)][key] = np.array(variance_collection["{}-{}-Peer2Peer-Sync".format(comm1, snd1)][key][:length]) + np.array(variance_collection["{}-{}-Peer2Peer-Sync".format(comm1, snd1)]["First Transpose (Send Complete)"][:length]) + np.array(variance_collection["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)]["First Transpose (Send Complete)"][:length])



        # Write ascertained results to csv files
        pathlib.Path("eval/{}/exact/data".format(join(prefix, subdir))).mkdir(parents=True, exist_ok=True)
        with open('eval/{}/exact/data/data_{}_{}_{}.csv'.format(join(prefix, subdir), opt, P_str, 1 if cuda_aware else 0), mode='w') as out_file:
            writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            pathlib.Path("eval/{}/exact/proportions".format(join(prefix, subdir))).mkdir(parents=True, exist_ok=True)
            with open('eval/{}/exact/proportions/proportions_{}_{}_{}.csv'.format(join(prefix, subdir), opt, P_str, 1 if cuda_aware else 0), mode='w') as out_file1:
                writer1 = csv.writer(out_file1, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                pathlib.Path("eval/{}/exact/runs".format(join(prefix, subdir))).mkdir(parents=True, exist_ok=True)
                with open('eval/{}/exact/runs/runs_{}_{}_{}.csv'.format(join(prefix, subdir), opt, P_str, 1 if cuda_aware else 0), mode='w') as out_file2:
                    writer2 = csv.writer(out_file2, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                    sizes = getSizes(opt, 0, 0, 0, 0, P_str, cuda_aware, subdir)
                    ref_sizes = sizes.copy()
                    runs1 = [{} for s in sizes]
                    runs2 = [{} for s in sizes]
                    writer2.writerow(["", "", "", ""] + sizes)

                    for comm1 in comm_methods:
                        for snd1 in send_methods[comm_methods[comm1]]:
                            for comm2 in comm_methods:
                                for snd2 in send_methods[comm_methods[comm2]]:
                                    if ((comm1 == "Peer2Peer" and snd1 == "Sync") or (comm2 == "Peer2Peer" and snd2 == "Sync")) and "{}-{}-{}-{}".format(comm1, snd1, comm2, snd2) in data_collection:
                                        sizes = getSizes(opt, comm_methods[comm1], send_methods[comm_methods[comm1]][snd1], comm_methods[comm2], send_methods[comm_methods[comm2]][snd2], P_str, cuda_aware, subdir)

                                        writer.writerow([comm1, snd1, comm2, snd2])
                                        writer.writerow([""] + sizes)
                                        writer1.writerow([comm1, snd1, comm2, snd2])
                                        writer1.writerow([""] + sizes)

                                        data = data_collection["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)]
                                        proportions = getProportions(data, comm_methods[comm1], comm_methods[comm2], forward)

                                        # write data
                                        for d in data:
                                            if sum(data[d]) != 0:
                                                writer.writerow([d]+data[d])
                                        
                                        # write proportions
                                        for p in proportions:
                                            if p != "Run":
                                                writer1.writerow([p] + [max(proportions[p][i] / proportions["Run"][i], 0) for i in range(len(proportions[p]))])
                                        writer1.writerow(["Run"] + [proportions["Run"][i] for i in range(len(proportions["Run"]))])

                                        # write run complete 
                                        if comm1 == "Peer2Peer" and snd1 == "Sync":
                                            if not (cuda_aware and snd2 == "Streams" and ((forward==1 and opt==1) or (forward==0 and opt==0))):
                                                writer2.writerow([comm1, snd1, comm2, snd2] + data["Run complete"])
                                                for i in range(len(sizes)):
                                                    runs1[ref_sizes.index(sizes[i])]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] = data["Run complete"][i]
                                        if comm2 == "Peer2Peer" and snd2 == "Sync":
                                            if not (cuda_aware and snd1 == "Streams" and forward==1 and opt==1):
                                                writer2.writerow([comm1, snd1, comm2, snd2] + data["Run complete"])
                                                for i in range(len(sizes)):
                                                    runs2[ref_sizes.index(sizes[i])]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] = data["Run complete"][i]  

                                        writer.writerow([])
                                        writer1.writerow([])   

                    writer2.writerow([])
                    writer2.writerow(["Scaled Difference to Minimum"])
                    diff1 = [{} for i in range(len(runs1))]
                    for i in range(len(runs1)):
                        for key in runs1[i]:
                            diff1[i][key] = (runs1[i][key])/min(runs1[i].values())

                    diff2 = [{} for i in range(len(runs2))]
                    for i in range(len(runs2)):
                        for key in runs2[i]:
                            diff2[i][key] = (runs2[i][key])/min(runs2[i].values())

                    for key in runs1[0]:
                        writer2.writerow(key.split("-") + [diff1[i][key] for i in range(len(diff1)) if key in diff1[i]])   
                    for key in runs2[0]:
                        writer2.writerow(key.split("-") + [diff2[i][key] for i in range(len(diff2)) if key in diff2[i]])          

                    writer2.writerow([])
                    writer2.writerow(["t-Student Approximation of Mean"])

                    # compute bounds for mean values
                    quantile = t.ppf(0.99, run_iterations*P-1)                    

                    fig11, ax11 = plt.subplots()
                    fig12, ax12 = plt.subplots()
                    fig21, ax21 = plt.subplots()
                    fig22, ax22 = plt.subplots()
                    fig31, ax31 = plt.subplots()
                    fig32, ax32 = plt.subplots()

                    labels1 = []; labels2 = []
                    legend1 = []; legend2 = []

                    m_count0 = 0; m_count1 = 0
                    max1 = 0; max2 = 0

                    for comm1 in comm_methods:
                        for snd1 in send_methods[comm_methods[comm1]]:
                            for comm2 in comm_methods:
                                for snd2 in send_methods[comm_methods[comm2]]:
                                    if ((comm1 == "Peer2Peer" and snd1 == "Sync") or (comm2 == "Peer2Peer" and snd2 == "Sync")) and "{}-{}-{}-{}".format(comm1, snd1, comm2, snd2) in variance_collection:
                                        sizes = getSizes(opt, comm_methods[comm1], send_methods[comm_methods[comm1]][snd1], comm_methods[comm2], send_methods[comm_methods[comm2]][snd2], P_str, cuda_aware, subdir)
                                        x_vals = ConvertSizesToLabels(sizes)
                                        sd_arr = [np.sqrt(v) for v in variance_collection["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)]["Run complete"]]
                                        eps = [quantile*sd/np.sqrt(P*run_iterations) for sd in sd_arr]

                                        if comm1 == "Peer2Peer" and snd1 == "Sync":
                                            linestyle = "-"
                                            if comm2 == "All2All":
                                                linestyle = ":"
                                            if cuda_aware and snd2 == "Streams" and ((forward==1 and opt==1) or (forward==0 and opt==0)):
                                                writer2.writerow([comm1, snd1, comm2, snd2] + [runs1[ref_sizes.index(sizes[i])]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)]-eps[i] for i in range(len(sd_arr))])
                                                writer2.writerow([comm1, snd1, comm2, snd2] + [runs1[ref_sizes.index(sizes[i])]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)]+eps[i] for i in range(len(sd_arr))])

                                                next(ax12._get_lines.prop_cycler) 
                                                next(ax22._get_lines.prop_cycler) 
                                                next(ax22._get_patches_for_fill.prop_cycler)
                                                next(ax32._get_lines.prop_cycler) 
                                                next(ax32._get_patches_for_fill.prop_cycler)
                                                m_count0 += 1
                                            else:
                                                label, = ax12.plot(x_vals, [runs1[ref_sizes.index(sizes[i])]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] for i in range(len(sizes))], linestyle=linestyle, marker=markers[m_count0%len(markers)], zorder=3, linewidth=5, markersize=15)
                                                ax22.plot(x_vals, [diff1[ref_sizes.index(sizes[i])]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] for i in range(len(sizes))], linestyle=linestyle, marker=markers[m_count0%len(markers)], zorder=3, linewidth=5, markersize=15)
                                                ax22.errorbar(x_vals, [diff1[ref_sizes.index(sizes[i])]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] for i in range(len(sizes))], [eps[i] / min(runs1[ref_sizes.index(sizes[i])].values()) for i in range(0, len(eps))], fmt='.k', elinewidth=3, capsize=5)
                                                ax22.fill_between(x_vals, [diff1[ref_sizes.index(sizes[i])]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] - eps[i] / min(runs1[ref_sizes.index(sizes[i])].values()) for i in range(len(sizes))], [diff1[ref_sizes.index(sizes[i])]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] + eps[i] / min(runs1[ref_sizes.index(sizes[i])].values()) for i in range(len(sizes))], zorder=3, alpha=0.3)
                                                labels2.append(label)
                                                legend2.append("{}-{}".format(comm2, snd2))
                                                if snd2 != "MPI\_Type":
                                                    ax32.plot(x_vals, [diff1[ref_sizes.index(sizes[i])]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] for i in range(len(sizes))], linestyle=linestyle, marker=markers[m_count0%len(markers)], zorder=3, linewidth=5, markersize=15)
                                                    ax32.errorbar(x_vals, [diff1[ref_sizes.index(sizes[i])]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] for i in range(len(sizes))], [eps[i] / min(runs1[ref_sizes.index(sizes[i])].values()) for i in range(0, len(eps))], fmt='.k', elinewidth=3, capsize=5)
                                                    ax32.fill_between(x_vals, [diff1[ref_sizes.index(sizes[i])]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] - eps[i] / min(runs1[ref_sizes.index(sizes[i])].values()) for i in range(len(sizes))], [diff1[ref_sizes.index(sizes[i])]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] + eps[i] / min(runs1[ref_sizes.index(sizes[i])].values()) for i in range(len(sizes))], zorder=3, alpha=0.3)
                                                    # max1 = max(max([diff1[i]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] for i in range(len(sizes))]), max1)
                                                else:
                                                    next(ax32._get_lines.prop_cycler) 
                                                    next(ax32._get_patches_for_fill.prop_cycler)
                                                m_count0 += 1
                                                if comm2 == "Peer2Peer" and snd2 == "Sync":
                                                    label, = ax11.plot(x_vals, [runs2[ref_sizes.index(sizes[i])]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] for i in range(len(sizes))], linestyle=linestyle, marker=markers[m_count1%len(markers)], zorder=3, linewidth=5, markersize=15)
                                                    ax21.plot(x_vals, [diff2[ref_sizes.index(sizes[i])]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] for i in range(len(sizes))], linestyle=linestyle, marker=markers[m_count1%len(markers)], zorder=3, linewidth=5, markersize=15)
                                                    ax21.errorbar(x_vals, [diff2[ref_sizes.index(sizes[i])]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] for i in range(len(sizes))], [eps[i] / min(runs2[ref_sizes.index(sizes[i])].values()) for i in range(0, len(eps))], fmt='.k', elinewidth=3, capsize=5)
                                                    ax21.fill_between(x_vals, [diff2[ref_sizes.index(sizes[i])]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] - eps[i] / min(runs2[ref_sizes.index(sizes[i])].values()) for i in range(len(sizes))], [diff2[ref_sizes.index(sizes[i])]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] + eps[i] / min(runs2[ref_sizes.index(sizes[i])].values()) for i in range(len(sizes))], zorder=3, alpha=0.3)
                                                    ax31.plot(x_vals, [diff2[ref_sizes.index(sizes[i])]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] for i in range(len(sizes))], linestyle=linestyle, marker=markers[m_count1%len(markers)], zorder=3, linewidth=5, markersize=15)
                                                    ax31.errorbar(x_vals, [diff2[ref_sizes.index(sizes[i])]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] for i in range(len(sizes))], [eps[i] / min(runs2[ref_sizes.index(sizes[i])].values()) for i in range(0, len(eps))], fmt='.k', elinewidth=5, capsize=5)
                                                    ax31.fill_between(x_vals, [diff2[ref_sizes.index(sizes[i])]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] - eps[i] / min(runs2[ref_sizes.index(sizes[i])].values()) for i in range(len(sizes))], [diff2[ref_sizes.index(sizes[i])]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] + eps[i] / min(runs2[ref_sizes.index(sizes[i])].values()) for i in range(len(sizes))], zorder=3, alpha=0.3)
                                                    labels1.append(label)
                                                    legend1.append("{}-{}".format(comm1, snd1))
                                                    m_count1 += 1
                                                    # max2 = max(max([diff2[i]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] for i in range(len(sizes))]), max2)
                                        else:                    
                                            linestyle = "-"
                                            if comm1 == "All2All":
                                                linestyle = ":"                            
                                            if cuda_aware and snd1 == "Streams" and forward==1 and opt==1:
                                                next(ax11._get_lines.prop_cycler) 
                                                next(ax21._get_lines.prop_cycler) 
                                                next(ax21._get_patches_for_fill.prop_cycler)
                                                next(ax31._get_lines.prop_cycler) 
                                                next(ax31._get_patches_for_fill.prop_cycler)
                                                m_count1 += 1
                                            else:
                                                writer2.writerow([comm1, snd1, comm2, snd2] + [runs2[ref_sizes.index(sizes[i])]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)]-eps[i] for i in range(len(sd_arr))])
                                                writer2.writerow([comm1, snd1, comm2, snd2] + [runs2[ref_sizes.index(sizes[i])]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)]+eps[i] for i in range(len(sd_arr))])

                                                label, = ax11.plot(x_vals, [runs2[ref_sizes.index(sizes[i])]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] for i in range(len(sizes))], linestyle=linestyle, marker=markers[m_count1%len(markers)], zorder=3, linewidth=5, markersize=15)
                                                ax21.plot(x_vals, [diff2[ref_sizes.index(sizes[i])]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] for i in range(len(sizes))], linestyle=linestyle, marker=markers[m_count1%len(markers)], zorder=3, linewidth=5, markersize=15)
                                                ax21.errorbar(x_vals, [diff2[ref_sizes.index(sizes[i])]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] for i in range(len(sizes))], [eps[i] / min(runs2[ref_sizes.index(sizes[i])].values()) for i in range(0, len(eps))], fmt='.k', elinewidth=3, capsize=5)
                                                ax21.fill_between(x_vals, [diff2[ref_sizes.index(sizes[i])]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] - eps[i] / min(runs2[ref_sizes.index(sizes[i])].values()) for i in range(len(sizes))], [diff2[ref_sizes.index(sizes[i])]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] + eps[i] / min(runs2[ref_sizes.index(sizes[i])].values()) for i in range(len(sizes))], zorder=3, alpha=0.3)
                                                labels1.append(label)
                                                legend1.append("{}-{}".format(comm1, snd1))
                                                if snd1 != "MPI\_Type":
                                                    ax31.plot(x_vals, [diff2[ref_sizes.index(sizes[i])]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] for i in range(len(sizes))], linestyle=linestyle, marker=markers[m_count1%len(markers)], zorder=3, linewidth=5, markersize=15)
                                                    ax31.errorbar(x_vals, [diff2[ref_sizes.index(sizes[i])]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] for i in range(len(sizes))], [eps[i] / min(runs2[ref_sizes.index(sizes[i])].values()) for i in range(0, len(eps))], fmt='.k', elinewidth=3, capsize=5)
                                                    ax31.fill_between(x_vals, [diff2[ref_sizes.index(sizes[i])]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] - eps[i] / min(runs2[ref_sizes.index(sizes[i])].values()) for i in range(len(sizes))], [diff2[ref_sizes.index(sizes[i])]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] + eps[i] / min(runs2[ref_sizes.index(sizes[i])].values()) for i in range(len(sizes))], zorder=3, alpha=0.3)
                                                    # max2 = max(max([diff2[i]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] for i in range(len(sizes))]), max2)
                                                else:
                                                    next(ax31._get_lines.prop_cycler) 
                                                    next(ax31._get_patches_for_fill.prop_cycler)
                                                m_count1 += 1

                                    elif comm2 == "Peer2Peer" and snd2 == "Sync":
                                        next(ax11._get_lines.prop_cycler) 
                                        next(ax21._get_lines.prop_cycler) 
                                        next(ax21._get_patches_for_fill.prop_cycler)
                                        next(ax31._get_lines.prop_cycler) 
                                        next(ax31._get_patches_for_fill.prop_cycler)
                                        m_count1 += 1
                                    
                                    elif comm1 == "Peer2Peer" and snd1 == "Sync":
                                        next(ax12._get_lines.prop_cycler) 
                                        next(ax22._get_lines.prop_cycler) 
                                        next(ax22._get_patches_for_fill.prop_cycler)
                                        next(ax32._get_lines.prop_cycler) 
                                        next(ax32._get_patches_for_fill.prop_cycler)
                                        m_count0 += 1


                    # ax31.set_ylim(top=max2*1.1, bottom=1-(max2-1)*0.05)
                    # ax32.set_ylim(top=max1*1.1, bottom=1-(max1-1)*0.05)

                    for p in [[fig11, fig12, ax11, ax12], [fig21, fig22, ax21, ax22], [fig31, fig32, ax31, ax32]]:
                        fig1 = p[0]; fig2 = p[1]; ax1 = p[2]; ax2 = p[3]
                            
                        ax1.legend(labels1, legend1, prop={"size":(32 if Table else 25)})
                        ax2.legend(labels2, legend2, prop={"size":(32 if Table else 25)})
                        for ax in [ax1, ax2]:
                            if fig1 == fig11:
                                if not Table:
                                    ax.set_ylabel(r"Time [ms]", fontsize=36)
                                ax.set_yscale('symlog', base=10)
                            elif not Table:
                                ax.set_ylabel(r"Relative", fontsize=36)
                                
                            ax.grid(zorder=0, color="grey")

                            if not Table:
                                plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
                                ax.tick_params(axis='x', labelsize=30)
                                ax.tick_params(axis='y', labelsize=30)
                            else:
                                ax.tick_params(axis='x', labelsize=32)
                                ax.tick_params(axis='y', labelsize=32, pad=9)
                                ax.set_xticks(x_vals)
                                ax.set_xticklabels([x_vals[i] if i%3==0 else "" for i in range(len(x_vals))])
                            
                        fig1.set_size_inches(13, 8)
                        fig2.set_size_inches(13, 8)

                    path = ""
                    if not Table:
                        path = 'eval/{}/exact/plots_legend/{}_{}_{}'.format(join(prefix, subdir), opt, P_str, 1 if cuda_aware else 0)
                    else:
                        path = 'eval/{}/exact/plots_tables_legend/{}_{}_{}'.format(join(prefix, subdir), opt, P_str, 1 if cuda_aware else 0)
                    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
                    print(path)

                    fig11.savefig("{}/plot_comm1.png".format(path))
                    fig11.savefig("{}/plot_comm1.pdf".format(path), bbox_inches='tight')
                    fig21.savefig("{}/diff_comm1.png".format(path))
                    fig21.savefig("{}/diff_comm1.pdf".format(path), bbox_inches='tight')
                    fig31.savefig("{}/reduced_diff_comm1.png".format(path))
                    fig31.savefig("{}/reduced_diff_comm1.pdf".format(path), bbox_inches='tight')
                    fig12.savefig("{}/plot_comm2.png".format(path))
                    fig12.savefig("{}/plot_comm2.pdf".format(path), bbox_inches='tight')
                    fig22.savefig("{}/diff_comm2.png".format(path))
                    fig22.savefig("{}/diff_comm2.pdf".format(path), bbox_inches='tight')
                    fig32.savefig("{}/reduced_diff_comm2.png".format(path))
                    fig32.savefig("{}/reduced_diff_comm2.pdf".format(path), bbox_inches='tight')


                    if not Table:
                        path = 'eval/{}/exact/plots/{}_{}_{}'.format(join(prefix, subdir), opt, P_str, 1 if cuda_aware else 0)
                    else:
                        path = 'eval/{}/exact/plots_tables/{}_{}_{}'.format(join(prefix, subdir), opt, P_str, 1 if cuda_aware else 0)
                    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
                    
                    for ax in [ax11, ax12, ax21, ax22, ax31, ax32]:
                        ax.get_legend().remove()
                    
                    fig11.savefig("{}/plot_comm1.png".format(path))
                    fig11.savefig("{}/plot_comm1.pdf".format(path), bbox_inches='tight')
                    fig21.savefig("{}/diff_comm1.png".format(path))
                    fig21.savefig("{}/diff_comm1.pdf".format(path), bbox_inches='tight')
                    fig31.savefig("{}/reduced_diff_comm1.png".format(path))
                    fig31.savefig("{}/reduced_diff_comm1.pdf".format(path), bbox_inches='tight')
                    fig12.savefig("{}/plot_comm2.png".format(path))
                    fig12.savefig("{}/plot_comm2.pdf".format(path), bbox_inches='tight')
                    fig22.savefig("{}/diff_comm2.png".format(path))
                    fig22.savefig("{}/diff_comm2.pdf".format(path), bbox_inches='tight')
                    fig32.savefig("{}/reduced_diff_comm2.png".format(path))
                    fig32.savefig("{}/reduced_diff_comm2.pdf".format(path), bbox_inches='tight') 

                    plt.close()

        # Write approximations   
        pathlib.Path("eval/{}/approx/data".format(join(prefix, subdir))).mkdir(parents=True, exist_ok=True)
        with open('eval/{}/approx/data/data_{}_{}_{}.csv'.format(join(prefix, subdir), opt, P_str, 1 if cuda_aware else 0), mode='w') as out_file:
            writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            pathlib.Path("eval/{}/approx/proportions".format(join(prefix, subdir))).mkdir(parents=True, exist_ok=True)
            with open('eval/{}/approx/proportions/proportions_{}_{}_{}.csv'.format(join(prefix, subdir), opt, P_str, 1 if cuda_aware else 0), mode='w') as out_file1:
                writer1 = csv.writer(out_file1, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                pathlib.Path("eval/{}/approx/runs".format(join(prefix, subdir))).mkdir(parents=True, exist_ok=True)
                with open('eval/{}/approx/runs/runs_{}_{}_{}.csv'.format(join(prefix, subdir), opt, P_str, 1 if cuda_aware else 0), mode='w') as out_file2:
                    writer2 = csv.writer(out_file2, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    pathlib.Path("eval/{}/approx/sd".format(join(prefix, subdir))).mkdir(parents=True, exist_ok=True)
                    with open('eval/{}/approx/sd/sd_{}_{}_{}.csv'.format(join(prefix, subdir), opt, P_str, 1 if cuda_aware else 0), mode='w') as out_file3:
                        writer3 = csv.writer(out_file3, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                        sizes = getSizes(opt, 0, 0, 0, 0, P_str, cuda_aware, subdir)
                        runs = [{} for s in sizes]
                        writer2.writerow(["", "", "", ""] + sizes)

                        for comm1 in comm_methods:
                            for snd1 in send_methods[comm_methods[comm1]]:
                                for comm2 in comm_methods:
                                    for snd2 in send_methods[comm_methods[comm2]]:
                                        if "{}-{}-{}-{}".format(comm1, snd1, comm2, snd2) in data_collection:
                                            sizes0 = getSizes(opt, 0, 0, comm_methods[comm2], send_methods[comm_methods[comm2]][snd2], P_str, cuda_aware, subdir)
                                            sizes1 = getSizes(opt, comm_methods[comm1], send_methods[comm_methods[comm1]][snd1], 0, 0, P_str, cuda_aware, subdir)

                                            sizes = [x for x in sizes0 if x in sizes1]

                                            writer.writerow([comm1, snd1, comm2, snd2])
                                            writer.writerow([""] + sizes)
                                            writer1.writerow([comm1, snd1, comm2, snd2])
                                            writer1.writerow([""] + sizes)
                                            writer3.writerow([comm1, snd1, comm2, snd2])
                                            writer3.writerow([""] + sizes)

                                            data = data_collection["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)]
                                            variance = variance_collection["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)]
                                            proportions = getProportions(data, comm_methods[comm1], comm_methods[comm2], forward)

                                            # write data
                                            for d in data:
                                                if sum(data[d]) != 0:
                                                    writer.writerow(list([d])+list(data[d]))
                                            
                                            # write proportions
                                            for p in proportions:
                                                if p != "Run":
                                                    writer1.writerow([p] + [max(proportions[p][i] / proportions["Run"][i], 0) for i in range(len(proportions[p]))])
                                            writer1.writerow(["Run"] + [proportions["Run"][i] for i in range(len(proportions["Run"]))])

                                            # write run complete 
                                            writer2.writerow(list([comm1, snd1, comm2, snd2]) + list(data["Run complete"]))
                                            for i in range(len(sizes)):
                                                runs[i]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] = data["Run complete"][i]

                                            # write sd complete 
                                            for d in variance:
                                                if sum(variance[d]) != 0:
                                                    writer3.writerow(list([d])+list([np.sqrt(e) for e in variance[d]]))

                                            writer.writerow([])
                                            writer1.writerow([])   
                                            writer3.writerow([])

                        writer2.writerow([])
                        writer2.writerow(["Squared Difference to Minimum"])
                        diff = [{} for i in range(len(runs))]
                        for i in range(len(runs)):
                            for key in runs[i]:
                                diff[i][key] = (runs[i][key] - min(runs[i].values()))**2

                        for key in runs[0]:
                            writer2.writerow(key.split("-") + [diff[i][key] for i in range(len(sizes)) if key in diff[i]])    

                        writer2.writerow([])
                        writer2.writerow(["t-Student Approximation of Mean"])

                        for comm1 in comm_methods:
                            for snd1 in send_methods[comm_methods[comm1]]:
                                for comm2 in comm_methods:
                                    for snd2 in send_methods[comm_methods[comm2]]:
                                        if "{}-{}-{}-{}".format(comm1, snd1, comm2, snd2) in variance_collection:
                                            sizes = getSizes(opt, comm_methods[comm1], send_methods[comm_methods[comm1]][snd1], comm_methods[comm2], send_methods[comm_methods[comm2]][snd2], P_str, cuda_aware, subdir)
                                            sd_arr = [np.sqrt(v) for v in variance_collection["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)]["Run complete"]]
                                            # print("{}-{}-{}-{}".format(comm1, snd1, comm2, snd2))
                                            # print(variance_collection["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)]["Run complete"])
                                            # print(sd_arr)
                                            eps = [quantile*sd/np.sqrt(P*run_iterations) for sd in sd_arr]

                                            writer2.writerow([comm1, snd1, comm2, snd2] + [runs[i]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)]-eps[i] for i in range(len(sd_arr)) if i < len(runs) and "{}-{}-{}-{}".format(comm1, snd1, comm2, snd2) in runs[i]])
                                            writer2.writerow([comm1, snd1, comm2, snd2] + [runs[i]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)]+eps[i] for i in range(len(sd_arr)) if i < len(runs) and "{}-{}-{}-{}".format(comm1, snd1, comm2, snd2) in runs[i]])
                                        

def main():
    for c in range(0, 2):
        cuda_aware = True if c == 1 else False
        for forward in range(0, 2):
            forward = 1 - forward
            for opt in range(0, 2):
                subdir = "{}/pencil".format("forward" if forward==1 else "inverse")
                pathlib.Path("eval/{}".format(join(prefix, subdir))).mkdir(parents=True, exist_ok=True)
                
                partitions, partitions_strings = getPartitions(opt, cuda_aware, subdir)
                for i in range(len(partitions_strings)):
                    compareMethods(opt, partitions_strings[i], partitions[i], cuda_aware, forward, subdir)

if __name__ == "__main__":
    main()
