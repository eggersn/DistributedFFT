from os import listdir
from os.path import isfile, join
import pathlib
import re
import csv
import matplotlib.transforms as bb
import numpy as np
from scipy.stats import t
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
import matplotlib as mpl
mpl.use("pgf")
import matplotlib.pyplot as plt
plt.rcParams.update({
  "text.usetex": True,
  "pgf.rcfonts": False,
  "font.size": 14,
  'pgf.rcfonts': False
})

comm_methods = {"Peer2Peer": 0, "All2All": 1}
send_methods = [{"Sync": 0, "Streams": 1, "MPI_Type": 2}, {"Sync": 0, "MPI_Type": 2}]
markers = ["D", "X", "o", "s", "v"]
linestyles = ["solid", "dotted", "dashed", "dashdot", (0, (3, 1, 1, 1, 1, 1))]

prefix = "benchmarks/bwunicluster/gpu8/small"
if len(sys.argv) > 1:
    prefix = "benchmarks" + str(sys.argv[1])

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
    if forward:
        return [d["1D FFT Z-Direction"][i] for i in range(length)]
    else:
        return [d["1D FFT Z-Direction"][i]-d["Second Transpose (Unpacking)"][i] for i in range(length)]

def getYFFTDuration(d, forward):
    length = len(d["Run complete"])
    return [d["1D FFT Y-Direction"][i]-d["First Transpose (Unpacking)"][i] for i in range(length)]


def getXFFTDuration(d, forward):
    length = len(d["Run complete"])
    if forward:
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
        if forward:
            fft_dur = getXFFTDuration(d, forward)
        else:
            fft_dur = getZFFTDuration(d, forward)
        return [d["Run complete"][i] - d["Second Transpose (First Send)"][i] - (d["Second Transpose (Unpacking)"][i] - d["Second Transpose (Finished Receive)"][i]) - fft_dur[i] for i in range(length)]
    else:
        return [d["Second Transpose (Finished All2All)"][i] - d["Second Transpose (Start All2All)"][i] for i in range(length)]

def getFirstPackingDuration(d, peer2peer, forward):
    length = len(d["Run complete"])
    if forward:
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

    proportions["First_Comm_Duration"] = np.array(getFirstCommDuration(data, comm1==0, forward))
    proportions["First_Packing_Duration"] = np.array(getFirstPackingDuration(data, comm1==0, forward))
    proportions["First_Unpacking_Duration"] = np.array(getFirstUnpackingDuration(data, comm1==0))
    proportions["First_Packing_Comm_Overlap"] = np.array(getFirstPackingCommOverlap(data, comm1==0))
    proportions["First_Unpacking_Comm_Overlap"] = np.array(getFirstUnpackingCommOverlap(data, comm1==0))
    proportions["Second_Comm_Duration"] = np.array(getSecondCommDuration(data, comm2==0, forward))
    proportions["Second_Packing_Duration"] = np.array(getSecondPackingDuration(data, comm2==0))
    proportions["Second_Unpacking_Duration"] = np.array(getSecondUnpackingDuration(data, comm2==0))
    proportions["Second_Packing_Comm_Overlap"] = np.array(getSecondPackingCommOverlap(data, comm2==0))
    proportions["Second_Unpacking_Comm_Overlap"] = np.array(getSecondUnpackingCommOverlap(data, comm2==0))
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
    
    pathlib.Path("evaluation/{}/exact/sd".format(join(prefix, subdir))).mkdir(parents=True, exist_ok=True)
    with open('evaluation/{}/exact/sd/sd_{}_{}_{}.csv'.format(join(prefix, subdir), opt, P_str, 1 if cuda_aware else 0), mode='w') as out_file:
        writer_sd = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for comm1 in comm_methods:
            for snd1 in send_methods[comm_methods[comm1]]:
                for comm2 in comm_methods:
                    for snd2 in send_methods[comm_methods[comm2]]:
                        if (comm1 == "Peer2Peer" and snd1 == "Sync") or (comm2 == "Peer2Peer" and snd2 == "Sync"):
                            sizes = getSizes(opt, comm_methods[comm1], send_methods[comm_methods[comm1]][snd1], comm_methods[comm2], send_methods[comm_methods[comm2]][snd2], P_str, cuda_aware, subdir)

                            data, sd = reduceTestcase(opt, comm_methods[comm1], send_methods[comm_methods[comm1]][snd1], comm_methods[comm2], send_methods[comm_methods[comm2]][snd2], P_str, P, cuda_aware, sizes, forward, subdir)

                            if data != {}:
                                data_collection["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] = data 
                                variance_collection["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] = [sd["Run complete"][i]**2 for i in range(len(sizes))]

                                # Write sd 
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
                                if forward:
                                    if comm1 == "Peer2Peer":
                                        first_sec = ["1D FFT Z-Direction", "First Transpose (First Send)", "First Transpose (Packing)", "First Transpose (Start Local Transpose)", "First Transpose (Start Receive)", "First Transpose (First Receive)", "First Transpose (Finished Receive)", "First Transpose (Unpacking)", "First Transpose (Send Complete)", "1D FFT Y-Direction"]
                                    else:
                                        first_sec = ["1D FFT Z-Direction", "First Transpose (Packing)", "First Transpose (Start All2All)", "First Transpose (Finished All2All)", "First Transpose (Unpacking)", "First Transpose (Send Complete)", "1D FFT Y-Direction"]
                                    
                                    if comm2 == "Peer2Peer":
                                        second_sec = ["Second Transpose (First Send)", "Second Transpose (Packing)", "Second Transpose (Start Local Transpose)", "Second Transpose (Start Receive)", "Second Transpose (First Receive)", "Second Transpose (Finished Receive)", "Second Transpose (Unpacking)", "1D FFT X-Direction", "Run complete"]
                                    else:
                                        second_sec = ["Second Transpose (Packing)", "Second Transpose (Start All2All)", "Second Transpose (Finished All2All)", "Second Transpose (Unpacking)", "1D FFT X-Direction", "Run complete"]

                                else:
                                    if comm1 == "Peer2Peer":
                                        first_sec = ["1D FFT X-Direction", "First Transpose (First Send)", "First Transpose (Packing)", "First Transpose (Start Local Transpose)", "First Transpose (Start Receive)", "First Transpose (First Receive)", "First Transpose (Finished Receive)", "First Transpose (Unpacking)", "First Transpose (Send Complete)", "1D FFT Y-Direction"]
                                    else:
                                        first_sec = ["1D FFT X-Direction", "First Transpose (Packing)", "First Transpose (Start All2All)", "First Transpose (Finished All2All)", "First Transpose (Unpacking)", "First Transpose (Send Complete)", "1D FFT Y-Direction"]

                                    if comm2 == "Peer2Peer":
                                        second_sec = ["Second Transpose (First Send)", "Second Transpose (Packing)", "Second Transpose (Start Local Transpose)", "Second Transpose (Start Receive)", "Second Transpose (First Receive)", "Second Transpose (Finished Receive)", "Second Transpose (Unpacking)", "1D FFT Z-Direction", "Run complete"]
                                    else:
                                        second_sec = ["Second Transpose (Packing)", "Second Transpose (Start All2All)", "Second Transpose (Finished All2All)", "Second Transpose (Unpacking)", "1D FFT Z-Direction", "Run complete"]                                


                                data_collection["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] = {}
                                for key in first_sec:
                                    data_collection["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)][key] = data_collection["{}-{}-Peer2Peer-Sync".format(comm1, snd1)][key]
                                for key in second_sec:
                                    length = min(len(data_collection["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)]["First Transpose (Send Complete)"]), len(data_collection["Peer2Peer-Sync-{}-{}".format(comm2, snd2)][key]))
                                    data_collection["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)][key] = np.array(data_collection["Peer2Peer-Sync-{}-{}".format(comm2, snd2)][key][:length]) - np.array(data_collection["Peer2Peer-Sync-{}-{}".format(comm2, snd2)]["First Transpose (Send Complete)"][:length]) + np.array(data_collection["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)]["First Transpose (Send Complete)"][:length])

        # Write ascertained results to csv files
        pathlib.Path("evaluation/{}/exact/data".format(join(prefix, subdir))).mkdir(parents=True, exist_ok=True)
        with open('evaluation/{}/exact/data/data_{}_{}_{}.csv'.format(join(prefix, subdir), opt, P_str, 1 if cuda_aware else 0), mode='w') as out_file:
            writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            pathlib.Path("evaluation/{}/exact/proportions".format(join(prefix, subdir))).mkdir(parents=True, exist_ok=True)
            with open('evaluation/{}/exact/proportions/proportions_{}_{}_{}.csv'.format(join(prefix, subdir), opt, P_str, 1 if cuda_aware else 0), mode='w') as out_file1:
                writer1 = csv.writer(out_file1, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                pathlib.Path("evaluation/{}/exact/runs".format(join(prefix, subdir))).mkdir(parents=True, exist_ok=True)
                with open('evaluation/{}/exact/runs/runs_{}_{}_{}.csv'.format(join(prefix, subdir), opt, P_str, 1 if cuda_aware else 0), mode='w') as out_file2:
                    writer2 = csv.writer(out_file2, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                    sizes = getSizes(opt, 0, 0, 0, 0, P_str, cuda_aware, subdir)
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
                                            writer1.writerow([p] + [max(proportions[p][i] / proportions["Run"][i], 0) for i in range(len(proportions[p]))])

                                        # write run complete 
                                        writer2.writerow([comm1, snd1, comm2, snd2] + data["Run complete"])
                                        if comm1 == "Peer2Peer" and snd1 == "Sync":
                                            for i in range(len(sizes)):
                                                runs1[i]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] = data["Run complete"][i]
                                        if comm2 == "Peer2Peer" and snd2 == "Sync":
                                            for i in range(len(sizes)):
                                                runs2[i]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] = data["Run complete"][i]  

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
                        writer2.writerow(key.split("-") + [diff1[i][key] for i in range(len(sizes)) if key in diff1[i]])   
                    for key in runs2[0]:
                        writer2.writerow(key.split("-") + [diff2[i][key] for i in range(len(sizes)) if key in diff2[i]])          

                    writer2.writerow([])
                    writer2.writerow(["t-Student Approximation of Mean"])

                    # compute bounds for mean values
                    quantile = t.ppf(0.99, run_iterations*P-1)                    

                    fig1, (ax11, ax12) = plt.subplots(1, 2)
                    fig2, (ax21, ax22) = plt.subplots(1, 2)
                    fig3, (ax31, ax32) = plt.subplots(1, 2)
                    title1 = r"Pencil Communication Methods [${}x{}$, {}{}]".format(P_str.split("_")[0], P_str.split("_")[1], "default" if opt==0 else "data realignment", ", CUDA-aware" if cuda_aware else "")
                    title2 = r"Pencil Communication Methods Proportions [${}x{}$, {}{}]".format(P_str.split("_")[0], P_str.split("_")[1], "default" if opt==0 else "data realignment", ", CUDA-aware" if cuda_aware else "")
                    title3 = r"Pencil Communication Methods Proportions [${}x{}$, {}{}]".format(P_str.split("_")[0], P_str.split("_")[1], "default" if opt==0 else "data realignment", ", CUDA-aware" if cuda_aware else "")
                    fig1.suptitle(title1, fontsize=20)
                    fig2.suptitle(title2, fontsize=20)
                    fig3.suptitle(title3, fontsize=20)
                    labels1 = []; labels2 = []
                    legend1 = []; legend2 = []

                    m_count0 = 0; m_count1 = 0

                    if not forward:
                        for ax in [[ax11, ax12], [ax21, ax22], [ax31, ax32]]:
                            a = ax[0]
                            ax[0] = ax[1]
                            ax[1] = a 

                    for comm1 in comm_methods:
                        for snd1 in send_methods[comm_methods[comm1]]:
                            for comm2 in comm_methods:
                                for snd2 in send_methods[comm_methods[comm2]]:
                                    if ((comm1 == "Peer2Peer" and snd1 == "Sync") or (comm2 == "Peer2Peer" and snd2 == "Sync")) and "{}-{}-{}-{}".format(comm1, snd1, comm2, snd2) in variance_collection:
                                        sizes = getSizes(opt, comm_methods[comm1], send_methods[comm_methods[comm1]][snd1], comm_methods[comm2], send_methods[comm_methods[comm2]][snd2], P_str, cuda_aware, subdir)
                                        x_vals = ConvertSizesToLabels(sizes)
                                        sd_arr = [np.sqrt(v) for v in variance_collection["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)]]
                                        eps = [quantile*sd/np.sqrt(P*run_iterations) for sd in sd_arr]

                                        if comm1 == "Peer2Peer" and snd1 == "Sync":
                                            writer2.writerow([comm1, snd1, comm2, snd2] + [runs1[i]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)]-eps[i] for i in range(len(sd_arr))])
                                            writer2.writerow([comm1, snd1, comm2, snd2] + [runs1[i]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)]+eps[i] for i in range(len(sd_arr))])

                                            label, = ax12.plot(x_vals, [runs1[i]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] for i in range(len(sizes))], linestyle=linestyles[m_count0], marker=markers[m_count0], zorder=3, linewidth=3, markersize=10)
                                            ax22.plot(x_vals, [diff1[i]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] for i in range(len(sizes))], linestyle=linestyles[m_count0], marker=markers[m_count0], zorder=3, linewidth=3, markersize=10)
                                            ax22.errorbar(x_vals, [diff1[i]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] for i in range(len(sizes))], [eps[i] / min(runs1[i].values()) for i in range(0, len(eps))], fmt='.k', elinewidth=3, capsize=5)
                                            ax22.fill_between(x_vals, [diff1[i]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] - eps[i] / min(runs1[i].values()) for i in range(len(sizes))], [diff1[i]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] + eps[i] / min(runs1[i].values()) for i in range(len(sizes))], zorder=3, alpha=0.3)
                                            if snd2 != "MPI_Type":
                                                ax32.plot(x_vals, [diff1[i]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] for i in range(len(sizes))], linestyle=linestyles[m_count0], marker=markers[m_count0], zorder=3, linewidth=3, markersize=10)
                                                ax32.errorbar(x_vals, [diff1[i]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] for i in range(len(sizes))], [eps[i] / min(runs1[i].values()) for i in range(0, len(eps))], fmt='.k', elinewidth=3, capsize=5)
                                                ax32.fill_between(x_vals, [diff1[i]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] - eps[i] / min(runs1[i].values()) for i in range(len(sizes))], [diff1[i]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] + eps[i] / min(runs1[i].values()) for i in range(len(sizes))], zorder=3, alpha=0.3)
                                            else:
                                                next(ax32._get_lines.prop_cycler) 
                                                next(ax32._get_patches_for_fill.prop_cycler)
                                            labels2.append(label)
                                            legend2.append("{}-{}".format(comm2, snd2))
                                            m_count0 += 1
                                            if comm2 == "Peer2Peer" and snd2 == "Sync":
                                                label, = ax11.plot(x_vals, [runs2[i]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] for i in range(len(sizes))], linestyle=linestyles[m_count1], marker=markers[m_count1], zorder=3, linewidth=3, markersize=10)
                                                ax21.plot(x_vals, [diff2[i]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] for i in range(len(sizes))], linestyle=linestyles[m_count1], marker=markers[m_count1], zorder=3, linewidth=3, markersize=10)
                                                ax21.errorbar(x_vals, [diff2[i]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] for i in range(len(sizes))], [eps[i] / min(runs2[i].values()) for i in range(0, len(eps))], fmt='.k', elinewidth=3, capsize=5)
                                                ax21.fill_between(x_vals, [diff2[i]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] - eps[i] / min(runs2[i].values()) for i in range(len(sizes))], [diff2[i]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] + eps[i] / min(runs2[i].values()) for i in range(len(sizes))], zorder=3, alpha=0.3)
                                                ax31.plot(x_vals, [diff2[i]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] for i in range(len(sizes))], linestyle=linestyles[m_count1], marker=markers[m_count1], zorder=3, linewidth=3, markersize=10)
                                                ax31.errorbar(x_vals, [diff2[i]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] for i in range(len(sizes))], [eps[i] / min(runs2[i].values()) for i in range(0, len(eps))], fmt='.k', elinewidth=3, capsize=5)
                                                ax31.fill_between(x_vals, [diff2[i]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] - eps[i] / min(runs2[i].values()) for i in range(len(sizes))], [diff2[i]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] + eps[i] / min(runs2[i].values()) for i in range(len(sizes))], zorder=3, alpha=0.3)
                                                labels1.append(label)
                                                legend1.append("{}-{}".format(comm1, snd1))
                                                m_count1 += 1

                                        else:
                                            writer2.writerow([comm1, snd1, comm2, snd2] + [runs2[i]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)]-eps[i] for i in range(len(sd_arr))])
                                            writer2.writerow([comm1, snd1, comm2, snd2] + [runs2[i]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)]+eps[i] for i in range(len(sd_arr))])
                                            label, = ax11.plot(x_vals, [runs2[i]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] for i in range(len(sizes))], linestyle=linestyles[m_count1], marker=markers[m_count1], zorder=3, linewidth=3, markersize=10)
                                            ax21.plot(x_vals, [diff2[i]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] for i in range(len(sizes))], linestyle=linestyles[m_count1], marker=markers[m_count1], zorder=3, linewidth=3, markersize=10)
                                            ax21.errorbar(x_vals, [diff2[i]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] for i in range(len(sizes))], [eps[i] / min(runs2[i].values()) for i in range(0, len(eps))], fmt='.k', elinewidth=3, capsize=5)
                                            ax21.fill_between(x_vals, [diff2[i]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] - eps[i] / min(runs2[i].values()) for i in range(len(sizes))], [diff2[i]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] + eps[i] / min(runs2[i].values()) for i in range(len(sizes))], zorder=3, alpha=0.3)
                                            if snd1 != "MPI_Type":
                                                ax31.plot(x_vals, [diff2[i]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] for i in range(len(sizes))], linestyle=linestyles[m_count1], marker=markers[m_count1], zorder=3, linewidth=3, markersize=10)
                                                ax31.errorbar(x_vals, [diff2[i]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] for i in range(len(sizes))], [eps[i] / min(runs2[i].values()) for i in range(0, len(eps))], fmt='.k', elinewidth=3, capsize=5)
                                                ax31.fill_between(x_vals, [diff2[i]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] - eps[i] / min(runs2[i].values()) for i in range(len(sizes))], [diff2[i]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] + eps[i] / min(runs2[i].values()) for i in range(len(sizes))], zorder=3, alpha=0.3)
                                            else:
                                                next(ax31._get_lines.prop_cycler) 
                                                next(ax31._get_patches_for_fill.prop_cycler)
                                            labels1.append(label)
                                            legend1.append("{}-{}".format(comm1, snd1))
                                            m_count1 += 1

                    for p in [[fig1, ax11, ax12], [fig2, ax21, ax22], [fig3, ax31, ax32]]:
                        fig = p[0]; ax1 = p[1]; ax2 = p[2]
                        if forward:
                            ax1.set_title("First Communication Phase", fontsize=18)
                            ax2.set_title("Second Communication Phase", fontsize=18)
                        else:
                            ax2.set_title("First Communication Phase", fontsize=18)
                            ax1.set_title("Second Communication Phase", fontsize=18)
                            
                        ax1.legend(labels1, legend1, prop={"size":22})
                        ax2.legend(labels2, legend2, prop={"size":22})
                        for ax in [ax1, ax2]:
                            if fig == fig1:
                                ax.set_ylabel("Time [ms]", fontsize=24)
                                ax.set_yscale('symlog', base=10)
                            else:
                                ax.set_ylabel("Proportion", fontsize=24)
                                
                            ax.grid(zorder=0, color="grey")
                            ax.tick_params(axis='x', labelsize=22)
                            ax.tick_params(axis='y', labelsize=22, pad=6)
                            plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
                        fig.set_size_inches(25, 8)

                    path = 'evaluation/{}/exact/plots/{}_{}_{}'.format(join(prefix, subdir), opt, P_str, 1 if cuda_aware else 0)
                    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
                    fig1.savefig("{}/plot.png".format(path), dpi=100)
                    fig1.savefig("{}/plot.pgf".format(path), dpi=100, bbox_inches='tight')
                    fig2.savefig("{}/diff.png".format(path), dpi=100)
                    fig2.savefig("{}/diff.pgf".format(path), dpi=100, bbox_inches='tight')
                    fig3.savefig("{}/reduced_diff.png".format(path), dpi=100)
                    fig3.savefig("{}/reduced_diff.pgf".format(path), dpi=100, bbox_inches='tight')
                    extent = ax11.get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
                    ax11.set_title("")
                    fig1.savefig("{}/plot_a.png".format(path), bbox_inches=bb.Bbox([extent.min-np.array([1,1]), extent.max+np.array([0.3, 0.3])]))
                    fig1.savefig("{}/plot_a.pgf".format(path), bbox_inches=bb.Bbox([extent.min-np.array([1,1]), extent.max+np.array([0.3, 0.3])]))
                    extent = ax12.get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
                    ax12.set_title("")
                    fig1.savefig("{}/plot_b.png".format(path), bbox_inches=bb.Bbox([extent.min-np.array([1,1]), extent.max+np.array([0.3, 0.3])]))
                    fig1.savefig("{}/plot_b.pgf".format(path), bbox_inches=bb.Bbox([extent.min-np.array([1,1]), extent.max+np.array([0.3, 0.3])]))
                    extent = ax21.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
                    ax21.set_title("")
                    fig2.savefig("{}/diff_a.png".format(path), bbox_inches=bb.Bbox([extent.min-np.array([1,1]), extent.max+np.array([0.3, 0.3])]))
                    fig2.savefig("{}/diff_a.pgf".format(path), bbox_inches=bb.Bbox([extent.min-np.array([1,1]), extent.max+np.array([0.3, 0.3])]))                    
                    extent = ax22.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
                    ax22.set_title("")
                    fig2.savefig("{}/diff_b.png".format(path), bbox_inches=bb.Bbox([extent.min-np.array([1,1]), extent.max+np.array([0.3, 0.3])]))
                    fig2.savefig("{}/diff_b.pgf".format(path), bbox_inches=bb.Bbox([extent.min-np.array([1,1]), extent.max+np.array([0.3, 0.3])]))  
                    extent = ax31.get_window_extent().transformed(fig3.dpi_scale_trans.inverted())
                    ax31.set_title("")
                    fig3.savefig("{}/reduced_diff_a.png".format(path), bbox_inches=bb.Bbox([extent.min-np.array([1,1]), extent.max+np.array([0.3, 0.3])]))                    
                    fig3.savefig("{}/reduced_diff_a.pgf".format(path), bbox_inches=bb.Bbox([extent.min-np.array([1,1]), extent.max+np.array([0.3, 0.3])]))                    
                    extent = ax32.get_window_extent().transformed(fig3.dpi_scale_trans.inverted())
                    ax32.set_title("")
                    fig3.savefig("{}/reduced_diff_b.png".format(path), bbox_inches=bb.Bbox([extent.min-np.array([1,1]), extent.max+np.array([0.3, 0.3])]))                    
                    fig3.savefig("{}/reduced_diff_b.pgf".format(path), bbox_inches=bb.Bbox([extent.min-np.array([1,1]), extent.max+np.array([0.3, 0.3])]))                                      
                    plt.close()

        # Write approximations   
        pathlib.Path("evaluation/{}/approx/data".format(join(prefix, subdir))).mkdir(parents=True, exist_ok=True)
        with open('evaluation/{}/approx/data/data_{}_{}_{}.csv'.format(join(prefix, subdir), opt, P_str, 1 if cuda_aware else 0), mode='w') as out_file:
            writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            pathlib.Path("evaluation/{}/approx/proportions".format(join(prefix, subdir))).mkdir(parents=True, exist_ok=True)
            with open('evaluation/{}/approx/proportions/proportions_{}_{}_{}.csv'.format(join(prefix, subdir), opt, P_str, 1 if cuda_aware else 0), mode='w') as out_file1:
                writer1 = csv.writer(out_file1, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                pathlib.Path("evaluation/{}/approx/runs".format(join(prefix, subdir))).mkdir(parents=True, exist_ok=True)
                with open('evaluation/{}/approx/runs/runs_{}_{}_{}.csv'.format(join(prefix, subdir), opt, P_str, 1 if cuda_aware else 0), mode='w') as out_file2:
                    writer2 = csv.writer(out_file2, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

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

                                        data = data_collection["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)]
                                        proportions = getProportions(data, comm_methods[comm1], comm_methods[comm2], forward)

                                        # write data
                                        for d in data:
                                            if sum(data[d]) != 0:
                                                writer.writerow(list([d])+list(data[d]))
                                        
                                        # write proportions
                                        for p in proportions:
                                            writer1.writerow([p] + [max(proportions[p][i] / proportions["Run"][i], 0) for i in range(len(proportions[p]))])

                                        # write run complete 
                                        writer2.writerow(list([comm1, snd1, comm2, snd2]) + list(data["Run complete"]))
                                        for i in range(len(sizes)):
                                            runs[i]["{}-{}-{}-{}".format(comm1, snd1, comm2, snd2)] = data["Run complete"][i]

                                        writer.writerow([])
                                        writer1.writerow([])   

                    writer2.writerow([])
                    writer2.writerow(["Squared Difference to Minimum"])
                    diff = [{} for i in range(len(runs))]
                    for i in range(len(runs)):
                        for key in runs[i]:
                            diff[i][key] = (runs[i][key] - min(runs[i].values()))**2

                    for key in runs[0]:
                        writer2.writerow(key.split("-") + [diff[i][key] for i in range(len(sizes)) if key in diff[i]])          
                                        

def main():
    for c in range(0, 2):
        cuda_aware = True if c == 1 else False
        for forward in range(0, 2):
            for opt in range(0, 2):
                subdir = "{}/pencil".format("forward" if forward==1 else "inverse")
                pathlib.Path("evaluation/{}".format(join(prefix, subdir))).mkdir(parents=True, exist_ok=True)
                
                partitions, partitions_strings = getPartitions(opt, cuda_aware, subdir)
                for i in range(len(partitions_strings)):
                    compareMethods(opt, partitions_strings[i], partitions[i], cuda_aware, forward, subdir)

if __name__ == "__main__":
    main()
