from os import listdir
from os.path import isfile, join
import pathlib
import re
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

comm_methods = {"Peer2Peer": 0, "All2All": 1}
send_methods = [{"Sync": 0, "Streams": 1, "MPI_Type": 2}, {"Sync": 0, "MPI_Type": 2}]

prefix = "benchmarks/bwunicluster/old"
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
            labels.append(r"${}^2x{}$".format(dims[0], dims[2]))
        elif dims[1] == dims[2]:
            labels.append(r"${}x{}^2$".format(dims[0], dims[2]))
        else:
            labels.append(r"${}x{}x{}$".format(dims[0], dims[1], dims[2]))

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
        return [d["Transpose (Packing)"][i] - d["Transpose (First Send)"][i] for i in range(length)]
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

    for size in sizes:
        run_data, run_sd = reduceRun(opt, comm, snd, P, cuda_aware, size, subdir)
        if len(data.keys()) == 0:
            for key in run_data:
                data[key] = []
                sd[key] = []
        for key in run_data:
            data[key].append(run_data[key])
            sd[key].append(run_sd[key])

    return data, sd

def getProportions(data, seq, comm, forward):
    proportions = {}
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

    pathlib.Path("evaluation/{}/sd".format(join(prefix, subdir))).mkdir(parents=True, exist_ok=True)
    with open('evaluation/{}/sd/sd_{}_{}_{}.csv'.format(join(prefix, subdir), opt, P, 1 if cuda_aware else 0), mode='w') as out_file:
        writer_sd = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for comm in comm_methods:
            for snd in send_methods[comm_methods[comm]]:
                sizes = getSizes(opt, comm_methods[comm], send_methods[comm_methods[comm]][snd], P, cuda_aware, subdir)

                data, sd = reduceTestcase(opt, comm_methods[comm], send_methods[comm_methods[comm]][snd], P, cuda_aware, sizes, forward, seq, subdir)

                if data != {}:
                    data_collection["{}-{}".format(comm, snd)] = data 
                    variance_collection["{}-{}".format(comm, snd)] = [sd["Run complete"][i]**2 for i in range(len(sizes))]

                    # Write sd 
                    for d in sd:
                        if sum(data[d]) != 0:
                            writer_sd.writerow([d]+sd[d])

                    writer_sd.writerow([])

    pathlib.Path("evaluation/{}/data".format(join(prefix, subdir))).mkdir(parents=True, exist_ok=True)
    with open('evaluation/{}/data/data{}_{}_{}.csv'.format(join(prefix, subdir), opt, P, 1 if cuda_aware else 0), mode='w') as out_file:
        writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        pathlib.Path("evaluation/{}/proportions".format(join(prefix, subdir))).mkdir(parents=True, exist_ok=True)
        with open('evaluation/{}/proportions/proportions_{}_{}_{}.csv'.format(join(prefix, subdir), opt, P, 1 if cuda_aware else 0), mode='w') as out_file1:
            writer1 = csv.writer(out_file1, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            pathlib.Path("evaluation/{}/runs".format(join(prefix, subdir))).mkdir(parents=True, exist_ok=True)
            with open('evaluation/{}/runs/runs_{}_{}_{}.csv'.format(join(prefix, subdir), opt, P, 1 if cuda_aware else 0), mode='w') as out_file2:
                writer2 = csv.writer(out_file2, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                sizes = getSizes(opt, 0, 0, P, cuda_aware, subdir)
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
                            # proportions = getProportions(data, seq, comm_methods[comm], forward)

                            for d in data:
                                if sum(data[d]) != 0:
                                    writer.writerow([d]+data[d])
                            
                            # for p in proportions:
                            #     writer1.writerow([p] + [max(proportions[p][i] / proportions["Run"][i], 0) for i in range(len(proportions[p]))])

                            writer2.writerow([comm, snd] + data["Run complete"])
                            for i in range(len(sizes)):
                                runs[i]["{}-{}".format(comm, snd)] = data["Run complete"][i]  

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

                writer2.writerow([])
                writer2.writerow(["t-Student Approximation of Mean"])

                quantile = t.ppf(0.99, run_iterations*P-1)

                fig1, ax1 = plt.subplots()
                fig2, ax2 = plt.subplots()
                labels = []; legend = []

                for comm in comm_methods:
                    for snd in send_methods[comm_methods[comm]]:
                        if "{}-{}".format(comm, snd) in variance_collection:
                            sizes = getSizes(opt, comm_methods[comm], send_methods[comm_methods[comm]][snd], P, cuda_aware, subdir)
                            sd_arr = [np.sqrt(v) for v in variance_collection["{}-{}".format(comm, snd)]]
                            eps = [quantile*sd/np.sqrt(P*run_iterations) for sd in sd_arr]

                            writer2.writerow([comm, snd] + [runs[i]["{}-{}".format(comm, snd)]-eps[i] for i in range(len(sd_arr))])
                            writer2.writerow([comm, snd] + [runs[i]["{}-{}".format(comm, snd)]+eps[i] for i in range(len(sd_arr))])

                            x_vals = ConvertSizesToLabels(sizes)

                            label, = ax1.plot(x_vals, [runs[i]["{}-{}".format(comm, snd)] for i in range(len(sizes))], "D-", zorder=3, linewidth=3, markersize=10)
                            ax2.plot(x_vals, [np.sqrt(diff[i]["{}-{}".format(comm, snd)]) for i in range(len(sizes))], "D-", zorder=3, linewidth=3, markersize=10)
                            labels.append(label)
                            legend.append("{}-{}".format(comm, snd))

                ax1.set_title(r"Slab Communication Methods [{}, {}, {}{}]".format(P, "ZY_Then_X" if seq==0 else "Z_Then_YX", "default" if opt==0 else "opt 1", ", CUDA-aware" if cuda_aware else ""), fontsize=18)
                ax2.set_title(r"Slab Communication Methods Difference [{}, {}, {}{}]".format(P, "ZY_Then_X" if seq==0 else "Z_Then_YX", "default" if opt==0 else "opt 1", ", CUDA-aware" if cuda_aware else ""), fontsize=18)
                for p in [[fig1, ax1], [fig2, ax2]]:
                    fig = p[0]; ax = p[1]

                    ax.legend(labels, legend, prop={"size":16})
                    ax.grid(zorder=0, color="grey")
                    ax.set_yscale('symlog', base=10)
                    ax.set_ylabel("Time [ms]", fontsize=16)
                    ax.tick_params(axis='x', labelsize=14)
                    ax.tick_params(axis='y', labelsize=14)
                    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
                    fig.set_size_inches(12.5, 8)

                pathlib.Path("evaluation/{}/plots".format(join(prefix, subdir))).mkdir(parents=True, exist_ok=True)
                fig1.savefig('evaluation/{}/plots/plot_{}_{}_{}.png'.format(join(prefix, subdir), opt, P, 1 if cuda_aware else 0), dpi=100)
                fig2.savefig('evaluation/{}/plots/diff_{}_{}_{}.png'.format(join(prefix, subdir), opt, P, 1 if cuda_aware else 0), dpi=100)
                plt.close()

                
def main():
    for c in range(0, 2):
        cuda_aware = True if c == 1 else False
        for forward in range(0, 2):
            for seq in range(0, 2):
                for opt in range(0, 2):
                    subdir = "{}/slab_{}".format("forward" if forward==1 else "inverse", "default" if seq==0 else "z_then_yx")
                    pathlib.Path("evaluation/{}".format(join(prefix, subdir))).mkdir(parents=True, exist_ok=True)
                    
                    partitions = getPartitions(opt, cuda_aware, subdir)
                    for P in partitions:
                        compareMethods(opt, P, cuda_aware, forward, seq, subdir)

if __name__ == "__main__":
    main()
