from os import listdir
from os.path import isfile, join
import re
import csv
import matplotlib.pyplot as plt

comm_methods = {"Peer2Peer": 0, "All2All": 1}
send_methods = [{"Sync": 0, "Streams": 1, "MPI_Type": 2}, {"Sync": 0, "MPI_Type": 2}]

prefix = "benchmarks/home/slab_default_forward"

forward = True
seq = 0

opt = 0
P = 4
cuda_aware = False
prec = "double"

def getSizes(opt, comm, snd, P, cuda_aware):
    files = [f for f in listdir(prefix) if isfile(join(prefix, f)) and re.match("test_{}_{}_{}_\d*_\d*_\d*_{}_{}".format(opt, comm, snd, (1 if cuda_aware else 0), P), f)]
    sizes = []
    for f in files:
        sizes.append(f.split("_")[4] + "_" + f.split("_")[5] + "_" + f.split("_")[6])
    sizes.sort(key=lambda e: int(e.split("_")[0]))
    sizes.sort(key=lambda e: int(e.split("_")[1]))
    sizes.sort(key=lambda e: int(e.split("_")[2]))
    return sizes

def reduceRun(opt, comm, snd, P, cuda_aware, size):
    file = join(prefix, "test_{}_{}_{}_{}_{}_{}.csv".format(opt, comm, snd, size, (1 if cuda_aware else 0), P))
    
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        data = {}
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

        return data  

def getFFTDuration(d):
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

def getCommDuration(d, peer2peer):
    length = len(d["init"])
    if peer2peer:
        fft_dur = getFFTDuration(d)
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

def getPackingDuration(d):
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

def reduceTestcase(opt, comm, snd, P, cuda_aware, sizes):
    data = {}
    proportions = {}
    for size in sizes:
        run_data = reduceRun(opt, comm, snd, P, cuda_aware, size)
        if len(data.keys()) == 0:
            for key in run_data:
                data[key] = []
        for key in run_data:
            data[key].append(run_data[key])

    proportions["FFT_Duration"] = getFFTDuration(data)
    proportions["Comm_Duration"] = getCommDuration(data, comm==0)
    proportions["Packing_Duration"] = getPackingDuration(data)
    proportions["Unpacking_Duration"] = getUnpackingDuration(data, comm==0)
    proportions["Packing_Comm_Overlap"] = getPackingCommOverlap(data, comm==0)
    proportions["Unpacking_Comm_Overlap"] = getUnpackingCommOverlap(data, comm==0)
    proportions["Run"] = data["Run complete"]

    return data, proportions

def compareMethods(opt, P, cuda_aware):
    with open('evaluation/slab/data.csv', mode='w') as out_file:
        writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        with open('evaluation/slab/proportions.csv', mode='w') as out_file1:
            writer1 = csv.writer(out_file1, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            for comm in comm_methods:
                for snd in send_methods[comm_methods[comm]]:
                    sizes = getSizes(opt, comm_methods[comm], send_methods[comm_methods[comm]][snd], P, cuda_aware)
                    writer.writerow([comm, snd])
                    writer.writerow([""] + sizes)
                    writer1.writerow([comm, snd])
                    writer1.writerow([""] + sizes)

                    data, proportions = reduceTestcase(opt, comm_methods[comm], send_methods[comm_methods[comm]][snd], P, cuda_aware, sizes)
                    for d in data:
                        if sum(data[d]) != 0:
                            writer.writerow([d]+data[d])
                    
                    for p in proportions:
                        writer1.writerow([p] + [max(proportions[p][i] / proportions["Run"][i], 0) for i in range(len(proportions[p]))])

                    writer.writerow([])
                    writer1.writerow([])

def main():
    compareMethods(opt, P, cuda_aware)

if __name__ == "__main__":
    main()
