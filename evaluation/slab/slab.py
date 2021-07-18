from os import listdir
from os.path import isfile, join
import re
import csv
import matplotlib.pyplot as plt

comm_methods = {"Peer2Peer": 0, "All2All": 1}
send_methods = [{"Sync": 0, "Streams": 1, "MPI_Type": 2}, {"Sync": 0, "MPI_Type": 2}]

prefix = "../../benchmarks_temp/slab_default"
opt = 0
P = 4
cuda_aware = True
prec = "double"

title = "Slab Decomposition (Krypton)"

count = 0
labels = []
legend_labels = []

sizes = ["256_256_256", "512_256_256", "256_512_256", "256_256_512"]

with open('out.csv', mode='w') as out_file:
    writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    writer.writerow(["Comm_Method", "Send_Method"] + sizes)

    for comm1 in comm_methods:
        for snd1 in send_methods[comm_methods[comm1]]:
            files = [f for f in listdir(prefix) if isfile(join(prefix, f)) and re.match("test_{}_{}_{}_\d*_\d*_\d*_{}_{}".format(opt, comm_methods[comm1], send_methods[comm_methods[comm1]][snd1], (1 if cuda_aware else 0), P), f)]
            
            x_vals = sizes
            mean_vals = [0 for x in x_vals]
            sd_vals = [0 for x in x_vals]
            f_count = 0
            for f in files:
                for i in range(0, len(sizes)):
                    if f.split("_")[4] + "_" + f.split("_")[5] + "_" + f.split("_")[6] == sizes[i]:
                        f_count = i

                with open(join(prefix, f)) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',')
                    line_counter = 0
                    for row in csv_reader:
                        if len(row) > 0 and row[0] == "Run complete":
                            mean_vals[f_count] += sum([float(x) for x in row[1:-1]])
                            line_counter += 1
                    mean_vals[f_count] /= (line_counter*P)
                    
                with open(join(prefix, f)) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',')
                    line_counter = 0
                    for row in csv_reader:
                        if len(row) > 0 and row[0] == "Run complete":
                            sd_vals[f_count] += (mean_vals[f_count]-sum([float(x) for x in row[1:-1]])/P)**2
                            line_counter += 1
                    sd_vals[f_count] /= (line_counter-1)

            writer.writerow([str(comm1), str(snd1)] + mean_vals)

            print(count)
            print(comm1 + " " + snd1)
            print(x_vals)
            print(mean_vals)
            print(sd_vals)
            print()

            count += 1







