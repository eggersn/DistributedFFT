from os import listdir
from os.path import isfile, join
import re
import csv
import matplotlib.pyplot as plt

comm_methods = {"Peer2Peer": 0, "All2All": 1}
send_methods = [{"Sync": 0, "Streams": 1, "MPI_Type": 2}, {"Sync": 0, "MPI_Type": 2}]

prefix = "../../benchmarks_krypton/slab_z_then_yx"
opt = 1
P = 4
cuda_aware = False
prec = "double"

title = "Slab Decomposition (Krypton)"

count = 0
labels = []
legend_labels = []

with open('out.csv', mode='w') as out_file:
    writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    writer.writerow(["Comm_Method", "Send_Method", 32, 64, 128, 256, 512])

    for comm1 in comm_methods:
        for snd1 in send_methods[comm_methods[comm1]]:
            files = [f for f in listdir(prefix) if isfile(join(prefix, f)) and re.match("test_{}_{}_{}_\d*_{}_{}".format(opt, comm_methods[comm1], send_methods[comm_methods[comm1]][snd1], (1 if cuda_aware else 0), P), f)]
            
            x_vals = [int(f.split("_")[4]) for f in files]
            y_vals = [0 for x in x_vals]
            f_count = 0
            for f in files:
                with open(join(prefix, f)) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',')
                    line_counter = 0
                    for row in csv_reader:
                        if len(row) > 0 and row[0] == "Run complete":
                            y_vals[f_count] += sum([float(x) for x in row[1:-1]])
                            line_counter += 1
                    y_vals[f_count] /= (line_counter*P)
                    f_count += 1

            y_vals = [x for _, x in sorted(zip(x_vals, y_vals), key=lambda pair: pair[0])]
            x_vals = sorted(x_vals)
            writer.writerow([str(comm1), str(snd1)] + y_vals)

            print(count)
            print(comm1 + " " + snd1)
            print(x_vals)
            print(y_vals)
            print()

            count += 1







