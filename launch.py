import json
import os, sys

with open('job.json') as f:
  data = json.load(f)

def menu_main():
    valid = False
    os.system("clear")
    while not valid:
        print("Select a Category:\n" +
        "-" * 35 + "\n" + 
        "[0] Run Specified Job (job.json)\n" +
        "[1] Pencil Decomposition \n" + 
        "[2] Slab Decomposition \n\n" +
        "Selection: ", end='')
        try: 
            opt = int(input())
            if opt >= 0 and opt <= 2:
                return opt    
        except ValueError: 
            os.system("clear")
            print("Please enter an integer!\n")
        except KeyboardInterrupt:
            print("\nexiting...")
            sys.exit()

    return 0

opt = menu_main()
os.chdir("build")
print()
if opt == 0:
    for s in data["size"]:
        print("Starting computation for size {}".format(s))
        for test in data["tests"]:
            ranks = 0
            if test["name"][0:6] == "Pencil":
                print("-> Executing test {} \n   (size: {}, P1: {}, P2: {}, cuda_aware: {}, precision: {})".format(test["name"], s, test["P1"], test["P2"], str(test["cuda_aware"]).lower(), test["precision"]))
                ranks = test["P1"] * test["P2"]
            else:
                print("-> Executing test {} \n   (size: {}, P: {}, cuda_aware: {}, precision: {})".format(test["name"], s, test["P"], str(test["cuda_aware"]).lower(), test["precision"]))
                ranks = test["P"]
            # One additional process for coordination
            if test["testcase"] == 1:
                ranks += 1
            command = "mpiexec -n " + str(ranks) + " " + data["additional-flags"] + " " + "tests_exec " + str(test["name"]) + " " + str(test["testcase"]) + " " 
            command += str(test["option"]) + " " + str(data["warmup-rounds"] + test["repetitions"]) + " " + str(s) + " " + str(s) + " " + str(s) + " " + str(test["cuda_aware"]).lower() + " " + test["precision"]
                
            benchmarkfile = test["name"].lower()
            if test["name"][0:6] == "Pencil":
                command += " " + str(test["P1"]) + " " + str(test["P2"])
                benchmarkfile = "pencil"

            if test["option"] != 0:
                benchmarkfile += "_opt" + str(test["option"])

            os.system(command)

            if not os.path.isdir("../benchmarks/" + benchmarkfile):
                os.system("mkdir ../benchmarks/" + benchmarkfile)

            with open("../benchmarks/" + benchmarkfile + ".csv", 'r') as f_in:
                f_data = f_in.read().splitlines(True)
                index = [i for i, n in enumerate(f_data) if n == "\n"][data["warmup-rounds"]-1]
            if test["name"][0:6] == "Pencil":
                os.system("mv ../benchmarks/" + benchmarkfile + ".csv ../benchmarks/" + benchmarkfile + "/test_" + str(s) + "_" + str(test["P1"]) + "_" + str(test["P2"]) + "_" + str(test["cuda_aware"]).lower() + "_" + test["precision"])
                with open("../benchmarks/" + benchmarkfile + "/test_" + str(s) + "_" + str(test["P1"]) + "_" + str(test["P2"]) + "_" + str(test["cuda_aware"]).lower() + "_" + test["precision"], 'w') as f_out:
                    f_out.writelines(f_data[0])
                    f_out.writelines(f_data[index+1:])
            else:
                os.system("mv ../benchmarks/" + benchmarkfile + ".csv ../benchmarks/" + benchmarkfile + "/test_" + str(s) + "_" + str(test["P"]) + "_" + str(test["cuda_aware"]).lower() + "_" + test["precision"])
                with open("../benchmarks/" + benchmarkfile + "/test_" + str(s) + "_" + str(test["P"]) + "_" + str(test["cuda_aware"]).lower() + "_" + test["precision"], 'w') as f_out:
                    f_out.writelines(f_data[0])
                    f_out.writelines(f_data[index+1:])
            
        print()
elif opt == 1:
    print("todo")
else:
    print("todo")


