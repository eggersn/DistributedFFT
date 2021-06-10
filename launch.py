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
            print("-> Executing test {}".format(test["name"]))
            ranks = 0
            if test["name"][0:6] == "Pencil":
                ranks = test["P1"] * test["P2"]
            else:
                ranks = test["P"]
            # One additional process for coordination
            if test["testcase"] == 1:
                ranks += 1
            command = "mpiexec -n " + str(ranks) + " " + data["additional-flags"] + " " + "tests_exec " + str(test["name"]) + " " + str(test["testcase"]) + " " 
            command += str(test["option"]) + " " + str(data["warmup-rounds"] + test["repetitions"]) + " " + str(s) + " " + str(s) + " " + str(s) + " " + str(test["cuda_aware"]) + " " + test["precision"]
                
            if test["name"][0:6] == "Pencil":
                command += " " + str(test["P1"]) + " " + str(test["P2"])

            os.system(command)
        print()
elif opt == 1:
    print("todo")
else:
    print("todo")


