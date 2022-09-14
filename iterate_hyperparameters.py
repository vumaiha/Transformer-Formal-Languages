import subprocess
import sys
import json

# This scripts runs the configurations given in the following 
# list of dictionaries
DATASET="Tomita-4"
PYTHON_COMMAND=sys.executable

hyperparameters={
        "d_model": {"begin":2, "end":32},
        "depth": {"begin":1, "end":4},
        "heads": {"begin":1, "end":4}, 
        "lr": [0.01,0.001],
        "run_params": ["-model_type SAN", "-model_type SAN -pos_encode","-model_type SAN-Rel"]
        }


def run_cmd(cmd):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT )
    out, err = p.communicate()
    return (out,err, p.returncode)

# Find the specific key's value recursively
# From : https://stackoverflow.com/questions/21028979/how-to-recursively-find-specific-key-in-nested-json
def item_generator(json_input, lookup_key):
    if isinstance(json_input, dict):
        for k, v in json_input.items():
            if k == lookup_key:
                yield v
            else:
                yield from item_generator(v, lookup_key)
    elif isinstance(json_input, list):
        for item in json_input:
            yield from item_generator(item, lookup_key)


def combine(elems):
    if len(elems) == 0:
        return [[]]
    return [[x, *y] for x in elems[0] for y in combine(elems[1:])]

results_file = open ("results.tsv", "w")
for par in hyperparameters:
    results_file.write(par + "\t")
results_file.write("max_val_acc_bin0\tmax_val_acc_bin1\n")

# Iterate over hyperparameter combinations
iteration_dict = {}
for par in hyperparameters:
    item = hyperparameters[par]
    
    if type(item) == dict:
        if "begin" in item and "end" in item:
            iteration_dict[par] =[par+"$$" +str(i) for i in list(range(item["begin"], item["end"]+1))]

    elif type(item) == list:
        iteration_dict[par]=[ par+"$$"+str(i) for i in item]
iteration_list=[]
for par in iteration_dict:
    iteration_list.append(iteration_dict[par])

combinations = combine(iteration_list)
for combination in combinations:
    command = PYTHON_COMMAND+" -m src.main -mode train -dataset " + DATASET
    name =""
    config=dict()
    for par in combination:
        par = par.split("$$")
        if par[0]=="run_params":
            name += " " + par[1]
        else:
            name += " -" + par[0] + " " + par[1]
        config[par[0]]=par[1]
    for par in hyperparameters:
        results_file.write(config[par] + "\t")

    run_name = name.strip().replace(" ", "_").replace("-","").replace("_run_params","")
    command += " -epochs 1 -run_name " + run_name + name + " -gpu 0"
    print("Running: " + command)
    out, err, return_code = run_cmd(command.split(" "))
    out = str(out)
    err = str(err)

    # If there is no proper output:
    if return_code!= 0:
        print(err)
        print ("No output for the command : " + command)
        continue

    # Try to get the saved file name
    filename=""
    index1=out.find("Scores saved at ")
    if index1!= -1:
        index2=out.find("\\n", index1)
        filename=out[index1+16:index2].strip()

    # Read the file
    if filename!="":
        f = open(filename)
        data = json.load(f)
        data = data[run_name]
        gen=item_generator(data,"max_val_acc_bin0")

        num=0
        for i in gen:
            max_val_acc_bin0 = i
            num +=1
        if num != 1:
            print("The is no or multiple max_val_acc_bin0 values")
            results_file.write("NA\t")
            continue

        results_file.write(str(max_val_acc_bin0) + "\t")

        gen=item_generator(data,"max_val_acc_bin1")
        num=0
        for i in gen:
            max_val_acc_bin1 = i
            num+=1
        if num != 1:
            print("The is no or multiple max_val_acc_bin1 values")
            results_file.write("NA\n")
            continue

        results_file.write(str(max_val_acc_bin1) + "\n")

    else:
        prin(err)
        print ("No output for the command : " + command)
        continue


#run_command = "python -m src.main -mode train -run_name testrun -dataset Tomita-4 -model_type SAN -gpu 0"
