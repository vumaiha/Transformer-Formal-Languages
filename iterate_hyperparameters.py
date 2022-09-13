import subprocess
import sys

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

def combine(elems):
    if len(elems) == 0:
        return [[]]
    return [[x, *y] for x in elems[0] for y in combine(elems[1:])]

combinations = combine(iteration_list)
for combination in combinations:
    command = PYTHON_COMMAND+" -m src.main -mode train -dataset " + DATASET
    name =""
    for par in combination:
        par = par.split("$$")
        if par[0]=="run_params":
            name += " " + par[1]
        else:
            name += " -" + par[0] + " " + par[1]
    command += " -run_name " + name.strip().replace(" ", "_").replace("-","").replace("_run_params","") + name  + " -gpu 0"
    print("Running: " + command)
    out, err, return_code = run_cmd(command.split(" "))
    
    print(out)
    print(err)
    print(return_code)
    exit(0)



#run_command = "python -m src.main -mode train -run_name testrun -dataset Tomita-4 -model_type SAN -gpu 0"
