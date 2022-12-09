#!/bin/bash
#condapath=$(conda info | grep -i 'base environment'|cut -d ":" -f 2 | cut -d " " -f 2)
condapath=/storage/ahmetyi/anaconda3
source $condapath/etc/profile.d/conda.sh
conda activate transformerFL
"$@"
conda deactivate
