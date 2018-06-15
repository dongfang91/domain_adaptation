#!/bin/bash
#BSUB -n 16
#BSUB -R gpu
#BSUB -q "standard"
#BSUB -o log/%J.out
#BSUB -e log/%J.err
#BSUB -x
module load python/3.5.2
module load cuda/7.5.18
source ~/xdf/bin/activate
python3 span_level_performance_med_new.py
