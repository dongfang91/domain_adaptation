#!/bin/bash
# Your job will use 1 node, 7 cores, and 42gb of memory total.
#PBS -q windfall
#PBS -l select=1:ncpus=28:mem=168gb:pcmem=8gb:ngpus=1
### Specify a name for the job
#PBS -N newkeras_colon_portion_9
### Specify the group name
#PBS -W group_list=nlp
### Used if job requires partial node only
#PBS -l place=pack:exclhost
### CPUtime required in hhh:mm:ss.
### Leading 0's can be omitted e.g 48:0:0 sets 48 hours
#PBS -l cput=4032:00:00
### Walltime is created by cputime divided by total cores.
### This field can be overwritten by a longer time
#PBS -l walltime=144:00:00
#PBS -e /home/u25/dongfangxu9/domain_adaptation/log/colon_9_err
#PBS -o /home/u25/dongfangxu9/domain_adaptation/log/colon_9_output

module load python/3.6/3.6.5
module load cuda80/neuralnet/6/6.0
module load cuda80/toolkit/8.0.61
source /home/u25/dongfangxu9/df36/bin/activate

cd $PBS_O_WORKDIR

THEANO_FLAGS='gcc.cxxflags=-march=corei7,base_compiledir=/home/u25/dongfangxu9/.theano/theano25' python3.6 trained_on_colon_portion_date_newkeras.py
