#!/bin/bash
job_name="training_news"
# Your job will use 1 node, 7 cores, and 42gb of memory total.
#PBS -q standard
#PBS -l select=1:ncpus=28:mem=168gb:pcmem=6gb:ngpus=1
### Specify the group name
#PBS -W group_list=nlp
### Used if job requires partial node only
#PBS -l place=pack:exclhost
### CPUtime required in hhh:mm:ss.
### Leading 0's can be omitted e.g 48:0:0 sets 48 hours
#PBS -l cput=999:59:59
### Walltime is created by cputime divided by total cores.
### This field can be overwritten by a longer time
#PBS -l walltime=192:59:59
#PBS -e /home/u25/dongfangxu9/domain_adaptation/log/err+$job_name
#PBS -o /home/u25/dongfangxu9/domain_adaptation/log/out+$job_name
### Specify a name for the job
#PBS -N $job_name

module load python/3.6/3.6.5
module load cuda80/neuralnet/5/5.1
module load cuda80/toolkit/8.0.61

source /home/u25/dongfangxu9/df36/bin/activate

cd $PBS_O_WORKDIR
python3.6 trained_on_news_newkeras.py

