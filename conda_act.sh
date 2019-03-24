#!/bin/bash
. /home/sankalp/anaconda3/bin/activate tf-gpu
cd /home/sankalp/models/research
export PYTHONPATH=$PYTHONPATH:/home/sankalp/models/research:/home/sankalp/models/research/slim
cd /home/sankalp/models/research/object_detection
python check1.py $1 $2 $3
