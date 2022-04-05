#!/bin/bash/
cd selavi_distributed_orig/
source /home/lriesch/miniconda3/bin/activate lab_vid
export CUDA_VISIBLE_DEVICES=2,3
export OMP_NUM_THREADS=32
python3 get_clusters.py --root_dir /home/lriesch/master_thesis/dat/AudioSetZSL/ASZSL_selavi/ --weights_path /shared-network/lriesch/models/selavi_vgg_sound.pth --mode train --pretrained True --aud_sample_rate 44100 --use_mlp False --dataset audioset_zsl --headcount 10 --exp_desc as_zsl_train_classes1024_numsecaud8 --num_sec_aud 8 --batch_size 36
