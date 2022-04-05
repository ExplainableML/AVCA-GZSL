#First stage training
python vae_gan_d2_xu_fsl.py --root_dir avgzsl_benchmark_datasets/UCF --feature_extraction_method main_features --input_size_audio 512 --input_size_video 512  --epochs 50 --lr_scheduler --dataset_name UCF --zero_shot_split main_split  --lr 0.001 --n_batches 500 --syn_num 1000 > f_vaegan_ucf_val.txt
#Second stage training
python vae_gan_d2_xu_fsl.py --root_dir avgzsl_benchmark_datasets/UCF --feature_extraction_method main_features --input_size_audio 512 --input_size_video 512  --epochs 50 --lr_scheduler --dataset_name UCF --zero_shot_split main_split   --lr 0.001 --retrain_all --save_checkpoints --n_batches 500 --syn_num 1000 > f_vaegan_ucf_all.txt
