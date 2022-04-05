#First stage training
python vae_gan_d2_xu_fsl.py --root_dir avgzsl_benchmark_datasets/VGGSound/ --feature_extraction_method main_features --input_size_audio 512 --input_size_video 512  --epochs 50 --lr_scheduler --dataset_name VGGSound --zero_shot_split main_split  --lr 0.001 --n_batches 500  --reg_loss --syn_num 10000 > f_vaegan_vggsound_val.txt
#Second stage training
python vae_gan_d2_xu_fsl.py --root_dir avgzsl_benchmark_datasets/VGGSound/ --feature_extraction_method main_features --input_size_audio 512 --input_size_video 512 --epochs 50 --lr_scheduler --dataset_name VGGSound --zero_shot_split main_split   --lr 0.001 --retrain_all --save_checkpoints --n_batches 500 --syn_num 10000 > f_vaegan_vggsound_all.txt

