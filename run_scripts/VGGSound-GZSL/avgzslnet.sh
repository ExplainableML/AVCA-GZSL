#First stage training
python main.py --root_dir avgzsl_benchmark_datasets/VGGSound/ --feature_extraction_method main_features --input_size_audio 512 --input_size_video 512 --lr_scheduler --dataset_name VGGSound --zero_shot_split main_split --epochs 50 --lr 0.001 --n_batches 500 --decoder_use_bn --embedding_use_bn  --normalize_decoder_outputs --exp_name avgzslnet_vggsound_val_main
python main.py --root_dir avgzsl_benchmark_datasets/VGGSound/ --feature_extraction_method cls_features --input_size_audio 128 --input_size_video 4096 --lr_scheduler --dataset_name VGGSound --zero_shot_split cls_split --epochs 50 --lr 0.001 --n_batches 500 --decoder_use_bn --embedding_use_bn  --normalize_decoder_outputs  --exp_name avgzslnet_vggsound_val_cls
#Second stage training
python main.py --root_dir avgzsl_benchmark_datasets/VGGSound/ --feature_extraction_method main_features --input_size_audio 512 --input_size_video 512 --lr_scheduler --dataset_name VGGSound --zero_shot_split main_split --epochs 50 --lr 0.001 --n_batches 500 --decoder_use_bn --embedding_use_bn  --normalize_decoder_outputs  --retrain_all  --save_checkpoints --exp_name avgzslnet_vggsound_all_main
python main.py --root_dir avgzsl_benchmark_datasets/VGGSound/ --feature_extraction_method cls_features --input_size_audio 128 --input_size_video 4096 --lr_scheduler --dataset_name VGGSound --zero_shot_split cls_split --epochs 50 --lr 0.001 --n_batches 500 --decoder_use_bn --embedding_use_bn   --normalize_decoder_outputs  --retrain_all  --save_checkpoints --exp_name avgzslnet_vggsound_all_cls


#Evaluation scripts
python get_evaluation.py --load_path_stage_A runs/avgzslnet_vggsound_val_main --load_path_stage_B runs/avgzslnet_vggsound_all_main  --dataset_name VGGSound
python get_evaluation.py --load_path_stage_A runs/avgzslnet_vggsound_val_cls --load_path_stage_B runs/avgzslnet_vggsound_all_cls  --dataset_name VGGSound  
