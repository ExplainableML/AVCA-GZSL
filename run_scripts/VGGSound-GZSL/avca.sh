#First stage training for main/cls features

python main.py --root_dir avgzsl_benchmark_datasets/VGGSound/ --feature_extraction_method main_features --input_size_audio 512 --input_size_video 512  --epochs 50 --lr_scheduler --dataset_name VGGSound --zero_shot_split main_split --AVCA --lr 0.001 --n_batches 500 --embeddings_hidden_size 512  --decoder_hidden_size 512 --embedding_dropout 0.0 --decoder_dropout 0.0 --additional_dropout 0.1 --depth_transformer 1 --additional_triplets_loss --first_additional_triplet 1  --second_additional_triplet 1 --reg_loss --momentum 0.1 --bs 64  --exp_name attention_vgg_val_main
python main.py --root_dir avgzsl_benchmark_datasets/VGGSound/ --feature_extraction_method cls_features --input_size_audio 128 --input_size_video 4096  --epochs 50 --lr_scheduler --dataset_name VGGSound --zero_shot_split cls_split --AVCA --lr 0.001 --n_batches 500 --embeddings_hidden_size 512  --decoder_hidden_size 512 --embedding_dropout 0.0 --decoder_dropout 0.0 --additional_dropout 0.0 --depth_transformer 1 --additional_triplets_loss --first_additional_triplet 1  --second_additional_triplet 1 --momentum 0.99 --reg_loss --exp_name attention_vgg_val_cls


#Second stage training for main/cls features

python main.py --root_dir avgzsl_benchmark_datasets/VGGSound/ --feature_extraction_method main_features --input_size_audio 512 --input_size_video 512 --epochs 50 --lr_scheduler --dataset_name VGGSound --zero_shot_split main_split --AVCA --lr 0.001 --retrain_all --n_batches 500 --embeddings_hidden_size 512  --decoder_hidden_size 512 --embedding_dropout 0.0 --decoder_dropout 0.0 --additional_dropout 0.1 --save_checkpoints --depth_transformer 1 --additional_triplets_loss --first_additional_triplet 1  --second_additional_triplet 1 --reg_loss --momentum 0.1 --bs 64 --exp_name attention_vgg_all_main
python main.py --root_dir avgzsl_benchmark_datasets/VGGSound/ --feature_extraction_method cls_features --input_size_audio 128 --input_size_video 4096 --epochs 50 --lr_scheduler --dataset_name VGGSound --zero_shot_split cls_split --AVCA --lr 0.001 --retrain_all --n_batches 500 --embeddings_hidden_size 512  --decoder_hidden_size 512 --embedding_dropout 0.0 --decoder_dropout 0.0 --additional_dropout 0.0 --save_checkpoints --depth_transformer 1 --additional_triplets_loss --first_additional_triplet 1  --second_additional_triplet 1 --momentum 0.99 --reg_loss --exp_name attention_vgg_all_cls



#Evaluation scripts
python get_evaluation.py --load_path_stage_A runs/attention_vgg_val_main --load_path_stage_B runs/attention_vgg_all_main  --dataset_name VGGSound --AVCA
python get_evaluation.py --load_path_stage_A runs/attention_vgg_val_cls --load_path_stage_B runs/attention_vgg_all_cls  --dataset_name VGGSound --AVCA
