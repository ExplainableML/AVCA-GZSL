import argparse
import pathlib


def args_main(*args, **kwargs):
    parser = argparse.ArgumentParser(description="Explainable Audio Visual Low Shot Learning")

    ### Filesystem ###
    parser.add_argument(
        "--root_dir",
        help="Path to dataset directory. Expected subfolder structure: '{root_dir}/features/{feature_extraction_method}/{audio,video,text}'",
        required=True,
        type=pathlib.Path
    )
    parser.add_argument(
        "--feature_extraction_method",
        help="Name of folder containing respective extracted features. Has to match {feature_extraction_method} in --root_dir argument.",
        required=True,
        type=pathlib.Path
    )
    parser.add_argument(
        "--dropout_baselines",
        help="Dropout to use for baselines",
        default=0.2,
        type=float
    )
    parser.add_argument(
        "--dataset_name",
        help="Name of the dataset to use",
        choices=["AudioSetZSL", "VGGSound", "UCF", "ActivityNet"],
        default="AudioSetZSL",
        type=str
    )

    parser.add_argument(
        "--momentum",
        help="Momentum for batch norm",
        default = 0.99,
        type = float
    )


    parser.add_argument(
        "--zero_shot_split",
        help="Name of zero shot split to use.",
        choices=["", "main_split", "cls_split"],
        default=""
    )

    parser.add_argument(
        "--manual_text_word2vec",
        help="Flag to use the manual word2vec text embeddings. CARE: Need to create cache files again!",
        action="store_true"
    )

    parser.add_argument(
        "--val_all_loss",
        help="Validate loss with seen + unseen",
        action="store_true"
    )

    parser.add_argument(
        "--additional_triplets_loss",
        help="Flag for using more triplets loss",
        action="store_true"
    )

    parser.add_argument(
        "--reg_loss",
        help="Flag for setting the regularization loss",
        action="store_true"

    )

    parser.add_argument(
        "--cycle_loss",
        help="Flag for using cycle loss",
        action="store_true"
    )

    parser.add_argument(
        "--retrain_all",
        help="Retrain with all data from train and validation",
        action="store_true"
    )

    parser.add_argument(
        "--save_checkpoints",
        help="Save checkpoints of the model every epoch",
        action="store_true"
    )

    ### Development options ###
    parser.add_argument(
        "--debug",
        help="Run the program in debug mode",
        action="store_true"
    )
    parser.add_argument(
        "--verbose",
        help="Run verbosely",
        action="store_true",
    )
    parser.add_argument(
        "--debug_comment",
        help="Custom comment string for the summary writer",
        default="",
        type=str
    )
    parser.add_argument(
        "--epochs",
        help="Number of epochs",
        default=100,
        type=int
    )

    parser.add_argument(
        "--norm_inputs",
        help="Normalize inputs before model",
        action="store_true"
    )

    parser.add_argument(
        "--z_score_inputs",
        help="Z-Score standardize inputs before model",
        action="store_true"
    )

    ### Hyperparameters ###
    parser.add_argument(
        "--lr",
        help="Learning rate",
        default=3e-4,
        type=float
    )
    parser.add_argument(
        "--bs",
        help="Batch size",
        default=256,
        type=int
    )
    parser.add_argument(
        "--n_batches",
        help="Number of batches for the balanced batch sampler",
        default=250,
        type=int
    )
    parser.add_argument(
        "--input_size",
        help="Dimension of the extracted features",
        type=int,
        required=False,
    )
    parser.add_argument(
        "--input_size_audio",
        help="Dimension of the extracted audio features",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--input_size_video",
        help="Dimension of the extracted video features",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--embeddings_hidden_size",
        help="Hidden layer size for the embedding networks",
        default=1024,
        type=int
    )
    parser.add_argument(
        "--decoder_hidden_size",
        help="Hidden layer size for the decoder loss network",
        default=64,
        type=int
    )
    parser.add_argument(
        "--embedding_dropout",
        help="Dropout in the embedding networks",
        default=0.8,
        type=float
    )
    parser.add_argument(
        "--decoder_dropout",
        help="Dropout in the decoder loss network",
        default=0.5,
        type=float
    )
    parser.add_argument(
        "--embedding_use_bn",
        help="Use batchnorm in the embedding networks",
        action="store_true",
    )
    parser.add_argument(
        "--decoder_use_bn",
        help="Use batchnorm in the decoder network",
        action="store_true",
    )
    parser.add_argument(
        "--normalize_decoder_outputs",
        help="L2 normalize the outputs of the decoder",
        action="store_true"
    )
    parser.add_argument(
        "--margin",
        help="Margin for the contrastive loss calculation",
        default=1.,
        type=float
    )
    parser.add_argument(
        "--distance_fn",
        help="Distance function for the contrastive loss calculation",
        choices=["L2Loss", "SquaredL2Loss"],
        default="L2Loss",
        type=str
    )
    parser.add_argument(
        "--lr_scheduler",
        help="Use LR_scheduler",
        action="store_true",
    )

    # defaults
    parser.add_argument(
        "--seed",
        help="Random seed",
        default=42,
        type=int
    )
    parser.add_argument(
        "--dump_path",
        help="Path where to create experiment log dirs",
        default=pathlib.Path("."),
        type=pathlib.Path
    )
    parser.add_argument(
        "--device",
        help="Device to run on.",
        choices=["cuda", "cpu"],
        default="cuda"
    )

    parser.add_argument(
        "--baseline",
        help="Flag to use the baseline where we have two ALEs, one for each modality and we just try to push the modalities to text embeddings",
        action="store_true"
    )

    parser.add_argument(
        "--audio_baseline",
        help="Flag to use the audio baseline",
        action="store_true"
    )
    parser.add_argument(
        "--video_baseline",
        help="Flag to use the video baseline",
        action="store_true"

    )
    parser.add_argument(
        "--concatenated_baseline",
        help="Flag to use the concatenated baseline",
        action="store_true"

    )
    parser.add_argument(
        "--cjme",
        help="Flag to use the CJME baseline",
        action="store_true"
    )

    parser.add_argument(
        "--new_model",
        help="Flag to use the new model",
        action="store_true"
    )

    parser.add_argument(
        "--new_model_early_fusion",
        help="Flag to use the early fusion new model",
        action="store_true"
    )

    parser.add_argument(
        "--new_model_middle_fusion",
        help="Flag to set the middle fusion new model",
        action="store_true"
    )

    parser.add_argument(
        "--AVCA",
        help="Flag to set the attention to the new model",
        action="store_true"

    )

    parser.add_argument(
        "--new_model_attention_both_heads",
        help="Flag to set if attention should provide output from both branches",
        action="store_true"

    )

    parser.add_argument(
        "--depth_transformer",
        help="Flag to se the number of layers of the transformer",
        default=1,
        type=int
    )

    parser.add_argument(
        "--exp_name",
        help="Flag to set the name of the experiment",
        default="",
        type=str
    )
    parser.add_argument(
        "--ale",
        help="Flag to set the ale",
        action="store_true"
    )
    parser.add_argument(
        "--devise",
        help="Flag to set the devise model",
        action="store_true"
    )
    parser.add_argument(
        "--sje",
        help="Flag to set the sje model",
        action="store_true"
    )

    parser.add_argument(
        "--apn",
        help="Flag to set the apn model",
        action="store_true"
    )

    parser.add_argument(
        "--first_additional_triplet",
        help="flag to set the first pair of additional triplets",
        default=1,
        type=int
    )

    parser.add_argument(
        "--second_additional_triplet",
        help="flag to set the second pair of additional triplets",
        default=1,
        type=int

    )

    parser.add_argument(
        "--third_additional_triplet",
        help="flag to set the third pair of additional triplets",
        default=1,
        type=int
    )
    parser.add_argument(
        "--additional_dropout",
        help="flag to set the additional dropouts",
        default=0,
        type=float

    )
    return parser.parse_args(*args, **kwargs)


def args_eval():
    parser = argparse.ArgumentParser(description="Explainable Audio Visual Low Shot Learning [Evaluation]")
    parser.add_argument(
        "--load_path_stage_A",
        help="Path to experiment log folder of stage A",
        required=True,
        type=pathlib.Path
    )
    parser.add_argument(
        "--root_dir",
        help="Path to dataset directory. Expected subfolder structure: '{root_dir}/features/{feature_extraction_method}/{audio,video,text}'",
        type=pathlib.Path
    )

    parser.add_argument(
        "--load_path_stage_B",
        help="Path to experiment log folder of stage B",
        required=True,
        type=pathlib.Path
    )

    """
    parser.add_argument(
        "--weights_path",
        help="Path to trained model weights. If not stated, random weights will be used!",
        type=pathlib.Path
    )

    parser.add_argument(
        "--weights_path_stage_A",
        help="Path to trained model weights from stage A. If not stated, random weights will be used!",
        type=pathlib.Path
    )

    parser.add_argument(
        "--weights_path_stage_B",
        help="Path to trained model weights from stage B. If not stated, random weights will be used!",
        type=pathlib.Path
    )
    """
    parser.add_argument(
        "--eval_name",
        help="Evaluation name to be displayed in the final output string",
        type=str,
    )

    parser.add_argument(
        "--dataset_name",
        help="Name of the dataset to use",
        choices=["AudioSetZSL", "VGGSound", "UCF", "ActivityNet"],
        default="AudioSetZSL",
        type=str
    )

    parser.add_argument(
        "--bs",
        help="Batch size",
        default=256,
        type=int
    )
    parser.add_argument(
        "--num_workers",
        help="Number of dataloader workers",
        default=8,
        type=int
    )
    parser.add_argument(
        "--pin_memory",
        help="Flag for pin_memory in dataloader",
        default=True,
        type=bool
    )
    parser.add_argument(
        "--drop_last",
        help="Drop last batch in dataloader",
        default=True,
        type=bool
    )
    parser.add_argument(
        "--device",
        help="Device to run on.",
        choices=["cuda", "cpu"],
        default="cuda"
    )

    parser.add_argument(
        "--baseline",
        help="Flag for setting baseline",
        action="store_true"

    )

    parser.add_argument(
        "--audio_baseline",
        help="Flag to use the audio baseline",
        action="store_true"
    )
    parser.add_argument(
        "--video_baseline",
        help="Flag to use the video baseline",
        action="store_true"

    )

    parser.add_argument(
        "--concatenated_baseline",
        help="Flag to use the concatenated baseline",
        action="store_true"

    )

    parser.add_argument(
        "--new_model",
        help="Flag to use the new model",
        action="store_true"
    )

    parser.add_argument(
        "--new_model_early_fusion",
        help="Flag to use the early fusion new model",
        action="store_true"
    )

    parser.add_argument(
        "--new_model_middle_fusion",
        help="Flag to set the middle fusion new model",
        action="store_true"
    )

    parser.add_argument(
        "--AVCA",
        help="Flag to set the attention to the new model",
        action="store_true"

    )
    parser.add_argument(
        "--cjme",
        help="Flag to use the CJME baseline",
        action="store_true"
    )

    parser.add_argument(
        "--ale",
        help="Flag to set the ale",
        action="store_true"
    )
    parser.add_argument(
        "--devise",
        help="Flag to set the devise model",
        action="store_true"
    )
    parser.add_argument(
        "--sje",
        help="Flag to se the sje model",
        action="store_true"
    )
    parser.add_argument(
        "--apn",
        help="flag to set apn model",
        action="store_true"
    )

    parser.add_argument(
        "--save_performances",
        help="Save class performances to disk",
        action="store_true"
    )

    return parser.parse_args()
