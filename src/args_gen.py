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
        "--dataset_name",
        help="Name of the dataset to use",
        choices=["AudioSetZSL", "VGGSound", "UCF", "ActivityNet"],
        default="AudioSetZSL",
        type=str
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
        "--bs",
        help="Batch size",
        default=64,
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
        "--new_model_attention",
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

    parser.add_argument('--dataset', default='FLO', help='FLO')
    parser.add_argument('--dataroot', default='data/', help='path to dataset')
    parser.add_argument('--matdataset', default=True, help='Data in matlab format')
    parser.add_argument('--image_embedding', default='res101')
    parser.add_argument('--class_embedding', default='att')
    parser.add_argument('--syn_num', type=int, default=10000, help='number features to generate per class')
    parser.add_argument('--gfsl', action='store_true', default=True, help='enable generalized zero-shot learning')
    parser.add_argument('--image_att10', action='store_true', default=False,
                        help='enable generalized zero-shot learning')
    parser.add_argument('--image_att', action='store_true', default=False, help='enable generalized zero-shot learning')
    parser.add_argument('--preprocessing', action='store_true', default=False,
                        help='enbale MinMaxScaler on visual features')
    parser.add_argument('--standardization', action='store_true', default=False)
    parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--resSize', type=int, default=1024, help='size of visual features')
    parser.add_argument('--attSize', type=int, default=300, help='size of semantic features')
    parser.add_argument('--nz', type=int, default=300, help='size of the latent z vector')
    parser.add_argument('--latent_size', type=int, default=300, help='size of the latent z vector')
    parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
    parser.add_argument('--ndh', type=int, default=512, help='size of the hidden units in discriminator')
    parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--nepoch_classifier', type=int, default=30, help='number of epochs to train for')
    parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
    parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
    parser.add_argument('--cls_weight', type=float, default=1, help='weight of the classification loss')
    parser.add_argument('--gan_weight', type=float, default=100000, help='weight of the classification loss')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GANs ')
    parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--pretrain_classifier', default='', help="path to pretrain classifier (to continue training)")
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--netD2', default='', help="path to netD (to continue training)")
    parser.add_argument('--Encoder', default='', help="path to netD (to continue training)")
    parser.add_argument('--netG_name', default='')
    parser.add_argument('--netD_name', default='')
    parser.add_argument('--outf', default='./checkpoint/', help='folder to output data and model checkpoints')
    parser.add_argument('--outname', help='folder to output data and model checkpoints')
    parser.add_argument('--save_every', type=int, default=100)
    parser.add_argument('--save_after', type=int, default=200)
    parser.add_argument('--print_every', type=int, default=1)
    parser.add_argument('--val_every', type=int, default=10)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')
    parser.add_argument('--encoder_layer_sizes', type=list, nargs='+', default=[1024, 300], help='number of all classes')
    parser.add_argument('--decoder_layer_sizes', type=list, nargs='+',default=[512, 1024], help='number of all classes')
    parser.add_argument('--ud_weight', type=float, default=1, help='weight of the classification loss')
    parser.add_argument('--vae_weight', type=float, default=1, help='weight of the classification loss')
    parser.add_argument('--kshot', type=int, default=1, help='number of all classes')
    parser.add_argument('--splitid', default='1', help='folder to output data and model checkpoints')
    parser.add_argument('--novel_weight', type=float, default=1, help='size of the latent z vector')

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
        required=True
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
        default=64,
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
        "--new_model_attention",
        help="Flag to set the attention to the new model",
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

    parser.add_argument('--dataset', default='FLO', help='FLO')
    parser.add_argument('--dataroot', default='data/', help='path to dataset')
    parser.add_argument('--matdataset', default=True, help='Data in matlab format')
    parser.add_argument('--image_embedding', default='res101')
    parser.add_argument('--class_embedding', default='att')
    parser.add_argument('--syn_num', type=int, default=100, help='number features to generate per class')
    parser.add_argument('--gfsl', action='store_true', default=True, help='enable generalized zero-shot learning')
    parser.add_argument('--image_att10', action='store_true', default=False,
                        help='enable generalized zero-shot learning')
    parser.add_argument('--image_att', action='store_true', default=False, help='enable generalized zero-shot learning')
    parser.add_argument('--preprocessing', action='store_true', default=False,
                        help='enbale MinMaxScaler on visual features')
    parser.add_argument('--standardization', action='store_true', default=False)
    parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--resSize', type=int, default=1024, help='size of visual features')
    parser.add_argument('--attSize', type=int, default=300, help='size of semantic features')
    parser.add_argument('--nz', type=int, default=312, help='size of the latent z vector')
    parser.add_argument('--latent_size', type=int, default=312, help='size of the latent z vector')
    parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
    parser.add_argument('--ndh', type=int, default=1024, help='size of the hidden units in discriminator')
    parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
    parser.add_argument('--nepoch_classifier', type=int, default=30, help='number of epochs to train for')
    parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
    parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
    parser.add_argument('--cls_weight', type=float, default=1, help='weight of the classification loss')
    parser.add_argument('--gan_weight', type=float, default=100000, help='weight of the classification loss')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GANs ')
    parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--pretrain_classifier', default='', help="path to pretrain classifier (to continue training)")
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--netD2', default='', help="path to netD (to continue training)")
    parser.add_argument('--Encoder', default='', help="path to netD (to continue training)")
    parser.add_argument('--netG_name', default='')
    parser.add_argument('--netD_name', default='')
    parser.add_argument('--outf', default='./checkpoint/', help='folder to output data and model checkpoints')
    parser.add_argument('--outname', help='folder to output data and model checkpoints')
    parser.add_argument('--save_every', type=int, default=100)
    parser.add_argument('--save_after', type=int, default=200)
    parser.add_argument('--print_every', type=int, default=1)
    parser.add_argument('--val_every', type=int, default=10)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')
    parser.add_argument('--encoder_layer_sizes', type=list, default=[2048, 1024], help='number of all classes')
    parser.add_argument('--decoder_layer_sizes', type=list, default=[1024, 2048], help='number of all classes')
    parser.add_argument('--ud_weight', type=float, default=1, help='weight of the classification loss')
    parser.add_argument('--vae_weight', type=float, default=1, help='weight of the classification loss')
    parser.add_argument('--kshot', type=int, default=1, help='number of all classes')
    parser.add_argument('--splitid', default='1', help='folder to output data and model checkpoints')
    parser.add_argument('--novel_weight', type=float, default=1, help='size of the latent z vector')

    return parser.parse_args()
