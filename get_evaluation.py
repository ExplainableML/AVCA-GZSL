import copy
import logging

import torch

from src.args import args_eval
from src.dataset import ActivityNetDataset, AudioSetZSLDataset, VGGSoundDataset, UCFDataset
from src.model import AVGZSLNet, DeviseModel, APN, CJME
from src.model_improvements import AVCA
from src.utils_improvements import get_model_params
from src.test import test
from src.utils import fix_seeds, load_args, load_model_parameters, setup_evaluation, load_model_weights
from pathlib import Path

def get_evaluation():
    args = args_eval()
    config = load_args(args.load_path_stage_B)
    assert config.retrain_all, f"--retrain_all flag is not set in load_path_stage_B. Are you sure this is the correct path?. {args.load_path_stage_B}"
    fix_seeds(config.seed)

    logger, eval_dir, test_stats = setup_evaluation(args, config.__dict__.keys())

    if args.dataset_name == "AudioSetZSL":
        val_all_dataset = AudioSetZSLDataset(
            args=config,
            dataset_split="val",
            zero_shot_mode="all",
        )
        test_dataset = AudioSetZSLDataset(
            args=config,
            dataset_split="test",
            zero_shot_mode="all",
        )
    elif args.dataset_name == "VGGSound":
        val_all_dataset = VGGSoundDataset(
            args=config,
            dataset_split="val",
            #dataset_split="test",
            zero_shot_mode=None,
        )
        test_dataset = VGGSoundDataset(
            args=config,
            dataset_split="test",
            zero_shot_mode=None,
        )
    elif args.dataset_name == "UCF":
        val_all_dataset = UCFDataset(
            args=config,
            dataset_split="val",
            #dataset_split="test",
            zero_shot_mode=None,
        )
        test_dataset = UCFDataset(
            args=config,
            dataset_split="test",
            zero_shot_mode=None,
        )
    elif args.dataset_name == "ActivityNet":
        val_all_dataset = ActivityNetDataset(
            args=config,
            dataset_split="val",
            #dataset_split="test",
            zero_shot_mode=None,
        )
        test_dataset = ActivityNetDataset(
            args=config,
            dataset_split="test",
            zero_shot_mode=None,
        )
    else:
        raise NotImplementedError()

    if args.AVCA==True:
        model_params = get_model_params(config.lr, config.first_additional_triplet, config.second_additional_triplet, \
                                        config.reg_loss, config.additional_triplets_loss, config.embedding_dropout, \
                                        config.decoder_dropout, config.additional_dropout,
                                        config.embeddings_hidden_size, \
                                        config.decoder_hidden_size, config.depth_transformer, config.momentum)

    if  args.ale==False and args.sje==False and args.devise==False and args.apn==False and args.cjme==False and args.AVCA==False:
        model_A = AVGZSLNet(config)
    elif args.ale==True or args.sje==True or args.devise==True:
        model_A=DeviseModel(config)
    elif args.apn==True:
        model_A=APN(config)
    elif args.cjme==True:
        model_A=CJME(config)
    elif args.AVCA==True:
        model_A = AVCA(params_model=model_params, input_size_audio=config.input_size_audio,input_size_video=config.input_size_video)

    logger.info(model_A)

    model_B = copy.deepcopy(model_A)

    weights_path_stage_A = list(args.load_path_stage_A.glob("*_score.pt"))[0]
    epoch_A = load_model_weights(weights_path_stage_A, model_A)
    weights_path_stage_B = list((args.load_path_stage_B / "checkpoints").glob(f"*_ckpt_{epoch_A - 1}.pt"))[0]
    _ = load_model_weights(weights_path_stage_B, model_B)

    model_A.to(config.device)
    model_B.to(config.device)



    test(
        eval_name=args.eval_name,
        val_dataset=val_all_dataset,
        test_dataset=test_dataset,
        model_A=model_A,
        model_B=model_B,
        device=args.device,
        distance_fn=config.distance_fn,
        devise_model=args.ale or args.sje or args.devise,
        new_model_attention=config.AVCA,
        apn=args.apn,
        args=config
    )

    logger.info("FINISHED")


if __name__ == "__main__":
    get_evaluation()
