import json
import logging
import pickle
import socket
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.logger import PD_Stats, create_logger


def read_features(path):
    hf = h5py.File(path, 'r')
    # keys = list(hf.keys())
    data = hf['data']
    url = [str(u, 'utf-8') for u in list(hf['video_urls'])]

    return data, url


def fix_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def setup_experiment(args, *stats):
    if args.exp_name == "":
        exp_name = f"runs/{datetime.now().strftime('%b%d_%H-%M-%S')}_{socket.gethostname()}"
    else:
        exp_name = "runs/" + str(args.exp_name)
        #exp_name = "/mnt/store_runs/" + str(args.exp_name)
    log_dir = (args.dump_path / exp_name)
    log_dir.mkdir(parents=True)
    (log_dir / "checkpoints").mkdir()
    pickle.dump(args, (log_dir / "args.pkl").open("wb"))
    train_stats = PD_Stats(log_dir / "train_stats.pkl", stats)
    val_stats = PD_Stats(log_dir / "val_stats.pkl", stats)
    logger = create_logger(log_dir / "train.log")

    logger.info(f"Start experiment {exp_name}")
    logger.info(
        "\n".join(f"{k}: {str(v)}" for k, v in sorted(dict(vars(args)).items()))
    )
    logger.info(f"The experiment will be stored in {log_dir.resolve()}\n")
    logger.info("")
    if args.exp_name == "":
        writer = SummaryWriter()
    else:
        writer = SummaryWriter(log_dir=exp_name)
    return logger, log_dir, writer, train_stats, val_stats


def setup_evaluation(args, *stats):
    eval_dir = args.load_path_stage_B
    assert eval_dir.exists()
    # pickle.dump(args, (eval_dir / "args.pkl").open("wb"))
    test_stats = PD_Stats(eval_dir / "test_stats.pkl", list(sorted(stats)))
    logger = create_logger(eval_dir / "eval.log")

    logger.info(f"Start evaluation {eval_dir}")
    logger.info(
        "\n".join(f"{k}: {str(v)}" for k, v in sorted(dict(vars(args)).items()))
    )
    logger.info(f"Loaded configuration {args.load_path_stage_B / 'args.pkl'}")
    logger.info(
        "\n".join(f"{k}: {str(v)}" for k, v in sorted(dict(vars(load_args(args.load_path_stage_B))).items()))
    )
    logger.info(f"The evaluation will be stored in {eval_dir.resolve()}\n")
    logger.info("")

    return logger, eval_dir, test_stats


def save_best_model(epoch, best_metric, model, optimizer, log_dir, metric="", checkpoint=False):
    logger = logging.getLogger()
    logger.info(f"Saving model to {log_dir} with {metric} = {best_metric:.4f}")
    save_dict = {
        "epoch": epoch + 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metric": metric
    }
    if checkpoint:
        torch.save(
            save_dict,
            log_dir / f"{model.__class__.__name__}_{metric}_ckpt_{epoch}.pt"
        )
    else:
        torch.save(
            save_dict,
            log_dir / f"{model.__class__.__name__}_{metric}.pt"
        )


def check_best_loss(epoch, best_loss, val_loss, model, optimizer, log_dir):
    if not best_loss:
        save_best_model(epoch, val_loss, model, optimizer, log_dir, metric="loss")
        return val_loss
    if val_loss < best_loss:
        best_loss = val_loss
        save_best_model(epoch, best_loss, model, optimizer, log_dir, metric="loss")
    return best_loss


def check_best_score(epoch, best_score, hm_score, model, optimizer, log_dir):
    if not best_score:
        save_best_model(epoch, hm_score, model, optimizer, log_dir, metric="score")
        return hm_score
    if hm_score > best_score:
        best_score = hm_score
        save_best_model(epoch, best_score, model, optimizer, log_dir, metric="score")
    return best_score


def load_model_parameters(model, model_weights):
    logger = logging.getLogger()
    loaded_state = model_weights
    self_state = model.state_dict()
    for name, param in loaded_state.items():
        param = param
        if 'module.' in name:
            name = name.replace('module.', '')
        if name in self_state.keys():
            self_state[name].copy_(param)
        else:
            logger.info("didnt load ", name)


def load_args(path):
    return pickle.load((path / "args.pkl").open("rb"))


def cos_dist(a, b):
    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    res = torch.mm(a_norm, b_norm.transpose(0, 1))
    return res


def evaluate_dataset_baseline(dataset, model, device, distance_fn, best_beta=None,
                              new_model_attention=False, model_devise=False, apn=False,
                              args=None, save_performances=False):
    data = dataset.all_data
    data_a = data["audio"].to(device)
    data_v = data["video"].to(device)
    data_t = data["text"].to(device)

    data_num = data["target"].to(device)
    if new_model_attention == True or model_devise == True or apn == True:
        all_data = (
            data_a, data_v, data_num, data_t
        )
    else:
        all_data = (
            data_a, data_v, data_t
        )
    try:
        if args.z_score_inputs:
            all_data = tuple([(x - torch.mean(x)) / torch.sqrt(torch.var(x)) for x in all_data])
    except AttributeError:
        print("Namespace has no fitting attribute. Continuing")

    all_targets = dataset.targets.to(device)
    model.eval()

    if new_model_attention == False and model_devise == False and apn == False:
        outputs_all = model(*all_data)
    elif apn == True:
        input_features = torch.cat((all_data[1], all_data[0]), 1)
        output_final, pre_attri, attention, pre_class, attributes = model(input_features, all_data[3])
        outputs_all = (pre_attri["final"], attributes)
    elif model_devise == True:
        input_features = torch.cat((all_data[1], all_data[0]), 1)
        outputs_all, projected_features, embeddings = model(input_features, all_data[3])
        outputs_all = (projected_features, embeddings)
    elif new_model_attention == True:
        audio_emb, video_emb, emb_cls = model.get_embeddings(all_data[0], all_data[1], all_data[3])
        outputs_all = (audio_emb, video_emb, emb_cls)

    if model_devise == True or apn == True:
        a_p, t_p = outputs_all
        v_p = None
    elif new_model_attention == True:
        a_p, v_p, t_p = outputs_all
        # a_p = None

    if  model_devise == True or apn == True:
        audio_evaluation = get_best_evaluation(dataset, all_targets, a_p, v_p, t_p, mode="audio", device=device,
                                               distance_fn=distance_fn, best_beta=best_beta, save_performances=save_performances, args=args)
    if new_model_attention == True:
        video_evaluation = get_best_evaluation(dataset, all_targets, a_p, v_p, t_p, mode="video", device=device,
                                               distance_fn=distance_fn, best_beta=best_beta, save_performances=save_performances,args=args)

    if  new_model_attention == True:
        return {
            "audio": video_evaluation,
            "video": video_evaluation,
            "both": video_evaluation
        }
    elif model_devise == True or apn == True:
        return {
            "audio": audio_evaluation,
            "video": audio_evaluation,
            "both": audio_evaluation
        }



def get_best_evaluation(dataset, targets, a_p, v_p, t_p, mode, device, distance_fn, best_beta=None, save_performances=False, args=None, attention_weights=None):
    seen_scores = []
    zsl_scores = []
    unseen_scores = []
    hm_scores = []
    per_class_recalls = []
    start = 0
    end = 3
    steps = (end - start) * 5 + 1
    betas = torch.tensor([best_beta], dtype=torch.float, device=device) if best_beta else torch.linspace(start, end, steps,
                                                                                                         device=device)
    seen_label_array = torch.tensor(dataset.seen_class_ids, dtype=torch.long, device=device)
    unseen_label_array = torch.tensor(dataset.unseen_class_ids, dtype=torch.long, device=device)
    seen_unseen_array = torch.tensor(np.sort(np.concatenate((dataset.seen_class_ids, dataset.unseen_class_ids))),
                                     dtype=torch.long, device=device)

    classes_embeddings = t_p
    with torch.no_grad():
        for beta in betas:
            if a_p == None:
                distance_mat = torch.zeros((v_p.shape[0], len(dataset.all_class_ids)), dtype=torch.float,
                                           device=device) + 99999999999999
                distance_mat_zsl = torch.zeros((v_p.shape[0], len(dataset.all_class_ids)), dtype=torch.float,
                                               device=device) + 99999999999999
            else:
                distance_mat = torch.zeros((a_p.shape[0], len(dataset.all_class_ids)), dtype=torch.float,
                                           device=device) + 99999999999999
                distance_mat_zsl = torch.zeros((a_p.shape[0], len(dataset.all_class_ids)), dtype=torch.float,
                                               device=device) + 99999999999999
            if mode == "audio":
                distance_mat[:, seen_unseen_array] = torch.cdist(a_p, classes_embeddings)  # .pow(2)
                mask = torch.zeros(len(dataset.all_class_ids), dtype=torch.long, device=device)
                mask[seen_label_array] = 99999999999999
                distance_mat_zsl = distance_mat + mask
                if distance_fn == "SquaredL2Loss":
                    distance_mat[:, seen_unseen_array] = distance_mat[:, seen_unseen_array].pow(2)
                    distance_mat_zsl[:, unseen_label_array] = distance_mat_zsl[:, unseen_label_array].pow(2)
            elif mode == "video":
                distance_mat[:, seen_unseen_array] = torch.cdist(v_p, classes_embeddings)  # .pow(2)
                mask = torch.zeros(len(dataset.all_class_ids), dtype=torch.long, device=device)
                mask[seen_label_array] = 99999999999999
                distance_mat_zsl = distance_mat + mask
                if distance_fn == "SquaredL2Loss":
                    distance_mat[:, seen_unseen_array] = distance_mat[:, seen_unseen_array].pow(2)
                    distance_mat_zsl[:, unseen_label_array] = distance_mat_zsl[:, unseen_label_array].pow(2)
            elif mode == "both":
                # L2
                audio_distance = torch.cdist(a_p, classes_embeddings, p=2)  # .pow(2)
                video_distance = torch.cdist(v_p, classes_embeddings, p=2)  # .pow(2)

                if distance_fn == "SquaredL2Loss":
                    audio_distance = audio_distance.pow(2)
                    video_distance = video_distance.pow(2)

                # Sum
                if args.cjme==True:
                    distance_mat[:, seen_unseen_array]=(1-attention_weights)*audio_distance+attention_weights*video_distance
                else:
                    distance_mat[:, seen_unseen_array] = (audio_distance + video_distance)

                mask = torch.zeros(len(dataset.all_class_ids), dtype=torch.long, device=device)
                mask[seen_label_array] = 99999999999999
                distance_mat_zsl = distance_mat + mask

            mask = torch.zeros(len(dataset.all_class_ids), dtype=torch.long, device=device) + beta
            mask[unseen_label_array] = 0
            neighbor_batch = torch.argmin(distance_mat + mask, dim=1)
            match_idx = neighbor_batch.eq(targets.int()).nonzero().flatten()
            match_counts = torch.bincount(neighbor_batch[match_idx], minlength=len(dataset.all_class_ids))[
                seen_unseen_array]
            target_counts = torch.bincount(targets, minlength=len(dataset.all_class_ids))[seen_unseen_array]
            per_class_recall = torch.zeros(len(dataset.all_class_ids), dtype=torch.float, device=device)
            per_class_recall[seen_unseen_array] = match_counts / target_counts
            seen_recall_dict = per_class_recall[seen_label_array]
            unseen_recall_dict = per_class_recall[unseen_label_array]
            s = seen_recall_dict.mean()
            u = unseen_recall_dict.mean()

            if save_performances:
                seen_dict = {k: v for k, v in zip(np.array(dataset.all_class_names)[seen_label_array.cpu().numpy()], seen_recall_dict.cpu().numpy())}
                unseen_dict = {k: v for k, v in zip(np.array(dataset.all_class_names)[unseen_label_array.cpu().numpy()], unseen_recall_dict.cpu().numpy())}
                save_class_performances(seen_dict, unseen_dict, dataset.dataset_name)

            hm = (2 * u * s) / ((u + s) + np.finfo(float).eps)

            neighbor_batch_zsl = torch.argmin(distance_mat_zsl, dim=1)
            match_idx = neighbor_batch_zsl.eq(targets.int()).nonzero().flatten()
            match_counts = torch.bincount(neighbor_batch_zsl[match_idx], minlength=len(dataset.all_class_ids))[
                seen_unseen_array]
            target_counts = torch.bincount(targets, minlength=len(dataset.all_class_ids))[seen_unseen_array]
            per_class_recall = torch.zeros(len(dataset.all_class_ids), dtype=torch.float, device=device)
            per_class_recall[seen_unseen_array] = match_counts / target_counts
            zsl = per_class_recall[unseen_label_array].mean()

            zsl_scores.append(zsl.item())
            seen_scores.append(s.item())
            unseen_scores.append(u.item())
            hm_scores.append(hm.item())
            per_class_recalls.append(per_class_recall.tolist())
        argmax_hm = np.argmax(hm_scores)
        max_seen = seen_scores[argmax_hm]
        max_zsl = zsl_scores[argmax_hm]
        max_unseen = unseen_scores[argmax_hm]
        max_hm = hm_scores[argmax_hm]
        max_recall = per_class_recalls[argmax_hm]
        best_beta = betas[argmax_hm].item()
    return {
        "seen": max_seen,
        "unseen": max_unseen,
        "hm": max_hm,
        "recall": max_recall,
        "zsl": max_zsl,
        "beta": best_beta
    }


def evaluate_dataset(dataset, model, device, distance_fn, best_beta=None, args=None):
    data = dataset.all_data
    data_a = data["audio"].to(device)
    data_v = data["video"].to(device)
    data_t = data["text"].to(device)
    all_data = (
        data_a, data_v, data_t
    )
    try:
        if args.z_score_inputs:
            all_data = tuple([(x - torch.mean(x)) / torch.sqrt(torch.var(x)) for x in all_data])
    except AttributeError:
        print("Namespace has no fitting attribute. Continuing")

    all_targets = dataset.targets.to(device)
    model.eval()
    outputs_all = model(*all_data, *all_data)
    if args.cjme==True:
        a_p, v_p, t_p, a_q, v_q, t_q, attention_weights, threshold_attention=outputs_all
    else:
        x_t_p, a_p, v_p, t_p, a_q, v_q, t_q, x_ta_p, x_tv_p, x_tt_p, x_ta_q, x_tv_q = outputs_all
        threshold_attention=None
    audio_evaluation = get_best_evaluation(dataset, all_targets, a_p, v_p, t_p, mode="audio", device=device,
                                           distance_fn=distance_fn, best_beta=best_beta, args=args)
    video_evaluation = get_best_evaluation(dataset, all_targets, a_p, v_p, t_p, mode="video", device=device,
                                           distance_fn=distance_fn, best_beta=best_beta, args=args)
    both_evaluation = get_best_evaluation(dataset, all_targets, a_p, v_p, t_p, mode="both", device=device,
                                          distance_fn=distance_fn, best_beta=best_beta, args=args, attention_weights=threshold_attention)
    return {
        "audio": audio_evaluation,
        "video": video_evaluation,
        "both": both_evaluation
    }


def get_class_names(path):
    if isinstance(path, str):
        path = Path(path)
    with path.open("r") as f:
        classes = sorted([line.strip() for line in f])
    return classes


def load_model_weights(weights_path, model):
    logging.info(f"Loading model weights from {weights_path}")
    load_dict = torch.load(weights_path)
    model_weights = load_dict["model"]
    epoch = load_dict["epoch"]
    logging.info(f"Load from epoch: {epoch}")
    load_model_parameters(model, model_weights)
    return epoch
    
def plot_hist_from_dict(dict):
    plt.bar(range(len(dict)), list(dict.values()), align="center")
    plt.xticks(range(len(dict)), list(dict.keys()), rotation='vertical')
    plt.tight_layout()
    plt.show()

def save_class_performances(seen_dict, unseen_dict, dataset_name):
    seen_path = Path(f"doc/cvpr2022/fig/final/class_performance_{dataset_name}_seen.pkl")
    unseen_path = Path(f"doc/cvpr2022/fig/final/class_performance_{dataset_name}_unseen.pkl")
    with seen_path.open("wb") as f:
        pickle.dump(seen_dict, f)
        logging.info(f"Saving seen class performances to {seen_path}")
    with unseen_path.open("wb") as f:
        pickle.dump(unseen_dict, f)
        logging.info(f"Saving unseen class performances to {unseen_path}")
