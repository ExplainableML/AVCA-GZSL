import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import h5py



features_split1_ex_others_nooverlap_respect_test_small_retrain = "main_features"
splits_name_ex_others_nooverlap_respect_test_small_retrain = "main_split"

# Path to the class splits
class_split_path="/home/lriesch29/ExplainableAudioVisualLowShotLearning/dat/VGGSound/class-split/"

# Path to where to save the features in the right folder structure
output_path_features="/home/lriesch29/ExplainableAudioVisualLowShotLearning/dat/VGGSound/features/"

# Path to load the extracted feature to arrange them into the desired folder structure
path_to_features="/home/lriesch29/selavi_distributed_orig/outputs/pretrained_VGGSound/vggsound/out512_avg/vggsound_all_classes512_numsecaud1_avg_cleaned.pkl"



def get_features(path):
    features = pickle.load(path.open("rb"))
    video, labels, audio, filenames = features["video"], features["labels"], features["audio"], features["filenames"]
    diff_len = len(video) - len(filenames)
    assert diff_len == 0
    assert video.device == labels.device == audio.device == torch.device("cpu")
    filenames = np.array(filenames)
    assert len(video) == len(labels) == len(audio) == len(filenames)

    return {"video": video, "labels": labels, "audio": audio, "filenames": filenames}


def get_indices(dataframe, features):
    names = set(dataframe.filename.values)
    filenames = features["filenames"]
    assert len(dataframe) == len(names)
    indices = np.array([idx for idx, filename in enumerate(filenames) if filename in names])

    return indices


def test_correct_labels(splits_name):
    features_path = Path(path_to_features)
    features = get_features(features_path)

    splits = [
        "stage_1_train", "stage_1_val_seen", "stage_1_val_unseen",
        "stage_2_train", "stage_2_test_seen", "stage_2_test_unseen"
    ]
    for split in splits:
        base_path = Path(class_split_path)
        df_path = sorted((base_path / splits_name).glob(f"*{split}.csv"))[0]
        df = pd.read_csv(df_path)
        wrong_labels = 0
        for category in df.label_code.unique():
            idx = np.where(features["labels"] == category)[0]
            feat_names = features["filenames"][idx]
            df_names = df[df.label_code == category]["filename"].values
            if df_names[0] not in set(feat_names):
                wrong_labels += 1

        assert wrong_labels == 0


test_correct_labels(splits_name_ex_others_nooverlap_respect_test_small_retrain)


def get_vggsound_classes():
    df = pd.read_csv("../dat/VGGSound/class-split/vggsound.csv", header=None, names=["youtube_id", "start_seconds", "label", "split"])
    classes = sorted(df.label.unique())
    assert classes[36] == "cattle mooing"
    assert classes[37] == "cattle, bovinae cowbell"
    assert len(classes) == 309
    return classes


def main(features_split_name, splits_name):
    out_path = Path(output_path_features) / features_split_name
    features_path = Path(path_to_features)
    features = get_features(features_path)
    video, labels, audio, filenames = features["video"], features["labels"], features["audio"], features["filenames"],
    classes = get_vggsound_classes()
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    for split in ["stage_1_train", "stage_1_val_seen", "stage_1_val_unseen", "stage_2_train", "stage_2_test_seen", "stage_2_test_unseen"]:
        base_path = Path(class_split_path)
        df_path = sorted((base_path / splits_name).glob(f"*{split}.csv"))[0]
        assert df_path.exists()
        df = pd.read_csv(df_path)

        split_classes = sorted(df.label.unique())
        split_indices = get_indices(df, features)
        video_split, labels_split, audio_split, filenames_split = video[split_indices], labels[split_indices], audio[
            split_indices], filenames[split_indices]
        assert len(video_split) == len(labels_split) == len(audio_split) == len(filenames_split) == len(split_indices)

        audio_path = out_path / f"audio/{split}"
        print(audio_path)
        Path(audio_path).mkdir(exist_ok=True, parents=True)

        video_path = out_path / f"video/{split}"
        print(video_path)
        Path(video_path).mkdir(exist_ok=True, parents=True)

        for c in split_classes:
            idx = class_to_idx[c]
            class_indices = np.where(labels_split == idx)[0]
            file_name = f"{c}.h5"

            with h5py.File(audio_path / file_name, "w") as hf:
                hf.create_dataset("data", data=audio_split[class_indices])
                tmp_names = [n.encode("ascii", "ignore") for n in filenames_split[class_indices]]
                hf.create_dataset("video_urls", dtype=h5py.special_dtype(vlen=str), data=tmp_names)

            with h5py.File(video_path / file_name, "w") as hf:
                hf.create_dataset("data", data=video_split[class_indices])
                tmp_names = [n.encode("ascii", "ignore") for n in filenames_split[class_indices]]
                hf.create_dataset("video_urls", dtype=h5py.special_dtype(vlen=str), data=tmp_names)


main(features_split1_ex_others_nooverlap_respect_test_small_retrain, splits_name_ex_others_nooverlap_respect_test_small_retrain)


def test(features_split_name, splits_name):
    root = Path(class_split_path) / splits_name

    df_train_train = pd.read_csv(root / "stage_1_train.csv")
    df_val_seen = pd.read_csv(root / "stage_1_val_seen.csv")
    df_val_unseen = pd.read_csv(root / "stage_1_val_unseen.csv")

    df_test_train = pd.read_csv(root / "stage_2_train.csv")
    df_test_seen = pd.read_csv(root / "stage_2_test_seen.csv")
    df_test_unseen = pd.read_csv(root / "stage_2_test_unseen.csv")

    def read_features(path):
        hf = h5py.File(path, 'r')
        # keys = list(hf.keys())
        data = hf['data']
        # url = [str(u, 'utf-8') for u in list(hf['video_urls'])]
        url = [str(u) for u in list(hf['video_urls'])]
        return data, url

    tmp_root = Path(output_path_features) / features_split_name
    test_name = sorted((tmp_root / "audio/stage_1_train").iterdir())[0].name
    print(f"Test name: {test_name}")
    audio_train_tmp = read_features(tmp_root / f"audio/stage_1_train/{test_name}")
    video_train_tmp = read_features(tmp_root / f"video/stage_1_train/{test_name}")
    # audio_train_tmp = read_features(tmp_root / "audio/train_train/air horn.h5")
    # video_train_tmp = read_features(tmp_root / "video/train_train/air horn.h5")
    assert len(audio_train_tmp[0]) == len(video_train_tmp[0])
    assert audio_train_tmp[1][0] == video_train_tmp[1][0]
    assert not np.array_equal(audio_train_tmp[0][0], video_train_tmp[0][0])

    tmp_root = Path(output_path_features) / features_split_name
    test_name = sorted((tmp_root / "audio/stage_2_test_unseen/").iterdir())[0].name
    print(f"Test name: {test_name}")
    audio_test_tmp = read_features(tmp_root / f"audio/stage_2_test_unseen/{test_name}")
    video_test_tmp = read_features(tmp_root / f"video/stage_2_test_unseen/{test_name}")
    # audio_test_tmp = read_features(tmp_root / "audio/test_unseen/bird chirping, tweeting.h5")
    # video_test_tmp = read_features(tmp_root / "video/test_unseen/bird chirping, tweeting.h5")
    assert len(audio_test_tmp[0]) == len(video_test_tmp[0])
    assert audio_test_tmp[1][0] == video_test_tmp[1][0]
    assert not np.array_equal(audio_test_tmp[0][0], video_test_tmp[0][0])


test(features_split1_ex_others_nooverlap_respect_test_small_retrain, splits_name_ex_others_nooverlap_respect_test_small_retrain)