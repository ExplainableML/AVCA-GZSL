import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import torch
import h5py

# Path to class splits
class_split_path="/home/lriesch29/ExplainableAudioVisualLowShotLearning/dat/UCF/class-split/"

# Path to the text embeddings
text_embedding_path="/home/lriesch29/ExplainableAudioVisualLowShotLearning/dat/UCF/features/main_features/text/"

# Path to where to put the data files for the features
output_path_features= "/home/lriesch29/ExplainableAudioVisualLowShotLearning/dat/UCF/features/"

# Path to the SeLaVi features
features_self_sup_path = Path("/home/lriesch29/selavi_distributed_orig/outputs/pretrained_VGGSound/ucf/out512_avg/ucf_all_classes512_numsecaud1_avg_cleaned.pkl")

# Path to the C3D/VGGish features
features_sup_path = Path("/home/lriesch29/akata-shared/shared/avzsl/UCF/supervised_extractions/ucf_averaged_all.pkl")





def get_features(path):
    features = pickle.load(path.open("rb"))
    video, labels, audio, filenames = features["video"], features["labels"], features["audio"], features["filenames"]
    diff_len = len(video) - len(filenames)
    assert diff_len == 0
    assert video.device == labels.device == audio.device == torch.device("cpu")
    filenames = np.array(filenames)
    assert len(video) == len(labels) == len(audio) == len(filenames)

    return {"video": video, "labels": labels, "audio": audio, "filenames": filenames}


def get_features_supervised(path):
    features = pickle.load(path.open("rb"))
    video = torch.tensor([f[0] for f in features])
    labels = torch.tensor([f[1] for f in features])
    audio = torch.tensor([f[2] for f in features])
    filenames = np.array([f[3].split(".")[0] for f in features])

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


def test_correct_labels(splits_name, features_path, mode):
    if mode == "self_sup":
        features = get_features(features_path)
    elif mode == "sup":
        # features = get_features_supervised(features_path)
        features = get_features(features_path)
    else:
        raise NotImplementedError()

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
                # import pdb; pdb.set_trace()
                wrong_labels += 1

        assert wrong_labels == 0


def get_ucf_classes():
    path = Path("/home/lriesch29/akata-shared/datasets/UCF101/UCF-101")
    classes = [p.stem for p in sorted(path.iterdir())]
    assert len(classes) == 101
    return classes


def main(features_split_name, splits_name, features_path, mode):
    out_path = Path(output_path_features) / features_split_name

    if mode == "self_sup":
        features = get_features(features_path)
    elif mode == "sup":
        # features = get_features_supervised(features_path)
        features = get_features(features_path)
    else:
        raise NotImplementedError()

    video, labels, audio, filenames = features["video"], features["labels"], features["audio"], features["filenames"],
    classes = get_ucf_classes()
    class_to_idx = {classes[i].lower(): i for i in range(len(classes))}

    path_text_emb = Path(text_embedding_path)
    # to_file = out_path
    out_path_text = out_path / 'text/'
    if not out_path_text.exists():
        print(f"Copying text embedding to {out_path_text.resolve()}")
        shutil.copytree(path_text_emb, out_path_text)

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
            idx = class_to_idx[c.lower()]
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
    assert len(audio_train_tmp[0]) == len(video_train_tmp[0])
    assert audio_train_tmp[1][0] == video_train_tmp[1][0]
    assert not np.array_equal(audio_train_tmp[0][0], video_train_tmp[0][0])

    tmp_root = Path(output_path_features) / features_split_name
    test_name = sorted((tmp_root / "audio/stage_2_test_unseen/").iterdir())[0].name
    print(f"Test name: {test_name}")
    audio_test_tmp = read_features(tmp_root / f"audio/stage_2_test_unseen/{test_name}")
    video_test_tmp = read_features(tmp_root / f"video/stage_2_test_unseen/{test_name}")
    assert len(audio_test_tmp[0]) == len(video_test_tmp[0])
    assert audio_test_tmp[1][0] == video_test_tmp[1][0]
    assert not np.array_equal(audio_test_tmp[0][0], video_test_tmp[0][0])


def copy_to_shared():
    my_file = Path("/home/lriesch29/ExplainableAudioVisualLowShotLearning/dat/UCF/")
    to_file = Path('/home/lriesch29/akata-shared/shared/avzsl/UCF/')

    shutil.copytree(my_file, to_file, dirs_exist_ok=True)



features_name = f"main_features"
splits_name = f"main_split"
print("Test correct labels")
test_correct_labels(splits_name, features_sup_path, mode="sup")
print("Main")
main(features_name, splits_name, features_sup_path, mode="sup")
print("Test")
test(features_name, splits_name)
print()

print("DONE")
features_name = f"cls_features"
splits_name = f"cls_split"
print("Test correct labels")
test_correct_labels(splits_name, features_self_sup_path, mode="self_sup")
print("Main")
main(features_name, splits_name, features_self_sup_path, mode="self_sup")
print("Test")
test(features_name, splits_name)
print()
print("DONE")
