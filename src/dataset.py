import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils import data
from tqdm import tqdm

from src.utils import read_features, get_class_names


class VGGSoundDataset(data.Dataset):
    @property
    def training_file(self):
        return self.features_processed_folder / self.feature_extraction_method / f"training{self.zero_shot_split}.pkl"

    @property
    def val_file(self):
        return self.features_processed_folder / self.feature_extraction_method / f"val{self.zero_shot_split}.pkl"

    @property
    def train_val_file(self):
        return self.features_processed_folder / self.feature_extraction_method / f"train_val{self.zero_shot_split}.pkl"

    @property
    def test_file(self):
        return self.features_processed_folder / self.feature_extraction_method / f"test{self.zero_shot_split}.pkl"

    @property
    def targets(self):
        classes_mask = np.where(np.isin(self.data["audio"]["target"], self.classes))[0]
        return self.data["audio"]["target"][classes_mask]

    @property
    def all_data(self):
        classes_mask = np.where(np.isin(self.data["audio"]["target"], self.classes))[0]
        return {
            "audio": self.data["audio"]["data"][classes_mask],
            "video": self.data["video"]["data"][classes_mask],
            "text": self.data["text"]["data"][sorted(self.classes)],
            "target": self.data["audio"]["target"][classes_mask],
            "url": self.data["audio"]["url"][classes_mask]
        }

    @property
    def map_embeddings_target(self):
        w2v_embedding = self.data["text"]["data"][sorted(self.classes)].cuda()
        sorted_classes = sorted(self.classes)
        mapping_dict = {}
        for i in range(len(sorted_classes)):
            mapping_dict[int(sorted_classes[i])] = i
        return w2v_embedding, mapping_dict

    @property
    def features_processed_folder(self):
        return Path().cwd() / "avgzsl_benchmark_datasets/VGGSound/_features_processed"

    @property
    def all_class_ids(self):
        return np.asarray([self.class_to_idx[name] for name in self.all_class_names])

    @property
    def train_train_ids(self):
        return np.asarray([self.class_to_idx[name] for name in self.train_train_class_names])

    @property
    def val_seen_ids(self):
        return np.asarray([self.class_to_idx[name] for name in self.val_seen_class_names])

    @property
    def val_unseen_ids(self):
        return np.asarray([self.class_to_idx[name] for name in self.val_unseen_class_names])

    @property
    def test_train_ids(self):
        return np.asarray([self.class_to_idx[name] for name in self.test_train_class_names])

    @property
    def test_seen_ids(self):
        return np.asarray([self.class_to_idx[name] for name in self.test_seen_class_names])

    @property
    def test_unseen_ids(self):
        return np.asarray([self.class_to_idx[name] for name in self.test_unseen_class_names])

    @property
    def text_label_mapping(self):
        df = pd.read_csv(self.root / "class-split/vggsound_w2v_class_names.csv")
        return {val: df.original[idx] for idx, val in enumerate(df.manual)}

    @property
    def classes(self):
        if self.zero_shot_split:
            return np.sort(np.concatenate((self.seen_class_ids, self.unseen_class_ids)))

        else:
            if self.zero_shot_mode == "all":
                return self.all_class_ids
            elif self.zero_shot_mode == "seen":
                return self.seen_class_ids
            elif self.zero_shot_mode == "unseen":
                return self.unseen_class_ids
            else:
                raise AttributeError(f"Zero shot mode has to be either all, seen or unseen. Is {self.zero_shot_mode}")

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(sorted(self.all_class_names))}

    @property
    def all_class_names(self):
        return get_class_names(self.root / "class-split/all_class.txt")

    @property
    def seen_class_names(self):
        if self.dataset_split == "train":
            return self.train_train_class_names
        elif self.dataset_split == "val":
            return self.val_seen_class_names
        elif self.dataset_split == "train_val":
            return np.concatenate((self.train_train_class_names, self.val_unseen_class_names))
        elif self.dataset_split == "test":
            return self.test_seen_class_names
        else:
            raise AttributeError("Dataset split has to be in {train,val,train_val,test}")

    @property
    def unseen_class_names(self):
        if self.dataset_split == "train":
            return np.array([])
        elif self.dataset_split == "val":
            return self.val_unseen_class_names
        elif self.dataset_split == "train_val":
            return np.array([])
        elif self.dataset_split == "test":
            return self.test_unseen_class_names
        else:
            raise AttributeError("Dataset split has to be in {train,val,train_val,test}")

    @property
    # @lru_cache(maxsize=128)
    def seen_class_ids(self):
        return np.asarray([self.class_to_idx[name] for name in self.seen_class_names])

    @property
    # @lru_cache(maxsize=128)
    def unseen_class_ids(self):
        return np.asarray([self.class_to_idx[name] for name in self.unseen_class_names])

    @property
    def train_train_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_1_train.txt")

    @property
    def val_seen_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_1_val_seen.txt")

    @property
    def val_unseen_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_1_val_unseen.txt")

    @property
    def test_train_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_2_train.txt")

    @property
    def test_seen_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_2_test_seen.txt")

    @property
    def test_unseen_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_2_test_unseen.txt")

    def __init__(self, args, dataset_split, zero_shot_mode, download=False, transform=None):
        super(VGGSoundDataset, self).__init__()
        self.logger = logging.getLogger()
        self.logger.info(
            f"Initializing Dataset {self.__class__.__name__}\t"
            f"Dataset split: {dataset_split}\t"
            f"Zero shot mode: {zero_shot_mode}")
        self.args = args
        self.root = args.root_dir
        self.dataset_name = args.dataset_name
        self.feature_extraction_method = args.feature_extraction_method
        self.dataset_split = dataset_split
        self.zero_shot_mode = zero_shot_mode
        self.zero_shot_split = args.zero_shot_split

        self.transform = transform

        self.preprocess()
        self.data = self.get_data()

    def __getitem__(self, item):
        raise NotImplementedError()

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return self.training_file.exists() and self.val_file.exists() and self.test_file.exists() and self.train_val_file.exists()

    def preprocess(self):
        if self._check_exists():
            return

        (self.features_processed_folder / self.feature_extraction_method).mkdir(parents=True, exist_ok=True)

        self.logger.info('Processing extracted features for faster training (only done once)...')
        self.logger.info(
            f"Processed files will be stored locally in {(self.features_processed_folder / self.feature_extraction_method).resolve()}"
        )

        training_set = self.read_dataset(dataset_type="train")
        val_set = self.read_dataset(dataset_type="val")
        train_val_set = self.read_dataset(dataset_type="train_val")
        test_set = self.read_dataset(dataset_type="test")

        with self.training_file.open('wb') as f:
            self.logger.info(f"Dumping to {self.training_file}")
            pickle.dump(training_set, f, pickle.HIGHEST_PROTOCOL)
        with self.val_file.open('wb') as f:
            self.logger.info(f"Dumping to {self.val_file}")
            pickle.dump(val_set, f, pickle.HIGHEST_PROTOCOL)
        with self.train_val_file.open('wb') as f:
            self.logger.info(f"Dumping to {self.train_val_file}")
            pickle.dump(train_val_set, f, pickle.HIGHEST_PROTOCOL)
        with self.test_file.open('wb') as f:
            self.logger.info(f"Dumping to {self.test_file}")
            pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)

        if not self._check_exists():
            raise RuntimeError("Dataset not found after preprocessing!")
        self.logger.info("Successfully finished preprocessing.")

    def get_data(self):
        if self.dataset_split == "train":
            data_file = self.training_file
        elif self.dataset_split == "val":
            data_file = self.val_file
        elif self.dataset_split == "train_val":
            data_file = self.train_val_file
        elif self.dataset_split == "test":
            data_file = self.test_file
        else:
            raise AttributeError("Dataset_split has to be either train, val or test.")

        load_path = (self.features_processed_folder / data_file).resolve()
        self.logger.info(f"Loading processed data from disk from {load_path}")
        with load_path.open('rb') as f:
            return pickle.load(f)

    def read_dataset(self, dataset_type):
        result_audio = self.get_data_by_modality(modality="audio", dataset_type=dataset_type)
        result_video = self.get_data_by_modality(modality="video", dataset_type=dataset_type)
        assert torch.equal(result_audio["target"], result_video["target"])
        assert np.array_equal(result_audio["url"], result_video["url"])
        result_text = self.get_data_by_modality(modality="text", dataset_type=dataset_type)
        return {"audio": result_audio, "video": result_video, "text": result_text}

    def get_data_by_modality(self, modality, dataset_type="train"):
        # import pdb; pdb.set_trace()
        result = {"data": [], "target": [], "url": []}
        if modality == "text":
            data_raw = np.load(
                (
                        self.root / "features" / self.feature_extraction_method / "text/word_embeddings_vggsound_normed.npy").resolve(),
                allow_pickle=True).item()
            data_raw_sorted = dict(sorted(data_raw.items()))
            result["data"] = list(data_raw_sorted.values())
            result["target"] = [self.class_to_idx[self.text_label_mapping[key]] for key in list(data_raw_sorted.keys())]

        elif modality == "audio" or modality == "video":
            split_names = []
            if dataset_type == "train":
                split_names.append("stage_1_train")
            elif dataset_type == "val":
                split_names.append("stage_1_val_seen")
                split_names.append("stage_1_val_unseen")
            elif dataset_type == "train_val":
                split_names.append("stage_1_train")
                split_names.append("stage_1_val_seen")
                split_names.append("stage_1_val_unseen")
            elif dataset_type == "test":
                split_names.append("stage_2_test_seen")
                split_names.append("stage_2_test_unseen")
            else:
                raise AttributeError("Dataset type incompatible. Has to be either train, val or test.")

            for split_name in split_names:
                modality_path = (
                        self.root / "features" / self.feature_extraction_method / f"{modality}/{split_name}").resolve()
                files = modality_path.iterdir()
                for file in tqdm(files, total=len(list(modality_path.glob('*'))),
                                 desc=f"{dataset_type}:{modality}:{split_name}"):
                    data, url = read_features(file)
                    #assert len(data[
                    #               0]) == self.args.input_size, f"Feature size {len(data[0])} is not compatible with specified --input_size {self.args.input_size}"
                    for i, d in enumerate(data):
                        result["data"].append(d)
                        result["target"].append(self.class_to_idx[file.stem])
                        result["url"].append(url[i])
        else:
            raise AttributeError("Modality has to be either audio, video or text")
        result["data"] = torch.FloatTensor(result["data"])
        result["target"] = torch.LongTensor(result["target"])
        result["url"] = np.array(result["url"])
        return result


class AudioSetZSLDataset(data.Dataset):
    """
    MNIST-like dataset for AudioSetZSL. This is heavily inspired by the torchvision implementation of MNIST.
    """

    @property
    def training_file(self):
        return self.features_processed_folder / self.feature_extraction_method / "training.pkl"

    @property
    def val_file(self):
        return self.features_processed_folder / self.feature_extraction_method / "val.pkl"

    @property
    def train_val_file(self):
        return self.features_processed_folder / self.feature_extraction_method / "trn_val.pkl"

    @property
    def test_file(self):
        return self.features_processed_folder / self.feature_extraction_method / "test.pkl"

    @property
    def targets(self):
        classes_mask = np.where(np.isin(self.data["audio"]["target"], self.classes))[0]
        return self.data["audio"]["target"][classes_mask]

    @property
    def all_data(self):
        classes_mask = np.where(np.isin(self.data["audio"]["target"], self.classes))[0]
        return {
            "audio": self.data["audio"]["data"][classes_mask],
            "video": self.data["video"]["data"][classes_mask],
            "text": self.data["text"]["data"][sorted(self.classes)],
            "target": self.data["audio"]["target"][classes_mask],
            "url": self.data["audio"]["url"][classes_mask]
        }

    @property
    def features_processed_folder(self):
        return Path().cwd() / "avgzsl_benchmark_datasets/AudioSetZSL/_features_processed"

    @property
    # @lru_cache(maxsize=128)
    def all_class_ids(self):
        return np.asarray([self.class_to_idx[name] for name in self.all_class_names])

    @property
    # @lru_cache(maxsize=128)
    def seen_class_ids(self):
        return np.asarray([self.class_to_idx[name] for name in self.seen_class_names])

    @property
    # @lru_cache(maxsize=128)
    def unseen_class_ids(self):
        return np.asarray([self.class_to_idx[name] for name in self.unseen_class_names])

    @property
    def classes(self):
        if self.zero_shot_mode == "all":
            return self.all_class_ids
        elif self.zero_shot_mode == "seen":
            return self.seen_class_ids
        elif self.zero_shot_mode == "unseen":
            return self.unseen_class_ids
        else:
            raise AttributeError(f"Zero shot mode has to be either all, seen or unseen. Is {self.zero_shot_mode}")

    @property
    # @lru_cache(maxsize=128)
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(sorted(self.all_class_names))}

    @property
    def all_class_names(self):
        class_path = self.root / "class-split/all_class.txt"
        all_classes = np.loadtxt(class_path, dtype=str)
        all_classes = sorted([s.replace("\'", "").replace(",", "") for s in all_classes])
        return all_classes
        # return get_class_names(self.root / "class-split/all_class.txt")

    @property
    def seen_class_names(self):
        class_path = self.root / "class-split/seen_class.txt"
        seen_classes = np.loadtxt(class_path, dtype=str)
        seen_classes = sorted([s.replace("\'", "").replace(",", "") for s in seen_classes])
        return seen_classes
        # return get_class_names(self.root / "class-split/seen_class.txt")

    @property
    def unseen_class_names(self):
        class_path = self.root / "class-split/unseen_class.txt"
        unseen_classes = np.loadtxt(class_path, dtype=str)
        unseen_classes = sorted([s.replace("\'", "").replace(",", "") for s in unseen_classes])
        return unseen_classes
        # return get_class_names(self.root / "class-split/unseen_class.txt")

    def __init__(self, args, dataset_split, zero_shot_mode, download=False, transform=None):

        super(AudioSetZSLDataset, self).__init__()
        self.logger = logging.getLogger()
        self.logger.info(
            f"Initializing Dataset {self.__class__.__name__}\t"
            f"Dataset split: {dataset_split}\t"
            f"Zero shot mode: {zero_shot_mode}")
        self.args = args
        self.root = args.root_dir
        self.dataset_name = args.dataset_name
        self.feature_extraction_method = args.feature_extraction_method
        self.dataset_split = dataset_split
        self.zero_shot_mode = zero_shot_mode
        self.transform = transform

        self.preprocess()
        self.data = self.get_data()

    def __getitem__(self, index):
        target = self.targets[index]

        audio = self.data["audio"]["data"][index]
        video = self.data["video"]["data"][index]
        text = self.data["text"]["data"][target]
        url = self.data["audio"]["url"][index]


        return {
                   "audio": audio,
                   "video": video,
                   "text": text,
                   "url": url
               }, target

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return self.training_file.exists() and self.val_file.exists() and self.train_val_file.exists() and self.test_file.exists()

    def download(self):

        self.logger.info("Downloading dataset...")

        raise NotImplementedError()

    def preprocess(self):
        if self._check_exists():
            return

        (self.features_processed_folder / self.feature_extraction_method).mkdir(parents=True, exist_ok=True)


        self.logger.info('Processing extracted features for faster training (only done once)...')
        self.logger.info(
            f"Processed files will be stored locally in {(self.features_processed_folder / self.feature_extraction_method).resolve()}"
        )

        training_set = self.read_dataset(dataset_type="trn")
        val_set = self.read_dataset(dataset_type="val")
        train_val_set = self.read_dataset(dataset_type="trn_val")
        test_set = self.read_dataset(dataset_type="tst")

        with self.training_file.open('wb') as f:
            self.logger.info(f"Dumping to {self.training_file}")
            pickle.dump(training_set, f, pickle.HIGHEST_PROTOCOL)
        with self.val_file.open('wb') as f:
            self.logger.info(f"Dumping to {self.val_file}")
            pickle.dump(val_set, f, pickle.HIGHEST_PROTOCOL)
        with self.train_val_file.open('wb') as f:
            self.logger.info(f"Dumping to {self.train_val_file}")
            pickle.dump(train_val_set, f, pickle.HIGHEST_PROTOCOL)
        with self.test_file.open('wb') as f:
            self.logger.info(f"Dumping to {self.test_file}")
            pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)

        if not self._check_exists():
            raise RuntimeError("Dataset not found after preprocessing!")
        self.logger.info("Successfully finished preprocessing.")

    def get_data(self):
        if self.dataset_split == "train":
            data_file = self.training_file
        elif self.dataset_split == "val":
            data_file = self.val_file
        elif self.dataset_split == "train_val":
            data_file = self.train_val_file
        elif self.dataset_split == "test":
            data_file = self.test_file
        else:
            raise AttributeError("Dataset_split has to be either train, val or test.")

        load_path = (self.features_processed_folder / data_file).resolve()
        self.logger.info(f"Loading processed data from disk from {load_path}")
        with load_path.open('rb') as f:
            return pickle.load(f)

    def read_dataset(self, dataset_type):
        result_audio = self.get_data_by_modality(modality="audio", dataset_type=dataset_type)
        result_video = self.get_data_by_modality(modality="video", dataset_type=dataset_type)
        assert torch.equal(result_audio["target"], result_video["target"])
        assert np.array_equal(result_audio["url"], result_video["url"])
        result_text = self.get_data_by_modality(modality="text", dataset_type=dataset_type)
        return {"audio": result_audio, "video": result_video, "text": result_text}

    def get_data_by_modality(self, modality, dataset_type="trn"):
        result = {"data": [], "target": [], "url": []}
        if modality == "text":
            if self.args.manual_text_word2vec:
                file_path = (
                        self.root / "features" / self.feature_extraction_method / "text/word_embeddings_audiosetzsl_normed.npy"
                ).resolve()
            else:
                file_path = (
                        self.root / "features" / self.feature_extraction_method / "text/word_embeddings-dict-33.npy"
                ).resolve()

            data_raw = np.load(file_path, allow_pickle=True).item()
            data_raw_sorted = dict(sorted(data_raw.items()))
            result["data"] = list(data_raw_sorted.values())
            result["target"] = [self.class_to_idx[key] for key in list(data_raw_sorted.keys())]

        elif modality == "audio" or modality == "video":
            split_names = []
            if dataset_type == "trn":
                split_names.append("trn")
            elif dataset_type == "val":
                split_names.append("val")
            elif dataset_type == "trn_val":
                split_names.append("trn")
                split_names.append("val")
            elif dataset_type == "tst":
                split_names.append("tst")
            else:
                raise AttributeError("Dataset type incompatible. Has to be either train, val or test.")

            for split_name in split_names:
                modality_path = (
                        self.root / "features" / self.feature_extraction_method / f"{modality}/{split_name}").resolve()
                files = modality_path.iterdir()
                for file in tqdm(files, total=len(list(modality_path.glob('*'))),
                                 desc=f"{dataset_type}:{modality}:{split_name}"):
                    data, url = read_features(file)
                    assert len(data[
                                   0]) == self.args.input_size, f"Feature size {len(data[0])} is not compatible with specified --input_size {self.args.input_size}"
                    for i, d in enumerate(data):
                        result["data"].append(d)
                        result["target"].append(self.class_to_idx[file.stem])
                        result["url"].append(url[i])
        else:
            raise AttributeError("Modality has to be either audio, video or text")
        result["data"] = torch.FloatTensor(result["data"])
        result["target"] = torch.LongTensor(result["target"])
        result["url"] = np.array(result["url"])
        return result


class ContrastiveDataset(data.Dataset):
    def __init__(self, zsl_dataset):
        super(ContrastiveDataset, self).__init__()
        self.logger = logging.getLogger()
        self.logger.info(
            f"Initializing Dataset {self.__class__.__name__}\t"
            f"Based on Dataset: {zsl_dataset.__class__.__name__}\t"
            f"with split: {zsl_dataset.dataset_split}")
        self.zsl_dataset = zsl_dataset
        self.dataset_split = self.zsl_dataset.dataset_split
        self.classes = self.zsl_dataset.classes

        if self.dataset_split == "train" or self.dataset_split == "train_val":
            self.targets = self.zsl_dataset.targets
            self.data = self.zsl_dataset.all_data
            self.targets_set = set(self.targets.tolist())
            self.target_to_indices = {target: np.where(self.zsl_dataset.targets == target)[0]
                                      for target in self.targets_set}

        elif self.dataset_split == "val" or self.dataset_split == "test":
            self.targets = self.zsl_dataset.targets
            self.data = self.zsl_dataset.all_data
            # generate fixed pairs for testing
            self.targets_set = set(self.targets.tolist())
            self.target_to_indices = {target: np.where(self.zsl_dataset.targets == target)[0]
                                      for target in self.targets_set}

            random_state = np.random.RandomState(29)

            # pos_neg_pairs = [i,j] -> list of all targets i with random respective negative index j
            pos_neg_pairs = [[i,
                              random_state.choice(self.target_to_indices[
                                                      np.random.choice(
                                                          list(self.targets_set - set([self.targets[i].item()]))
                                                      )
                                                  ])
                              ]
                             for i in range(len(self.targets))]
            self.val_pairs = pos_neg_pairs
        else:
            raise AttributeError("Dataset_split has to be either train, val, train_val or test.")

    def __len__(self):
        classes_mask = np.where(np.isin(self.zsl_dataset.targets, self.classes))[0]
        return len(self.zsl_dataset.targets[classes_mask])

    def __getitem__(self, index):
        if self.dataset_split == "train" or self.dataset_split == "train_val":
            positive_target = self.targets[index].item()
            pos_target_index = list(self.targets_set).index(positive_target)
            x_a1 = self.data["audio"][index]
            x_v1 = self.data["video"][index]
            x_t1 = self.data["text"][pos_target_index]
            x_url1 = self.data["url"][index]
            # x_numeric1=self.data["target"][index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.target_to_indices[positive_target])
            negative_target = np.random.choice(list(self.targets_set - set([positive_target])))
            negative_index = np.random.choice(self.target_to_indices[negative_target])
            neg_target_index = list(self.targets_set).index(negative_target)
            x_a2 = self.data["audio"][negative_index]
            x_v2 = self.data["video"][negative_index]
            x_t2 = self.data["text"][neg_target_index]
            x_url2 = self.data["url"][negative_index]
            # x_numeric2=self.data["target"][negative_index].item()
        elif self.dataset_split == "val" or self.dataset_split == "test":
            positive_target = self.targets[self.val_pairs[index][0]].item()
            pos_target_index = list(self.targets_set).index(positive_target)
            x_a1 = self.data["audio"][self.val_pairs[index][0]]
            x_v1 = self.data["video"][self.val_pairs[index][0]]
            x_t1 = self.data["text"][pos_target_index]
            # x_numeric1=self.data["target"][self.val_pairs[index][0]].item()
            x_url1 = self.data["url"][self.val_pairs[index][0]]
            negative_target = self.targets[self.val_pairs[index][1]].item()
            neg_target_index = list(self.targets_set).index(negative_target)
            x_a2 = self.data["audio"][self.val_pairs[index][1]]
            x_v2 = self.data["video"][self.val_pairs[index][1]]
            x_t2 = self.data["text"][neg_target_index]
            # x_numeric2=self.data["target"][self.val_pairs[index][1]].item()
            x_url2 = self.data["url"][self.val_pairs[index][1]]
        else:
            raise AttributeError("Dataset_split has to be either train, val, train_val or test.")

        data = {
            "positive": {"audio": x_a1, "video": x_v1, "text": x_t1, "url": x_url1},
            "negative": {"audio": x_a2, "video": x_v2, "text": x_t2, "url": x_url2}
        }
        target = {
            "positive": positive_target,
            "negative": negative_target
        }
        return data, target


class UCFDataset(data.Dataset):

    @property
    def map_embeddings_target(self):
        w2v_embedding = self.data["text"]["data"][sorted(self.classes)].cuda()
        sorted_classes = sorted(self.classes)
        mapping_dict = {}
        for i in range(len(sorted_classes)):
            mapping_dict[int(sorted_classes[i])] = i
        return w2v_embedding, mapping_dict

    @property
    def training_file(self):
        return self.features_processed_folder / self.feature_extraction_method / f"training{self.zero_shot_split}.pkl"

    @property
    def val_file(self):
        return self.features_processed_folder / self.feature_extraction_method / f"val{self.zero_shot_split}.pkl"

    @property
    def train_val_file(self):
        return self.features_processed_folder / self.feature_extraction_method / f"train_val{self.zero_shot_split}.pkl"

    @property
    def test_file(self):
        return self.features_processed_folder / self.feature_extraction_method / f"test{self.zero_shot_split}.pkl"

    @property
    def targets(self):
        classes_mask = np.where(np.isin(self.data["audio"]["target"], self.classes))[0]
        return self.data["audio"]["target"][classes_mask]

    @property
    def all_data(self):
        classes_mask = np.where(np.isin(self.data["audio"]["target"], self.classes))[0]
        return {
            "audio": self.data["audio"]["data"][classes_mask],
            "video": self.data["video"]["data"][classes_mask],
            "text": self.data["text"]["data"][sorted(self.classes)],
            "target": self.data["audio"]["target"][classes_mask],
            "url": self.data["audio"]["url"][classes_mask]
        }

    @property
    def features_processed_folder(self):
        return Path().cwd() / "avgzsl_benchmark_datasets/UCF/_features_processed"

    @property
    def all_class_ids(self):
        return np.asarray([self.class_to_idx[name.lower()] for name in self.all_class_names])

    @property
    def train_train_ids(self):
        return np.asarray([self.class_to_idx[name.lower()] for name in self.train_train_class_names])

    @property
    def val_seen_ids(self):
        return np.asarray([self.class_to_idx[name.lower()] for name in self.val_seen_class_names])

    @property
    def val_unseen_ids(self):
        return np.asarray([self.class_to_idx[name.lower()] for name in self.val_unseen_class_names])

    @property
    def test_train_ids(self):
        return np.asarray([self.class_to_idx[name.lower()] for name in self.test_train_class_names])

    @property
    def test_seen_ids(self):
        return np.asarray([self.class_to_idx[name.lower()] for name in self.test_seen_class_names])

    @property
    def test_unseen_ids(self):
        return np.asarray([self.class_to_idx[name.lower()] for name in self.test_unseen_class_names])

    @property
    def text_label_mapping(self):
        df = pd.read_csv(self.root / "class-split/ucf_w2v_class_names.csv")
        return {val: df.original[idx] for idx, val in enumerate(df.manual)}

    @property
    def classes(self):
        if self.zero_shot_split:
            return np.sort(np.concatenate((self.seen_class_ids, self.unseen_class_ids)))

        else:
            if self.zero_shot_mode == "all":
                return self.all_class_ids
            elif self.zero_shot_mode == "seen":
                return self.seen_class_ids
            elif self.zero_shot_mode == "unseen":
                return self.unseen_class_ids
            else:
                raise AttributeError(f"Zero shot mode has to be either all, seen or unseen. Is {self.zero_shot_mode}")

    @property
    def class_to_idx(self):
        return {_class.lower(): i for i, _class in enumerate(sorted(self.all_class_names))}

    @property
    def all_class_names(self):
        return get_class_names(self.root / "class-split/all_class.txt")

    @property
    def seen_class_names(self):
        if self.dataset_split == "train":
            return self.train_train_class_names
        elif self.dataset_split == "val":
            return self.val_seen_class_names
        elif self.dataset_split == "train_val":
            return np.concatenate((self.train_train_class_names, self.val_unseen_class_names))
        elif self.dataset_split == "test":
            return self.test_seen_class_names
        else:
            raise AttributeError("Dataset split has to be in {train,val,train_val,test}")

    @property
    def unseen_class_names(self):
        if self.dataset_split == "train":
            return np.array([])
        elif self.dataset_split == "val":
            return self.val_unseen_class_names
        elif self.dataset_split == "train_val":
            return np.array([])
        elif self.dataset_split == "test":
            return self.test_unseen_class_names
        else:
            raise AttributeError("Dataset split has to be in {train,val,train_val,test}")

    @property
    # @lru_cache(maxsize=128)
    def seen_class_ids(self):
        return np.asarray([self.class_to_idx[name.lower()] for name in self.seen_class_names])

    @property
    # @lru_cache(maxsize=128)
    def unseen_class_ids(self):
        return np.asarray([self.class_to_idx[name.lower()] for name in self.unseen_class_names])

    @property
    def train_train_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_1_train.txt")

    @property
    def val_seen_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_1_val_seen.txt")

    @property
    def val_unseen_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_1_val_unseen.txt")

    @property
    def test_train_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_2_train.txt")

    @property
    def test_seen_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_2_test_seen.txt")

    @property
    def test_unseen_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_2_test_unseen.txt")

    def __init__(self, args, dataset_split, zero_shot_mode, download=False, transform=None):
        super(UCFDataset, self).__init__()
        self.logger = logging.getLogger()
        self.logger.info(
            f"Initializing Dataset {self.__class__.__name__}\t"
            f"Dataset split: {dataset_split}\t"
            f"Zero shot mode: {zero_shot_mode}")
        self.args = args
        self.root = args.root_dir
        self.dataset_name = args.dataset_name
        self.feature_extraction_method = args.feature_extraction_method
        self.dataset_split = dataset_split
        self.zero_shot_mode = zero_shot_mode
        self.zero_shot_split = args.zero_shot_split

        self.transform = transform

        self.preprocess()
        self.data = self.get_data()

    def __getitem__(self, item):
        raise NotImplementedError()

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return self.training_file.exists() and self.val_file.exists() and self.test_file.exists() and self.train_val_file.exists()

    def preprocess(self):
        if self._check_exists():
            return

        (self.features_processed_folder / self.feature_extraction_method).mkdir(parents=True, exist_ok=True)


        self.logger.info('Processing extracted features for faster training (only done once)...')
        self.logger.info(
            f"Processed files will be stored locally in {(self.features_processed_folder / self.feature_extraction_method).resolve()}"
        )

        training_set = self.read_dataset(dataset_type="train")
        val_set = self.read_dataset(dataset_type="val")
        train_val_set = self.read_dataset(dataset_type="train_val")
        test_set = self.read_dataset(dataset_type="test")

        with self.training_file.open('wb') as f:
            self.logger.info(f"Dumping to {self.training_file}")
            pickle.dump(training_set, f, pickle.HIGHEST_PROTOCOL)
        with self.val_file.open('wb') as f:
            self.logger.info(f"Dumping to {self.val_file}")
            pickle.dump(val_set, f, pickle.HIGHEST_PROTOCOL)
        with self.train_val_file.open('wb') as f:
            self.logger.info(f"Dumping to {self.train_val_file}")
            pickle.dump(train_val_set, f, pickle.HIGHEST_PROTOCOL)
        with self.test_file.open('wb') as f:
            self.logger.info(f"Dumping to {self.test_file}")
            pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)

        if not self._check_exists():
            raise RuntimeError("Dataset not found after preprocessing!")
        self.logger.info("Successfully finished preprocessing.")

    def get_data(self):
        if self.dataset_split == "train":
            data_file = self.training_file
        elif self.dataset_split == "val":
            data_file = self.val_file
        elif self.dataset_split == "train_val":
            data_file = self.train_val_file
        elif self.dataset_split == "test":
            data_file = self.test_file
        else:
            raise AttributeError("Dataset_split has to be either train, val or test.")

        load_path = (self.features_processed_folder / data_file).resolve()
        self.logger.info(f"Loading processed data from disk from {load_path}")
        with load_path.open('rb') as f:
            return pickle.load(f)

    def read_dataset(self, dataset_type):
        result_audio = self.get_data_by_modality(modality="audio", dataset_type=dataset_type)
        result_video = self.get_data_by_modality(modality="video", dataset_type=dataset_type)
        assert torch.equal(result_audio["target"], result_video["target"])
        assert np.array_equal(result_audio["url"], result_video["url"])
        result_text = self.get_data_by_modality(modality="text", dataset_type=dataset_type)
        return {"audio": result_audio, "video": result_video, "text": result_text}

    def get_data_by_modality(self, modality, dataset_type="train"):
        # import pdb; pdb.set_trace()
        result = {"data": [], "target": [], "url": []}
        if modality == "text":
            data_raw = np.load(
                (
                        self.root / "features" / self.feature_extraction_method / "text/word_embeddings_ucf_normed.npy").resolve(),
                allow_pickle=True).item()
            data_raw_sorted = dict(sorted(data_raw.items()))
            result["data"] = list(data_raw_sorted.values())
            result["target"] = [self.class_to_idx[self.text_label_mapping[key].lower()] for key in
                                list(data_raw_sorted.keys())]

        elif modality == "audio" or modality == "video":
            split_names = []
            if dataset_type == "train":
                split_names.append("stage_1_train")
            elif dataset_type == "val":
                split_names.append("stage_1_val_seen")
                split_names.append("stage_1_val_unseen")
            elif dataset_type == "train_val":
                split_names.append("stage_1_train")
                split_names.append("stage_1_val_seen")
                split_names.append("stage_1_val_unseen")
            elif dataset_type == "test":
                split_names.append("stage_2_test_seen")
                split_names.append("stage_2_test_unseen")
            else:
                raise AttributeError("Dataset type incompatible. Has to be either train, val or test.")

            for split_name in split_names:
                modality_path = (
                        self.root / "features" / self.feature_extraction_method / f"{modality}/{split_name}").resolve()
                files = modality_path.iterdir()
                for file in tqdm(files, total=len(list(modality_path.glob('*'))),
                                 desc=f"{dataset_type}:{modality}:{split_name}"):
                    data, url = read_features(file)
                    #assert len(data[
                    #               0]) == self.args.input_size, f"Feature size {len(data[0])} is not compatible with specified --input_size {self.args.input_size}"
                    for i, d in enumerate(data):
                        result["data"].append(d)
                        result["target"].append(self.class_to_idx[file.stem.lower()])
                        result["url"].append(url[i])
        else:
            raise AttributeError("Modality has to be either audio, video or text")
        result["data"] = torch.FloatTensor(result["data"])
        result["target"] = torch.LongTensor(result["target"])
        result["url"] = np.array(result["url"])
        return result


class ActivityNetDataset(data.Dataset):

    @property
    def map_embeddings_target(self):
        w2v_embedding = self.data["text"]["data"][sorted(self.classes)].cuda()
        sorted_classes = sorted(self.classes)
        mapping_dict = {}
        for i in range(len(sorted_classes)):
            mapping_dict[int(sorted_classes[i])] = i
        return w2v_embedding, mapping_dict

    @property
    def training_file(self):
        return self.features_processed_folder / self.feature_extraction_method / f"training{self.zero_shot_split}.pkl"

    @property
    def val_file(self):
        return self.features_processed_folder / self.feature_extraction_method / f"val{self.zero_shot_split}.pkl"

    @property
    def train_val_file(self):
        return self.features_processed_folder / self.feature_extraction_method / f"train_val{self.zero_shot_split}.pkl"

    @property
    def test_file(self):
        return self.features_processed_folder / self.feature_extraction_method / f"test{self.zero_shot_split}.pkl"

    @property
    def targets(self):
        classes_mask = np.where(np.isin(self.data["audio"]["target"], self.classes))[0]
        return self.data["audio"]["target"][classes_mask]

    @property
    def all_data(self):
        classes_mask = np.where(np.isin(self.data["audio"]["target"], self.classes))[0]
        return {
            "audio": self.data["audio"]["data"][classes_mask],
            "video": self.data["video"]["data"][classes_mask],
            "text": self.data["text"]["data"][sorted(self.classes)],
            "target": self.data["audio"]["target"][classes_mask],
            "url": self.data["audio"]["url"][classes_mask]
        }

    @property
    def features_processed_folder(self):
        return Path().cwd() / "avgzsl_benchmark_datasets/ActivityNet/_features_processed"

    @property
    def all_class_ids(self):
        return np.asarray([self.class_to_idx[name.lower()] for name in self.all_class_names])

    @property
    def train_train_ids(self):
        return np.asarray([self.class_to_idx[name.lower()] for name in self.train_train_class_names])

    @property
    def val_seen_ids(self):
        return np.asarray([self.class_to_idx[name.lower()] for name in self.val_seen_class_names])

    @property
    def val_unseen_ids(self):
        return np.asarray([self.class_to_idx[name.lower()] for name in self.val_unseen_class_names])

    @property
    def test_train_ids(self):
        return np.asarray([self.class_to_idx[name.lower()] for name in self.test_train_class_names])

    @property
    def test_seen_ids(self):
        return np.asarray([self.class_to_idx[name.lower()] for name in self.test_seen_class_names])

    @property
    def test_unseen_ids(self):
        return np.asarray([self.class_to_idx[name.lower()] for name in self.test_unseen_class_names])

    @property
    def text_label_mapping(self):
        df = pd.read_csv(self.root / "class-split/activitynet_w2v_class_names.csv")
        return {val: df.original[idx] for idx, val in enumerate(df.manual)}

    @property
    def classes(self):
        if self.zero_shot_split:
            return np.sort(np.concatenate((self.seen_class_ids, self.unseen_class_ids)))

        else:
            if self.zero_shot_mode == "all":
                return self.all_class_ids
            elif self.zero_shot_mode == "seen":
                return self.seen_class_ids
            elif self.zero_shot_mode == "unseen":
                return self.unseen_class_ids
            else:
                raise AttributeError(f"Zero shot mode has to be either all, seen or unseen. Is {self.zero_shot_mode}")

    @property
    def class_to_idx(self):
        return {_class.lower(): i for i, _class in enumerate(sorted(self.all_class_names))}

    @property
    def all_class_names(self):
        return get_class_names(self.root / "class-split/all_class.txt")

    @property
    def seen_class_names(self):
        if self.dataset_split == "train":
            return self.train_train_class_names
        elif self.dataset_split == "val":
            return self.val_seen_class_names
        elif self.dataset_split == "train_val":
            return np.concatenate((self.train_train_class_names, self.val_unseen_class_names))
        elif self.dataset_split == "test":
            return self.test_seen_class_names
        else:
            raise AttributeError("Dataset split has to be in {train,val,train_val,test}")

    @property
    def unseen_class_names(self):
        if self.dataset_split == "train":
            return np.array([])
        elif self.dataset_split == "val":
            return self.val_unseen_class_names
        elif self.dataset_split == "train_val":
            return np.array([])
        elif self.dataset_split == "test":
            return self.test_unseen_class_names
        else:
            raise AttributeError("Dataset split has to be in {train,val,train_val,test}")

    @property
    # @lru_cache(maxsize=128)
    def seen_class_ids(self):
        return np.asarray([self.class_to_idx[name.lower()] for name in self.seen_class_names])

    @property
    # @lru_cache(maxsize=128)
    def unseen_class_ids(self):
        return np.asarray([self.class_to_idx[name.lower()] for name in self.unseen_class_names])

    @property
    def train_train_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_1_train.txt")

    @property
    def val_seen_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_1_val_seen.txt")

    @property
    def val_unseen_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_1_val_unseen.txt")

    @property
    def test_train_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_2_train.txt")

    @property
    def test_seen_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_2_test_seen.txt")

    @property
    def test_unseen_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_2_test_unseen.txt")

    def __init__(self, args, dataset_split, zero_shot_mode, download=False, transform=None):
        super(ActivityNetDataset, self).__init__()
        self.logger = logging.getLogger()
        self.logger.info(
            f"Initializing Dataset {self.__class__.__name__}\t"
            f"Dataset split: {dataset_split}\t"
            f"Zero shot mode: {zero_shot_mode}")
        self.args = args
        self.root = args.root_dir
        self.dataset_name = args.dataset_name
        self.feature_extraction_method = args.feature_extraction_method
        self.dataset_split = dataset_split
        self.zero_shot_mode = zero_shot_mode
        self.zero_shot_split = args.zero_shot_split

        self.transform = transform

        self.preprocess()
        self.data = self.get_data()

    def __getitem__(self, item):
        raise NotImplementedError()

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return self.training_file.exists() and self.val_file.exists() and self.test_file.exists() and self.train_val_file.exists()

    def preprocess(self):
        if self._check_exists():
            return

        (self.features_processed_folder / self.feature_extraction_method).mkdir(parents=True, exist_ok=True)

        self.logger.info('Processing extracted features for faster training (only done once)...')
        self.logger.info(
            f"Processed files will be stored locally in {(self.features_processed_folder / self.feature_extraction_method).resolve()}"
        )

        training_set = self.read_dataset(dataset_type="train")
        val_set = self.read_dataset(dataset_type="val")
        train_val_set = self.read_dataset(dataset_type="train_val")
        test_set = self.read_dataset(dataset_type="test")

        with self.training_file.open('wb') as f:
            self.logger.info(f"Dumping to {self.training_file}")
            pickle.dump(training_set, f, pickle.HIGHEST_PROTOCOL)
        with self.val_file.open('wb') as f:
            self.logger.info(f"Dumping to {self.val_file}")
            pickle.dump(val_set, f, pickle.HIGHEST_PROTOCOL)
        with self.train_val_file.open('wb') as f:
            self.logger.info(f"Dumping to {self.train_val_file}")
            pickle.dump(train_val_set, f, pickle.HIGHEST_PROTOCOL)
        with self.test_file.open('wb') as f:
            self.logger.info(f"Dumping to {self.test_file}")
            pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)

        if not self._check_exists():
            raise RuntimeError("Dataset not found after preprocessing!")
        self.logger.info("Successfully finished preprocessing.")

    def get_data(self):
        if self.dataset_split == "train":
            data_file = self.training_file
        elif self.dataset_split == "val":
            data_file = self.val_file
        elif self.dataset_split == "train_val":
            data_file = self.train_val_file
        elif self.dataset_split == "test":
            data_file = self.test_file
        else:
            raise AttributeError("Dataset_split has to be either train, val or test.")

        load_path = (self.features_processed_folder / data_file).resolve()
        self.logger.info(f"Loading processed data from disk from {load_path}")
        with load_path.open('rb') as f:
            return pickle.load(f)

    def read_dataset(self, dataset_type):
        result_audio = self.get_data_by_modality(modality="audio", dataset_type=dataset_type)
        result_video = self.get_data_by_modality(modality="video", dataset_type=dataset_type)
        assert torch.equal(result_audio["target"], result_video["target"])
        assert np.array_equal(result_audio["url"], result_video["url"])
        result_text = self.get_data_by_modality(modality="text", dataset_type=dataset_type)
        return {"audio": result_audio, "video": result_video, "text": result_text}

    def get_data_by_modality(self, modality, dataset_type="train"):
        # import pdb; pdb.set_trace()
        result = {"data": [], "target": [], "url": []}
        if modality == "text":
            data_raw = np.load(
                (
                        self.root / "features" / self.feature_extraction_method / "text/word_embeddings_activity_normed.npy").resolve(),
                allow_pickle=True).item()
            data_raw_sorted = dict(sorted(data_raw.items()))
            result["data"] = list(data_raw_sorted.values())
            result["target"] = [self.class_to_idx[self.text_label_mapping[key].lower()] for key in
                                list(data_raw_sorted.keys())]

        elif modality == "audio" or modality == "video":
            split_names = []
            if dataset_type == "train":
                split_names.append("stage_1_train")
            elif dataset_type == "val":
                split_names.append("stage_1_val_seen")
                split_names.append("stage_1_val_unseen")
            elif dataset_type == "train_val":
                split_names.append("stage_1_train")
                split_names.append("stage_1_val_seen")
                split_names.append("stage_1_val_unseen")
            elif dataset_type == "test":
                split_names.append("stage_2_test_seen")
                split_names.append("stage_2_test_unseen")
            else:
                raise AttributeError("Dataset type incompatible. Has to be either train, val or test.")

            for split_name in split_names:
                modality_path = (
                        self.root / "features" / self.feature_extraction_method / f"{modality}/{split_name}").resolve()
                files = modality_path.iterdir()
                for file in tqdm(files, total=len(list(modality_path.glob('*'))),
                                 desc=f"{dataset_type}:{modality}:{split_name}"):
                    data, url = read_features(file)
                    #assert len(data[
                    #               0]) == self.args.input_size, f"Feature size {len(data[0])} is not compatible with specified --input_size {self.args.input_size}"
                    for i, d in enumerate(data):
                        result["data"].append(d)
                        result["target"].append(self.class_to_idx[file.stem.lower()])
                        result["url"].append(url[i])
        else:
            raise AttributeError("Modality has to be either audio, video or text")
        result["data"] = torch.FloatTensor(result["data"])
        result["target"] = torch.LongTensor(result["target"])
        result["url"] = np.array(result["url"])
        return result
