from pathlib import Path

import fasttext
import numpy as np
import pandas as pd


def main():
    # model = fasttext.load_model("/shared-local/datasets/word2vec/wiki.en.bin")
    model = fasttext.load_model("/shared-network/lriesch29/word2vec/wiki.en.bin")
    #audioset_classes = get_audioset_classes()
    #vggsound_classes = get_vggsound_classes()
    #data_audioset = extract_label_embeddings(model, audioset_classes)
    #data_vggsound = extract_label_embeddings(model, vggsound_classes)
    #ucf_classes = get_ucf_classes()
    activity_classes = _get_class_names(Path("/home/lriesch29/ExplainableAudioVisualLowShotLearning/dat/ActivityNet/class-split/all_class.txt"))
    #data_ucf = extract_label_embeddings(model, ucf_classes)
    data_activity = extract_label_embeddings(model, activity_classes)
    #print(len(data_audioset))
    #print(len(data_vggsound))
    #print(len(data_ucf))
    print(len(data_activity))
    #np.save('word_embeddings_audiosetzsl_normed.npy', data_audioset)
    #np.save('word_embeddings_vggsound_normed.npy', data_vggsound)
    #np.save('word_embeddings_ucf_normed.npy', data_ucf)
    np.save('word_embeddings_activity_normed.npy', data_activity)


def get_audioset_classes():
    path_audioset = Path("data/all_class_clean.txt")
    classes = []
    with path_audioset.open() as f:
        for line in f:
            classes.append(line.strip())
    return classes

#def get_ucf_classes():
#    path = Path("data/ucf_class_clean.txt")
#    classes = []
#    with path.open() as f:
#        for line in f:
#            classes.append(line.strip())
#    return classes

def get_ucf_classes():
    return list(pd.read_csv("/home/lriesch29/akata-shared/shared/avzsl/UCF/class-split/ucf_manual_names_ask.csv").manual)

def get_vggsound_classes():
    path_vggsound = Path("data/vggsound_class_clean.txt")
    classes = []
    with path_vggsound.open() as f:
        for line in f:
            classes.append(line.strip())
    return classes

def _get_class_names(path):
    if isinstance(path, str):
        path = Path(path)
    with path.open("r") as f:
        classes = sorted([line.strip() for line in f])
    return classes

def extract_label_embeddings(model, classes, normalize=True):
    result = {}
    for c in classes:
        value = np.array(model.get_word_vector(c))
        if normalize:
            value = value / np.linalg.norm(value)
            np.testing.assert_almost_equal(np.linalg.norm(value), 1)
        result[c] = value
    return result


if __name__ == '__main__':
    main()
