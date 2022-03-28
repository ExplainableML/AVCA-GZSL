# Audio-visual Generalised Zero-shot Learning with Cross-modal Attention and Language

This repository is the official implementation of [Audio-visual Generalised Zero-shot Learning with Cross-modal Attention and Language
](https://arxiv.org/abs/2203.03598).



# GZSL benchmark datasets: VGGSound-GZSL, ActivityNet-GZSL, and UCF-GZSL
Our GZSL benchmark splits are based on the VGGSound, ActivityNet, and UCF datasets.

## Dataset download
The VGGSound dataset (licensed under [CC BY 4.0](https://github.com/hche11/VGGSound/blob/master/LICENCE.txt)) can be downloaded from [https://www.robots.ox.ac.uk/~vgg/data/vggsound/](https://www.robots.ox.ac.uk/~vgg/data/vggsound/).

The ActivityNet1.3 dataset (licensed under [MIT license](https://github.com/activitynet/ActivityNet/blob/master/LICENSE)) can be downloaded from [http://activity-net.org/download.html](http://activity-net.org/download.html). We used the action annotations to trim all videos.

The UCF101 dataset can be downloaded from [https://www.crcv.ucf.edu/data/UCF101.php](https://www.crcv.ucf.edu/data/UCF101.php).

## GZSL training and evaluation protocol
We introduce a unified two-stage training and evaluation protocol for our GZSL benchmarks.

In the first stage, we train the models on the training set (stage_1_train), and evaluate on the subsets of seen validation classes (stage_1_val_seen) and unseen validation classes (stage_1_val_unseen) to determine the GZSL parameters (e.g. best epoch, calibrated stacking, etc).

In the second training stage, we re-train the models using the full training set (stage_2_train) which includes the first stage's training and validation sets using the GZSL parameters determined during the first training stage. Our final models are then evaluated on the test set which contains samples from seen (stage_2_test_seen) and unseen classes (stage_2_test_unseen). 


## Dataset structure
Our proposed GZSL benchmarks can be found in the ```avgzsl_benchmark_datasets/``` folder.

The structure of the dataset should be 
```
UCF                
├──class_split
|  ├── all_class.txt   # .txt file containing all classes of the UCF101 dataset.
|  ├── ucf_w2v_class_names.csv 
|  ├── main_split        # name of the class split. Can be changed.
|  |   ├── stage_2_test_seen.txt
|  |   ├── stage_2_test_seen.csv
|  |   ├── stage_2_test_unseen.txt
|  |   ├── stage_2_test_unseen.csv
|  |   ├── stage_2_train.txt
|  |   ├── stage_2_train.csv
|  |   ├── stage_1_train.txt
|  |   ├── stage_1_train.csv
|  |   ├── stage_1_val_seen.txt
|  |   ├── stage_1_val_seen.csv
|  |   ├── stage_1_val_unseen.txt
|  |   └── stage_1_val_unseen.csv
|  |
|  └── cls_split
|      └── ...
|             
└── features
    ├── main_features # name of the features. Can be changed.
    |   ├── audio
    |   |   ├── stage_2_test_seen
    |   |   |   ├── ApplyEyeMakeup.h5 #one .h5 file per class for each subsplit.
    |   |   |   └── .... 
    |   |   ├── stage_2_test_unseen
    |   |   |   ├── BandMarching.h5
    |   |   |   └── ....
    |   |   ├── stage_1_val_seen
    |   |   |   ├── Archery.h5
    |   |   |   └── ....
    |   |   ├── stage_1_val_unseen
    |   |   |   ├── ApplyEyeMakeup.h5
    |   |   |   └── ....
    |   |   ├── stage_2_train
    |   |   |   ├── ApplyEyeMakeup.h5
    |   |   |   └── ....
    |   |   ├── stage_1_train
    |   |   |   ├── Archery.h5
    |   |   |   └── ....
    |   ├── text
    |   |   └── word_embeddings_ucf_normed.npy  #this is the file containing the word2vec embeddings
    |   └── video   #same as audio
    |
    └── cls_features
        └── ...
```

The same folder structure is used for VGGSound/ActivityNet and for the SeLaVi features (main_split) and C3D/VGGish features (cls_split). For the SeLaVi features, the folder name is ```main_split/``` and this can be found in each dataset folder in ```avgzsl_benchmark_datasets/```. For C3D/VGGish features the folder name is ```cls_split/```.

The ```.h5``` files were saved as:    

```python
with h5py.File(path+'.h5', "w") as hf:
    hf.create_dataset("data", data=list_features)
    hf.create_dataset("video_urls", dtype="S80", data=list_video_names)
```

The ```data``` field contains either the audio/video features, depending on whether the parent folder is ```audio``` or ```video```. The ```video_urls``` contain the file name of each video. The .h5 files are named according to the class names.

The ```stage_*.txt``` files consist of lists of classes used for the two training stages and for testing.

The ```stage_*.csv``` files have one entry on each row with the following structure ```[video_name,class_name,label_code]```, where the ```video_name```  is the name of the video, ```class_name``` is the name of the class and ```label_code``` contains the index of the ```class_name``` in the alphabetically sorted list of all the classes in the original dataset.

The ```all_class.txt``` files contain all the class names in the original dataset.

The ```{dataset_name}_w2v_class_names.csv``` contains the mapping between the class names used to extract the word2vec embedding and the original class names in each dataset. This mapping is important, since some class annotations did not contain spaces between the words, while the class names used to extract the word2vec embeddings contain these spaces.

The ```.npy``` files contain the word2vec embeddings and the structure is a dictionary, where the key is the class name and the value is the w2v embedding corresponding to that class, which has a dimension of 300. These files are stored in ```w2v_features/```.




# Model training and evaluation
Code to train and evaluate our proposed AVCA model and baseline models on the VGGSound-GZSL, ActivityNet-GZSL, and UCF-GZSL datasets will be uploaded soon.


# References

If you find this code useful, please consider citing:

```
@inproceedings{mercea2022avca,
  author    = {Mercea, Otniel and Riesch, Lukas and Koepke, A. Sophia and Akata, Zeynep},
  title     = {Audio-visual Generalised Zero-shot Learning with Cross-modal Attention and Language},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022}
}
```
