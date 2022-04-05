# Audio-visual Generalised Zero-shot Learning with Cross-modal Attention and Language

This repository is the official implementation of [Audio-visual Generalised Zero-shot Learning with Cross-modal Attention and Language
](https://arxiv.org/abs/2203.03598).
<img src="/img/audio-visual-zsl.png" width="700" height="400">

## Requirements

Install all required dependencies into a new virtual environment via conda.
```shell
conda env create -f AVCA_env.yml
```

# Obtaining GZSL benchmark datasets: VGGSound-GZSL, ActivityNet-GZSL, and UCF-GZSL
Our GZSL benchmark splits are based on the VGGSound, ActivityNet, and UCF datasets.

## Downloading our features
In case you want to use the features that we extracted, you can download them from [here](https://drive.google.com/file/d/1NzpSVL-2nXRFH6pvHNOZeRhu_jfDL0ZE/view?usp=sharing). The zip file called ```avgzsl_benchmark_datasets.zip``` contains the data structure presented in [Dataset structure section](#dataset-structure).

Moreover, the zip files also contain the cache files, so you will not need to generate them as mentioned in the [Training section](#training).

The unzipped files should be placed in the ```avgzsl_benchmark_datasets/``` folder.


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

The ```data``` field contains either the audio/video features, depending on whether the parent folder is ```audio``` or ```video```. The ```video_urls``` contain the file name of each video. The ```.h5``` files are named according to the class names.

The ```stage_*.txt``` files consist of lists of classes used for the two training stages and for testing.

The ```stage_*.csv``` files have one entry on each row with the following structure ```[video_name,class_name,label_code]```, where the ```video_name```  is the name of the video, ```class_name``` is the name of the class and ```label_code``` contains the index of the ```class_name``` in the alphabetically sorted list of all the classes in the original dataset.

The ```all_class.txt``` files contain all the class names in the original dataset.

The ```{dataset_name}_w2v_class_names.csv``` contains the mapping between the class names used to extract the word2vec embedding and the original class names in each dataset. This mapping is important, since some class annotations did not contain spaces between the words, while the class names used to extract the word2vec embeddings contain these spaces.

The ```.npy``` files contain the word2vec embeddings and the structure is a dictionary, where the key is the class name and the value is the w2v embedding corresponding to that class, which has a dimension of 300. These files are stored in ```w2v_features/```.


## Extracting features for VGGSound-GZSL/UCF-GZSL/ActivityNet-GZSL

In case you want to extract the features from the dataset yourself, follow the instructions in this section.

### Dataset download
You first need to download the VGGSound, ActivityNet1.3, and UCF101 datasets.

The VGGSound dataset (licensed under [CC BY 4.0](https://github.com/hche11/VGGSound/blob/master/LICENCE.txt)) can be downloaded from [https://www.robots.ox.ac.uk/~vgg/data/vggsound/](https://www.robots.ox.ac.uk/~vgg/data/vggsound/).

The ActivityNet1.3 dataset (licensed under [MIT license](https://github.com/activitynet/ActivityNet/blob/master/LICENSE)) can be downloaded from [http://activity-net.org/download.html](http://activity-net.org/download.html). We used the action annotations to trim all videos.

The UCF101 dataset can be downloaded from [https://www.crcv.ucf.edu/data/UCF101.php](https://www.crcv.ucf.edu/data/UCF101.php).

!!! IMPORTANT !!! For C3D/VGGish features, we converted all videos to 25 fps for each datasets and then we extracted the features. 


### Obtaining C3D/VGGish features.
For obtaining the C3D/VGGish features we used 
```
python cls_feature_extraction/get_features_activitynet.py
python cls_feature_extraction/get_features_ucf.py
python cls_feature_extraction/get_features_vggsound.py
```
These scripts will extract both the visual/audio classification features using the C3D/VGGish.
The features are saved in a single ```.pkl``` file, where each video is represented as
```[video_features, class_id, audio_features, name_file]```. Again, the class_id is represented by the index of the class in the sorted list of all classes in the dataset.

For setting up the VGGish feature extraction, please refer to the ```audioset_vggish_tensorflow_to_pytorch``` folder in this github repository which is a modified version of this original repository https://github.com/tcvrick/audioset-vggish-tensorflow-to-pytorch. The only things that should be done is to download all the necessarily files in the ```audioset_vggish_tensorflow_to_pytorch``` folder as shown in the instructions there. 
For the C3D feature extraction, please follow this repository https://github.com/DavideA/c3d-pytorch. The only thing that should be done here is to save the C3D pretrained weights.


### Obtaining SeLaVi features

For obtaining the SeLaVi features we used the following command
```
python3 selavi_feature_extraction/get_clusters.py \
--root_dir <path_to_raw_videos> \
--weights_path <path_to_pretrained_selavi_vgg_sound.pth> \
--mode train \
--pretrained False \
--aud_sample_rate 44100 \
--use_mlp False \
--dataset {activity,ucf,vggsound} \
--headcount 2 \
--exp_desc <experiment_description> \
--output_dir <path_to_save_extracted_features> \
--batch_size 1 \
--workers 0
```
For the detailed setup, please refer to the ```selavi_feature_extraction``` folder.  
We use the provided SeLaVi baseline which was [pre-trained on VGGSound](https://github.com/facebookresearch/selavi#model-zoo). The script will extract the audio/visual features in a self-supervised fashion. For this, we slightly adapted the code of the [SeLaVi repository](https://github.com/facebookresearch/selavi). We extract audio/visual time windows of one second in a sliding-window manner and compute their average. All videos are stored in a single ```.pkl``` file, which is a dictionary with the following structure  
```{video: [video_features], labels:[class_ids], audio: [audio_features], filenames: [filenames]}```.  

### Splitting features  

Once the files are extracted using the above python scripts into the ```.pkl``` file, they will need to be arranged in the folder structure presented in the [Dataset section](#dataset-structure). Arranging them in such a folder structure can be done by running the scripts ```splitting_scripts/{dataset_name}_split_features.py```

# Training
For training the systems, we provide the commands in ```run_scripts/``` which contains folders for each dataset. There is a ```.sh``` file for each baselines in each of these folders. These ```.sh``` files contain commands for running both stages for SeLaVi features and C3D/VGGish features.

Here is an example how to train AVCA for both stages on the UCF-GZSL dataset using SeLaVi features.

```
python main.py --root_dir avgzsl_benchmark_datasets/UCF --feature_extraction_method main_features --input_size_audio 512 --input_size_video 512  --epochs 50 --lr_scheduler --dataset_name UCF --zero_shot_split main_split  --AVCA --lr 0.001 --n_batches 500 --embeddings_hidden_size 512  --decoder_hidden_size 512 --embedding_dropout 0.2 --decoder_dropout 0.3 --additional_dropout 0.5 --depth_transformer 1 --additional_triplets_loss --first_additional_triplet 1  --second_additional_triplet 1 --momentum 0.1 --reg_loss --exp_name attention_ucf_val_main

python main.py --root_dir avgzsl_benchmark_datasets/UCF --feature_extraction_method main_features --input_size_audio 512 --input_size_video 512 --epochs 50 --lr_scheduler --dataset_name UCF --zero_shot_split main_split   --AVCA --lr 0.001 --retrain_all --save_checkpoints --n_batches 500 --embeddings_hidden_size 512  --decoder_hidden_size 512 --embedding_dropout 0.2 --decoder_dropout 0.3 --additional_dropout 0.5 --depth_transformer 1 --additional_triplets_loss --first_additional_triplet 1  --second_additional_triplet 1 --momentum 0.1 --reg_loss --exp_name attention_ucf_all_main

```



!!! IMPORTANT !!! The networks generated by the above two commands will be saved in the ```runs``` folder which will be created inside the project directory tree. The path to the networks will be ```runs/attention_ucf_all_main``` and ```runs/attention_ucf_val_main```. During evaluation these paths will be required to evaluate the model. It can be observed that the directory where the networks are stored inside ```runs/``` is given by the ```--exp_name```.

Next we provide a description of the essential parameters that we use in main.py.

The ``--feature_extraction_method`` will look for a folder with the same name in ``avgzsl_benchmark_datasets/{dataset_name}/features``. The ``--zero_shot_split`` will look for a folder with the same name in ```avgzsl_benchmark_datasets/{dataset_name}/class_splits```

```
arguments:
  --root_dir  Path to dataset directory: '{root_dir}/{dataset_name}. Expected subfolder structure to be the same as mentioned in dataset structure
  --feature_extraction_method - Name of folder containing .h5 files.
  --zero_shot_split - Name of the folder containin the class splits .txt/.csv files.
  --dataset_name {VGGSound, UCF, ActivityNet} - Name of the dataset to use
  --exp_name - Name of the folder where to save the experiment (the model+logs).
  --save_checkpoints - Used to indicate if the model should be saved each epoch. Used only in the second stage training.
  --retrain_all - Used to indicate if we train for first stage or for second stage. 
  ```
  
```main.py``` will automatically create a cache file containing all extracted features during the first time the program is run.  
The cache files will be stored in the ```avgzsl_benchmark_datasets/{dataset_name}/_features_preprocessed``` having the same name as that of the features. On subsequent runs and during evaluation, the program will load these cache files. The ```avgzsl_benchmark_datasets``` folder will be located in the project file tree



# Evaluation

## Dowloading pre-trained models

[Here](https://drive.google.com/file/d/1xyxfnHjG1GF7VtcJRHGmI7dZ-qdoRrYI/view?usp=sharing), you can download our trained AVCA models and baselines which are located in ```model.zip``` The models that have in the naming ```val``` are the first stage models, and those that contain ```all``` are the second stage models. 

!!! IMPORTANT !!! Put the content of ```model.zip``` in the ```runs/``` folder and then simply use the ```get_evaluation.py``` commands from ```run_scripts/``` on these models to evaluate them. ```runs/``` folder should be created as ```{repository_root}/runs/``` if it is not already created. 

## Re-producing our results

Here is an examples for evaluating AVCA on UCF-GZSL using SeLaVi features. 

```
python get_evaluation.py --load_path_stage_A runs/attention_ucf_val_main --load_path_stage_B runs/attention_ucf_all_main  --dataset_name UCF --AVCA
```


Next we provide a description of the essential parameters that we need to use in get_evaluation.py.

```
arguments:
  --load_path_stage_A - Contains the path to the folder that stores the model and logs from stage 1
  --load_path_stage_B - Contains the path to the folder that stores the model and logs from stage 2
  --dataset_name - Represents the name of the dataset that will be used to evaluate the model
  
```

Additionally, the ```.sh``` files in ```run_scripts/``` also contain the evaluation script for each baseline that should be run in order to obtain the results.



# Results 
## Using SeLaVi features
### GZSL performance on VGGSound-GZSL, UCF-GZSL, ActivityNet-GZSL

| Method             | VGGSound-GZSL          | UCF-GZSL        | ActivityNet-GZSL |
|--------------------|------------------------|-----------------|------------------|
| ALE                |    0.53                |    23.66        |   3.94           |
| SJE                |    2.15                |    26.50        |   5.57           |
| DEVISE             |    2.08                |    23.56        |   4.91           |
| APN                |    5.11                |    20.61        |   7.27           |
| f-vaegan-d2        |    1.77                |    11.37        |   2.87           |
| CJME               |    6.17                |    12.48        |   5.12           |
| AVGZSLNET          |    5.83                |    18.05        |   6.44           |
| **AVCA**           |    **6.31**            |    **27.15**    |   **12.13**      |


### ZSL performance on VGGSound-GZSL, UCF-GZSL, ActivityNet-GZSL

| Method             | VGGSound-GZSL          | UCF-GZSL        | ActivityNet-GZSL |
|--------------------|------------------------|-----------------|------------------|
| ALE                |    5.48                |    16.32        |   7.90           |
| SJE                |    4.06                |    18.93        |   7.08           |
| DEVISE             |    5.59                |    16.09        |   8.53           |
| APN                |    4.49                |    16.44        |   6.34           |
| f-vaegan-d2        |    1.91                |    11.11        |   2.40           |
| CJME               |    5.16                |    8.29         |   5.84           |
| AVGZSLNET          |    5.28                |    13.65        |   5.40           |
| **AVCA**           |    **6.00**            |    **20.01**    |   **9.13**       |

## Using VGGish/C3D features
### GZSL performance on VGGSound-GZSL, UCF-GZSL, ActivityNet-GZSL

| Method             | VGGSound-GZSL          | UCF-GZSL        | ActivityNet-GZSL |
|--------------------|------------------------|-----------------|------------------|
| ALE                |  3.23                  |  35.37          |  1.55            |
| SJE                |  4.69                  |  24.28          |  2.35            |
| DEVISE             |  3.64                  |  31.98          |  0.33            |
| APN                |  6.29                  |  18.35          |  3.58            |
| CJME               |  3.68                  |  28.65          |  7.32            |
| AVGZSLNET          |  5.26                  |  36.51          |  8.30            |
| **AVCA**           |  **8.31**              |  **41.34**      |  **9.92**        |


### ZSL performance on VGGSound-GZSL, UCF-GZSL, ActivityNet-GZSL

| Method             | VGGSound-GZSL          | UCF-GZSL        | ActivityNet-GZSL |
|--------------------|------------------------|-----------------|------------------|
| ALE                |  4.97                  |  32.30          | 6.16             |
| SJE                |  3.22                  |  32.47          | 4.35             |
| DEVISE             |  4.72                  |  35.48          | 5.84             |
| APN                |  6.50                  |  29.69          | 3.97             |
| CJME               |  3.72                  |  29.01          | 6.29             |
| AVGZSLNET          |  4.81                  |  31.51          | 6.39             |
| **AVCA**           |  **6.91**              |  **37.72**      | **7.58**         |

# Project Structure
```audioset_vggish_tensorflow_to_pytorch``` - Contains the code which is used to obtain the audio features using VGGish.

```c3d``` - Folder contains the code for the C3D network.

```selavi_feature_extraction``` - Contains the code used to extract the SeLaVi features.

```src``` - Contains the code used throughout the project for dataloaders/models/training/testing.

```cls_feature_extraction``` - Contains the code used to extract the C3D/VGGish features from all 3 datasets.

```avgzsl_benchmark_datasets``` - Contains the class splits and the video splits for each dataset for both features from SeLaVi and features from C3D/VGGish.

```splitting_scripts``` - Contains files from spltting our dataset into the required structure. 

```w2v_features``` - Contains the w2v embeddings for each dataset.
```run_scripts``` - Contains the scripts for training/evaluation for all models for each dataset.


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

