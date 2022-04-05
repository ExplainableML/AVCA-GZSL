from pathlib import Path
import os
import sys
sys.path.append("..")
from tqdm import tqdm
import torch
from c3d.c3d import C3D
import torchvision
import csv
from csv import reader
import numpy as np
from timeit import default_timer as timer
from audioset_vggish_tensorflow_to_pytorch.vggish import VGGish
from audioset_vggish_tensorflow_to_pytorch.audioset import vggish_input, vggish_postprocess
import cv2
from pydub import AudioSegment
import pickle

output_list_no_average=[]
output_list_average=[]

path=Path("/home/omercea19/akata-shared/datasets/UCF101/UCF-101") # path to search for videos

dict_csv={}
list_classes=[]
count=0
dict_classes_ids={}


for f in tqdm(path.glob("**/*.avi")):
    class_name=str(f).split("/")[-2]
    if class_name not in list_classes:
        list_classes.append(class_name)

list_classes.sort()

for index,val in enumerate(sorted(list_classes)):
    dict_classes_ids[val]=index

device = 'cuda:0'
pytorch_model = VGGish()
pytorch_model.load_state_dict(torch.load('/home/omercea19/ExplainableAudioVisualLowShotLearning/audioset_vggish_tensorflow_to_pytorch/pytorch_vggish.pth')) # path of the vggish pretrained network
pytorch_model = pytorch_model.to(device)

pytorch_model.eval()

model=C3D().cuda()
model.load_state_dict(torch.load('/home/omercea19/ExplainableAudioVisualLowShotLearning/c3d.pickle'), strict=True)
model.eval()
counter=0

for f in tqdm(path.glob("**/*.avi")):

    counter+=1
    if counter%3000==0:
        with open('/mnt/ucf_features_averaged_fixed', 'wb') as handle:
            pickle.dump(output_list_average, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('/mnt/ucf_features_no_averaged_fixed', 'wb') as handle:
            pickle.dump(output_list_no_average, handle, protocol=pickle.HIGHEST_PROTOCOL)

    try:
        mp4_version = AudioSegment.from_file(str(f), "avi")
        mp4_version.export("/mnt/ucf_dummy.wav", format="wav")

        audio = torch.from_numpy(vggish_input.wavfile_to_examples("/mnt/ucf_dummy.wav"))
        audio=audio.float().to(device)
        audio = audio.unsqueeze(dim=1)
        vggish_output = pytorch_model(audio)
        vggish_output = vggish_output.detach().cpu().numpy()
        post_processor= vggish_postprocess.Postprocessor('/home/omercea19/ExplainableAudioVisualLowShotLearning/audioset_vggish_tensorflow_to_pytorch/vggish_pca_params.npz')
        vggish_output = post_processor.postprocess(vggish_output)
        vggish_output_average=np.average(vggish_output, axis=0)


        cap = cv2.VideoCapture(str(f))
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video = torch.zeros((frameCount, frameHeight,frameWidth, 3), dtype=torch.float32)

        fc = 0
        ret = True

        while (fc < frameCount and ret):
            ret, image=cap.read()
            if ret==True:
                torch_image=torch.from_numpy(image)
                video[fc]=torch_image
                fc += 1

        cap.release()

        list_clips=[]

        p= torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((128, 171)),
            torchvision.transforms.CenterCrop((112,112)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                                             std=[0.22803, 0.22145, 0.216989]),

        ])
        for i in range(0,fc, 16):
            dummy = torch.zeros((16,3, 112, 112), dtype=torch.float32)
            clip=video[i:i+16]
            for j in range(clip.shape[0]):
                frame=clip[j]
                frame=frame.permute(2, 0, 1)
                frame=p(frame)
                dummy[j]=frame

            dummy=dummy.permute(1,0,2,3)
            dummy=((torch.unsqueeze(dummy,0)).float()).cuda()
            output=model(dummy)
            output=torch.squeeze(output)
            output=output.cpu().detach().numpy()
            list_clips.append(output)

        list_clips=np.array(list_clips)
        list_clips_average=np.average(list_clips, axis=0)

    except Exception as e:
        print(e)
        print(f)
        continue

    name_file=str(f).split("/")[-1]
    class_name=str(f).split("/")[-2]
    class_id=dict_classes_ids[class_name]

    result_list=[list_clips, class_id, vggish_output, name_file]
    result_list_average=[list_clips_average, class_id, vggish_output_average, name_file]

    output_list_no_average.append(result_list)
    output_list_average.append(result_list_average)


with open('/mnt/ucf_features_averaged_fixed', 'wb') as handle:
    pickle.dump(output_list_average, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('/mnt/ucf_features_no_averaged_fixed', 'wb') as handle:
    pickle.dump(output_list_no_average, handle, protocol=pickle.HIGHEST_PROTOCOL)

