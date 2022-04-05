from pathlib import Path
import os
import sys
import argparse
sys.path.append("..")
from tqdm import tqdm
import torch
from c3d.c3d import C3D
import torchvision
import numpy as np
from audioset_vggish_tensorflow_to_pytorch.vggish import VGGish
from audioset_vggish_tensorflow_to_pytorch.audioset import vggish_input, vggish_postprocess
import cv2
from pydub import AudioSegment
import pickle

parser = argparse.ArgumentParser(description="GZSL with ESZSL")

parser.add_argument('--index', default=0, type=int)
args = parser.parse_args()

print(args.index)



output_list_no_average=[]
output_list_average=[]

path=Path("/home/omercea19/akata-shared/datasets/ActivityNet/v1-2-trim") # path to search for videos

dict_csv={}
list_classes=[]
count=0
dict_classes_ids={}
dict_files_name_to_class={}

list_of_files=[]


for f in tqdm(path.glob("**/*.mp4")):
    class_name=str(f).split("/")[-2]
    list_of_files.append(f)
    if class_name not in list_classes:
        list_classes.append(class_name)

path=Path("/home/omercea19/akata-shared/datasets/ActivityNet/v1-3-trim")

for f in tqdm(path.glob("**/*.mp4")):
    class_name=str(f).split("/")[-2]
    list_of_files.append(f)
    if class_name not in list_classes:
        list_classes.append(class_name)


chunk=int(len(list_of_files)/7)+1

list_of_files=list_of_files[args.index*chunk:(args.index+1)*chunk]


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
for f in tqdm(list_of_files):
    counter+=1

    if counter % 1000 == 0:
        with open('/mnt/activitynet_features_averaged' + str(args.index), 'wb') as handle:
            pickle.dump(output_list_average, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('/mnt/activitynet_features_no_averaged' + str(args.index), 'wb') as handle:
            pickle.dump(output_list_no_average, handle, protocol=pickle.HIGHEST_PROTOCOL)

    try:

        mp4_version = AudioSegment.from_file(str(f), "mp4")
        mp4_version.export("/mnt/activity_dummy"+str(args.index)+".wav", format="wav")

        audio = torch.from_numpy(vggish_input.wavfile_to_examples("/mnt/activity_dummy"+str(args.index)+".wav"))
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
        video = torch.zeros((frameCount,3, 112,112), dtype=torch.float32)

        fc = 0
        ret = True

        p = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((128, 171)),
            torchvision.transforms.CenterCrop((112, 112)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                                             std=[0.22803, 0.22145, 0.216989]),

        ])


        while (fc < frameCount and ret):
            ret, image=cap.read()
            if ret==True:
                torch_image=torch.from_numpy(image)
                torch_image=torch_image.permute(2,0,1)
                torch_image=p(torch_image)
                video[fc]=torch_image
                fc += 1

        cap.release()


        list_clips=[]



        for i in range(0,fc, 16):
            dummy = torch.zeros((16, 3, 112, 112), dtype=torch.float32)
            clip=video[i:i+16]
            for j in range(clip.shape[0]):
                frame=clip[j]
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

    result_list_no_average=[list_clips, class_id, vggish_output, name_file]
    output_list_no_average.append(result_list_no_average)

    results_list_average=[list_clips_average, class_id, vggish_output_average, name_file]
    output_list_average.append(results_list_average)




with open('/mnt/activitynet_features_averaged'+str(args.index), 'wb') as handle:
     pickle.dump(output_list_average, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('/mnt/activitynet_features_no_averaged'+str(args.index),'wb') as handle:
     pickle.dump(output_list_no_average, handle, protocol=pickle.HIGHEST_PROTOCOL)

