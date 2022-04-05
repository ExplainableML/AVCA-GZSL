import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from vggish import VGGish
from audioset import vggish_input, vggish_postprocess
import os
import json
import torch.nn as nn

def main():
    # Initialize the PyTorch model.

    exception_dict=[]
    device = 'cuda:0'

    pytorch_model = VGGish()
    pytorch_model.load_state_dict(torch.load('pytorch_vggish.pth'))
    pytorch_model = pytorch_model.to(device)
    pytorch_model.eval()
    path = Path("/home/omercea19/akata-shared/omercea19/full_ebird_download")

    root_saved=Path("/home/omercea19/akata-shared/omercea19/full_ebird_download_embeddings_new_model")

    dict = {}
    for file in tqdm(path.glob('**/*.wav')):
        try:
            dict[file] = 0
            loaded_wav=torch.from_numpy(vggish_input.wavfile_to_examples(str(file)))
            input_batch = loaded_wav.float().to(device)
            input_batch = input_batch.unsqueeze(dim=1)
            pytorch_output = pytorch_model(input_batch)
            pytorch_output = pytorch_output.detach().cpu().numpy()
            post_processor = vggish_postprocess.Postprocessor('vggish_pca_params.npz')
            postprocessed_output = post_processor.postprocess(pytorch_output)
            #postprocessed_output=postprocessed_output.mean(axis=0)
            new_path=str(file.relative_to(path))
            new_directory=Path.joinpath(root_saved,file.relative_to(path).parent)
            try:
                os.makedirs(new_directory)
            except:
                pass
            new_path=Path.joinpath(root_saved,file.relative_to(path))
            name = str(new_path.name).split('.')[0]
            name = name + ".npy"
            new_path = Path.joinpath(new_path.parent, Path(name))
            np.save(new_path,postprocessed_output)
            #zz=np.load(new_path)
            #print(zz)
        except:
            print(str(file))
            exception_dict.append(str(file))

    with open("./exception.json","w") as g:
        json.dump(exception_dict,g)



    ''' 
    
    THIS IS THE VARIANT WITH THE BATCHES, WHICH WILL MOST LIKELY WE USED DURING THE TRAINING/INFERENCE
    
    # Generate a sample input (as in the AudioSet repo smoke test).
    x=['../altele/5bS607UKT2U.wav','../altele/5bS607UKT2U.wav']
    input_batch=[]
    for i in x:
        input_batch.append(torch.from_numpy(vggish_input.wavfile_to_examples(i)))

    input_batch=torch.stack(input_batch)

    # Produce a batch of log mel spectrogram examples.
    input_batch = input_batch.float().to(device)
    input_batch=input_batch.unsqueeze(dim=2)
    input_batch=input_batch.view(-1,input_batch.shape[2],input_batch.shape[3],input_batch.shape[4])

    # Run the PyTorch model.
    pytorch_output = pytorch_model(input_batch)
    pytorch_output = pytorch_output.detach().cpu().numpy()
    print('Input Shape:', tuple(input_batch.shape))
    print('Output Shape:', tuple(pytorch_output.shape))

    # Post-processing.
    post_processor = vggish_postprocess.Postprocessor('vggish_pca_params.npz')
    postprocessed_output = post_processor.postprocess(pytorch_output)
    postprocessed_output=np.reshape(postprocessed_output,(len(x),-1,postprocessed_output.shape[1]))
    print("final")
    '''

if __name__ == '__main__':
    main()
