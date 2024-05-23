from model import TFC_TDF_net
import os
import glob
import torch
import soundfile as sf
import yaml
from ml_collections import ConfigDict
import argparse


# utiliser les poids du meilleur modele
parser = argparse.ArgumentParser()    
parser.add_argument("--model", type=str, default='', help="The model to use")
args = parser.parse_args()
device = torch.device('cuda')
    #fichier de configs
if args.model == 'tcfBest':
    with open('conf/config-tcf-ref.yaml') as f:
            config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
    model = TFC_TDF_net(config).eval().cuda()
    model.load_state_dict(torch.load('model/baseline.pth'))
else:
    with open('conf/config-tcf.yaml') as f:
            config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
    model = TFC_TDF_net(config).eval().cuda()
    model.load_state_dict(torch.load('model/trained_model.pth'))

instruments = [ 'vocals', 'other','drums','bass' ]

def load_music(path='input'):
    wav_file = glob.glob(os.path.join(path, '*.wav'))[0]
    x, samplerate = sf.read(wav_file, dtype='float32')
    return torch.tensor(x.T,dtype=torch.float32)

def save_music(outputs, output_sample_rate,path="output"):
    for instrument in instruments:
        full_path = os.path.join(path, f'{instrument}.wav')
        soundfile.write(full_path,data=outputs[instrument],samplerate=output_sample_rate)

def predict(model,mix):
    batch_size = config.inference.batch_size 
    C = config.audio.hop_length * (config.inference.dim_t-1) #taille de la musique
    N = config.inference.num_overlap # nb d'overlap
        
    H = C//N   # taille du hop
    L = mix.shape[1]    
    pad_size = H-(L-C)%H # assurer les c est divisable par N
    mix = torch.cat([torch.zeros(2,C-H), mix, torch.zeros(2,pad_size + C-H)], 1)
    mix = mix.cuda()

    chunks = []
    i = 0
    while i+C <= mix.shape[1]:
        chunks.append(mix[:, i:i+C])
        i += H
    chunks = torch.stack(chunks)

    batches = []
    i = 0
    while i < len(chunks):
        batches.append(chunks[i:i+batch_size])
        i = i + batch_size

    X = torch.zeros(len(instruments),2,C-H)
    X = X.cuda()
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            for batch in batches:
                x = model(batch) 
                for w in x:
                    a = X[...,:-(C-H)]
                    b = X[...,-(C-H):] + w[...,:(C-H)]
                    c = w[...,(C-H):]
                    X = torch.cat([a,b,c], -1)

    estimated_sources = X[..., C-H:-(pad_size+C-H)]/N
        
    return estimated_sources.cpu().numpy()
    
if __name__ == "__main__":
    torch.cuda.empty_cache()
    output = predict(model,load_music())
    for instrument,track in zip(instruments,output):
        sf.write(f'output/{instrument}.wav',data=track.T,samplerate=config.audio.sample_rate)
   