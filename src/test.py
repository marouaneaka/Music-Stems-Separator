from torch.utils.data import DataLoader
import yaml
from ml_collections import ConfigDict   
from data import Data
import soundfile as sf
with open('conf/config.yaml') as f:
        config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
train_loader = DataLoader(Data(config,"/home/marbel/projetML/demucs/dataset/train"), batch_size=config.training.batch_size, shuffle=True)

print(train_loader)

print(len(train_loader))

for i,data in enumerate(train_loader):
    print(data.numpy())
    sf.write('tmp/test{i}.wav',data=data.numpy().T,samplerate=config.audio.sample_rate)
    print(len(data))
    