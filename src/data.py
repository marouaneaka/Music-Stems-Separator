import os
import random
import numpy as np
import torch
import soundfile as sf
import pickle
from tqdm import tqdm
from glob import glob


class Data(torch.utils.data.Dataset):
    def __init__(self, config, data_path):  
        self.config = config
        self.instruments = instruments = config.training.instruments
        
        metadata_path = data_path+'/metadata'
        try:
            metadata = pickle.load(open(metadata_path, 'rb'))
        except Exception:   
            print('Collecting metadata for', data_path)
            metadata = []
            track_paths = sorted(glob(data_path+'/*'))
            track_paths = [path for path in track_paths if os.path.basename(path)[0]!='.' and os.path.isdir(path)]
            for path in tqdm(track_paths):
                length = len(sf.read(path+f'/{instruments[0]}.wav')[0])
                metadata.append((path, length))
            pickle.dump(metadata, open(metadata_path, 'wb'))              
        
        self.metadata = metadata    
        self.chunk_size = config.audio.chunk_size 
        self.min_mean_abs = config.audio.min_mean_abs
               
    def __len__(self):
        return self.config.training.num_steps * self.config.training.batch_size

    def load_chunk(self,path, length, chunk_size, offset=None):
        if chunk_size <= length:
            if offset is None:
                offset = np.random.randint(length - chunk_size + 1)
            x = sf.read(path, dtype='float32', start=offset, frames=chunk_size)[0]    
        else:
            x = sf.read(path, dtype='float32')[0]
            pad = np.zeros([chunk_size-length,2])
            x = np.concatenate([x, pad])
        return x.T
    def load_source(self, metadata, i):
        while True:
            track_path, track_length = random.choice(metadata)
            source = self.load_chunk(f'{track_path}/{i}.wav', track_length, self.chunk_size)
            if np.abs(source).mean() >= self.min_mean_abs:  # remove quiet chunks
                break
        return torch.tensor(source, dtype=torch.float32)
    
    def __getitem__(self, index):
        return torch.stack([self.load_source(self.metadata, i) for i in self.instruments])