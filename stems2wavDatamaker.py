import stempeg
import os
import glob
import soundfile as sf


instruments = ['mixture', 'drums', 'bass','other','vocals' ]
train = 'data/musdb18/train/'
test = 'data/musdb18/test/'

newname='musdb18wav'
if not os.path.exists(f'data/{newname}'):
    os.mkdir(f'data/{newname}')


out_train = f'data/{newname}/test/'
if not os.path.exists(out_train):
    os.mkdir(out_train)


track_name = "Ben Carrigan - We'll Talk About It All Tonight.stem.mp4"
S, rate = stempeg.read_stems(f'{test}{track_name}')

if not os.path.exists(f'{out_train}{track_name}'):
    os.mkdir(f'{out_train}{track_name}')

for track,instrument in zip(S,instruments):
    sf.write(f'{out_train}{track_name}/{instrument}.wav',data=track,samplerate=rate)