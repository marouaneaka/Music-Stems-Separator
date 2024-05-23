# Music Stems Separator
This is a python implementation for easy training and prediction for the TFC TDF UNET V3 model.
Feel free to use this repository for your own projects


pip install -r requirements.txt 

## Usage
Interact using the makefile 
make train for training
make predict for predictions

tensorboard logs and models are saved during training in tmp, 
!!! MOVE YOUR TMP FILES TO HISTORY AFTER TRAINING !!! as the make clean command deletes all the files in tmp

place your best model in the folder model for the prediction

## Dataset
I used the MUSDB18 HQ dataset
Put your dataset into the data folder structured as :
data/train
data/test

Data set-> https://zenodo.org/records/1117372 | goes to the data folder


If you want to use the regular musdb18 dataset with stems.mp4 format the stems2wav python script will help you convert your files into wav files
Paper -> https://arxiv.org/pdf/2306.09382






