import torch
import os
import datetime
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from data import Data
from model import TFC_TDF_net 
from metrics import myMetric as criterion
import yaml
from ml_collections import ConfigDict
from torch.utils.tensorboard import SummaryWriter 
from torch.optim import Adam, RMSprop, AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp.grad_scaler import GradScaler
import argparse





# utiliser les poids du meilleur modele
parser = argparse.ArgumentParser()    
parser.add_argument("--pre", type=bool, default=False, help="utiliser les poids du meilleur modele")
parser.add_argument("--model", type=str, default=False, help="utiliser les poids du meilleur modele")
args = parser.parse_args()

device = torch.device('cuda')
    #fichier de configs
with open('conf/config-tcf.yaml') as f:
        config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
model = TFC_TDF_net(config).to(device)

optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)

# chargement de donnees
train_loader = DataLoader(Data(config,"data/train"), batch_size=config.training.batch_size, shuffle=True)
val_loader = DataLoader(Data(config,"data/test"), batch_size=config.training.batch_size, shuffle=False)


run_name = f'{datetime.datetime.now()}Steps_{config.training.num_steps}_Epochs_{config.training.epochs}'
os.mkdir(f'tmp/{run_name}')
writer = SummaryWriter(log_dir=f'tmp/{run_name}/torch')




if args.pre == True:
    model.load_state_dict(torch.load('model/trained_model.pth'))

#entrainement
for epoch in range(config.training.epochs):
    model.train()
    running_loss = 0.0
    scaler = GradScaler() 
    for i, data in enumerate(tqdm(train_loader)):
        y = data.to(device)
        x = y.sum(1) # mix  
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():    
            outputs = model(x)
            loss = criterion(outputs, y,q=config.training.q)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        writer.add_scalar(f'Steps of epoch {epoch+1} Training Loss', loss.item(), i)
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")
    writer.add_scalar('Epoch Training Loss', running_loss / len(train_loader), epoch)    
    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for i, data in enumerate(tqdm(val_loader)):
            y = data.to(device)
            x = y.sum(1)  # mix  

            outputs = model(x)
            loss = criterion(outputs, y,q=config.training.q)

            val_loss += loss.item()
        print(f"Validation Loss: {val_loss / len(val_loader)}")
        writer.add_scalar('Epoch val Loss', val_loss / len(val_loader), epoch)
    #historique des modeles
    torch.save(model.state_dict(), f'tmp/{run_name}/trained_model.pth')    
# export du modele
torch.save(model.state_dict(), f'tmp/{run_name}/trained_model.pth')
