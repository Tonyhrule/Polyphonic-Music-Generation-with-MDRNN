import torch
import argparse
import os
    
import utils
from mdrnn import MDRNN, model_params
import torch.nn as nn
from torch.utils.data import DataLoader
from preprocessing import processMidi, noteToVector

def save_model():
    # Save model
    root_model_path = 'models/latest_model' + '.pt'
    model_dict = mdrnn_model.state_dict()
    state_dict = {'model': model_dict, 'optimizer': optimizer.state_dict()}
    torch.save(state_dict, root_model_path)
    print('Saved model')

# CUDA reset
torch.cuda.empty_cache()

# Hyperparams
max_epochs = 100
learning_rate = 1e-4

# Setup GPU stuff
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using',device)


# Load datasets
midi_file_path = './Mozart/mozart_-_Turkish_March_in_Bb.mid'
notes = processMidi(midi_file_path)

dataset_train = notes[:1000]
dataset_valid = notes[1000:]
dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True)
dataloader_valid = DataLoader(dataset_valid, batch_size=1, shuffle=True)

embedding = nn.Embedding(num_embedding = len(notes), embedding_dim = 64)

steps = len(dataset_train)
params = model_params()

#for next music prediction
inputs = notes[:-5]
gold = notes[5:]

#each file is one input
multi_input = [inputs]
multi_gold = [inputs]

# Model
mdrnn_model = MDRNN(
    input_size = multi_input.size(), 
    hidden_size = "placeholder", 
    output_size = multi_gold.size(), 
    time_steps = steps, 
    pitch_steps = steps, 
    chord_steps = steps, 
    params = model_params)

mdrnn_model.to(device)
optimizer = torch.optim.Adam(mdrnn_model.parameters(), lr=learning_rate)

#train
for epoch in range(max_epochs):
    x = embedding(multi_input)
    x, (h, c) = mdrnn_model.forward(x)
    #loss function
    loss = nn.CTCLoss(reduction = 'mean')
    #hyperbolic tan function
    a = torch.tanh(x)
    _, preds = a.max(dim = -1)

