import torch
import os

from postprocessing import postProcess
from mdrnn import MDRNN, model_params
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from custom_loss import CustomMidiLoss
import numpy as np


steps = 1

mdrnn_model = MDRNN(
        input_size = 3,
        hidden_size = 5, #Placeholder
        output_size = 3,
        time_steps = steps,
        pitch_steps = steps,
        duration_steps = steps,
        params = model_params()
    )
#Loading the model
save_dir = 'models'
state_dict = torch.load(os.path.join(save_dir, "model_state_dict.pth"))
mdrnn_model.load_state_dict(state_dict['model'])

#Loading the input data
input_data = np.load('output/midis-3 in G major.npy')

#for unnormalizing purposes
flattened_list = [value for x in input_data for value in x]
max_value = max(flattened_list)
min_value = min(flattened_list)

input_tensor = torch.tensor(input_data, dtype=torch.float32)

with torch.no_grad():
    mdrnn_model.eval()  #Set to evaluation mode
    output, _, _, _, _, next_pred = mdrnn_model(input_tensor)

scaler = MinMaxScaler()

content = np.load('output/midis-3 in G major.npy')
scaler.fit(content)

min_values = scaler.data_min_
max_values = scaler.data_max_

# For next music prediction
inputs = content[:-1]
gold = content[1:]

#normalizing data
scaler = MinMaxScaler()
inputs_normalized = scaler.fit_transform(inputs)
gold_normalized = scaler.fit_transform(gold)

# Inverse transform to get back the original data
output_unnormalized = abs(output * (max_values - min_values) + min_values).cpu().numpy()
print(output_unnormalized)
print(output)
postProcess(output_unnormalized)

print(len(output_unnormalized))
print(len(input_data))
