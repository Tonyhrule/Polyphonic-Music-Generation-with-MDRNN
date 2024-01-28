import torch
import numpy as np
from mdrnn import MDRNN, model_params
import torch.nn as nn
from torch.utils.data import DataLoader
from preprocessing import processMidi
#inputs = [[(72, 480, 120), (72, 480, 120)], [(72, 600, 120), (72, 600, 120)], [(70, 480, 120), (70, 480, 120)], [(70, 600, 120), (70, 600, 120)]]
#inputs = [(72,70), (480,600), (120,120)]

inputs = [(72, 480, 120), (70, 600, 120)]

steps = 1

multi_input = inputs
multi_gold = inputs

# Model
mdrnn_model = MDRNN(
    input_size = 3, 
    hidden_size = 5, #placeholder
    output_size = 3, 
    time_steps = steps, 
    pitch_steps = steps, 
    duration_steps = steps, 
    params = model_params()
)

embedding = nn.Embedding(num_embeddings = 2, embedding_dim = 64)
x = multi_input
output, pitch_out, time_out, duration_out, x = mdrnn_model.forward(x)
print(output)
print(f"This is pitch out: {pitch_out}")
print(f"This is time out: {time_out}")
print(f"This is duration out: {duration_out}")


# Number of predictions to generate
num_predictions = 5
initial_input = x

# Generate predictions and update input sequence

new_output = output.detach()
print(new_output.view(2,3))
size_along_dim1 = new_output.size()[1]
print(size_along_dim1)
print(initial_input)
    # Concatenate predictions to the input sequence
initial_input = torch.cat([initial_input, output.unsqueeze(0)], dim=0)

# Display the final input sequence with predictions
print("Final Input Sequence with Predictions:")
print(initial_input)

