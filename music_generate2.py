import torch
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from mdrnn import MDRNN, model_params
from postprocessing import postProcess

# Load pre-trained model
save_dir = 'models'
state_dict = torch.load(os.path.join(save_dir, "model_state_dict.pth"))
mdrnn_model = MDRNN(
    input_size=3,
    hidden_size=5,  # Placeholder
    output_size=3,
    time_steps=1,
    pitch_steps=1,
    duration_steps=1,
    params=model_params()
)
mdrnn_model.load_state_dict(state_dict['model'])

# Load input data
input_data = np.load('output/midis-Completely.npy')

# Normalize input data
scaler = MinMaxScaler()
input_data_normalized = scaler.fit_transform(input_data)
input_tensor = torch.tensor(input_data_normalized, dtype=torch.float32)
hi = input_tensor
# Define the number of predictions to generate
num_predictions = 100  # Adjust as needed

# List to store generated predictions
next_preds_list = []
# Generate predictions
with torch.no_grad():
    mdrnn_model.eval()
    for i in range(num_predictions):
        output, _, _, _, _, next_pred = mdrnn_model(input_tensor)
        next_pred = np.array(next_pred)
        # Convert next_pred to a NumPy array and reshape it
        next_pred_np = next_pred.reshape(1, -1)

        # Inverse transform to get back the original data scale
        output_unnormalized = scaler.inverse_transform(next_pred_np)

        # Round the values to the nearest integer
        output_unnormalized = np.round(output_unnormalized).astype(int)

        # Convert the rounded values back to a torch tensor
        # Inverse transform to get back the original data scale
        next_pred_unnormalized = torch.tensor(scaler.inverse_transform(next_pred_np), dtype=torch.float32)

        # Concatenate next_pred_unnormalized to input_tensor
        input_tensor = torch.cat((input_tensor, next_pred_unnormalized), 0)
        input_tensor = input_tensor[i + 1:]

        # Print the updated input_tensor
        print(input_tensor)

        next_preds_list.append(next_pred_unnormalized)

# Concatenate the list of predictions into a single tensor
all_next_preds = torch.cat(next_preds_list)
print(all_next_preds)
print(next_pred_unnormalized)
print(input_tensor)

# Perform post-processing on the concatenated predictions
postProcess(all_next_preds)