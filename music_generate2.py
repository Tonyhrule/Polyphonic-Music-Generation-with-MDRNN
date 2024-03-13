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
input_data = np.load('output/output40.npy')

# Normalize input data
scaler = MinMaxScaler()
input_data_normalized = scaler.fit_transform(input_data)
input_tensor = torch.tensor(input_data_normalized, dtype=torch.float32)
# Define the number of predictions to generate
num_predictions = 200

#list to store generated predictions
next_preds_list = []
#generate predictions
with torch.no_grad():
    mdrnn_model.eval()
    for i in range(num_predictions):
        x = torch.tensor(input_tensor).to(torch.float32)
        print(x.size())
        max_sequence_length = x.size(0)

        # Determine the desired sequence length you want after padding
        desired_sequence_length = 4992
        padding_needed = desired_sequence_length - max_sequence_length
        x = torch.nn.functional.pad(x, (0, 0, 0, padding_needed), value=0)
        # Check the size of the padded tensor
        print("Size of padded tensor:", x.size())
        x = x.view(8, -1, 3)
        print(f'this: {x.size()}')

        output, _, _, _, _, next_pred = mdrnn_model(x)
        next_pred = np.array(next_pred)
        #convert next_pred to a NumPy array and reshape it
        next_pred_np = torch.tensor(next_pred.reshape(1, -1))

        #inverse transform to get back the original data scale
        output_unnormalized = scaler.inverse_transform(next_pred_np)
        #round the values to the nearest integer
        output_unnormalized = np.round(output_unnormalized).astype(int)
        #convert the rounded values back to a torch tensor
        #inverse transform to get back the original data scale
        next_pred_unnormalized = torch.tensor(scaler.inverse_transform(next_pred_np), dtype=torch.float32)

        #concatenate next_pred_unnormalized to input_tensor
        input_tensor = torch.cat((input_tensor, next_pred_unnormalized), 0)
        print(f'original tensor {input_tensor}')
        input_tensor = torch.cat((input_tensor[:0], input_tensor[1:]), dim=0)

        #list of next notes in sequential order
        next_preds_list.append(next_pred_unnormalized)

# Concatenate the list of predictions into a single tensor
all_next_preds = np.array(torch.cat(next_preds_list))
#post-processing on the concatenated predictions
print(all_next_preds)
postProcess(all_next_preds)