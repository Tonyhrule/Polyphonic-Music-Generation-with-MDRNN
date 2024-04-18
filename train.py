import torch
import os
    
from mdrnn import MDRNN, model_params
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from custom_loss import CustomMidiLoss
import torch.nn.functional as F

bad_files = []

def save_model():
    save_dir = "models"

    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

# Save the model state_dict
    torch.save({'model': mdrnn_model.state_dict()}, os.path.join(save_dir, "model_state_dict.pth"))
    #Model saved
    print('Saved model')

# CUDA reset
torch.cuda.empty_cache()

# Hyperparams
max_epochs = 1
learning_rate = 1e-5

# Setup GPU stuff
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using',device)

# Going through each file in output
import os
# assign directory
directory = 'output'
count = 0
#for plotting
train_losses = []
# iterate over files in
# that directory
steps = 10

#model
mdrnn_model = MDRNN(
        input_size = 3,
        hidden_size = 5,  # Placeholder
        output_size = 3,
        time_steps = steps,
        pitch_steps = steps,
        duration_steps = steps,
        params = model_params()
)
#loss
custom_loss = CustomMidiLoss()

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        print(f)

    #'output/output146.npy' and 'output/output194.npy' are bad files
    if f in ['output/output146.npy', 'output/output194.npy', 'output/output984.npy', 'output/output738.npy', 'output/output722.npy', 'output/output622.npy']:
        continue

    #load datasets
    content = np.load(f)
    content = list(map(tuple, content))
    
    #print(content) 

    #MIGHT NEED TO CREATE AN EMBEDDING LAYER TO PASS NUMBERS INTO MDRNN

    dataset_train = content

    # For next music prediction
    inputs = dataset_train[:-1]
    gold = dataset_train[1:]

    #normalizing data
    scaler = MinMaxScaler()
    inputs_normalized = scaler.fit_transform(inputs)
    gold_normalized = scaler.fit_transform(gold)

    print(inputs_normalized)

    for param in mdrnn_model.parameters():
        param.requires_grad = True

    mdrnn_model.to(device)
    optimizer = torch.optim.Adam(mdrnn_model.parameters(), lr=learning_rate)

    #train
    for epoch in range(max_epochs):
        optimizer.zero_grad() #zeroing gradients

        x = torch.tensor(inputs_normalized).to(torch.float32)
        print(x.size())
        max_sequence_length = x.size(0)

        # Determine the desired sequence length you want after padding
        desired_sequence_length = 4992
        padding_needed = desired_sequence_length - max_sequence_length
        x = torch.nn.functional.pad(x, (0, 0, 0, padding_needed), value=0)
        # Check the size of the padded tensor
        print("Size of padded tensor:", x.size())

        print(x)
        x = x.view(8, -1, 3)
  
        output, pitch_out, time_out, duration_out, x, next_pred = mdrnn_model.forward(x)

        print(f'Next prediction: {next_pred}')
        # Post-processing output

        #processing for computing loss and implementing backpropagation
        #both output and target need to be in similar format
        next_pred = torch.tensor(next_pred, dtype=torch.float32, device=device, requires_grad=True)
        gold = torch.tensor(gold_normalized, dtype=torch.float32, device=device, requires_grad=True)

        num_classes_pitch = mdrnn_model.output_layer.out_features
        print(f'gold: {gold}')
        loss = custom_loss.forward(next_pred, gold[-1])

        #backpropagation
        #zero the gradients to avoid accumulation
        loss.backward()#compute gradients
        torch.nn.utils.clip_grad_norm_(mdrnn_model.parameters(), max_norm=1.0)
        #update weights
        optimizer.step()

        train_losses.append(loss.item())
        if loss.item() > 1:
            bad_files.append(f)

        print("Final Loss:", loss.item())

save_model()


# Plotting the losses
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epochs * Each File')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()