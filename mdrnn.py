import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain
from sklearn.decomposition import PCA

def model_params():
    params = dict()
    params['rnn_units'] = 512
    params['rnn_layers'] = 3
    return params

#function for mapping softmax output to the input
def softmax_output_to_class(output_vector, class_labels):
    #Finding the greatest probability of pitch, time, and duration in output vector
    max_pitch = max(output_vector[0]) #index 0 has pitch list
    max_time = max(output_vector[1]) #index 1 has time list
    max_duration = max(output_vector[2]) #index 2 has duration list

    #Finding index of the max probability for pitch, time, and duration
    pitch_index = output_vector[0].index(max_pitch)
    time_index = output_vector[1].index(max_time)
    duration_index = output_vector[2].index(max_duration)

    #transposing the tensor of class_labels to make it the same format as the output_vector
    transposed_labels = torch.tensor(class_labels, dtype=torch.float32).t()
    #list of predictions
    predictions = [transposed_labels[0][pitch_index], transposed_labels[1][time_index], transposed_labels[2][duration_index]]
    return predictions

class MDRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, time_steps, pitch_steps, duration_steps, params, dropout_prob=0.3):
        super(MDRNN, self).__init__()
        
        self.params = params
        self.time_steps = time_steps
        self.pitch_steps = pitch_steps
        self.duration_steps = duration_steps
        self.hidden_size = hidden_size

        # LSTMs for each dimension
        self.time_lstm = nn.LSTM(hidden_size * 2, hidden_size, params['rnn_layers'], bidirectional=True)
        self.pitch_lstm = nn.LSTM(input_size, hidden_size, params['rnn_layers'], bidirectional=True)
        self.duration_lstm = nn.LSTM(hidden_size * 2, hidden_size, params['rnn_layers'], bidirectional=True)

        #dropout
        #self.dropout = nn.Dropout(p=dropout_prob)

        #linear layer for output
        self.output_layer = nn.Linear(hidden_size * 2, output_size)

    def init_hidden(self, batch_size):
    # Initialize hidden state with zeros for bidirectional LSTM
        num_directions = 2

        hidden_pitch = (
            torch.zeros(self.params['rnn_layers'] * num_directions, batch_size, self.hidden_size),
            torch.zeros(self.params['rnn_layers'] * num_directions, batch_size, self.hidden_size)
        )

        hidden_time = (
            torch.zeros(self.params['rnn_layers'] * num_directions, 2, self.hidden_size),
            torch.zeros(self.params['rnn_layers'] * num_directions, 2, self.hidden_size)
        )

        hidden_duration = (
            torch.zeros(self.params['rnn_layers'] * num_directions, 4, self.hidden_size),
            torch.zeros(self.params['rnn_layers'] * num_directions, 4, self.hidden_size)
        )

        return hidden_pitch, hidden_time, hidden_duration

    def forward(self, x):
        #x = torch.tensor(x)
        batch_size = 1
        seq_len, feature_dim, _ = tuple(x.size())
        self.pitch_steps = seq_len 
        print(f'batch length: {batch_size}')
        print(f'seq quence length: {seq_len}')
        print(f'feature dimensions: {feature_dim}')
        
        #x = x.view(seq_len, batch_size, feature_dim)  #reshape to [seq_len, batch, features]

        #initialize hidden states
        hidden_pitch, hidden_time, hidden_duration = self.init_hidden(batch_size)
        #convert to 32 bit integer
        #x = x.to(torch.float32)
        #process each timestep
        for p in range(self.pitch_steps):
            #update hidden state for each pitch step

            pitch_out, hidden_pitch = self.pitch_lstm(x[:, p:p+1, :], hidden_pitch)
            resized_tensor = torch.zeros(8, 1, 10)
            
            #copy the values from the original tensor to the resized tensor
            resized_tensor[:, :, :3] = x[:, p:p+1, :]

            #update hidden state for each time step
            #assuming pitch_out has a size of [batch_size, 1, hidden_size]
            concatenated_input = torch.cat((pitch_out, resized_tensor), dim=1)
            time_out, hidden_time = self.time_lstm(concatenated_input, hidden_time)

            #update hidden state for each duration step
            concated = torch.cat((time_out, concatenated_input), dim=1)
            duration_out, hidden_duration = self.duration_lstm(concated, hidden_duration)

        output = self.output_layer(torch.cat([pitch_out[-1], time_out[-1], duration_out[-1]], dim=0))

        #length is amount of rows in input x
        length = x.shape[-2]
        print(f'length: {length}')
        #truncating output vector to size of the input
        output = output[:length]
        #DEBUG

        #transpose output for softmax function to have 3 rows, one for each dimension
        transposed_output = output.t()
        #probabilities for each output
        output_probs = F.softmax(transposed_output, dim=1)

        #processing data for finding greatest softmax index
        #output_probs = np.array(output_probs.detach())
        output_probs = output_probs.reshape(-1, output_probs.shape[-1]).tolist()
        class_labels = x
        class_labels = class_labels.reshape(-1, class_labels.shape[-1])

        #output probs are mapped onto class_labels
        predicted_class = softmax_output_to_class(output_probs, class_labels)

        return output, pitch_out, time_out, duration_out, x, predicted_class
