import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain
from sklearn.decomposition import PCA

def model_params():
    params = dict()
    params['rnn_units'] = 512
    params['rnn_layers'] = 2
    return params

def softmax_output_to_class(output_vector, class_labels):
    #print(f'output vector: {output_vector}')
    #arr = list(chain.from_iterable(output_vector))
    max_pitch = max(output_vector[0])
    max_time = max(output_vector[1])
    max_duration = max(output_vector[2])
    #print(f'max number: {max_num}')
    pitch_index = output_vector[0].index(max_pitch)
    time_index = output_vector[1].index(max_time)
    duration_index = output_vector[2].index(max_duration)
    #print(f'predicted index: {predicted_index}')
    #class_labels = list(chain.from_iterable(class_labels))
    #print(f'length of class_labels: {len(class_labels)}')
    #print(f'length of output: {len(arr)}')
    transposed_labels = torch.tensor(class_labels, dtype=torch.float32).t()
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
        self.dropout = nn.Dropout(p=dropout_prob)

        # Linear layer for output
        self.output_layer = nn.Linear(hidden_size * 2, output_size)

    def init_hidden(self, batch_size):
        # Initialize hidden state with zeros for bidirectional LSTM
        num_directions = 2

        hidden_pitch = (torch.zeros(self.params['rnn_layers'] * num_directions, batch_size, self.hidden_size),
                        torch.zeros(self.params['rnn_layers'] * num_directions, batch_size, self.hidden_size))
        
        hidden_time = (torch.zeros(self.params['rnn_layers'] * num_directions, batch_size, self.hidden_size),
                       torch.zeros(self.params['rnn_layers'] * num_directions, batch_size, self.hidden_size))
        
        hidden_duration = (torch.zeros(self.params['rnn_layers'] * num_directions, batch_size, self.hidden_size),
                           torch.zeros(self.params['rnn_layers'] * num_directions, batch_size, self.hidden_size))


        return hidden_pitch, hidden_time, hidden_duration

    def forward(self, x):

        x = torch.tensor(x)

        batch_size, feature_dim = tuple(x.size())
        seq_len = self.pitch_steps  # define appropriately based on your architecture
        input_size = torch.Size([seq_len, batch_size, feature_dim])

        #print(f"Input size of x: {input_size}")

        x = x.view(seq_len, batch_size, feature_dim)  # Reshape to [seq_len, batch, features]

        #Initialize hidden states
        hidden_pitch, hidden_time, hidden_duration = self.init_hidden(batch_size)
        
        #convert to 32 bit integer
        x = x.to(torch.float32)

        #print(f"Shape of input x: {x.shape}")
        #print(f"Expected input_size for pitch_lstm: {self.pitch_lstm.input_size}")
        
        #process each timestep
        for p in range(self.pitch_steps):
            #update hidden state for each pitch step
            for t in range(self.time_steps):
                #update hidden state for each chord step
                for d in range(self.duration_steps): 
                    #update hidden state for each pitch step
                    pitch_out, hidden_pitch = self.pitch_lstm(x, hidden_pitch)
                    pitch_out = self.dropout(pitch_out)

                    #update hidden state for each time step
                    time_out, hidden_time = self.time_lstm(pitch_out, hidden_time)
                    time_out = self.dropout(time_out)

                    #update hidden state for each duration step
                    duration_out, hidden_duration = self.duration_lstm(time_out, hidden_duration)
                    duration_out = self.dropout(duration_out)

        output = self.output_layer(torch.cat([pitch_out[-1], time_out[-1], duration_out[-1]], dim=0))
        #reshape the PCA-transformed output back to its original shape
        print(f'output{output}')
        if len(output) > 0 or len(output[0]) > 0:
            length = np.array(x).shape[-2]
            print(f'length: {length}')
            output = output[:length] 
            print(f'output: {output}')
            print(f'output length: {len(output)}')
            #transpose output for softmax function
            transposed_output = output.t()
            print(f'transposed: {transposed_output}')
            output_probs = F.softmax(transposed_output, dim=1)
            
            print(f'output probs: {output_probs}')
            #processing data for finding greatest softmax index
            output_probs = np.array(output_probs.detach())
            print(len(output_probs))
            output_probs = output_probs.reshape(-1, output_probs.shape[-1]).tolist()
            class_labels = np.array(x)
            class_labels = class_labels.reshape(-1, class_labels.shape[-1])

            #DEBUGGING
            predicted_class = softmax_output_to_class(output_probs, class_labels)
            print(f"Predicted class: {predicted_class}")  # This should print: Predicted class: Dog
            
            #predicted_step_index = torch.argmax(output_probs, dim=1)
        return output, pitch_out, time_out, duration_out, x, predicted_class
