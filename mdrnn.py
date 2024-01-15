import torch
import torch.nn as nn
import torch.nn.functional as F

def model_params():
    params = dict()
    params['rnn_units'] = 512
    params['rnn_layers'] = 2
    return params

class MDRNN_ver2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, time_steps, pitch_steps, params):
        super(MDRNN_ver2, self).__init__()
        
        self.params = params
        self.time_steps = time_steps
        self.pitch_steps = pitch_steps
        self.hidden_size = hidden_size

        # RNNs for each dimension
        self.time_lstm = nn.LSTM(input_size, hidden_size, params['rnn_layers'], bidirectional=True)
        self.pitch_lstm = nn.LSTM(hidden_size * 2, hidden_size, params['rnn_layers'], bidirectional=True)

        # Linear layer for output
        self.output_layer = nn.Linear(hidden_size * 2, output_size)

    def init_hidden(self, batch_size):
        # Initialize hidden state with zeros for bidirectional LSTM
        num_directions = 2  # 2 cus LSTM is bidirectional
        return (torch.zeros(self.params['rnn_layers'] * num_directions, batch_size, self.hidden_size),
                torch.zeros(self.params['rnn_layers'] * num_directions, batch_size, self.hidden_size))
    
    def forward(self, x):
    
        params = self.params
        x = x.view(batch_size, self.time_steps, self.pitch_steps, -1)
        batch_size, _ = int(x.size())
        # Conv blocks (2)
        x = self.b1(x)
        x = self.b2(x)

        x = x.permute(3, 0, 2, 1)  # U can change based on the output of CNN
        feature_dim = self.params['conv_filter_n'][-1] * self.params['img_height']
        x = x.reshape(-1, batch_size, feature_dim)  # Reshape to [seq_len, batch, features]

        # Initialize hidden states
        hidden_time = self.init_hidden(batch_size)
        hidden_pitch = self.init_hidden(batch_size)

        # Process each time step
        for t in range(self.time_steps):
            # Update hidden state for each pitch step
            for p in range(self.pitch_steps):

                time_out, hidden_time = self.time_lstm(x[:, t, p, :], hidden_time)
                pitch_out, hidden_pitch = self.pitch_lstm(time_out, hidden_pitch)

        # Final output
        output = self.output_layer(pitch_out)

        return output