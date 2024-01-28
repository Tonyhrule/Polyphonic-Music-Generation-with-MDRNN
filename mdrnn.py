import torch
import torch.nn as nn
import torch.nn.functional as F

def model_params():
    params = dict()
    params['rnn_units'] = 512
    params['rnn_layers'] = 2
    return params

class MDRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, time_steps, pitch_steps, duration_steps, params):
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

        print(f"Input size of x: {input_size}")

        x = x.view(seq_len, batch_size, feature_dim)  # Reshape to [seq_len, batch, features]

        # Initialize hidden states
        hidden_pitch, hidden_time, hidden_duration = self.init_hidden(batch_size)
        
        #convert to 32 bit integer
        x = x.to(torch.float32)

        print(f"Shape of input x: {x.shape}")
        print(f"Expected input_size for pitch_lstm: {self.pitch_lstm.input_size}")
        
        #process each timestep
        for p in range(self.pitch_steps):
            # Update hidden state for each pitch step
            for t in range(self.time_steps):
                #Update hidden state for each chord step
                for d in range(self.duration_steps): 
                    # Update hidden state for each pitch step
                    pitch_out, hidden_pitch = self.pitch_lstm(x, hidden_pitch)
                    
                    # Update hidden state for each time step
                    time_out, hidden_time = self.time_lstm(pitch_out, hidden_time)
                    
                    # Update hidden state for each duration step
                    duration_out, hidden_duration = self.duration_lstm(time_out, hidden_duration)
        """
        print(f"Shape of pitch_out[-1]: {pitch_out[-1].shape}")
        print(f"Shape of time_out[-1]: {time_out[-1].shape}")
        print(f"Shape of duration_out[-1]: {duration_out[-1].shape}")
        """

        # Final output
        output = self.output_layer(torch.cat([pitch_out[-1], time_out[-1], duration_out[-1]], dim=0))

        return output, pitch_out, time_out, duration_out, x
        
    def postprocessing(self, output, x):
        pass
