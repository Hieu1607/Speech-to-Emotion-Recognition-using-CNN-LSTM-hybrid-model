import torch.nn as nn
import torch.nn.functional as F
from ..preprocess.data_normalizaton import RMSNorm
# Define the CNN-LSTM hybrid model
class CNN_LSTM_Model(nn.Module):
    """
    A hybrid CNN-LSTM model for audio classification.
    - 3 layers CNN (Conv1D + BatchNorm + ELU + MaxPool)
    - 1 layer LSTM
    - 1 Fully Connected layer
    """
    def __init__(self, input_size=2376, input_dim = 1, n_emotions=7, lstm_hidden_size=64, num_layers_of_lstm=1):
        """
        Args:
            input_size: The number of features in input data
            n_emotions: The number of classes in output
            lstm_hidden_size: The number of unit in a layer of lstm.
            num_layers_of_lstm: How many layers in lstm.
        """
        super(CNN_LSTM_Model, self).__init__()
        self.rmsnorm = RMSNorm(normalized_shape=input_dim)
        # Convolutional layers
        # Conv1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=64, stride=4, padding=32)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=8, stride=4, padding=4)

        # Conv2
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=32, stride=4, padding=16)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=8, stride=4, padding=4)

        # Conv3
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=16, stride=4, padding=8)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=8, stride=4, padding=4)        

        # LSTM layers
        # We need to flatten CNN output to make input for LSTM
        self.lstm_hidden_size = lstm_hidden_size
        self.num_layers_of_lstm = num_layers_of_lstm
        self.lstm = nn.LSTM(input_size=128, hidden_size=lstm_hidden_size, num_layers=num_layers_of_lstm, batch_first=True) # batch first means the order of input is (batch_size, sequence_length, features)

        # Fully connected layer
        self.fc_output = nn.Linear(lstm_hidden_size, n_emotions)

    def forward(self, x):
        # Normalize data using Root Mean Square Normalization
        x = self.rmsnorm(x)
        # x got shape (batch size, 1, input_size)
        # Convert x if x just got 2 dimensions
        if x.dim() == 2:
            x = x.unsqueeze(1) # (batch size, 1, input_size)
        
        # 3 CNN layers
        # Conv1
        x = self.pool1(F.elu(self.bn1(self.conv1(x))))
        # Conv2
        x = self.pool2(F.elu(self.bn2(self.conv2(x))))
        # Conv3
        x = self.pool3(F.elu(self.bn3(self.conv3(x))))
        # x.shape now is (batch_size, 256, sequence_length_after_pooling)

        # LSTM layer
        # Convert x from (batch, channels, seq) to (batch, seq, channels) for LSTM
        # Activation in LSTM layer is tanh, which is built-in
        x = x.permute(0, 2, 1)
        # hn,cn are last hidden state and last cell state
        output, (hn, cn) = self.lstm(x)
        # hn.shape = (num_layers * num_directions, Batch, hidden_size)

        # Fully Connected layer
        final_features = hn[-1,:,:] # (Batch, hidden_size)
        out = self.fc_output(final_features) # (Batch, N_EMOTIONS)

        return out

def initialize_weights(m):
    # 1. Define for Convolutional Layer 1D (Conv1d)
    if isinstance(m, nn.Conv1d):
        # Use Kaiming Uniform (He Initialization) because it is most suitable for ReLU
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
            
    # 2. Define for Batch Normalization Layer 1D (BatchNorm1d)
    elif isinstance(m, nn.BatchNorm1d):
        # Initialize weight (gamma) to 1, bias (beta) to 0
        nn.init.constant_(m.weight.data, 1) 
        nn.init.constant_(m.bias.data, 0)
        
    # 3. Define for Linear Layer (Linear)
    elif isinstance(m, nn.Linear):
        # Use Xavier Uniform (Glorot Initialization) or Kaiming Uniform
        # Kaiming is good if you use ReLU after Linear, Xavier is good for Sigmoid/Tanh.
        # Usually use Xavier or Kaiming depending on the specific architecture.
        nn.init.xavier_uniform_(m.weight.data) 
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

    # 4. Define for Long Short-Term Memory Layer (LSTM)
    elif isinstance(m, nn.LSTM):
        # Initialize all weight matrices in LSTM (i, f, c, o gates) uniformly
        for name, param in m.named_parameters():
            if 'weight_ih' in name: # Weight from input to hidden
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name: # Weight from hidden to hidden
                nn.init.orthogonal_(param.data) # Orthogonal initialization is often preferred for recurrent weights
            elif 'bias' in name: # Bias
                # Initialize bias of Forget Gate to 1.0 (or 1.05) to prevent Vanishing Gradient
                # Other biases are usually set to 0.
                param.data.fill_(0)
                n = param.size(0)
                # Set bias of forget gate (second quarter) to 1
                param.data[(n//4):(n//2)].fill_(1.0)

def create_model(input_size=2376, n_emotions=7, lstm_hidden_size=64, num_layers_of_lstm=1):
    model = CNN_LSTM_Model(input_size=input_size, n_emotions=n_emotions, lstm_hidden_size=lstm_hidden_size, num_layers_of_lstm=num_layers_of_lstm)
    model.apply(initialize_weights)
    return model
