import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the CNN-LSTM hybrid model
class CNN_LSTM_Model(nn.Module):
    """
    A hybrid CNN-LSTM model for audio classification.
    - 3 layers CNN (Conv1D + BatchNorm + ELU + MaxPool)
    - 1 layer LSTM
    - 1 Fully Connected layer
    """
    def __init__(self, input_size=2376, n_emotions=7, lstm_hidden_size=64, num_layers_of_lstm=1):
        """
        Args:
            input_size: The number of features in input data
            n_emotions: The number of classes in output
            lstm_hidden_size: The number of unit in a layer of lstm.
            num_layers_of_lstm: How many layers in lstm.
        """
        super(CNN_LSTM_Model, self).__init__()

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
        self.lstm = nn.LSTM(input_size=256, hidden_size=lstm_hidden_size, num_layers=num_layers_of_lstm, batch_first=True) # batch first means the order of input is (batch_size, sequence_length, features)

        # Fully connected layer
        self.fc_output = nn.Linear(lstm_hidden_size, n_emotions)

    def forward(self, x):
        # x got shape (batch size, 1, input_size)
        # Convert x if x just got 2 dimensions
        if x.dim() == 2:
            x = x.unsqueeze(1) # (batch size, 1, input_size)
        
        # 3 CNN layers
        # Conv1
        x = self.pool1(F.elu(self.bn1(self.conv1(x))))
        # Conv2
        x = self.pool1(F.elu(self.bn1(self.conv1(x))))
        # Conv3
        x = self.pool1(F.elu(self.bn1(self.conv1(x))))
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

