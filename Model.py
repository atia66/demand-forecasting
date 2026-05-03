from torch import nn

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=3):
        super().__init__()
        self.RNN = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.RNN(x)
        out = self.fc(out[:, -1, :])
        return out
