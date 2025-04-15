import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512):
        super(TextEncoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        x = self.embed(x)
        _, (h_n, _) = self.lstm(x)
        return h_n.squeeze(0)  # shape: (batch, hidden_dim)
