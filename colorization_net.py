import torch
import torch.nn as nn

class ColorizationNet(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size=512):
        super(ColorizationNet, self).__init__()
        
        # Text embedding
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)

        # Image encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # L channel only
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # Fusion
        self.fusion = nn.Linear(hidden_size + 256, 256)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=3, padding=1),  # Output ab channels
            nn.Tanh()
        )

    def forward(self, L, captions):
        # Encode text
        embedded = self.embedding(captions)
        _, (hidden, _) = self.lstm(embedded)  # hidden: (1, B, H)
        hidden = hidden.squeeze(0)  # (B, H)

        # Encode image
        img_feat = self.encoder(L)  # (B, 256, H/4, W/4)
        B, C, H, W = img_feat.shape

        # Expand text features
        text_feat = hidden.unsqueeze(2).unsqueeze(3).expand(B, hidden.size(1), H, W)

        # Concatenate image and text features
        fused = torch.cat([img_feat, text_feat], dim=1)  # (B, C+H, H, W)
        fused = self.fusion(fused.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # Linear on channels

        # Decode to ab
        out_ab = self.decoder(fused)
        return out_ab


