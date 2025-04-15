import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
from models.colorization_net import ColorizationNet
from utils.preprocessing import clean_caption
from utils.vocab import Vocabulary
from skimage.color import rgb2lab
import nltk
nltk.download('punkt')

# ==== Config ====
IMAGE_DIR = "data/Images/"
CAPTION_FILE = "data/captions.txt"
IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Dataset ====
class FlickrColorDataset(Dataset):
    def __init__(self, img_dir, captions_df, vocab, transform=None):
        self.img_dir = img_dir
        self.captions_df = captions_df
        self.vocab = vocab
        self.transform = transform

    def __len__(self):
        return len(self.captions_df)

    def __getitem__(self, idx):
        img_name = self.captions_df.iloc[idx]["image"]
        caption = self.captions_df.iloc[idx]["caption"]
        img_path = os.path.join(self.img_dir, img_name)

        # Load and preprocess image
        img = Image.open(img_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        img = np.array(img) / 255.0
        img_lab = rgb2lab(img).astype("float32")
        
        # Separate channels
        L = img_lab[:, :, 0] / 100.0           # Shape: [H, W], normalize to 0-1
        ab = img_lab[:, :, 1:] / 128.0         # Shape: [H, W, 2], normalize to -1 to 1

        # Convert to tensors and add channel dim
        L = torch.from_numpy(L).unsqueeze(0)   # Shape: [1, H, W]
        ab = torch.from_numpy(ab).permute(2, 0, 1)  # [2, H, W]

        # Caption to tensor
        caption_tensor = torch.tensor(self.vocab.numericalize(caption), dtype=torch.long)

        return L.float(), ab.float(), caption_tensor


# ==== Collate Function ====
def collate_fn(batch):
    images, ab_channels, captions = zip(*batch)
    captions = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=0)
    return torch.stack(images), torch.stack(ab_channels), captions

# ==== Training Loop ====
def train():
    print("Loading captions...")
    df = pd.read_csv(CAPTION_FILE, names=["image", "caption"])
    df["caption"] = df["caption"].dropna().astype(str).apply(clean_caption)

    print("Building vocabulary...")
    vocab = Vocabulary(freq_threshold=5)
    vocab.build_vocab(df["caption"].tolist())

    print("Creating dataset and dataloader...")
    dataset = FlickrColorDataset(
        img_dir=IMAGE_DIR,
        captions_df=df,
        vocab=vocab,
        transform=transforms.ToTensor()
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    print("Initializing model...")
    model = ColorizationNet(vocab_size=len(vocab)).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for L, ab, captions in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            L, ab, captions = L.to(DEVICE), ab.to(DEVICE), captions.to(DEVICE)

            optimizer.zero_grad()
            output_ab = model(L, captions)
            loss = criterion(output_ab, ab)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), f"checkpoints/color_model_epoch{epoch+1}.pth")

if __name__ == "__main__":
    train()
