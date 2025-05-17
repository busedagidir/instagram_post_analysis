import torch
from torch.utils.data import Dataset
from PIL import Image
from io import BytesIO
import base64
import pandas as pd

class FusionDataset(Dataset):
    def __init__(self, df, label_encoder, transform, tokenizer, device):
        self.df = df
        self.label_encoder = label_encoder
        self.transform = transform
        self.tokenizer = tokenizer
        self.device = device

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # image
        base64_str = row["filename"]
        base64_str = base64_str.split(",")[1]
        image = Image.open(BytesIO(base64.b64decode(base64_str))).convert("RGB")
        image_tensor = self.transform(image)

        # Caption (tokenizer burada yapılmaz, batch’te yapılır)
        caption = str(row["body"])

        # Label
        label = self.label_encoder.transform([row["label_post"]])[0]

        return image_tensor, caption, label
