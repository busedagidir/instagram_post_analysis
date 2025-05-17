import torch
import torch.nn as nn


class FusionClassifier(nn.Module):
    def __init__(self, image_dim=2048, text_dim=768, hidden_dim=512, num_classes=3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(image_dim + text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, image_emb, text_emb):
        combined = torch.cat((image_emb, text_emb), dim=-1)
        return self.fc(combined)
