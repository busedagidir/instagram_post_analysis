import base64
from io import BytesIO

import torch
from PIL import Image
from torchvision import models, transforms


class ImageEncoder:
    def __init__(self, device, fine_tune=False):
        self.device = device
        self.fine_tune = fine_tune
        self.model = models.resnet50(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])  # removed last fully connected layer
        if not self.fine_tune:
            self.model.eval()
            
        self.model.to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def decode_base64(self, base64_str):
        #base64_str = base64_str.split(",")[1]
        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        return image
    
    def encode_from_base64(self, base64_str):
        image = self.decode_base64(base64_str)
        image = self.transform(image).unsqueeze(0).to(self.device)

        if self.fine_tune:
            embedding = self.model(image).squeeze()
        else:
            with torch.no_grad():
                embedding = self.model(image).squeeze()
        return embedding
