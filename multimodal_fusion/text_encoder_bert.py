import torch
from transformers import BertTokenizer, BertModel

class TextEncoder:
    def __init__(self, device='cpu', fine_tune=False):
        self.device = device
        self.fine_tune = fine_tune
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased").to(self.device)
        if not fine_tune:
            self.model.eval()
        
        self.model.to(self.device)

    def encode(self, text):
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        # print(text)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        if self.fine_tune:
            outputs = self.model(**inputs).pooler_output.squeeze()
        else:
            with torch.no_grad():
                outputs = self.model(**inputs).pooler_output.squeeze()
        # return outputs.pooler_output.squeeze()
        return outputs
