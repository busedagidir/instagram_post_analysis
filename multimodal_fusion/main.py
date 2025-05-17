from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from torchvision import transforms

from fusion_dataset import FusionDataset
from fusion_model import FusionClassifier
from image_encoder_resnet import ImageEncoder
from text_encoder_bert import TextEncoder

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using..: ", device)

def set_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


def train(epochs, model, train_loader, val_loader, text_encoder, image_encoder, criterion, optimizer, fine_tune_mode):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_val_loss = float("inf")
    patience = 3
    epochs_no_improve = 0
    best_model_path = None

    model_dir = Path("checkpoints")
    model_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = model_dir / f"best_model_resnet_{timestamp}.pt"


    for epoch in range(epochs):
        # training
        model.train()
        train_correct = 0
        train_total = 0
        batch_train_loss = 0

        for batch_idx, (images, captions, labels) in enumerate(train_loader):
            # DataLoader __getitem__(0), __getitem__(1), ..., __getitem__(7) çağırdı
            images = images.to(device)
            labels = labels.to(device)
            # print("Labels: ", labels)

            # Text tokenizer (batch)
            tokenized = text_encoder.tokenizer(
                list(captions),
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)

            if fine_tune_mode:
                text_embeddings = text_encoder.model(**tokenized).pooler_output
                image_embeddings = image_encoder.model(images).squeeze(-1).squeeze(-1)
            else:
                # Text embedding
                with torch.no_grad():
                    text_embeddings = text_encoder.model(**tokenized).pooler_output

                # Image embedding
                with torch.no_grad():
                    # image_embeddings = image_encoder.model(images).squeeze()
                    image_embeddings = image_encoder.model(images).squeeze(-1).squeeze(-1)

            # forward
            outputs = model(image_embeddings, text_embeddings)
            loss = criterion(outputs, labels)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # accuracy & loss tracking
            batch_train_loss += loss.item()
            predicted = outputs.argmax(dim=1)  # tensor([0, 0, 0, 0, 0, 0, 0, 0])
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)


        # Calculate the accuracy and loss for this epoch
        accuracy = 100 * train_correct / train_total
        avg_train_loss = batch_train_loss / len(train_loader)

        # validation
        model.eval()
        batch_val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_idx, (images, captions, labels) in enumerate(val_loader):
                # DataLoader __getitem__(0), __getitem__(1), ..., __getitem__(7) çağırdı
                images = images.to(device)
                labels = labels.to(device)

                # Text tokenizer (batch)
                tokenized = text_encoder.tokenizer(
                    list(captions),
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(device)

                # Text embedding
                text_embeddings = text_encoder.model(**tokenized).pooler_output

                # Image embedding
                # image_embeddings = image_encoder.model(images).squeeze()
                image_embeddings = image_encoder.model(images).squeeze(-1).squeeze(-1)

                outputs = model(image_embeddings, text_embeddings)
                loss = criterion(outputs, labels)

                batch_val_loss += loss.item()
                predicted = outputs.argmax(dim=1)  # tensor([0, 0, 0, 0, 0, 0, 0, 0])
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_accuracy = 100 * val_correct / val_total
        avg_val_loss = batch_val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(accuracy)
        val_accuracies.append(val_accuracy)

        print(f'Epoch {epoch + 1}: Train Accuracy = {accuracy:.2f}%')
        print(f'Epoch {epoch + 1}: Validation Accuracy = {val_accuracy:.2f}%')
        print(f"Epoch {epoch + 1}: Avg Train Loss = {avg_train_loss:.4f}")
        print(f"Epoch {epoch + 1}: Avg Validation Loss = {avg_val_loss:.4f}")

        # early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_path = model_path
            torch.save(model.state_dict(), best_model_path)
            print(f"Validation loss improved. Model saved at {best_model_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")

            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    return best_model_path, train_losses, val_losses, train_accuracies, val_accuracies


def test(model_path, model, test_loader, text_encoder, image_encoder, criterion, device, label_encoder=None, show_predictions=False):
    print(f"Loading best model from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_correct = 0
    test_total = 0
    test_loss = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, captions, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Text encoding
            tokenized = text_encoder.tokenizer(
                list(captions),
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)
            text_embeddings = text_encoder.model(**tokenized).pooler_output

            # Image encoding
            image_embeddings = image_encoder.model(images).squeeze(-1).squeeze(-1)

            # Prediction
            outputs = model(image_embeddings, text_embeddings)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            predicted = outputs.argmax(dim=1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Optional: Show sample predictions
            if show_predictions and label_encoder:
                for i in range(len(captions)):
                    pred_str = label_encoder.inverse_transform([predicted[i].cpu().item()])[0]
                    true_str = label_encoder.inverse_transform([labels[i].cpu().item()])[0]
                    caption_preview = captions[i][:50].replace("\n", " ")
                    print(f"Caption: {caption_preview}...")
                    print(f"Predicted: {pred_str} | True: {true_str}")
                    print("-" * 40)

    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * test_correct / test_total

    print(f"Final Test Results:")
    print(f"Test Accuracy : {test_accuracy:.2f}%")
    print(f"Avg Test Loss : {avg_test_loss:.4f}")

    label_names = label_encoder.classes_
    plot_confusion_matrix(all_labels, all_preds, labels=label_names)

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=label_names))



def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Over Epochs")
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Train Acc")
    plt.plot(epochs, val_accuracies, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Over Epochs")
    plt.legend()

    plt.tight_layout()
    plt.savefig("/mnt/ceph/storage/data-tmp/2024/gani7218/instagram-post-analysis/multimodal_fusion/accuracy_loss_metrics_resnet.png")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("/mnt/ceph/storage/data-tmp/2024/gani7218/instagram-post-analysis/multimodal_fusion/confusion_matrix_resnet.png")
    plt.show()


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "config/config.yml"
    # Load the configuration
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    csv_file_path = project_root / "annotated_data" / config["csv_file_path"]
    fine_tune_mode = config['fine-tune']

    df = pd.read_csv(csv_file_path)
    label_encoder = LabelEncoder()
    df["label_encoded"] = label_encoder.fit_transform(df["label_post"])

    image_encoder = ImageEncoder(device, fine_tune_mode)
    text_encoder = TextEncoder(device, fine_tune_mode)
    model = FusionClassifier().to(device)
    
    set_requires_grad(image_encoder.model, fine_tune_mode)
    set_requires_grad(text_encoder.model, fine_tune_mode)

    params = list(model.parameters())
    if fine_tune_mode:
        params += list(image_encoder.model.parameters())
        params += list(text_encoder.model.parameters())
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params, lr=1e-4 if not fine_tune_mode else 1e-5)
    
    X_train, X_val = train_test_split(df, test_size=0.3, random_state=42)  # label
    X_val, X_test = train_test_split(X_val, test_size=0.5, random_state=42)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Dataset & Dataloader
    train_dataset = FusionDataset(X_train, label_encoder, transform, text_encoder.tokenizer, device)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataset = FusionDataset(X_val, label_encoder, transform, text_encoder.tokenizer, device)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_dataset = FusionDataset(X_test, label_encoder, transform, text_encoder.tokenizer, device)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    print(f"Fine-tuning: {'ENABLED' if fine_tune_mode else 'DISABLED'}")

    model_path, train_losses, val_losses, train_accuracies, val_accuracies = train(config["epochs"], model, train_loader, val_loader, text_encoder, image_encoder, criterion, optimizer, fine_tune_mode)
    if model_path:
        test(
            model_path=model_path,
            model=model,
            test_loader=test_loader,
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            criterion=criterion,
            device=device,
            label_encoder=label_encoder,  # opsiyonel
            show_predictions=True  # örnek caption görmek istersen
        )
    else:
        print("No model was saved, skipping test.")

    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)
