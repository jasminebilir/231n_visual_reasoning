import torch
import torchvision
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from e2cnn import gspaces
from e2cnn import nn as enn
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import random
import matplotlib.pyplot as plt
import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torchvision.transforms import Resize
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score
import json
from tqdm import tqdm

class DeepNet(nn.Module):
    def __init__(self):
        super(DeepNet, self).__init__()
        self.nb_channels = 512
        self.conv1 = nn.Conv2d(3,  32, kernel_size=7, stride=4, padding=3)
        self.conv2 = nn.Conv2d( 32, self.nb_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(self.nb_channels, self.nb_channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(self.nb_channels, self.nb_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(self.nb_channels, self.nb_channels, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * self.nb_channels, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(x)
        #x = x.view(-1, 16 * self.nb_channels)
        #x = self.fc1(x)
        #x = F.relu(x)
        #x = self.fc2(x)
        #x = F.relu(x)
        #x = self.fc3(x)

        return x

class ViT(nn.Module):
    def __init__(self, num_features=8192, img_size=224, patch_size=16, num_classes=2, emb_size=768, num_heads=8, num_layers=6, hidden_dim=2048, dropout_rate=0.1):
        super(ViT, self).__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.emb_size = emb_size

        self.feature_embedding = nn.Linear(num_features, emb_size)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 2, emb_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))

        encoder_layer = TransformerEncoderLayer(emb_size, num_heads, hidden_dim, dropout_rate)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)

        self.to_cls_token = nn.Identity()

        self.fc = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x = self.feature_embedding(x)
        x = x.unsqueeze(1)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        x = self.transformer_encoder(x)
        x = self.to_cls_token(x[:, 0])
        x = self.fc(x)
        return x
    
class CNNTransformer(nn.Module):
    def __init__(self):
        super(CNNTransformer, self).__init__()
        self.cnn = DeepNet()
        self.vit = ViT()

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        output = self.vit(x)
        return output

# Define training, validation, and test loops
def train(model, device, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for data, target in tqdm(train_loader, desc="train steps"):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def validate(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    predictions = []
    true_labels = []
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
            preds = output.argmax(dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(target.cpu().numpy())
    accuracy = accuracy_score(true_labels, predictions)
    return val_loss / len(val_loader), accuracy

def test(model, device, test_loader):
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = output.argmax(dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(target.cpu().numpy())
    accuracy = accuracy_score(true_labels, predictions)
    return accuracy, predictions, true_labels

# Save model checkpoint
def save_checkpoint(model, optimizer, epoch, train_losses, val_losses, filepath):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    torch.save(checkpoint, filepath)

# Load model checkpoint
def load_checkpoint(filepath, model, optimizer):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    return epoch, train_losses, val_losses

# Hyperparameter tuning
def hyperparameter_tuning(params, train_dataset, val_dataset):
    best_val_accuracy = 0.0
    best_params = None
    for param in tqdm(params, desc="Param Combos"):
        model = CNNTransformer().to(device)
        optimizer = optim.Adam(model.parameters(), lr=param['lr'])
        criterion = nn.CrossEntropyLoss()
        train_loader = DataLoader(train_dataset, batch_size=param['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=param['batch_size'], shuffle=False)

        train_losses = []
        val_losses = []

        for epoch in tqdm(range(param['epochs']), desc="Remaining Epochs"):
            train_loss = train(model, device, train_loader, optimizer, criterion)
            val_loss, val_accuracy = validate(model, device, val_loader, criterion)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_params = param
                save_checkpoint(model, optimizer, epoch, train_losses, val_losses, 'best_model.pth')
            print(f"Epoch: {epoch}, Train_Loss: {train_loss}, Val_Loss: {val_loss}")
    return best_params, best_val_accuracy




if __name__ == "__main__":
    transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.ToTensor(),          # Convert images to tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize images (mean and std should be adjusted based on your dataset)
])
    train_dataset = datasets.ImageFolder(root='./results_problem_1/train', transform=transform)
    val_dataset = datasets.ImageFolder(root='./results_problem_1/val', transform=transform)
    test_dataset = datasets.ImageFolder(root='./results_problem_1/test', transform=transform)


    print("Created train_dataset, val_dataset, and test_dataset")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # Hyperparameter tuning
    params = [
        {'lr': 0.001, 'batch_size': 32, 'epochs': 5},
        #{'lr': 0.0001, 'batch_size': 64, 'epochs': 10}
    ]
    best_params, best_val_accuracy = hyperparameter_tuning(params, train_dataset, val_dataset)
    print(f"Best parameters: {best_params}, Best validation accuracy: {best_val_accuracy}")

    # Load best model and test
    model = CNNTransformer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])
    load_checkpoint('best_model.pth', model, optimizer)
    test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)
    test_accuracy, predictions, true_labels = test(model, device, test_loader)
    print(f"Test accuracy: {test_accuracy}")

    # Save predictions to file
    with open('predictions.json', 'w') as f:
        json.dump({'predictions': predictions, 'true_labels': true_labels}, f)