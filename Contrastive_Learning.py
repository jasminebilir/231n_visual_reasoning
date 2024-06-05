import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
import random
import matplotlib.pyplot as plt
import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.metrics import accuracy_score
import json
import argparse
import torch.backends.cudnn as cudnn
from PIL import Image
import numpy as np

# Contrastive Loss function
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

# Modified DeepNet with embeddings
class DeepNet2(nn.Module):
    def __init__(self):
        super(DeepNet, self).__init__()
        self.nb_channels = 512
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=4, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, self.nb_channels, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(self.nb_channels)
        self.conv3 = nn.Conv2d(self.nb_channels, self.nb_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(self.nb_channels)
        self.conv4 = nn.Conv2d(self.nb_channels, self.nb_channels, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(self.nb_channels)
        self.conv5 = nn.Conv2d(self.nb_channels, self.nb_channels, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(self.nb_channels)
        self.fc1 = nn.Linear(16 * self.nb_channels, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.bn_fc2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(x)
        x = x.view(-1, 16 * self.nb_channels)
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        embedding = self.fc2(x)
        x = self.bn_fc2(embedding)
        x = F.relu(x)
        x = self.fc3(x)
        return x, embedding




class DeepNet(nn.Module):
    def __init__(self):
        super(DeepNet, self).__init__()
        self.nb_channels = 512
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=4, padding=3)
        self.conv2 = nn.Conv2d(32, self.nb_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(self.nb_channels, self.nb_channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(self.nb_channels, self.nb_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(self.nb_channels, self.nb_channels, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * self.nb_channels, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)

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
        x = x.view(-1, 16 * self.nb_channels)
        x = self.fc1(x)
        x = F.relu(x)
        embedding = self.fc2(x)
        x = F.relu(embedding)
        x = self.fc3(x)
        return x, embedding



# Custom Dataset to create pairs of images
class SiameseNetworkDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img0_tuple[1] != img1_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)

# Updated training function with contrastive loss
def train(model, device, train_loader, optimizer, criterion, contrastive_criterion, alpha=0.5):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        # Siamese Network Dataset case
        img0, img1, label = batch
        img0, img1, label = img0.to(device), img1.to(device), label.to(device)
            
        optimizer.zero_grad()
        output1, embedding1 = model(img0)
        output2, embedding2 = model(img1)
        contrastive_loss = contrastive_criterion(embedding1, embedding2, label)
            
        combined_loss = (1 - alpha) * contrastive_loss
        combined_loss.backward()
        optimizer.step()
        running_loss += combined_loss.item()
            
    return running_loss / len(train_loader)

def validate(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    predictions = []
    true_labels = []
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            target = target.float().view(-1, 1)  # Ensure target is of shape (batch_size, 1)
            output, _ = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
            preds = torch.sigmoid(output).round().cpu().numpy()  # Apply sigmoid and round to get binary predictions
            predictions.extend(preds)
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
            target = target.float().view(-1, 1)  # Ensure target is of shape (batch_size, 1)
            output, _ = model(data)
            preds = torch.sigmoid(output).round().cpu().numpy()  # Apply sigmoid and round to get binary predictions
            predictions.extend(preds)
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
def hyperparameter_tuning(params, train_dataset, val_dataset, problem_number):
    best_val_accuracy = 0.0
    best_params = None
    for param in params:
        model = DeepNet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=param['lr'])
        criterion = nn.BCEWithLogitsLoss()
        contrastive_criterion = ContrastiveLoss()
        train_loader = DataLoader(train_dataset, batch_size=param['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=param['batch_size'], shuffle=False)
        
        train_losses = []
        val_losses = []

        for epoch in range(param['epochs']):
            train_loss = train(model, device, train_loader, optimizer, criterion, contrastive_criterion)
            val_loss, val_accuracy = validate(model, device, val_loader, criterion)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_params = param
                save_checkpoint(model, optimizer, epoch, train_losses, val_losses, problem_number + '_best_model.pth')
            print(f"Epoch: {epoch}, Train_Loss: {train_loss}, Val_Loss: {val_loss}")
    return best_params, best_val_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem_number', type=str, required=True, help='Problem Number')
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    problem_path = 'results_' + args.problem_number
    train_dataset = datasets.ImageFolder(root=f'recurrent_vision_transformers/{problem_path}/train', transform=transform)
    val_dataset = datasets.ImageFolder(root=f'recurrent_vision_transformers/{problem_path}/val', transform=transform)
    test_dataset = datasets.ImageFolder(root=f'recurrent_vision_transformers/{problem_path}/test', transform=transform)

    train_siamese_dataset = SiameseNetworkDataset(imageFolderDataset=train_dataset, transform=transform)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = [
        {'lr': 0.0001, 'batch_size': 64, 'epochs': 5, 'alpha': 0.5},
        {'lr': 0.001, 'batch_size': 64, 'epochs': 5, 'alpha': 0.5},
    ]

    best_params, best_val_accuracy = hyperparameter_tuning(params, train_siamese_dataset, val_dataset, args.problem_number)
    print(f"Best parameters: {best_params}, Best validation accuracy: {best_val_accuracy}")

    model = DeepNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])
    epoch, train_losses, val_losses = load_checkpoint(args.problem_number + '_best_model.pth', model, optimizer)
    test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)
    test_accuracy, predictions, true_labels = test(model, device, test_loader)
    print(f"Test accuracy: {test_accuracy}")

    predictions = [int(prediction) for prediction in predictions]
    true_labels = [int(true_label) for true_label in true_labels]
    train_losses = [loss for loss in train_losses]
    val_losses = [loss for loss in val_losses]

    results = {
        'predictions': predictions,
        'true_labels': true_labels,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_accuracy': test_accuracy
    }

    with open(args.problem_number+'_results.json', 'w') as f:
        json.dump(results, f)
