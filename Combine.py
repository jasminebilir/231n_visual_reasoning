import torch
import torchvision
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from e2cnn import gspaces
from e2cnn import nn as enn
import os
import argparse
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
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        #print("CNN", x.shape)
        return x
    
class ViTWithTorchTransformer(nn.Module):
    def __init__(self, num_classes, dim, depth, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.num_patches = 16  # 4x4 grid of patches from a 4x4 feature map
        patch_size = 4  # Size of one side of a square patch
        patch_depth = 512  # Number of channels in each patch
        self.patch_dim = patch_depth * patch_size * patch_size  # Not used for direct flattening

        self.patch_to_embedding = nn.Conv2d(patch_depth, dim, kernel_size=(4, 4), stride=4)
        
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

        self.transformer = nn.Transformer(
            d_model=dim,
            nhead=heads,
            num_encoder_layers=depth,
            dim_feedforward=mlp_dim,
            dropout=dropout
        )
        
        self.norm1 = nn.LayerNorm(dim)  # Layer normalization before entering the transformer
        self.norm2 = nn.LayerNorm(dim)  # Optional: additional normalization before final classification
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, x):
        # x should have dimensions [batch_size, channels, height, width] = [batch_size, 512, 4, 4]
        #print("ViT input shape:", x.shape)
        x = self.patch_to_embedding(x)  # This will convert each [4, 4, 512] patch directly into [dim]
        #print("After patch_to_embedding shape:", x.shape)  # Expected: [32, 768, 1, 1] given stride of 4 and kernel of 4
        x = x.flatten(2).permute(0, 2, 1)  # Rearrange to sequence format expected by Transformer
        #print("After flattening and permute:", x.shape)  # Expected: [32, 1, 768]

        b, n, _ = x.shape
        cls_tokens = self.cls_token.expand(b, 1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        #print("Before Transformer shape:", x.shape)  # Check if the dimensions are still correct
        x = self.norm1(x)
        x = self.transformer(x, x)
        #print("After Transformer shape:", x.shape)  # This should be [32, 1, 768] or similar, with '1' being the position of class token
        x = x[:, 0]
        #print("After extracting class token:", x.shape)# [32, 768]
        x = self.norm2(x)
        out = self.mlp_head(x)
       # print("Final output shape:", out.shape)  # Should be [32, 1]
        return out

class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()
        self.cnn = DeepNet()
        self.vit = ViTWithTorchTransformer(num_classes=1, dim=768, depth=6, heads=12, mlp_dim=3072)

    def forward(self, x):
        x = self.cnn(x)
        # The output from CNN is directly in the correct shape to be processed by patch_to_embedding in ViT
        x = self.vit(x)
        return x

def train(model, device, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for data, target in tqdm(train_loader, desc="train"):
        data, target = data.to(device), target.to(device)
        target = target.float().view(-1, 1)  # Ensure target is of shape (batch_size, 1)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            target = target.float().view(-1, 1)  # Ensure target is of shape (batch_size, 1)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
            preds = torch.sigmoid(output).round()  # Apply sigmoid and round to get binary predictions
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(target.cpu().numpy())
            print("Preds", preds)
            print("out", output)
           # input()
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
            output = model(data)
            preds = torch.sigmoid(output).round()  # Apply sigmoid and round to get binary predictions
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

def load_checkpoint(filepath, model, optimizer):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    return epoch, train_losses, val_losses

def init_weights_he(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# Hyperparameter tuning
def hyperparameter_tuning(params, train_dataset, val_dataset, problem_number):
    best_val_accuracy = 0.0
    best_params = None
    for param in params:
        model = CombinedModel().to(device)
        model.apply(init_weights_he)
        optimizer = optim.Adam(model.parameters(), lr=param['lr'])
        criterion = nn.BCEWithLogitsLoss()
        train_loader = DataLoader(train_dataset, batch_size=param['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=param['batch_size'], shuffle=False)
        
        train_losses = []
        val_losses = []

        for epoch in tqdm(range(param['epochs']), desc="epochs"):
            train_loss = train(model, device, train_loader, optimizer, criterion)
            val_loss, val_accuracy = validate(model, device, val_loader, criterion)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_params = param
                save_checkpoint(model, optimizer, epoch, train_losses, val_losses, problem_number + '_best_model.pth')
            print(f"Epoch: {epoch}, Train_Loss: {train_loss}, Val_Loss: {val_loss}, Val_Acc: {val_accuracy}, Best_acc: {best_val_accuracy}")
    return best_params, best_val_accuracy


def visualize_sample_predictions(test_loader, model, device, n_samples=5):
    model.eval()
    fig, axes = plt.subplots(1, n_samples, figsize=(15, 15))
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            if i >= n_samples:
                break
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = torch.sigmoid(output).round().cpu().numpy()
            for j in range(data.size(0)):
                if j >= n_samples:
                    break
                img = data[j].cpu().numpy().transpose((1, 2, 0))
                axes[j].imshow(img)
                axes[j].set_title(f'Pred: {int(preds[j][0])}, True: {int(target[j].item())}')
                axes[j].axis('off')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem_number', type=str, required=True, help='Problem Number')
    args = parser.parse_args()

    #torch.backends.cudnn.benchmark = False  # Try setting this to False
    #torch.backends.cudnn.deterministic = True  # Optionally set this to True
    #torch.backends.cudnn.enabled = False

    transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.ToTensor(),          # Convert images to tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize images (mean and std should be adjusted based on your dataset)
])
    problem_path = 'results_problem_' + args.problem_number
    train_dataset = datasets.ImageFolder(root=f'./train', transform=transform)
    val_dataset = datasets.ImageFolder(root=f'./val', transform=transform)
    test_dataset = datasets.ImageFolder(root=f'./test', transform=transform)

    print("Created train_dataset, val_dataset, and test_dataset")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # Hyperparameter tuning
    params = [
        {'lr': 0.000001, 'batch_size': 32, 'epochs': 10}
    ]

    best_params, best_val_accuracy = hyperparameter_tuning(params, train_dataset, val_dataset, args.problem_number)
    print(f"Best parameters: {best_params}, Best validation accuracy: {best_val_accuracy}")

    
    # Load best model and test
    model = CombinedModel().to(device)
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
