import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import os
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import argparse


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class MultiscaleFeatureExtractor(nn.Module):
    def __init__(self):
        super(MultiscaleFeatureExtractor, self).__init__()
        resnet = models.resnet152(pretrained=True)
        self.resnet_layers = nn.Sequential(*list(resnet.children())[:-2])
        
        self.conv3 = nn.Sequential(*list(resnet.children())[:6])
        self.conv4 = nn.Sequential(*list(resnet.children())[6])
        self.conv5 = nn.Sequential(*list(resnet.children())[7])
        
        self.upsample2x = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample4x = nn.Upsample(scale_factor=4, mode='nearest')
        
        self.conv5_1x1 = nn.Conv2d(2048, 1024, kernel_size=1)
        self.conv4_1x1 = nn.Conv2d(1024, 512, kernel_size=1)
    
    def forward(self, x):
        conv3_features = self.conv3(x)  # 28x28x512
        conv4_features = self.conv4(conv3_features)  # 14x14x1024
        conv5_features = self.conv5(conv4_features)  # 7x7x2048
        
        # Perform upsampling and fusion
        conv5_1x1 = self.conv5_1x1(conv5_features)  # 7x7x1024
        upsampled_conv5 = self.upsample2x(conv5_1x1)  # 14x14x1024
        fused_conv4 = conv4_features + upsampled_conv5  # 14x14x1024
        
        fused_conv4_1x1 = self.conv4_1x1(fused_conv4)  # 14x14x512
        upsampled_fused_conv4 = self.upsample2x(fused_conv4_1x1)  # 28x28x512
        fused_conv3 = conv3_features + upsampled_fused_conv4  # 28x28x512
        
        return fused_conv3

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class VQAResNet152(nn.Module):
    def __init__(self, hidden_dim=512, output_dim=1000):  # Example dimensions
        super(VQAResNet152, self).__init__()
        self.feature_extractor = MultiscaleFeatureExtractor()
        self.mlp = MLP(input_dim=28*28*512, hidden_dim=hidden_dim, output_dim=output_dim)
    
    def forward(self, x):
        # Extract and fuse features
        fused_features = self.feature_extractor(x)
        # Flatten the fused features
        flattened_features = fused_features.view(fused_features.size(0), -1)
        # Pass through the MLP for final prediction
        output = self.mlp(flattened_features)
        return output
    

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for _, data in tqdm(enumerate(train_loader, 0), unit="batch", total=len(train_loader)):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def evaluate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for _, data in tqdm(enumerate(val_loader, 0), unit="batch", total=len(val_loader)):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a VQA model on the SVRT dataset")
    parser.add_argument('--data_dir', type=str, help='Directory where the SVRT dataset is located')
    parser.add_argument('--model_name', type=str, help='Model pth output filename')
    parser.add_argument('--loss_file', type=str, help='Filename of file containing losses')
    args = parser.parse_args()

    data_dir = args.data_dir

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)

    print('Retrieved datasets')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print("Loaded Data")

    model = VQAResNet152(output_dim=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    losses = []

    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        losses.append(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

   #  with open(f'{args.loss_file}.txt', 'w') as f:
   #     for loss in losses:
   #         f.write(loss + "\n")
    
    torch.save(model.state_dict(), f'{args.model_name}.pth')

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    losses.append(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    with open(f'{args.loss_file}.txt', 'w') as f:
        for loss in losses:
            f.write(loss + "\n")