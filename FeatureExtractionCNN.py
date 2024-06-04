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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, )),
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

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(16 * 1 * 1, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc_layers(x)
        return x



class VQAResNet152(nn.Module):
    def __init__(self):
        super(VQAResNet152, self).__init__()
        self.feature_extractor = MultiscaleFeatureExtractor()
        self.cnn = CNN()

    def forward(self, x):
        fused_features = self.feature_extractor(x)
        output = self.cnn(fused_features)
        return output


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for _, data in tqdm(enumerate(train_loader, 0), unit="batch", total=len(train_loader)):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        labels = labels.float().view(-1, 1)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = torch.sigmoid(outputs).round()
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

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
            labels = labels.float().view(-1, 1)

            outputs = model(images)
            # print(outputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            # _, preds = torch.max(outputs, 1)
            preds = torch.sigmoid(outputs).round()
            # print(preds)
            # input()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

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

    model = VQAResNet152().to(device)

    criterion = nn.BCEWithLogitsLoss()  # Using BCEWithLogitsLoss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    losses = []

    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        losses.append(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    torch.save(model.state_dict(), f'{args.model_name}.pth')

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    losses.append(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    with open(f'{args.loss_file}.txt', 'w') as f:
        for loss in losses:
            f.write(loss + "\n")