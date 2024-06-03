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
import  json


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
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x



def train(model, device, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        target = target.float().view(-1, 1)  # Ensure target is of shape (batch_size, 1)
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
            target = target.float().view(-1, 1)  # Ensure target is of shape (batch_size, 1)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
            preds = torch.sigmoid(output).round()  # Apply sigmoid and round to get binary predictions
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
    for param in params:
        model = DeepNet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=param['lr'])
        criterion = nn.BCEWithLogitsLoss()
        train_loader = DataLoader(train_dataset, batch_size=param['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=param['batch_size'], shuffle=False)
        
        train_losses = []
        val_losses = []

        for epoch in range(param['epochs']):
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
    transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.ToTensor(),          # Convert images to tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize images (mean and std should be adjusted based on your dataset)
])
    #train_dataset = datasets.ImageFolder(root='recurrent_vision_transformers/results_problem_1/train', transform=transform)
    #val_dataset = datasets.ImageFolder(root='recurrent_vision_transformers/results_problem_1/val', transform=transform)
    test_dataset = datasets.ImageFolder(root='recurrent_vision_transformers/results_problem_1/test', transform=transform)

    
    print("Created train_dataset, val_dataset, and test_dataset")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # Hyperparameter tuning
    params = [
        {'lr': 0.0001, 'batch_size': 64, 'epochs': 10}
    ]
    #best_params, best_val_accuracy = hyperparameter_tuning(params, train_dataset, val_dataset)
    #print(f"Best parameters: {best_params}, Best validation accuracy: {best_val_accuracy}")

    best_params = params[0]
    # Load best model and test
    model = DeepNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])
    epoch, train_losses, val_losses = load_checkpoint('best_model.pth', model, optimizer)
    test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)
    test_accuracy, predictions, true_labels = test(model, device, test_loader)
    print(f"Test accuracy: {test_accuracy}")

   


    predictions = [int(prediction) for prediction in predictions]
    true_labels = [int(true_label) for true_label in true_labels]
    train_losses = [loss for loss in train_losses]
    val_losses = [loss for loss in val_losses]
    #train_accuracies = [float(acc) for acc in train_accuracies]
    #val_accuracies = [float(acc) for acc in val_accuracies]

    #print(type(predictions))
    #print(type(true_labels))
    #print(type(train_losses))
    #print(type(val_losses))
    # Save predictions and losses to file
    results = {
        'predictions': predictions,
        'true_labels': true_labels,
        'train_losses': train_losses,
        'val_losses': val_losses,
        #'train_accuracies': train_accuracies,
        #'val_accuracies': val_accuracies,
        'test_accuracy': test_accuracy
    }

    with open('results.json', 'w') as f:
        json.dump(results, f)


