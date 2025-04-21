# Import necessary libraries
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pywt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns
# Define the LeapGestRecog dataset class
class LeapGestRecog(Dataset):
    def __init__(self, root, transform=None, apply_wavelet=True):
        self.root = root
        self.transform = transform
        self.apply_wavelet = apply_wavelet
        self.x = []
        self.y = []
        self.idx_to_class = {
            0: "Palm", 1: "I", 2: "Fist", 3: "Fist_Moved",
            4: "Thumb", 5: "Index", 6: "OK", 7: "Palm_Moved",
            8: "C", 9: "Down"
        }
        # Load image file paths and labels
        folders = os.listdir(root)
        for folder in folders:
            for dirpath, _, filenames in os.walk(os.path.join(root, folder)):
                for filename in filenames:
                    self.x.append(os.path.join(dirpath, filename))
                    self.y.append(int(filename[9:11]))
        self.len = len(self.x)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img = Image.open(self.x[index]).convert('L')
        if self.apply_wavelet:
            img = self.apply_wavelet_transform(img)
        if self.transform:
            img = self.transform(img)
        y = torch.tensor(self.y[index] - 1)
        return img, y

    def apply_wavelet_transform(self, image, wavelet='haar', level=3, thresholding=True):
        image_np = np.array(image)
        coeffs = pywt.wavedec2(image_np, wavelet, level=level)
        cA = coeffs[0]
        coeffs_detail = coeffs[1:]
        if thresholding:
            threshold = np.median(np.abs(coeffs_detail[0])) / 0.6745
            coeffs_detail_thresholded = []
            for detail_level in coeffs_detail:
                cH, cV, cD = detail_level
                cH_thresh = pywt.threshold(cH, threshold, mode='soft')
                cV_thresh = pywt.threshold(cV, threshold, mode='soft')
                cD_thresh = pywt.threshold(cD, threshold, mode='soft')
                coeffs_detail_thresholded.append((cH_thresh, cV_thresh, cD_thresh))
        reconstructed = pywt.waverec2((cA, *coeffs_detail_thresholded), wavelet)
        transformed_image = Image.fromarray(np.clip(reconstructed, 0, 255).astype('uint8'))
        return transformed_image

# Set up data transformations and dataset
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])
root = "for data set contact  me on saurav.satyam132@gmail.com"
dataset = LeapGestRecog(root, transform=transform, apply_wavelet=True)

# Split dataset into training, validation, and testing
train_size = int(0.8 * 0.8 * len(dataset))
val_size = int(0.8 * 0.2 * len(dataset))
test_size = len(dataset) - (train_size + val_size)
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

# Define the neural network architecture with dropout
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(30 * 30 * 128, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 30 * 30 * 128)  # Flattening, adjust here as well
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Initialize the model and optimizer
model = Net()
# Assuming 'model' is already defined along with 'train_loader', 'val_loader', and 'loss_fn'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
min_val_loss = float('inf')
early_stop_counter = 0
train_loss_epoch = []
val_loss_epoch = []

# Train for more epochs and implement early stopping
for epoch in range(5):  # More epochs for detailed training curve
    model.train()
    train_loss = 0
    total_batches = len(train_loader)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)

        # Calculate progress
        progress = (batch_idx + 1) / total_batches * 100
        print(f'Train Epoch: {epoch} [{batch_idx + 1}/{total_batches} ({progress:.0f}%)]\tLoss: {loss.item():.6f}')
        
    train_loss /= len(train_loader.dataset)
    train_loss_epoch.append(train_loss)

    valid_loss = 0
    model.eval()
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target)
            valid_loss += loss.item() * data.size(0)

    valid_loss /= len(val_loader.dataset)
    val_loss_epoch.append(valid_loss)

    print(f'Epoch {epoch} \t Training Loss: {train_loss:.6f} \t Validation Loss: {valid_loss:.6f}')

    # Early stopping logic
    if valid_loss < min_val_loss:
        min_val_loss = valid_loss
        torch.save(model.state_dict(), 'model_best.pth')
        early_stop_counter = 10
    else:
        early_stop_counter += 1
        if early_stop_counter > 100:
            print("Early stopping triggered.")
            break


# Plotting the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_loss_epoch, label='Train Loss')
plt.plot(val_loss_epoch, label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_validation_loss_curve.png')

# Load the best model and test
model.load_state_dict(torch.load('model_best.pth'))
model.to(device)

# Import necessary libraries for confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Define the function to test the model and create confusion matrix
def test():
    model.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_preds.extend(pred.view(-1).tolist())
            all_targets.extend(target.view(-1).tolist())

    test_loss /= len(test_loader.dataset)
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='macro')
    recall = recall_score(all_targets, all_preds, average='macro')
    f1 = f1_score(all_targets, all_preds, average='macro')

    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')
    
    # Create and visualize confusion matrix
    conf_matrix = confusion_matrix(all_targets, all_preds)
    class_names = ['Palm', 'I', 'Fist', 'Fist_Moved', 'Thumb', 'Index', 'OK', 'Palm_Moved', 'C', 'Down']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Call the test function
test()