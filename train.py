import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


class OptimizedCNN(nn.Module):
    def __init__(self, num_classes):
        super(OptimizedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.relu1 = nn.LeakyReLU(0.1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.relu2 = nn.LeakyReLU(0.1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=6, stride=1, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.relu3 = nn.LeakyReLU(0.1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.batch_norm4 = nn.BatchNorm2d(256)
        self.relu4 = nn.LeakyReLU(0.1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(256, 128)
        self.batch_norm_fc = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5)
        self.relu_fc = nn.LeakyReLU(0.1)

        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        x = self.conv4(x)
        x = self.batch_norm4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.batch_norm_fc(x)
        x = self.relu_fc(x)
        x = self.dropout(x)

        x = self.fc2(x)
        return x


# Function to train the model
def train(model, train_loader, criterion, optimizer, scheduler, num_epochs=5, device='cuda'):
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        start_time = time.time()  # Record the start time for the epoch
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Learning rate scheduler step
        scheduler.step()

        end_time = time.time()  # Record the end time for the epoch
        epoch_time = end_time - start_time
        remaining_time = (num_epochs - (epoch + 1)) * epoch_time  # Estimate remaining time

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, LR: {scheduler.get_last_lr()[0]}, '
              f'Epoch Time: {epoch_time:.2f} sec, Estimated Remaining Time: {remaining_time:.2f} sec')


# Function to test the model
def test(model, test_loader, device='cuda'):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy * 100:.2f}%')


# Function to save the trained model
def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    print(f'Model saved to {filepath}')


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load and preprocess the MNIST dataset
    mnist_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor()
    ])

    mnist_train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=mnist_transform, download=True)
    mnist_test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

    # Load and preprocess the EMNIST dataset
    emnist_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor()
    ])

    emnist_train_dataset = torchvision.datasets.EMNIST(root='./data', split='letters', train=True, transform=emnist_transform, download=True)
    emnist_test_dataset = torchvision.datasets.EMNIST(root='./data', split='letters', train=False, transform=transforms.ToTensor(), download=True)

    # Separate MNIST and EMNIST datasets
    mnist_train_loader = torch.utils.data.DataLoader(dataset=mnist_train_dataset, batch_size=64, shuffle=True)
    emnist_train_loader = torch.utils.data.DataLoader(dataset=emnist_train_dataset, batch_size=64, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(dataset=mnist_test_dataset, batch_size=64, shuffle=False)
    emnist_test_loader = torch.utils.data.DataLoader(dataset=emnist_test_dataset, batch_size=64, shuffle=False)

    # Create an instance of the OptimizedCNN model for MNIST
    mnist_model = OptimizedCNN(num_classes=10)
    mnist_model.to(device)

    # Define loss function and optimizer for MNIST
    mnist_criterion = nn.CrossEntropyLoss()
    mnist_optimizer = optim.Adam(mnist_model.parameters(), lr=0.001)
    mnist_scheduler = optim.lr_scheduler.StepLR(mnist_optimizer, step_size=5, gamma=0.1)
    num_epochs = 20

    # Train the MNIST model
    print("Start training MNIST model")
    train(mnist_model, mnist_train_loader, mnist_criterion, mnist_optimizer, scheduler=mnist_scheduler, num_epochs=num_epochs, device=device)

    # Create an instance of the OptimizedCNN model for EMNIST
    emnist_model = OptimizedCNN(num_classes=27)
    emnist_model.to(device)

    # Define loss function and optimizer for EMNIST
    emnist_criterion = nn.CrossEntropyLoss()
    emnist_optimizer = optim.Adam(emnist_model.parameters(), lr=0.001)
    emnist_scheduler = optim.lr_scheduler.StepLR(emnist_optimizer, step_size=5, gamma=0.1)

    # Train the EMNIST model
    print("Start training EMNIST model")
    train(emnist_model, emnist_train_loader, emnist_criterion, emnist_optimizer, scheduler=emnist_scheduler, num_epochs=num_epochs, device=device)

    # Test both models and compare results
    test(mnist_model, mnist_test_loader, device=device)
    test(emnist_model, emnist_test_loader, device=device)

    # Save the trained models
    save_model(mnist_model, "mnist_model.pth")
    save_model(emnist_model, "emnist_model.pth")


if __name__ == '__main__':
    main()
