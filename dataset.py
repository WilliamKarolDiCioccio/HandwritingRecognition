import torch
import torchvision
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


class ImageBrowser:
    def __init__(self, train_loader):
        self.train_loader = train_loader
        self.current_index = 0

        # Create the figure and Cartesian axis
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)

        # Add buttons for navigation
        self.prev_button = Button(plt.axes([0.4, 0.01, 0.1, 0.075]), 'Previous')
        self.next_button = Button(plt.axes([0.5, 0.01, 0.1, 0.075]), 'Next')

        # Connect button press events to functions
        self.prev_button.on_clicked(self.prev_image)
        self.next_button.on_clicked(self.next_image)

        # Display the first training sample
        self.show_image()

        plt.show()

    def show_image(self):
        for images, labels in self.train_loader:
            self.ax.imshow(images[0][0], cmap='gray')
            self.ax.set_title(f'Training Label: {labels[0].item()}')
            plt.draw()
            break

    def prev_image(self, event):
        self.current_index = (self.current_index - 1) % len(self.train_loader)
        self.show_image()

    def next_image(self, event):
        self.current_index = (self.current_index + 1) % len(self.train_loader)
        self.show_image()


def load_data():
    # Define the transformations to be applied to the data
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    # Load the MNIST dataset
    train_dataset = torchvision.datasets.EMNIST(root='./data', split="letters", train=True, transform=transform, download=True)

    # Create a data loader for training
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)

    return train_loader


def main():
    train_loader = load_data()
    image_browser = ImageBrowser(train_loader)

    return 0


if __name__ == '__main__':
    main()
