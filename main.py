import tkinter as tk

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


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


def preprocess_image(image):
    # Preprocess the image for the model (reshape and normalize)
    image = image.resize((28, 28))
    image = np.array(image) / 255.0
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return torch.tensor(image)


class WhiteboardApp:
    def __init__(self, master, mnist_model, emnist_model):
        self.master = master
        self.master.title("Whiteboard App")

        self.canvas_width = 400
        self.canvas_height = 400
        self.canvas_resolution = 28

        self.canvas = tk.Canvas(self.master, bg="white", width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack(expand=tk.YES, fill=tk.BOTH)

        self.whiteboard_data = np.zeros((self.canvas_resolution, self.canvas_resolution), dtype=np.uint8)

        self.mnist_model = mnist_model
        self.emnist_model = emnist_model

        self.text_label = tk.StringVar()
        self.label_text = tk.Label(self.master, textvariable=self.text_label, font=("Helvetica", 16))
        self.label_text.pack()

        self.clear_button = tk.Button(self.master, text="Clear", command=self.clear_whiteboard)
        self.clear_button.pack()

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=2)

        radius = 2
        self.update_whiteboard_data(event.x, event.y, radius)

        # Predict the digit whenever the whiteboard is updated
        symbol_name = self.predict_digit()
        self.text_label.set(f"Predicted symbol: {symbol_name}")

    def update_whiteboard_data(self, x, y, radius):
        x_scaled = int(x * self.canvas_resolution / self.canvas_width)
        y_scaled = int(y * self.canvas_resolution / self.canvas_height)

        x_min, x_max = max(0, x_scaled - radius), min(self.canvas_resolution, x_scaled + radius)
        y_min, y_max = max(0, y_scaled - radius), min(self.canvas_resolution, y_scaled + radius)

        self.whiteboard_data[y_min:y_max, x_min:x_max] = 255

    def reset(self, event):
        # Save the whiteboard content as a NumPy array
        # You can perform any processing or save the data to a file/database here
        print("Whiteboard content saved as a NumPy array.")

    def clear_whiteboard(self):
        self.canvas.delete("all")
        self.whiteboard_data = np.zeros((self.canvas_resolution, self.canvas_resolution), dtype=np.uint8)
        self.text_label.set("Predicted symbol: ")

    def predict_letter(self):
        # Convert the whiteboard data to an image
        img = Image.fromarray(self.whiteboard_data)

        # Preprocess the image for the EMNIST model
        input_image = preprocess_image(img)

        # Make a prediction using the EMNIST model
        with torch.no_grad():
            output = self.emnist_model(input_image)
            prediction = torch.argmax(output, dim=1).item()

        # Map the letter prediction to a symbol name (you can customize this part)
        symbol_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                        'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        symbol_name = symbol_names[prediction]

        return symbol_name

    def predict_digit(self):
        # Convert the whiteboard data to an image
        img = Image.fromarray(self.whiteboard_data)

        # Preprocess the image for the MNIST model
        input_image = preprocess_image(img)

        # Make a prediction using the MNIST model
        with torch.no_grad():
            output = self.mnist_model(input_image)
            digit_prediction = torch.argmax(output, dim=1).item()

        symbol_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
        symbol_name = symbol_names[digit_prediction]

        return symbol_name


if __name__ == "__main__":
    # Create an instance of the SimpleCNN model
    mnist_model = OptimizedCNN(10)
    emnist_model = OptimizedCNN(27)

    # Load the MNIST and EMNIST models
    mnist_model.load_state_dict(torch.load('mnist_model.pth'))
    emnist_model.load_state_dict(torch.load('emnist_model.pth'))

    # Set the model to evaluation mode
    mnist_model.eval()
    emnist_model.eval()

    root = tk.Tk()
    app = WhiteboardApp(root, mnist_model, emnist_model)
    root.mainloop()
