import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

class DiffusionModel(nn.Module):
    def __init__(self):
        super(DiffusionModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

# Define the image size
IMAGE_SIZE = (256, 256)

def generate_fingerprint(img_path):
    # Load the saved model
    model_save_path = 'diffusion_model.pth'  # Path where the model is saved
    checkpoint = torch.load(model_save_path, map_location=torch.device('cpu'))

    # Initialize the model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DiffusionModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Load the model state_dict and optimizer state_dict
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Set the model to evaluation mode
    model.eval()

    # Load a partial fingerprint image for testing
    partial_img_path = img_path  # Example partial image
    partial_img = cv2.imread(partial_img_path)
    partial_img = cv2.resize(partial_img, IMAGE_SIZE)
    partial_img = partial_img / 255.0

    # Convert the partial image to PyTorch tensor
    partial_img_tensor = torch.from_numpy(partial_img).permute(2, 0, 1).float().unsqueeze(0).to(device)

    # Use the model to reconstruct the complete fingerprint
    with torch.no_grad():
        reconstructed_img_tensor = model(partial_img_tensor)

    # Convert the reconstructed image tensor to a numpy array
    reconstructed_img = reconstructed_img_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    cv2.imwrite('reconstructed_fingerprint.jpg', reconstructed_img * 255.0)
    # plt.subplot(1, 3, 3)
    # plt.title("Reconstructed")
    # plt.imshow(reconstructed_img)
    # plt.axis("off")

    # # Show the plots
    # plt.show()

#generate_fingerprint("saved_files\6824989d82d0b9664b071601\6824989d82d0b9664b071601_partial.png")
# For visualization, we need the original image (if you have it)
# For now, I'll assume you have the original corresponding image, or you can load it from your dataset
# original_img = cv2.imread("1__M_Left_index_finger.BMP")
# original_img = cv2.resize(original_img, IMAGE_SIZE)
# original_img = original_img / 255.0

# # Visualize the partial, original, and reconstructed images
# plt.figure(figsize=(12, 4))

# # Display the partial image
# plt.subplot(1, 3, 1)
# plt.title("Partial")
# plt.imshow(partial_img)
# plt.axis("off")

# # Display the original image
# plt.subplot(1, 3, 2)
# plt.title("Original")
# plt.imshow(original_img)
# plt.axis("off")

# # Display the reconstructed image
# plt.subplot(1, 3, 3)
# plt.title("Reconstructed")
# plt.imshow(reconstructed_img)
# plt.axis("off")

# # Show the plots
# plt.show()

# # Optionally, save the reconstructed image
# cv2.imwrite('reconstructed_fingerprint12.jpg', reconstructed_img * 255.0)