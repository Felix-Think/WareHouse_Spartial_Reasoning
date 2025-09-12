import torch
from PIL import Image
import numpy as np
import sys
import os

from training.scripts.ResUnet.ResUnet34 import ResUNet34

def load_model(file_name):
    checkpoint = torch.load(file_name, map_location='cpu')
    
    # Get model configuration from checkpoint
    config = checkpoint.get('config', {})
    model_config = config.get('model', {})
    in_channels = model_config.get('in_channels', 3)
    num_classes = model_config.get('num_classes', 4)
    
    # Create model instance
    model = ResUNet34(in_channels=in_channels, num_classes=num_classes)
    
    # Load the state dict into the model
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to evaluation mode
    model.eval()
    
    optimizer_state = checkpoint['optimizer_state_dict']
    return model, optimizer_state

def visualize(output, save_path):
    # Apply softmax to get probabilities and take argmax to get class predictions
    output = torch.softmax(output, dim=1)  # Convert logits to probabilities
    output = torch.argmax(output, dim=1)   # Get class predictions
    output = output.squeeze(0).detach().numpy()  # Remove batch dimension
    
    # Convert class indices to RGB image for better visualization
    # Create a colormap for the 4 classes
    colormap = np.array([
        [0, 0, 0],       # Class 0: Black
        [255, 0, 0],     # Class 1: Red  
        [0, 255, 0],     # Class 2: Green
        [0, 0, 255],     # Class 3: Blue
    ], dtype=np.uint8)
    
    # Map class indices to colors
    colored_output = colormap[output]
    
    # Convert to PIL Image and save
    img = Image.fromarray(colored_output)
    img.save(save_path)
if __name__ == "__main__":
    model, optimizer = load_model("models/checkpoint/best_resunet34_epoch_39.pth")
    input_image = Image.open('Datasets/test/images/000000.png')
    input_array = np.array(input_image)
    input_tensor = torch.tensor(input_array).unsqueeze(0).permute(0, 3, 1, 2).float() / 255.0
    
    with torch.no_grad():  # Disable gradient computation for inference
        output = model(input_tensor)
    
    visualize(output, 'output.png')  # Save as PNG image
    print("Model inference completed successfully!")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print("Segmentation result saved as 'output.png'")
     