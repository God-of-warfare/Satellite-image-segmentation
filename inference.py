
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2

from model import UNet
from PIL import Image
import os
from data import LandCoverDataset
from main import LandCoverModel as MyModel


class ModelInference:
    def __init__(self, checkpoint_path, device=None):
        """
        Initialize the inference class
        Args:
            checkpoint_path (str): Path to the model checkpoint
            device (str, optional): Device to run inference on ('cuda', 'cpu').
                                  If None, automatically selects available device.
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model from checkpoint
        self.model = MyModel.load_from_checkpoint(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, input_data):
        """
        Preprocess input data before inference
        Args:
            input_data: Input data (numpy array, list, tensor, etc.)
        Returns:
            torch.Tensor: Processed input tensor
        """

        self.image_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        ])
        # Add batch dimension if needed

        image = self.image_transforms(input_data)
        image = image.unsqueeze(0)
        return image.float().to(self.device)

    def postprocess(self, output, input_tensor):
        """
        Postprocess model output for segmentation.

        Args:
            output (torch.Tensor): Model output tensor with shape (1, num_classes, H, W)
            input_tensor
        Returns:
            colored_mask (numpy.ndarray): Color-coded segmentation mask (H, W, 3)
        """
        # Get the predicted labels (shape: H, W) by taking the argmax across the class dimension
        _, predicted_labels = torch.max(output, dim=1)
        predicted_labels = predicted_labels.squeeze(0)  # Remove the batch dimension if it’s 1
        print(torch.unique(predicted_labels))  # Print unique labels for verification

        # Define label-to-color mapping
        label_map_colors = {
            0: [0, 255, 255],
            1: [255, 255, 0],
            2: [255, 0, 255],
            3: [0, 255, 0],
            4: [0, 0, 255],
            5: [255, 255, 255],
            6: [0, 0, 0]
        }

        # Create an empty color mask
        H, W = predicted_labels.shape
        colored_mask = np.zeros((H, W, 3), dtype=np.uint8)

        # Apply color mapping based on label
        for label, color in label_map_colors.items():
            colored_mask[predicted_labels.cpu().numpy() == label] = color

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        input_image = input_tensor.squeeze(0).cpu()
        input_image = input_image * std + mean  # Denormalize
        input_image = input_image.permute(1, 2, 0).numpy()
        input_image = (input_image * 255).clip(0, 255).astype(np.uint8)

        alpha = 0.4  # Opacity of the mask
        overlay_img = cv2.addWeighted(input_image, 1, colored_mask, alpha, 0)

        plt.figure(figsize=(30, 8))

        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(input_image)
        plt.title('Original Image', fontsize=16)
        plt.axis('off')

        # Segmentation mask
        plt.subplot(1, 3, 2)
        plt.imshow(colored_mask)
        plt.title('Segmentation Mask', fontsize=16)
        plt.axis('off')

        # Overlay
        plt.subplot(1, 3, 3)
        plt.imshow(overlay_img)
        plt.title('Overlay', fontsize=16)
        plt.axis('off')

        plt.tight_layout()  # Adjust spacing between subplots
        plt.show()

        return colored_mask

    def predict(self, input_data, return_tensor=False):
        """
        Make predictions on input data
        Args:
            input_data: Input data to make predictions on
            return_tensor (bool): If True, return torch.Tensor instead of numpy array
        Returns:
            Model predictions
        """
        # Preprocess input
        input_tensor = self.preprocess(input_data)

        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)

        # Return results
        if return_tensor:
            return output
        return self.postprocess(output, input_tensor)

    def predict_batch(self, input_data, batch_size=32, return_tensor=False):
        """
        Make predictions on a large batch of data
        Args:
            input_data: Input data to make predictions on
            batch_size (int): Size of batches to process
            return_tensor (bool): If True, return torch.Tensor instead of numpy array
        Returns:
            Model predictions
        """
        input_tensor = self.preprocess(input_data)
        predictions = []

        # Process in batches
        for i in range(0, len(input_tensor), batch_size):
            batch = input_tensor[i:i + batch_size]
            with torch.no_grad():
                batch_output = self.model(batch)
            predictions.append(batch_output)

        # Concatenate results
        predictions = torch.cat(predictions, dim=0)

        if return_tensor:
            return predictions
        return self.postprocess(predictions)


# Example usage
if __name__ == "__main__":
    # Initialize inference class
    checkpoint_path = ""
    inferencer = ModelInference(checkpoint_path)

    image_path = ""
    # Example: Single prediction
    sample_input = Image.open(image_path)
    # sample_input = np.array(sample_input)
    prediction = inferencer.predict(sample_input)


    # Example: Batch prediction
    # batch_input = np.random.randn(100, input_size)  # Replace with your batch input
    # batch_predictions = inferencer.predict_batch(batch_input, batch_size=32)
    # print(f"Batch predictions shape: {batch_predictions.shape}")
