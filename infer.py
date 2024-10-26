import os
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
import torchvision.transforms as transforms

from models import get_model
from utils import convert_state_dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DewarpNetPredictor:
    def __init__(
        self,
        wc_model_path: str = './eval/models/unetnc_doc3d.pkl',
        bm_model_path: str = './eval/models/dnetccnl_doc3d.pkl',
        device: Optional[torch.device] = None,
    ):
        """
        Initializes the DewarpNet predictor by loading the models.

        Args:
            wc_model_path (str, optional): Path to the wireframe correction (wc) model.
                                           Defaults to './eval/models/unetnc_doc3d.pkl'.
            bm_model_path (str, optional): Path to the basis matrix (bm) model.
                                           Defaults to './eval/models/dnetccnl_doc3d.pkl'.
            device (torch.device, optional): Device to run the models on. Defaults to GPU if available.
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.wc_model = self._load_model(wc_model_path, model_type='wc')
        self.bm_model = self._load_model(bm_model_path, model_type='bm')
        self.htan = nn.Hardtanh(0, 1.0)
        self.wc_img_size = (256, 256)
        self.bm_img_size = (128, 128)
        self.transform = transforms.Compose([
            transforms.Resize(self.wc_img_size),
            transforms.ToTensor(),
        ])
        logger.info("Models loaded and predictor initialized.")

    def _load_model(self, model_path: str, model_type: str) -> nn.Module:
        """
        Loads a model from the given path.

        Args:
            model_path (str): Path to the model file.
            model_type (str): Type of the model ('wc' or 'bm').

        Returns:
            nn.Module: The loaded model.
        """
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model_file_name = os.path.basename(model_path)
        model_name = model_file_name.split('_')[0]

        n_classes = 3 if model_type == 'wc' else 2
        model = get_model(model_name, n_classes, in_channels=3)

        try:
            if self.device.type == 'cpu':
                state = convert_state_dict(
                    torch.load(model_path, map_location='cpu')['model_state']
                )
            else:
                state = convert_state_dict(
                    torch.load(model_path)['model_state']
                )
            model.load_state_dict(state)
            model.to(self.device)
            model.eval()
            logger.info(f"{model_type.upper()} model loaded successfully.")
            return model
        except Exception as e:
            logger.exception(f"Failed to load {model_type.upper()} model.")
            raise e

    def predict(self, image: Image.Image) -> np.ndarray:
        """
        Performs dewarping prediction on the given image.

        Args:
            image (PIL.Image.Image): The input image.

        Returns:
            np.ndarray: The dewarped image as a NumPy array.
        """
        try:
            # Preprocess the image
            img = self.transform(image).unsqueeze(0).to(self.device)

            # Wireframe Correction Model Prediction
            with torch.no_grad():
                wc_output = self.wc_model(img)
                pred_wc = self.htan(wc_output)

            # Prepare input for Basis Matrix Model
            bm_input = F.interpolate(pred_wc, size=self.bm_img_size, mode='bilinear', align_corners=False)

            # Basis Matrix Model Prediction
            with torch.no_grad():
                bm_output = self.bm_model(bm_input)

            # Unwarp the image
            dewarped_image = self._unwarp(image, bm_output)
            logger.info("Prediction completed successfully.")
            return dewarped_image
        except Exception as e:
            logger.exception("Prediction failed.")
            raise e

    def _unwarp(self, img: Image.Image, bm: torch.Tensor) -> np.ndarray:
        """
        Applies the unwarping transformation to the image using the predicted basis matrix.

        Args:
            img (PIL.Image.Image): The original image.
            bm (torch.Tensor): The predicted basis matrix from the model.

        Returns:
            np.ndarray: The dewarped image as a NumPy array.
        """
        # Convert image to NumPy array
        img_np = np.array(img).astype(np.float32) / 255.0
        h, w = img_np.shape[:2]

        # Process basis matrix
        bm_np = bm.permute(0, 2, 3, 1).squeeze(0).detach().cpu().numpy()
        bm0 = gaussian_filter(bm_np[:, :, 0], sigma=1)
        bm1 = gaussian_filter(bm_np[:, :, 1], sigma=1)

        # Resize basis matrices to match image dimensions
        bm0_resized = np.array(Image.fromarray(bm0).resize((w, h), Image.BILINEAR))
        bm1_resized = np.array(Image.fromarray(bm1).resize((w, h), Image.BILINEAR))

        # Stack and prepare grid for sampling
        bm_resized = np.stack([bm0_resized, bm1_resized], axis=-1)
        bm_tensor = torch.from_numpy(bm_resized).unsqueeze(0).double().to(self.device)

        # Prepare image tensor
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).double().to(self.device)

        # Grid sampling
        with torch.no_grad():
            resampled = F.grid_sample(
                img_tensor,
                bm_tensor,
                mode='bilinear',
                padding_mode='border',
                align_corners=False
            )

        # Convert back to NumPy array
        dewarped_img = resampled.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        dewarped_img = (dewarped_img * 255).astype(np.uint8)
        return dewarped_img

# Example usage:
if __name__ == '__main__':
    # Initialize the predictor with default model paths
    predictor = DewarpNetPredictor()

    # Load the input image (replace 'path/to/input/image.jpg' with your image path)
    input_image_path = 'eval/inp/4_2.png'
    input_image = Image.open(input_image_path).convert('RGB')

    # Perform prediction
    dewarped_image = predictor.predict(input_image)

    # Save the output image (replace 'path/to/output/image.jpg' with your desired output path)
    output_image_path = '4_2.png'
    Image.fromarray(dewarped_image).save(output_image_path)
    logger.info(f"Dewarped image saved to: {output_image_path}")
