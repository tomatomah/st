import os

import numpy as np
import onnxruntime as ort
from PIL import Image


def preprocess_numpy(image, input_size):
    """Preprocess image using numpy"""
    # 1. Resize (bicubic interpolation)
    image = image.resize((input_size, input_size), Image.BICUBIC)

    # 2. Convert to numpy array and normalize to [0, 1]
    image_array = np.array(image).astype(np.float32) / 255.0

    # 3. Convert from HWC to CHW format
    image_array = np.transpose(image_array, (2, 0, 1))

    # 4. Normalize using ImageNet statistics
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32).reshape(3, 1, 1)
    image_array = (image_array - mean) / std

    # 5. Add batch dimension
    return np.expand_dims(image_array, axis=0).astype(np.float32)


class SimpleStaffDetector:
    """Simple ONNX-based staff detector"""

    def __init__(self, onnx_model_path, input_size=224):
        """
        Initialize detector

        Args:
            onnx_model_path: Path to ONNX model file
            input_size: Input image size (default: 224)
        """
        if not os.path.exists(onnx_model_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")

        self.input_size = input_size
        self.session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

        print(f"Loaded ONNX model: {onnx_model_path}")
        print(f"Input size: {input_size}x{input_size}")

    def extract_features(self, image_path):
        """Extract features from image"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        image_array = preprocess_numpy(image, self.input_size)

        # ONNX inference
        features = self.session.run(None, {self.input_name: image_array})[0]

        # L2 normalize
        return features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)

    def calculate_similarity(self, uniform_path, person_path):
        """Calculate similarity between uniform and person images"""
        uniform_features = self.extract_features(uniform_path)
        person_features = self.extract_features(person_path)

        similarity = np.dot(uniform_features, person_features.T).item()
        return similarity


def main():
    """Main function"""
    # ===== CONFIGURATION =====
    # Specify your file paths here
    ONNX_MODEL_PATH = "onnx_models/clip_RN50.onnx"  # Path to ONNX model
    UNIFORM_IMAGE_PATH = "uniform.jpg"  # Path to uniform reference image
    PERSON_IMAGE_PATH = "staff1.jpg"  # Path to person image to check
    INPUT_SIZE = 224  # Input image size (224 for most models)
    THRESHOLD = 0.6  # Similarity threshold for staff detection

    try:
        # Initialize detector
        detector = SimpleStaffDetector(ONNX_MODEL_PATH, INPUT_SIZE)

        # Calculate similarity
        print("\nCalculating similarity...")
        similarity = detector.calculate_similarity(UNIFORM_IMAGE_PATH, PERSON_IMAGE_PATH)

        # Results
        is_staff = similarity > THRESHOLD
        status = "✓ STAFF" if is_staff else "✗ NOT STAFF"

        print(f"\nResults:")
        print(f"Uniform image: {UNIFORM_IMAGE_PATH}")
        print(f"Person image:  {PERSON_IMAGE_PATH}")
        print(f"Similarity:    {similarity:.6f}")
        print(f"Threshold:     {THRESHOLD}")
        print(f"Decision:      {status}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
