import os

import clip
import numpy as np
import onnxruntime as ort
import torch
import torchvision.transforms as transforms
from PIL import Image

# Available CLIP models and their input sizes
CLIP_MODELS = ["RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"]
MODEL_INPUT_SIZES = {
    "RN50": 224,
    "RN101": 224,
    "RN50x4": 288,
    "RN50x16": 384,
    "RN50x64": 448,
    "ViT-B/32": 224,
    "ViT-B/16": 224,
    "ViT-L/14": 224,
    "ViT-L/14@336px": 336,
}


def get_onnx_filename(model_name):
    """Convert model name to ONNX filename"""
    safe_name = model_name.replace("/", "_").replace("@", "_")
    return f"clip_{safe_name}.onnx"


def preprocess_numpy(image, input_size):
    """Preprocess image using numpy"""
    # 1. Resize (bicubic interpolation)
    image = image.resize((input_size, input_size), Image.BICUBIC)

    # 2. Center crop (already resized to target size, so no cropping needed)

    # 3. Convert to numpy array and normalize to [0, 1]
    image_array = np.array(image).astype(np.float32) / 255.0

    # 4. Convert from HWC to CHW format
    image_array = np.transpose(image_array, (2, 0, 1))

    # 5. Normalize using ImageNet statistics (ensure float32)
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32).reshape(3, 1, 1)
    image_array = (image_array - mean) / std

    # 6. Add batch dimension and ensure float32
    return np.expand_dims(image_array, axis=0).astype(np.float32)


class PyTorchStaffDetector:
    """PyTorch CLIP staff detector"""

    def __init__(self, model_name):
        self.model_name = model_name
        self.input_size = MODEL_INPUT_SIZES[model_name]

        # Load model
        self.model, _ = clip.load(model_name, device="cpu", jit=False)
        self.model.visual.eval()
        torch.set_grad_enabled(False)

    def extract_features(self, image_path):
        """Extract features from image"""
        image = Image.open(image_path).convert("RGB")
        image_array = preprocess_numpy(image, self.input_size)

        # Convert numpy array to torch tensor
        image_tensor = torch.from_numpy(image_array)

        with torch.no_grad():
            features = self.model.encode_image(image_tensor)
            return features / features.norm(dim=-1, keepdim=True)

    def calculate_similarity(self, uniform_path, person_path):
        """Calculate similarity between uniform and person images"""
        uniform_features = self.extract_features(uniform_path)
        person_features = self.extract_features(person_path)

        similarity = torch.cosine_similarity(uniform_features, person_features).item()
        return similarity


class ONNXStaffDetector:
    """ONNX CLIP staff detector"""

    def __init__(self, model_name):
        self.model_name = model_name
        self.input_size = MODEL_INPUT_SIZES[model_name]

        # Load ONNX model
        onnx_filename = get_onnx_filename(model_name)
        onnx_path = os.path.join("onnx_models", onnx_filename)

        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def extract_features(self, image_path):
        """Extract features from image"""
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


def compare_models(model_name, uniform_path, person_path):
    """Compare PyTorch and ONNX model similarity scores"""
    print(f"Comparing {model_name}...")

    try:
        # PyTorch
        pytorch_detector = PyTorchStaffDetector(model_name)
        pytorch_similarity = pytorch_detector.calculate_similarity(uniform_path, person_path)

        # ONNX
        onnx_detector = ONNXStaffDetector(model_name)
        onnx_similarity = onnx_detector.calculate_similarity(uniform_path, person_path)

        # Comparison
        diff = abs(pytorch_similarity - onnx_similarity)

        print(f"PyTorch:  {pytorch_similarity:.6f}")
        print(f"ONNX:     {onnx_similarity:.6f}")
        print(f"Diff:     {diff:.6f}")

        return {
            "model_name": model_name,
            "pytorch_similarity": pytorch_similarity,
            "onnx_similarity": onnx_similarity,
            "diff": diff,
        }

    except Exception as e:
        print(f"Error with {model_name}: {e}")
        return None


def compare_all_models(uniform_path, person_path):
    """Compare all available ONNX models"""
    onnx_dir = "onnx_models"
    if not os.path.exists(onnx_dir):
        print("No ONNX models found. Please convert models first.")
        return

    # Find available models
    available_models = []
    for model_name in CLIP_MODELS:
        onnx_filename = get_onnx_filename(model_name)
        if os.path.exists(os.path.join(onnx_dir, onnx_filename)):
            available_models.append(model_name)

    if not available_models:
        print("No matching ONNX models found.")
        return

    print(f"Comparing {len(available_models)} models...")
    print(f"Uniform: {uniform_path}")
    print(f"Person:  {person_path}")
    print()

    results = []
    for i, model_name in enumerate(available_models, 1):
        print(f"[{i}/{len(available_models)}] " + "=" * 40)
        result = compare_models(model_name, uniform_path, person_path)
        if result:
            results.append(result)
        print()

    # Summary
    if results:
        print("=" * 50)
        print("=== SIMILARITY SUMMARY ===")
        print(f"{'Model':<15} {'PyTorch':<10} {'ONNX':<10} {'Diff':<8}")
        print("-" * 43)
        for r in results:
            print(
                f"{r['model_name']:<15} {r['pytorch_similarity']:<10.6f} {r['onnx_similarity']:<10.6f} {r['diff']:<8.6f}"
            )

        # Find highest similarity
        highest = max(results, key=lambda x: x["onnx_similarity"])
        print(f"\n🏆 Highest similarity: {highest['model_name']} ({highest['onnx_similarity']:.6f})")


def compare_specific_model(uniform_path, person_path):
    """Compare specific model"""
    onnx_dir = "onnx_models"
    if not os.path.exists(onnx_dir):
        print("No ONNX models found. Please convert models first.")
        return

    # Find available models
    available_models = []
    for model_name in CLIP_MODELS:
        onnx_filename = get_onnx_filename(model_name)
        if os.path.exists(os.path.join(onnx_dir, onnx_filename)):
            available_models.append(model_name)

    if not available_models:
        print("No matching ONNX models found.")
        return

    print("Available models:")
    for i, model_name in enumerate(available_models, 1):
        print(f"{i}. {model_name}")

    try:
        choice = int(input(f"\nSelect model (1-{len(available_models)}): ")) - 1
        if 0 <= choice < len(available_models):
            selected_model = available_models[choice]
            print()
            compare_models(selected_model, uniform_path, person_path)
        else:
            print("Invalid selection")
    except ValueError:
        print("Invalid input")


def get_image_paths():
    """Get uniform and person image paths from user"""
    uniform_path = input("Enter uniform image path: ").strip()
    if not uniform_path or not os.path.exists(uniform_path):
        print("Uniform image not found")
        return None, None

    person_path = input("Enter person image path: ").strip()
    if not person_path or not os.path.exists(person_path):
        print("Person image not found")
        return None, None

    return uniform_path, person_path


def main():
    """Main function"""
    while True:
        print("\n=== Staff Similarity Detector ===")
        print("1. Compare all models")
        print("2. Compare specific model")
        print("3. Exit")

        choice = input("\nSelect: ").strip()

        if choice == "1":
            uniform_path, person_path = get_image_paths()
            if uniform_path and person_path:
                compare_all_models(uniform_path, person_path)

        elif choice == "2":
            uniform_path, person_path = get_image_paths()
            if uniform_path and person_path:
                compare_specific_model(uniform_path, person_path)

        elif choice == "3":
            break

        else:
            print("Invalid option")


if __name__ == "__main__":
    main()
