import os
import time

import clip
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchvision.transforms as transforms
from PIL import Image


class StaffDetector:
    """Simple staff detection using CLIP"""

    def __init__(self, model_name="RN50", device="cpu", threshold=0.6):
        self.threshold = threshold
        self.uniform_features = None
        self.uniform_name = None

        print(f"Loading CLIP model: {model_name}")
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.device = device

        # CPU optimization settings
        torch.set_num_threads(4)
        torch.set_grad_enabled(False)

        # Warm up the model
        dummy_image = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            _ = self.model.encode_image(dummy_image)

    def register_uniform(self, image_path, name="staff_uniform"):
        """Register uniform image"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        print(f"Registering uniform: {image_path}")
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model.encode_image(image_tensor)
            self.uniform_features = features / features.norm(dim=-1, keepdim=True)

        self.uniform_name = name
        print(f"Uniform registered: {name}")

    def detect_staff(self, image_path):
        """Detect if person is staff"""
        if self.uniform_features is None:
            raise ValueError("No uniform registered")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Extract features
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            person_features = self.model.encode_image(image_tensor)
            person_features = person_features / person_features.norm(dim=-1, keepdim=True)

        # Calculate similarity
        similarity = torch.cosine_similarity(person_features, self.uniform_features).item()

        return {"is_staff": similarity > self.threshold, "similarity": similarity, "threshold": self.threshold}


class ONNXStaffDetector:
    """ONNX-optimized staff detector"""

    def __init__(self, onnx_path="clip_image_encoder.onnx", threshold=0.6):
        self.threshold = threshold
        self.uniform_features = None
        self.uniform_name = None

        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        print("Loading ONNX model...")

        # Optimized session options
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=["CPUExecutionProvider"])

        self.preprocess = self._create_preprocess()
        self.input_name = self.session.get_inputs()[0].name

        # Warm up the model
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        _ = self.session.run(None, {self.input_name: dummy_input})
        print("ONNX model loaded and warmed up")

    def _create_preprocess(self):
        """Create preprocessing pipeline"""
        return transforms.Compose(
            [
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
                ),
            ]
        )

    def register_uniform(self, image_path, name="staff_uniform"):
        """Register uniform image"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        print(f"Registering uniform: {image_path}")
        self.uniform_features = self._extract_features(image_path)
        self.uniform_name = name
        print(f"Uniform registered: {name}")

    def detect_staff(self, image_path):
        """Detect if person is staff"""
        if self.uniform_features is None:
            raise ValueError("No uniform registered")

        person_features = self._extract_features(image_path)
        similarity = np.dot(person_features, self.uniform_features.T).item()

        return {"is_staff": similarity > self.threshold, "similarity": similarity, "threshold": self.threshold}

    def _extract_features(self, image_path):
        """Extract features using ONNX"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        image_array = self.preprocess(image).unsqueeze(0).numpy().astype(np.float32)

        # ONNX inference
        features = self.session.run(None, {self.input_name: image_array})[0]

        # L2 normalize
        norm = np.linalg.norm(features, axis=1, keepdims=True)
        return features / (norm + 1e-8)


def convert_to_onnx(model_name="RN50", output_path="clip_image_encoder.onnx"):
    """Convert CLIP model to ONNX format"""
    print(f"Converting {model_name} to ONNX...")

    try:
        model, _ = clip.load(model_name, device="cpu", jit=False)
        model.visual.eval()

        dummy_input = torch.randn(1, 3, 224, 224)

        torch.onnx.export(
            model.visual,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=17,
            input_names=["image"],
            output_names=["features"],
            dynamic_axes={"image": {0: "batch_size"}, "features": {0: "batch_size"}},
            do_constant_folding=True,
        )

        onnx.checker.check_model(onnx.load(output_path))
        print(f"Conversion successful: {output_path}")
        return True

    except Exception as e:
        print(f"Conversion failed: {e}")
        return False


def benchmark(uniform_path="uniform.jpg", iterations=10):
    """Benchmark PyTorch vs ONNX performance"""
    if not os.path.exists(uniform_path):
        print(f"Test image not found: {uniform_path}")
        return

    print("=== Performance Benchmark ===")

    # Test PyTorch
    print("Testing PyTorch...")
    pytorch_detector = StaffDetector()
    pytorch_detector.register_uniform(uniform_path)

    # Warmup runs (not counted)
    for _ in range(3):
        _ = pytorch_detector.detect_staff(uniform_path)

    pytorch_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        pytorch_result = pytorch_detector.detect_staff(uniform_path)
        pytorch_times.append(time.perf_counter() - start)

    # Test ONNX
    print("Testing ONNX...")
    onnx_path = "clip_image_encoder.onnx"
    if not os.path.exists(onnx_path):
        print("Converting to ONNX...")
        if not convert_to_onnx():
            return

    onnx_detector = ONNXStaffDetector()
    onnx_detector.register_uniform(uniform_path)

    # Warmup runs (not counted)
    for _ in range(3):
        _ = onnx_detector.detect_staff(uniform_path)

    onnx_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        onnx_result = onnx_detector.detect_staff(uniform_path)
        onnx_times.append(time.perf_counter() - start)

    # Results
    pytorch_avg = np.mean(pytorch_times[2:]) * 1000  # Skip first 2 runs
    onnx_avg = np.mean(onnx_times[2:]) * 1000
    speedup = pytorch_avg / onnx_avg
    accuracy_diff = abs(pytorch_result["similarity"] - onnx_result["similarity"])

    print(f"\nResults (average of {iterations-2} runs after warmup):")
    print(f"PyTorch: {pytorch_avg:.2f}ms")
    print(f"ONNX:    {onnx_avg:.2f}ms")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Accuracy diff: {accuracy_diff:.6f}")

    # Detailed timing analysis
    print(f"\nDetailed timing:")
    print(f"PyTorch - Min: {min(pytorch_times[2:])*1000:.2f}ms, Max: {max(pytorch_times[2:])*1000:.2f}ms")
    print(f"ONNX    - Min: {min(onnx_times[2:])*1000:.2f}ms, Max: {max(onnx_times[2:])*1000:.2f}ms")


def demo():
    """Interactive demo"""
    print("=== Staff Detection Demo ===")

    # Create detector
    detector = StaffDetector()

    # Register uniform
    uniform_path = input("Uniform image path: ").strip()
    if not uniform_path or not os.path.exists(uniform_path):
        print("Uniform image not found")
        return

    detector.register_uniform(uniform_path)

    # Test detection
    person_path = input("Person image path: ").strip()
    if not person_path or not os.path.exists(person_path):
        print("Person image not found")
        return

    result = detector.detect_staff(person_path)
    status = "Staff" if result["is_staff"] else "Not Staff"
    print(f"\nResult: {status}")
    print(f"Similarity: {result['similarity']:.4f}")


def main():
    """Main menu"""
    while True:
        print("\n=== CLIP Staff Detection ===")
        print("1. Convert to ONNX")
        print("2. Run demo")
        print("3. Run benchmark")
        print("4. Exit")

        choice = input("\nSelect: ").strip()

        if choice == "1":
            convert_to_onnx()
        elif choice == "2":
            demo()
        elif choice == "3":
            benchmark()
        elif choice == "4":
            break
        else:
            print("Invalid option")


if __name__ == "__main__":
    main()
