import os
import time

import clip
import numpy as np
import onnxruntime as ort
import torch

# Available CLIP models in order
CLIP_MODELS = ["RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"]

# Model input sizes
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


def benchmark_model(model_name):
    """Compare ONNX and PyTorch model performance for a single model"""
    onnx_dir = "onnx_models"
    onnx_filename = get_onnx_filename(model_name)
    onnx_path = os.path.join(onnx_dir, onnx_filename)

    if not os.path.exists(onnx_path):
        print(f"ONNX model not found: {onnx_path}")
        return None

    print(f"Benchmarking: {model_name}")
    input_size = MODEL_INPUT_SIZES[model_name]
    print(f"Input size: {input_size}x{input_size}")

    try:
        # Load PyTorch model
        torch_model, _ = clip.load(model_name, device="cpu", jit=False)
        torch_model.visual.eval()
        torch.set_grad_enabled(False)

        # Load ONNX model
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name

        # Create test input
        test_input = np.random.randn(1, 3, input_size, input_size).astype(np.float32)
        torch_input = torch.from_numpy(test_input)

        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = torch_model.encode_image(torch_input)
            _ = session.run(None, {input_name: test_input})

        # Benchmark PyTorch (5 iterations)
        torch_times = []
        for _ in range(5):
            start_time = time.perf_counter()
            with torch.no_grad():
                torch_features = torch_model.encode_image(torch_input)
            torch_times.append(time.perf_counter() - start_time)

        # Benchmark ONNX (5 iterations)
        onnx_times = []
        for _ in range(5):
            start_time = time.perf_counter()
            onnx_features = session.run(None, {input_name: test_input})[0]
            onnx_times.append(time.perf_counter() - start_time)

        # Normalize features for comparison
        torch_features_norm = torch_features / torch_features.norm(dim=-1, keepdim=True)
        onnx_features_norm = onnx_features / (np.linalg.norm(onnx_features, axis=1, keepdims=True) + 1e-8)

        # Calculate metrics
        torch_avg = np.mean(torch_times) * 1000  # ms
        onnx_avg = np.mean(onnx_times) * 1000  # ms
        speedup = torch_avg / onnx_avg
        file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB

        # Feature comparison
        feature_diff = np.abs(torch_features_norm.numpy() - onnx_features_norm)
        max_diff = np.max(feature_diff)

        # Display results
        print(f"PyTorch:  {torch_avg:.2f}ms")
        print(f"ONNX:     {onnx_avg:.2f}ms")
        print(f"Speedup:  {speedup:.2f}x")
        print(f"File size: {file_size:.1f}MB")
        print(f"Max diff: {max_diff:.6f}")

        return {
            "model_name": model_name,
            "onnx_time": onnx_avg,
            "speedup": speedup,
            "file_size": file_size,
            "max_diff": max_diff,
        }

    except Exception as e:
        print(f"Benchmark failed: {e}")
        return None


def benchmark_all():
    """Benchmark all available models"""
    onnx_dir = "onnx_models"
    if not os.path.exists(onnx_dir):
        print("No ONNX models found. Please convert models first.")
        return

    available_models = []
    for model_name in CLIP_MODELS:
        onnx_filename = get_onnx_filename(model_name)
        if os.path.exists(os.path.join(onnx_dir, onnx_filename)):
            available_models.append(model_name)

    if not available_models:
        print("No matching ONNX models found.")
        return

    print(f"Benchmarking {len(available_models)} models...")
    results = []

    for i, model_name in enumerate(available_models, 1):
        print(f"\n[{i}/{len(available_models)}] " + "=" * 50)
        result = benchmark_model(model_name)
        if result:
            results.append(result)

    # Show recommendations
    if results:
        print("\n" + "=" * 60)
        print("=== RECOMMENDATIONS ===")

        smallest = min(results, key=lambda x: x["file_size"])
        fastest = min(results, key=lambda x: x["onnx_time"])

        print(f"\n🏆 Smallest: {smallest['model_name']} ({smallest['file_size']:.1f}MB)")
        print(f"🚀 Fastest:  {fastest['model_name']} ({fastest['onnx_time']:.2f}ms)")

        print(f"\n{'Model':<15} {'Size(MB)':<9} {'Speed(ms)':<11} {'Speedup':<8}")
        print("-" * 43)
        for r in sorted(results, key=lambda x: x["onnx_time"]):
            print(f"{r['model_name']:<15} {r['file_size']:<9.1f} {r['onnx_time']:<11.2f} {r['speedup']:<8.2f}")


def benchmark_specific():
    """Benchmark a specific model"""
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
            print()
            benchmark_model(available_models[choice])
        else:
            print("Invalid selection")
    except ValueError:
        print("Invalid input")


def main():
    """Main function"""
    while True:
        print("\n=== ONNX vs PyTorch Benchmark ===")
        print("1. Benchmark all models")
        print("2. Benchmark specific model")
        print("3. Exit")

        choice = input("\nSelect: ").strip()

        if choice == "1":
            benchmark_all()
        elif choice == "2":
            benchmark_specific()
        elif choice == "3":
            break
        else:
            print("Invalid option")


if __name__ == "__main__":
    main()
