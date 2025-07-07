import os

import clip
import numpy as np
import onnx
import onnxruntime as ort
import torch


def get_input_size(model_name):
    """Get appropriate input size for each model"""
    if "@336px" in model_name or "336px" in model_name:
        return 336
    elif "RN50x" in model_name:
        # These models use different resolutions
        if "RN50x4" in model_name:
            return 288
        elif "RN50x16" in model_name:
            return 384
        elif "RN50x64" in model_name:
            return 448
    return 224


def convert_clip_to_onnx(model_name, output_dir="onnx_models"):
    """Convert a single CLIP model to ONNX format"""
    try:
        print(f"Converting {model_name} to ONNX...")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Load CLIP model and preprocess
        model, preprocess = clip.load(model_name, device="cpu", jit=False)
        model.visual.eval()

        # Get appropriate input size
        input_size = get_input_size(model_name)
        print(f"Using input size: {input_size}x{input_size}")

        # Create dummy input with appropriate size
        dummy_input = torch.randn(1, 3, input_size, input_size)

        # Generate output filename
        safe_name = model_name.replace("/", "_").replace("@", "_")
        output_path = os.path.join(output_dir, f"clip_{safe_name}.onnx")

        # Export to ONNX
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

        # Verify the model
        onnx.checker.check_model(onnx.load(output_path))

        print(f"✓ Successfully converted: {output_path}")
        return True

    except Exception as e:
        print(f"✗ Failed to convert {model_name}: {e}")
        return False


def convert_all_clip_models():
    """Convert all available CLIP models to ONNX"""
    # Available CLIP models
    models = ["RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"]

    print("=== CLIP to ONNX Conversion ===")
    print(f"Converting {len(models)} models...")

    success_count = 0
    failed_models = []

    for model_name in models:
        if convert_clip_to_onnx(model_name):
            success_count += 1
        else:
            failed_models.append(model_name)

    print("\n=== Conversion Summary ===")
    print(f"Successfully converted: {success_count}/{len(models)} models")

    if failed_models:
        print(f"Failed models: {', '.join(failed_models)}")

    print("All conversions completed!")


def convert_specific_model():
    """Convert a specific CLIP model selected by user"""
    # Available CLIP models
    models = ["RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"]

    print("\n=== Convert Specific Model ===")
    print("Available CLIP models:")
    for i, model_name in enumerate(models, 1):
        print(f"{i}. {model_name}")

    try:
        choice = int(input(f"\nSelect model number (1-{len(models)}): ")) - 1
        if choice < 0 or choice >= len(models):
            print("Invalid selection")
            return

        selected_model = models[choice]
        print(f"Selected: {selected_model}")
        convert_clip_to_onnx(selected_model)

    except ValueError:
        print("Invalid input. Please enter a number.")
    except KeyboardInterrupt:
        print("\nCancelled by user.")


def test_onnx_model():
    """Test all ONNX models feature extraction with random input"""
    print("\n=== ONNX Model Test ===")

    # List available ONNX models
    onnx_dir = "onnx_models"
    if not os.path.exists(onnx_dir):
        print("No ONNX models found. Please convert models first.")
        return

    onnx_files = [f for f in os.listdir(onnx_dir) if f.endswith(".onnx")]
    if not onnx_files:
        print("No ONNX models found in onnx_models directory.")
        return

    print(f"Found {len(onnx_files)} ONNX models to test:")
    for filename in onnx_files:
        print(f"  - {filename}")
    print()

    success_count = 0
    failed_models = []

    # Test each model
    for i, model_file in enumerate(onnx_files, 1):
        print(f"[{i}/{len(onnx_files)}] Testing: {model_file}")
        model_path = os.path.join(onnx_dir, model_file)

        try:
            # Determine input size from model name
            input_size = get_input_size(model_file)

            print(f"  Input size: {input_size}x{input_size}")

            # Load ONNX model
            session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            input_name = session.get_inputs()[0].name

            # Create random input tensor
            random_input = np.random.randn(1, 3, input_size, input_size).astype(np.float32)

            # Extract features
            features = session.run(None, {input_name: random_input})[0]

            # Normalize features (L2 normalization)
            features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)

            print(f"  ✓ Success - Feature shape: {features.shape}, Norm: {np.linalg.norm(features_norm):.6f}")
            success_count += 1

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed_models.append(model_file)

        print()  # Add blank line between tests

    # Summary
    print("=== Test Summary ===")
    print(f"Successfully tested: {success_count}/{len(onnx_files)} models")

    if failed_models:
        print(f"Failed models: {', '.join(failed_models)}")
    else:
        print("All models passed the test!")

    return success_count == len(onnx_files)


def main():
    """Main function with menu"""
    while True:
        print("\n=== CLIP to ONNX Converter ===")
        print("1. Convert all models")
        print("2. Convert specific model")
        print("3. Test ONNX model")
        print("4. Exit")

        choice = input("\nSelect option: ").strip()

        if choice == "1":
            convert_all_clip_models()

        elif choice == "2":
            convert_specific_model()

        elif choice == "3":
            test_onnx_model()

        elif choice == "4":
            print("Exiting...")
            break

        else:
            print("Invalid option")


if __name__ == "__main__":
    main()
