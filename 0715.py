import glob
import os
import shutil

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

    def process_directory(self, uniform_path, input_dir, output_dir, threshold=0.6):
        """
        Process multiple images in directory and classify as staff/non-staff

        Args:
            uniform_path: Path to uniform reference image
            input_dir: Directory containing images to process
            output_dir: Output directory for classification results
            threshold: Similarity threshold for staff detection

        Returns:
            dict: Processing results with statistics
        """
        # Supported image extensions
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]

        # Get all image files from input directory
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_dir, ext)))
            image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))

        if not image_files:
            raise ValueError(f"No image files found in: {input_dir}")

        print(f"Found {len(image_files)} images to process")

        # Create output directories
        staff_dir = os.path.join(output_dir, "staff")
        non_staff_dir = os.path.join(output_dir, "non_staff")
        os.makedirs(staff_dir, exist_ok=True)
        os.makedirs(non_staff_dir, exist_ok=True)

        # Extract uniform features once
        print("Extracting uniform features...")
        uniform_features = self.extract_features(uniform_path)

        # Process each image
        results = []
        staff_count = 0

        print("Processing images...")
        for i, image_path in enumerate(image_files):
            try:
                # Extract features and calculate similarity
                person_features = self.extract_features(image_path)
                similarity = np.dot(uniform_features, person_features.T).item()

                # Determine classification
                is_staff = similarity > threshold

                # Copy file to appropriate directory
                filename = os.path.basename(image_path)
                if is_staff:
                    dest_path = os.path.join(staff_dir, filename)
                    staff_count += 1
                else:
                    dest_path = os.path.join(non_staff_dir, filename)

                shutil.copy2(image_path, dest_path)

                # Store result
                results.append(
                    {
                        "filename": filename,
                        "similarity": similarity,
                        "is_staff": is_staff,
                        "source_path": image_path,
                        "dest_path": dest_path,
                    }
                )

                # Progress indicator
                print(
                    f"  [{i+1:3d}/{len(image_files)}] {filename}: {similarity:.4f} -> {'STAFF' if is_staff else 'NON-STAFF'}"
                )

            except Exception as e:
                print(f"  Error processing {image_path}: {e}")
                continue

        # Calculate statistics
        total_processed = len(results)
        non_staff_count = total_processed - staff_count
        staff_detection_rate = (staff_count / total_processed) * 100 if total_processed > 0 else 0

        # Summary
        summary = {
            "total_images": len(image_files),
            "processed_images": total_processed,
            "staff_count": staff_count,
            "non_staff_count": non_staff_count,
            "staff_detection_rate": staff_detection_rate,
            "threshold": threshold,
            "results": results,
        }

        return summary


def print_summary(summary):
    """Print processing summary"""
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total images found:     {summary['total_images']}")
    print(f"Successfully processed: {summary['processed_images']}")
    print(f"Staff detected:         {summary['staff_count']}")
    print(f"Non-staff detected:     {summary['non_staff_count']}")
    print(f"Staff detection rate:   {summary['staff_detection_rate']:.2f}%")
    print(f"Threshold used:         {summary['threshold']}")
    print("=" * 60)


def main():
    """Main function"""
    # ===== CONFIGURATION =====
    # Specify your file paths here
    ONNX_MODEL_PATH = "onnx_models/clip_RN50.onnx"  # Path to ONNX model
    UNIFORM_IMAGE_PATH = "uniform.jpg"  # Path to uniform reference image

    # For single image processing
    PERSON_IMAGE_PATH = "staff1.jpg"  # Path to person image to check

    # For batch processing
    INPUT_DIRECTORY = "test_images"  # Directory containing images to process
    OUTPUT_DIRECTORY = "results"  # Output directory for classification

    INPUT_SIZE = 224  # Input image size (224 for most models)
    THRESHOLD = 0.6  # Similarity threshold for staff detection

    # Processing mode
    BATCH_MODE = True  # Set to True for batch processing, False for single image

    try:
        # Initialize detector
        detector = SimpleStaffDetector(ONNX_MODEL_PATH, INPUT_SIZE)

        if BATCH_MODE:
            # Batch processing mode
            print(f"\n🔄 BATCH PROCESSING MODE")
            print(f"Uniform image: {UNIFORM_IMAGE_PATH}")
            print(f"Input directory: {INPUT_DIRECTORY}")
            print(f"Output directory: {OUTPUT_DIRECTORY}")
            print(f"Threshold: {THRESHOLD}")

            # Process directory
            summary = detector.process_directory(
                uniform_path=UNIFORM_IMAGE_PATH,
                input_dir=INPUT_DIRECTORY,
                output_dir=OUTPUT_DIRECTORY,
                threshold=THRESHOLD,
            )

            # Print summary
            print_summary(summary)

            # Show file distribution
            print(f"\nFiles have been sorted into:")
            print(f"  📁 Staff:     {os.path.join(OUTPUT_DIRECTORY, 'staff')}")
            print(f"  📁 Non-staff: {os.path.join(OUTPUT_DIRECTORY, 'non_staff')}")

        else:
            # Single image processing mode
            print(f"\n🔄 SINGLE IMAGE MODE")
            print("Calculating similarity...")
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
        print(f"❌ Error: {e}")
        print("\n💡 Make sure the following files/directories exist:")
        print(f"   - ONNX model: {ONNX_MODEL_PATH}")
        print(f"   - Uniform image: {UNIFORM_IMAGE_PATH}")
        if BATCH_MODE:
            print(f"   - Input directory: {INPUT_DIRECTORY}")
        else:
            print(f"   - Person image: {PERSON_IMAGE_PATH}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
