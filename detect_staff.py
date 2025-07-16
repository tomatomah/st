import glob
import json
import os
import shutil
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import seaborn as sns
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
    """Simple ONNX-based staff detector with diagnostic capabilities"""

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

    def diagnose_similarity_distribution(self, uniform_path, test_dirs, output_dir="diagnostics"):
        """
        Diagnose similarity distribution for different categories

        Args:
            uniform_path: Path to uniform reference image
            test_dirs: Dictionary of category names and their directories
                      e.g., {"staff": "path/to/staff", "non_staff": "path/to/non_staff"}
            output_dir: Directory to save diagnostic results

        Returns:
            dict: Diagnostic results with statistics
        """
        os.makedirs(output_dir, exist_ok=True)

        # Supported image extensions
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]

        # Extract uniform features once
        print("Extracting uniform features...")
        uniform_features = self.extract_features(uniform_path)

        # Collect results by category
        all_results = []
        category_stats = {}

        for category, directory in test_dirs.items():
            print(f"\nAnalyzing category: {category}")

            # Get all image files
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(directory, ext)))
                image_files.extend(glob.glob(os.path.join(directory, ext.upper())))

            if not image_files:
                print(f"  No images found in {directory}")
                continue

            print(f"  Found {len(image_files)} images")

            # Calculate similarities
            similarities = []
            for img_path in image_files:
                try:
                    person_features = self.extract_features(img_path)
                    similarity = np.dot(uniform_features, person_features.T).item()

                    all_results.append(
                        {
                            "category": category,
                            "image": os.path.basename(img_path),
                            "path": img_path,
                            "similarity": similarity,
                        }
                    )
                    similarities.append(similarity)

                except Exception as e:
                    print(f"  Error processing {img_path}: {e}")
                    continue

            # Calculate statistics
            if similarities:
                category_stats[category] = {
                    "count": len(similarities),
                    "mean": np.mean(similarities),
                    "std": np.std(similarities),
                    "min": np.min(similarities),
                    "max": np.max(similarities),
                    "median": np.median(similarities),
                    "q25": np.percentile(similarities, 25),
                    "q75": np.percentile(similarities, 75),
                    "similarities": similarities,
                }

                print(f"  Statistics for {category}:")
                print(f"    Count:  {category_stats[category]['count']}")
                print(f"    Mean:   {category_stats[category]['mean']:.4f}")
                print(f"    Std:    {category_stats[category]['std']:.4f}")
                print(f"    Range:  [{category_stats[category]['min']:.4f}, {category_stats[category]['max']:.4f}]")
                print(f"    Median: {category_stats[category]['median']:.4f}")

        # Generate visualizations
        self._plot_diagnostics(category_stats, output_dir)

        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(output_dir, f"diagnostic_results_{timestamp}.json")

        with open(results_file, "w") as f:
            json.dump(
                {
                    "uniform_path": uniform_path,
                    "category_stats": {
                        k: {key: val for key, val in v.items() if key != "similarities"}
                        for k, v in category_stats.items()
                    },
                    "all_results": all_results,
                },
                f,
                indent=2,
            )

        print(f"\nDiagnostic results saved to: {results_file}")

        # Find optimal threshold
        optimal_threshold = self._find_optimal_threshold(category_stats)

        return {
            "category_stats": category_stats,
            "all_results": all_results,
            "optimal_threshold": optimal_threshold,
            "results_file": results_file,
        }

    def _plot_diagnostics(self, category_stats, output_dir):
        """Create diagnostic plots"""

        # 1. Histogram of similarity distributions
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        for category, stats in category_stats.items():
            plt.hist(stats["similarities"], bins=30, alpha=0.6, label=category, density=True)

        plt.xlabel("Similarity Score")
        plt.ylabel("Density")
        plt.title("Similarity Score Distribution by Category")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. Box plot
        plt.subplot(1, 2, 2)
        data_for_box = []
        labels_for_box = []

        for category, stats in category_stats.items():
            data_for_box.append(stats["similarities"])
            labels_for_box.append(f"{category}\n(n={stats['count']})")

        plt.boxplot(data_for_box, labels=labels_for_box)
        plt.ylabel("Similarity Score")
        plt.title("Similarity Score Distribution (Box Plot)")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = os.path.join(output_dir, "similarity_distribution.png")
        plt.savefig(plot_file, dpi=150)
        plt.close()

        # 3. Overlap visualization
        plt.figure(figsize=(10, 6))

        # Create bins for histogram
        all_similarities = []
        for stats in category_stats.values():
            all_similarities.extend(stats["similarities"])

        if all_similarities:
            bins = np.linspace(min(all_similarities), max(all_similarities), 50)

            for category, stats in category_stats.items():
                counts, bin_edges = np.histogram(stats["similarities"], bins=bins)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                plt.plot(bin_centers, counts, label=category, linewidth=2)

            plt.xlabel("Similarity Score")
            plt.ylabel("Count")
            plt.title("Similarity Score Distribution Overlap")
            plt.legend()
            plt.grid(True, alpha=0.3)

            overlap_file = os.path.join(output_dir, "similarity_overlap.png")
            plt.savefig(overlap_file, dpi=150)
            plt.close()

        # 4. Statistical summary plot
        if len(category_stats) > 0:
            plt.figure(figsize=(10, 6))

            categories = list(category_stats.keys())
            means = [stats["mean"] for stats in category_stats.values()]
            stds = [stats["std"] for stats in category_stats.values()]

            x = np.arange(len(categories))
            plt.bar(x, means, yerr=stds, capsize=5, alpha=0.7)
            plt.xticks(x, categories)
            plt.xlabel("Category")
            plt.ylabel("Similarity Score")
            plt.title("Mean Similarity Score by Category (with std dev)")
            plt.grid(True, alpha=0.3, axis="y")

            stats_file = os.path.join(output_dir, "similarity_stats.png")
            plt.savefig(stats_file, dpi=150)
            plt.close()

        print(f"\nDiagnostic plots saved to: {output_dir}")

    def _find_optimal_threshold(self, category_stats):
        """Find optimal threshold based on category distributions"""

        # Assuming binary classification (staff vs non_staff)
        if "staff" in category_stats and "non_staff" in category_stats:
            staff_sims = category_stats["staff"]["similarities"]
            non_staff_sims = category_stats["non_staff"]["similarities"]

            # Try different thresholds
            thresholds = np.linspace(
                min(min(staff_sims), min(non_staff_sims)), max(max(staff_sims), max(non_staff_sims)), 100
            )

            best_threshold = 0
            best_accuracy = 0

            for threshold in thresholds:
                # Calculate accuracy
                true_positives = sum(1 for s in staff_sims if s > threshold)
                true_negatives = sum(1 for s in non_staff_sims if s <= threshold)

                accuracy = (true_positives + true_negatives) / (len(staff_sims) + len(non_staff_sims))

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_threshold = threshold

            print(f"\nOptimal threshold: {best_threshold:.4f} (accuracy: {best_accuracy:.2%})")

            # Calculate metrics at optimal threshold
            tp = sum(1 for s in staff_sims if s > best_threshold)
            fn = len(staff_sims) - tp
            tn = sum(1 for s in non_staff_sims if s <= best_threshold)
            fp = len(non_staff_sims) - tn

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            print(f"  Precision: {precision:.2%}")
            print(f"  Recall:    {recall:.2%}")
            print(f"  F1-Score:  {f1:.2%}")

            return {
                "threshold": best_threshold,
                "accuracy": best_accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }

        return None


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
    UNIFORM_IMAGE_PATH = "test/uniform_images/uniform1.jpg"  # Path to uniform reference image

    # For single image processing
    PERSON_IMAGE_PATH = "staff1.jpg"  # Path to person image to check

    # For batch processing
    INPUT_DIRECTORY = "test/staff_images"  # Directory containing images to process
    OUTPUT_DIRECTORY = "results"  # Output directory for classification

    # For diagnostic mode
    DIAGNOSTIC_DIRS = {
        "staff": "test/staff_images",  # Directory with staff images
        "non_staff": "test/non_staff_images",  # Directory with non-staff images
    }
    DIAGNOSTIC_OUTPUT = "diagnostics"  # Output directory for diagnostic results

    INPUT_SIZE = 224  # Input image size (224 for most models)
    THRESHOLD = 0.6  # Similarity threshold for staff detection

    # Processing mode
    MODE = "diagnostic"  # Options: "single", "batch", "diagnostic"

    try:
        # Initialize detector
        detector = SimpleStaffDetector(ONNX_MODEL_PATH, INPUT_SIZE)

        if MODE == "diagnostic":
            # Diagnostic mode - analyze similarity distributions
            print(f"\n🔬 DIAGNOSTIC MODE")
            print(f"Uniform image: {UNIFORM_IMAGE_PATH}")
            print(f"Test directories: {DIAGNOSTIC_DIRS}")

            # Run diagnostics
            diagnostic_results = detector.diagnose_similarity_distribution(
                uniform_path=UNIFORM_IMAGE_PATH, test_dirs=DIAGNOSTIC_DIRS, output_dir=DIAGNOSTIC_OUTPUT
            )

            # Show optimal threshold recommendation
            if diagnostic_results["optimal_threshold"]:
                opt = diagnostic_results["optimal_threshold"]
                print(f"\n📊 RECOMMENDATION:")
                print(f"   Use threshold: {opt['threshold']:.4f}")
                print(f"   Expected accuracy: {opt['accuracy']:.2%}")

        elif MODE == "batch":
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

        else:  # single mode
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

        if MODE == "diagnostic":
            print("   - Test directories:")
            for category, path in DIAGNOSTIC_DIRS.items():
                print(f"     - {category}: {path}")
        elif MODE == "batch":
            print(f"   - Input directory: {INPUT_DIRECTORY}")
        else:
            print(f"   - Person image: {PERSON_IMAGE_PATH}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
