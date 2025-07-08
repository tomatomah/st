import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import seaborn as sns
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


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


class PPEStaffEvaluator:
    """PPE-based Staff Detection Evaluator"""

    def __init__(self, onnx_model_path, input_size=224):
        """
        Initialize evaluator

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

    def calculate_similarity(self, reference_features, test_features):
        """Calculate similarity between reference and test features"""
        similarity = np.dot(reference_features, test_features.T).item()
        return similarity

    def prepare_ppe_dataset(self, ppe_dataset_path, output_dir="evaluation_dataset"):
        """
        Prepare PPE dataset for staff detection evaluation

        Args:
            ppe_dataset_path: Path to PPE dataset directory
            output_dir: Output directory for organized dataset
        """
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/staff_uniform", exist_ok=True)
        os.makedirs(f"{output_dir}/no_uniform", exist_ok=True)
        os.makedirs(f"{output_dir}/reference", exist_ok=True)

        # PPE dataset structure (assuming YOLO format)
        # Look for images with safety vest/reflective clothing as "staff"
        # Look for person without vest as "no_uniform"

        print("Organizing PPE dataset for staff detection evaluation...")

        # This is a template - adjust based on actual PPE dataset structure
        image_files = list(Path(ppe_dataset_path).glob("*.jpg")) + list(Path(ppe_dataset_path).glob("*.png"))

        # Randomly split for demonstration (in real scenario, use actual annotations)
        random.shuffle(image_files)

        # Simulate PPE annotations mapping
        staff_count = 0
        no_uniform_count = 0

        for i, img_file in enumerate(image_files[:200]):  # Limit for demo
            if i % 3 == 0:  # Simulate vest/uniform images
                dest_path = f"{output_dir}/staff_uniform/{img_file.name}"
                staff_count += 1
            else:  # Simulate no vest/uniform images
                dest_path = f"{output_dir}/no_uniform/{img_file.name}"
                no_uniform_count += 1

            # Copy file (or create symlink)
            import shutil

            shutil.copy2(img_file, dest_path)

        # Select reference images
        ref_images = list(Path(f"{output_dir}/staff_uniform").glob("*.jpg"))[:5]
        for ref_img in ref_images:
            shutil.copy2(ref_img, f"{output_dir}/reference/{ref_img.name}")

        print(f"Dataset organized:")
        print(f"  Staff uniform images: {staff_count}")
        print(f"  No uniform images: {no_uniform_count}")
        print(f"  Reference images: {len(ref_images)}")

        return output_dir

    def evaluate_detection(self, dataset_dir, reference_image_path, threshold=0.75):
        """
        Evaluate staff detection performance

        Args:
            dataset_dir: Path to organized dataset directory
            reference_image_path: Path to reference uniform image
            threshold: Similarity threshold for staff detection
        """
        print(f"\n=== Staff Detection Evaluation ===")
        print(f"Reference image: {reference_image_path}")
        print(f"Threshold: {threshold}")

        # Extract reference features
        print("Extracting reference features...")
        reference_features = self.extract_features(reference_image_path)

        # Evaluate on staff uniform images (should be positive)
        staff_images = list(Path(f"{dataset_dir}/staff_uniform").glob("*.jpg"))
        staff_similarities = []
        staff_predictions = []

        print(f"\nEvaluating {len(staff_images)} staff uniform images...")
        for img_path in staff_images:
            try:
                test_features = self.extract_features(str(img_path))
                similarity = self.calculate_similarity(reference_features, test_features)
                staff_similarities.append(similarity)
                staff_predictions.append(1 if similarity > threshold else 0)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        # Evaluate on no uniform images (should be negative)
        no_uniform_images = list(Path(f"{dataset_dir}/no_uniform").glob("*.jpg"))
        no_uniform_similarities = []
        no_uniform_predictions = []

        print(f"Evaluating {len(no_uniform_images)} no uniform images...")
        for img_path in no_uniform_images:
            try:
                test_features = self.extract_features(str(img_path))
                similarity = self.calculate_similarity(reference_features, test_features)
                no_uniform_similarities.append(similarity)
                no_uniform_predictions.append(1 if similarity > threshold else 0)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        # Calculate metrics
        y_true = [1] * len(staff_predictions) + [0] * len(no_uniform_predictions)
        y_pred = staff_predictions + no_uniform_predictions
        all_similarities = staff_similarities + no_uniform_similarities

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Display results
        print(f"\n=== Evaluation Results ===")
        print(f"Total images evaluated: {len(y_true)}")
        print(f"  Staff uniform: {len(staff_predictions)}")
        print(f"  No uniform: {len(no_uniform_predictions)}")
        print(f"\nPerformance Metrics:")
        print(f"  Accuracy:  {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  F1-Score:  {f1:.3f}")

        # Similarity statistics
        print(f"\nSimilarity Statistics:")
        print(f"  Staff uniform - Mean: {np.mean(staff_similarities):.3f}, Std: {np.std(staff_similarities):.3f}")
        print(
            f"  No uniform - Mean: {np.mean(no_uniform_similarities):.3f}, Std: {np.std(no_uniform_similarities):.3f}"
        )

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"Actual    No_Staff  Staff")
        print(f"No_Staff     {cm[0,0]:4d}    {cm[0,1]:4d}")
        print(f"Staff        {cm[1,0]:4d}    {cm[1,1]:4d}")

        # Return detailed results
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "staff_similarities": staff_similarities,
            "no_uniform_similarities": no_uniform_similarities,
            "confusion_matrix": cm,
            "threshold": threshold,
        }

    def plot_results(self, results, save_path="evaluation_results.png"):
        """Plot evaluation results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Similarity distributions
        axes[0, 0].hist(results["staff_similarities"], bins=20, alpha=0.7, label="Staff Uniform", color="blue")
        axes[0, 0].hist(results["no_uniform_similarities"], bins=20, alpha=0.7, label="No Uniform", color="red")
        axes[0, 0].axvline(
            results["threshold"], color="black", linestyle="--", label=f"Threshold ({results['threshold']})"
        )
        axes[0, 0].set_xlabel("Similarity Score")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].set_title("Similarity Score Distribution")
        axes[0, 0].legend()

        # Confusion matrix
        sns.heatmap(
            results["confusion_matrix"],
            annot=True,
            fmt="d",
            ax=axes[0, 1],
            xticklabels=["No Staff", "Staff"],
            yticklabels=["No Staff", "Staff"],
        )
        axes[0, 1].set_title("Confusion Matrix")

        # Performance metrics
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
        values = [results["accuracy"], results["precision"], results["recall"], results["f1_score"]]
        bars = axes[1, 0].bar(metrics, values, color=["skyblue", "lightgreen", "orange", "pink"])
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].set_title("Performance Metrics")
        axes[1, 0].set_ylabel("Score")

        # Add value labels on bars
        for bar, value in zip(bars, values):
            axes[1, 0].text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{value:.3f}", ha="center", va="bottom"
            )

        # Threshold analysis (simplified)
        thresholds = np.linspace(0.5, 0.9, 20)
        accuracies = []

        y_true = [1] * len(results["staff_similarities"]) + [0] * len(results["no_uniform_similarities"])
        all_similarities = results["staff_similarities"] + results["no_uniform_similarities"]

        for th in thresholds:
            y_pred = [1 if sim > th else 0 for sim in all_similarities]
            acc = accuracy_score(y_true, y_pred)
            accuracies.append(acc)

        axes[1, 1].plot(thresholds, accuracies, marker="o")
        axes[1, 1].axvline(
            results["threshold"], color="red", linestyle="--", label=f"Current ({results['threshold']})"
        )
        axes[1, 1].set_xlabel("Threshold")
        axes[1, 1].set_ylabel("Accuracy")
        axes[1, 1].set_title("Threshold vs Accuracy")
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"Results plot saved to: {save_path}")


def main():
    """Main evaluation function"""
    # ===== CONFIGURATION =====
    ONNX_MODEL_PATH = "onnx_models/clip_RN50.onnx"  # Path to ONNX model
    PPE_DATASET_PATH = "ppe_dataset"  # Path to PPE dataset
    REFERENCE_IMAGE_PATH = "reference_uniform.jpg"  # Path to reference uniform image
    INPUT_SIZE = 224  # Input image size
    THRESHOLD = 0.75  # Similarity threshold

    try:
        # Initialize evaluator
        evaluator = PPEStaffEvaluator(ONNX_MODEL_PATH, INPUT_SIZE)

        # Check if PPE dataset exists
        if not os.path.exists(PPE_DATASET_PATH):
            print(f"PPE dataset not found at: {PPE_DATASET_PATH}")
            print("Please download and extract a PPE dataset (e.g., from Mendeley, Roboflow)")
            print("Example datasets:")
            print("- Mendeley PPE Dataset: https://data.mendeley.com/datasets/zkzghjvpn2/2")
            print("- Roboflow PPE datasets: https://universe.roboflow.com/")
            return

        # Prepare dataset for evaluation
        dataset_dir = evaluator.prepare_ppe_dataset(PPE_DATASET_PATH)

        # Check reference image
        if not os.path.exists(REFERENCE_IMAGE_PATH):
            # Use first staff uniform image as reference
            reference_candidates = list(Path(f"{dataset_dir}/reference").glob("*.jpg"))
            if reference_candidates:
                REFERENCE_IMAGE_PATH = str(reference_candidates[0])
                print(f"Using auto-selected reference: {REFERENCE_IMAGE_PATH}")
            else:
                print("No reference image found. Please provide a reference uniform image.")
                return

        # Run evaluation
        results = evaluator.evaluate_detection(dataset_dir, REFERENCE_IMAGE_PATH, THRESHOLD)

        # Plot results
        evaluator.plot_results(results)

        # Threshold optimization suggestion
        print(f"\n=== Optimization Suggestions ===")
        if results["precision"] < 0.8:
            print("💡 Low precision: Consider increasing threshold to reduce false positives")
        if results["recall"] < 0.8:
            print("💡 Low recall: Consider decreasing threshold to reduce false negatives")
        if results["f1_score"] > 0.8:
            print("🎯 Good F1-score: Current threshold seems well-balanced")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure all required files exist.")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
