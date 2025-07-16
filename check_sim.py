import os

import matplotlib.pyplot as plt
import numpy as np
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


def diagnose_similarity_distribution(detector, uniform_path):
    """類似度の分布を可視化して問題を診断"""

    # 1. スタッフ画像とそれ以外の画像の類似度を収集
    staff_similarities = []
    non_staff_similarities = []

    # テスト画像のパスを手動で分類（一時的に）
    # 実際のスタッフ画像のディレクトリ
    staff_dir = "test/staff_images"
    non_staff_dir = "test/person_images"

    print("📊 類似度分析を開始...")

    # スタッフ画像の類似度計算
    import glob

    staff_images = glob.glob(f"{staff_dir}/*.jpg") + glob.glob(f"{staff_dir}/*.png")
    for img_path in staff_images:
        try:
            sim = detector.calculate_similarity(uniform_path, img_path)
            staff_similarities.append(sim)
            print(f"Staff: {img_path} -> {sim:.4f}")
        except:
            pass

    # 非スタッフ画像の類似度計算
    non_staff_images = glob.glob(f"{non_staff_dir}/*.jpg") + glob.glob(f"{non_staff_dir}/*.png")
    for img_path in non_staff_images:
        try:
            sim = detector.calculate_similarity(uniform_path, img_path)
            non_staff_similarities.append(sim)
            print(f"Non-staff: {img_path} -> {sim:.4f}")
        except:
            pass

    # 2. 統計情報の表示
    print("\n📈 統計情報:")
    print(
        f"スタッフ画像の類似度: 平均={np.mean(staff_similarities):.4f}, "
        f"最小={np.min(staff_similarities):.4f}, 最大={np.max(staff_similarities):.4f}"
    )
    print(
        f"非スタッフ画像の類似度: 平均={np.mean(non_staff_similarities):.4f}, "
        f"最小={np.min(non_staff_similarities):.4f}, 最大={np.max(non_staff_similarities):.4f}"
    )

    # 3. ヒストグラムで可視化
    plt.figure(figsize=(10, 6))
    plt.hist(staff_similarities, bins=20, alpha=0.5, label="Staff", color="blue")
    plt.hist(non_staff_similarities, bins=20, alpha=0.5, label="Non-staff", color="red")
    plt.xlabel("Similarity Score")
    plt.ylabel("Count")
    plt.title("Similarity Distribution: Staff vs Non-staff")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("similarity_distribution.png")
    plt.close()

    print("\n💡 分析結果を 'similarity_distribution.png' に保存しました")

    # 4. 問題の診断
    overlap = calculate_overlap(staff_similarities, non_staff_similarities)
    print(f"\n⚠️ 分布の重なり度: {overlap:.2%}")

    if overlap > 0.7:
        print("❌ 重大な問題: スタッフと非スタッフの類似度がほぼ区別できません")
    elif overlap > 0.3:
        print("⚠️ 中程度の問題: かなりの重なりがあります")
    else:
        print("✓ 分布は比較的分離しています")

    return {
        "staff_similarities": staff_similarities,
        "non_staff_similarities": non_staff_similarities,
        "overlap": overlap,
    }


def calculate_overlap(list1, list2):
    """2つの分布の重なり度を計算"""
    min_val = min(min(list1), min(list2))
    max_val = max(max(list1), max(list2))

    hist1, _ = np.histogram(list1, bins=50, range=(min_val, max_val))
    hist2, _ = np.histogram(list2, bins=50, range=(min_val, max_val))

    # 正規化
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)

    # 重なり部分の計算
    overlap = np.sum(np.minimum(hist1, hist2))
    return overlap


# 使用例
if __name__ == "__main__":
    detector = SimpleStaffDetector("onnx_models/clip_RN50.onnx")
    results = diagnose_similarity_distribution(detector, "test/uniform_images/uniform1.jpg")
