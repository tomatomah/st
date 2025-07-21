import glob
import os
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
from clip_image_similarity import CLIPSimilarityDetector


class StaffDetectionAnalyzer:
    """スタッフ判定の誤判定要因を分析"""

    def __init__(self, model_path):
        self.detector = CLIPSimilarityDetector(model_path)

    def analyze_resolution(self, evaluation_dir):
        """評価結果フォルダから解像度を分析"""
        # 各カテゴリの画像を収集
        categories = {
            "staff_correct": "Staff (Correct)",
            "staff_incorrect": "Staff (Incorrect)",
            "non_staff_correct": "Non-staff (Correct)",
            "non_staff_incorrect": "Non-staff (Incorrect)",
        }

        resolution_data = defaultdict(list)

        # 各カテゴリの画像解像度を取得
        for folder, label in categories.items():
            folder_path = os.path.join(evaluation_dir, folder)
            if not os.path.exists(folder_path):
                continue

            images = glob.glob(os.path.join(folder_path, "*.jpg"))
            print(f"\n{label}: {len(images)} images")

            for img_path in images:
                img = cv2.imread(img_path)
                if img is not None:
                    height, width = img.shape[:2]
                    resolution_data[label].append(
                        {
                            "width": width,
                            "height": height,
                            "pixels": width * height,
                            "aspect_ratio": width / height,
                            "filename": os.path.basename(img_path),
                        }
                    )

        # 統計情報を表示
        self._print_statistics(resolution_data)

        # グラフを作成
        self._plot_resolution_analysis(resolution_data)

        return resolution_data

    def _print_statistics(self, resolution_data):
        """解像度の統計情報を表示"""
        print("\n=== Resolution Statistics ===")

        for label, data in resolution_data.items():
            if not data:
                continue

            pixels = [d["pixels"] for d in data]
            widths = [d["width"] for d in data]
            heights = [d["height"] for d in data]
            aspects = [d["aspect_ratio"] for d in data]

            print(f"\n{label}:")
            print(f"  Resolution (pixels): {np.mean(pixels):.0f} ± {np.std(pixels):.0f}")
            print(f"  Width: {np.mean(widths):.0f} ± {np.std(widths):.0f}")
            print(f"  Height: {np.mean(heights):.0f} ± {np.std(heights):.0f}")
            print(f"  Aspect ratio: {np.mean(aspects):.2f} ± {np.std(aspects):.2f}")

            # 極端なケースを検出
            if pixels:
                min_idx = np.argmin(pixels)
                max_idx = np.argmax(pixels)
                print(f"  Smallest: {data[min_idx]['filename']} ({data[min_idx]['width']}x{data[min_idx]['height']})")
                print(f"  Largest: {data[max_idx]['filename']} ({data[max_idx]['width']}x{data[max_idx]['height']})")

    def _plot_resolution_analysis(self, resolution_data):
        """解像度分析のグラフを作成"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # カラーマップ
        colors = {
            "Staff (Correct)": "green",
            "Staff (Incorrect)": "red",
            "Non-staff (Correct)": "blue",
            "Non-staff (Incorrect)": "orange",
        }

        # 1. 解像度分布（ヒストグラム）
        ax = axes[0, 0]
        for label, data in resolution_data.items():
            if data:
                pixels = [d["pixels"] / 1000 for d in data]  # K pixels
                ax.hist(pixels, bins=20, alpha=0.5, label=label, color=colors.get(label, "gray"))
        ax.set_xlabel("Resolution (K pixels)")
        ax.set_ylabel("Count")
        ax.set_title("Resolution Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. 幅と高さの散布図
        ax = axes[0, 1]
        for label, data in resolution_data.items():
            if data:
                widths = [d["width"] for d in data]
                heights = [d["height"] for d in data]
                ax.scatter(widths, heights, alpha=0.6, label=label, color=colors.get(label, "gray"))
        ax.set_xlabel("Width (pixels)")
        ax.set_ylabel("Height (pixels)")
        ax.set_title("Image Dimensions")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. アスペクト比分布
        ax = axes[1, 0]
        for label, data in resolution_data.items():
            if data:
                aspects = [d["aspect_ratio"] for d in data]
                ax.hist(aspects, bins=20, alpha=0.5, label=label, color=colors.get(label, "gray"))
        ax.set_xlabel("Aspect Ratio (width/height)")
        ax.set_ylabel("Count")
        ax.set_title("Aspect Ratio Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. 正解/不正解の解像度比較（箱ひげ図）
        ax = axes[1, 1]
        correct_pixels = []
        incorrect_pixels = []

        for label, data in resolution_data.items():
            if "Correct" in label:
                correct_pixels.extend([d["pixels"] / 1000 for d in data])
            else:
                incorrect_pixels.extend([d["pixels"] / 1000 for d in data])

        if correct_pixels and incorrect_pixels:
            ax.boxplot([correct_pixels, incorrect_pixels], labels=["Correct", "Incorrect"])
            ax.set_ylabel("Resolution (K pixels)")
            ax.set_title("Resolution Comparison: Correct vs Incorrect")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("resolution_analysis.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("\nSaved: resolution_analysis.png")

    def analyze_with_similarity(self, registered_path, evaluation_dir):
        """解像度と類似度スコアの相関を分析"""
        # 登録画像の特徴量を取得
        if os.path.isfile(registered_path):
            registered_features = self.detector.extract_features(registered_path)
        else:
            # フォルダの場合は最初の画像を使用
            images = glob.glob(os.path.join(registered_path, "*.jpg"))
            registered_features = self.detector.extract_features(images[0]) if images else None

        if registered_features is None:
            print("No registered image found")
            return

        # 各画像の解像度と類似度を計算
        results = []
        for category in ["staff_correct", "staff_incorrect", "non_staff_correct", "non_staff_incorrect"]:
            folder_path = os.path.join(evaluation_dir, category)
            if not os.path.exists(folder_path):
                continue

            images = glob.glob(os.path.join(folder_path, "*.jpg"))
            for img_path in images:
                # 解像度取得
                img = cv2.imread(img_path)
                if img is None:
                    continue

                height, width = img.shape[:2]

                # 類似度計算
                test_features = self.detector.extract_features(img_path)
                similarity = float(np.dot(test_features, registered_features))

                results.append({"category": category, "pixels": width * height, "similarity": similarity})

        # 散布図で可視化
        self._plot_resolution_vs_similarity(results)

        return results

    def _plot_resolution_vs_similarity(self, results):
        """解像度と類似度の相関を可視化"""
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = {
            "staff_correct": "green",
            "staff_incorrect": "red",
            "non_staff_correct": "blue",
            "non_staff_incorrect": "orange",
        }

        for category, color in colors.items():
            data = [r for r in results if r["category"] == category]
            if data:
                pixels = [d["pixels"] / 1000 for d in data]
                similarities = [d["similarity"] for d in data]
                ax.scatter(pixels, similarities, alpha=0.6, label=category.replace("_", " ").title(), color=color)

        ax.set_xlabel("Resolution (K pixels)")
        ax.set_ylabel("Similarity Score")
        ax.set_title("Resolution vs Similarity Score")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("resolution_vs_similarity.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("\nSaved: resolution_vs_similarity.png")


if __name__ == "__main__":
    analyzer = StaffDetectionAnalyzer("onnx_models/clip_RN50.onnx")

    # 基本的な解像度分析
    analyzer.analyze_resolution("results/evaluation")

    # 解像度と類似度の相関分析
    analyzer.analyze_with_similarity("data/uniform.jpg", "results/evaluation")
