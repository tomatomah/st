import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from clip_image_similarity import CLIPSimilarityDetector


class DetectionErrorAnalyzer:
    """誤検出・未検出画像の特性分析ツール"""

    def __init__(self, model_path, registered_path):
        self.detector = CLIPSimilarityDetector(model_path)
        self.registered_path = registered_path

    def analyze_image_properties(self, image_path):
        """画像の基本特性を分析"""
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 基本統計
        properties = {
            "filename": os.path.basename(image_path),
            "brightness": np.mean(img_hsv[:, :, 2]),  # 明度
            "saturation": np.mean(img_hsv[:, :, 1]),  # 彩度
            "contrast": np.std(img_gray),  # コントラスト
            "edge_amount": np.mean(cv2.Canny(img_gray, 100, 200)),  # エッジ量
            "color_variance": np.mean(np.std(img_rgb, axis=(0, 1))),  # 色の分散
        }

        # 類似度スコア
        similarity = self.detector.calculate_similarity(self.registered_path, image_path)
        properties["similarity"] = similarity

        return properties

    def analyze_groups(self, correct_dir, incorrect_dir, output_dir="analysis_results"):
        """正解・不正解グループの比較分析"""
        os.makedirs(output_dir, exist_ok=True)

        # 各グループの画像を分析
        print("正解画像を分析中...")
        correct_data = []
        for img_path in glob.glob(os.path.join(correct_dir, "*.jpg")):
            correct_data.append(self.analyze_image_properties(img_path))

        print("不正解画像を分析中...")
        incorrect_data = []
        for img_path in glob.glob(os.path.join(incorrect_dir, "*.jpg")):
            incorrect_data.append(self.analyze_image_properties(img_path))

        # DataFrameに変換
        df_correct = pd.DataFrame(correct_data)
        df_incorrect = pd.DataFrame(incorrect_data)

        # 統計サマリ
        self._save_statistics(df_correct, df_incorrect, output_dir)

        # グラフ作成
        self._create_comparison_plots(df_correct, df_incorrect, output_dir)

        return df_correct, df_incorrect

    def _save_statistics(self, df_correct, df_incorrect, output_dir):
        """統計情報を保存"""
        stats = pd.DataFrame(
            {
                "正解_平均": df_correct.mean(numeric_only=True),
                "正解_標準偏差": df_correct.std(numeric_only=True),
                "不正解_平均": df_incorrect.mean(numeric_only=True),
                "不正解_標準偏差": df_incorrect.std(numeric_only=True),
                "差分": df_correct.mean(numeric_only=True) - df_incorrect.mean(numeric_only=True),
            }
        )

        stats.to_csv(os.path.join(output_dir, "statistics_summary.csv"))
        print(f"\n統計サマリ:\n{stats}")

    def _create_comparison_plots(self, df_correct, df_incorrect, output_dir):
        """比較グラフを作成"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        metrics = ["brightness", "saturation", "contrast", "edge_amount", "color_variance", "similarity"]

        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            # ヒストグラム作成
            bins = 20
            ax.hist(df_correct[metric], bins=bins, alpha=0.6, label="正解", color="blue")
            ax.hist(df_incorrect[metric], bins=bins, alpha=0.6, label="不正解", color="red")

            ax.set_xlabel(metric)
            ax.set_ylabel("頻度")
            ax.set_title(f"{metric}の分布比較")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "comparison_analysis.png"), dpi=150)
        plt.close()

        print(f"グラフ保存: {os.path.join(output_dir, 'comparison_analysis.png')}")


# 使用例
if __name__ == "__main__":
    analyzer = DetectionErrorAnalyzer(
        model_path="onnx_models/clip_RN50.onnx", registered_path="data/registered_staff/img_001_crop_0.jpg"
    )

    # スタッフ画像の正解・不正解を分析
    analyzer.analyze_groups(
        correct_dir="results/evaluation/staff_correct",
        incorrect_dir="results/evaluation/staff_incorrect",
        output_dir="results/error_analysis",
    )
