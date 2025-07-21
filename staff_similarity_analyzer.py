import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from clip_image_similarity import CLIPSimilarityDetector


class StaffSimilarityAnalyzer:
    def __init__(self, model_path):
        self.detector = CLIPSimilarityDetector(model_path)

    def analyze_and_save(self, registered_dir, staff_dir, non_staff_dir, output_dir="results/analysis"):
        os.makedirs(output_dir, exist_ok=True)

        # 画像パス取得
        get_images = lambda d: glob.glob(os.path.join(d, "*.jpg"))
        registered = get_images(registered_dir)
        staff_images = get_images(staff_dir)
        non_staff_images = get_images(non_staff_dir)

        all_results = []

        # 各登録画像を処理
        for reg_img in registered:
            name = os.path.basename(reg_img)

            # 類似度計算
            staff_sims = [self.detector.calculate_similarity(reg_img, img) for img in staff_images]
            non_staff_sims = [self.detector.calculate_similarity(reg_img, img) for img in non_staff_images]

            # 最適閾値の計算
            best_threshold = self.find_best_threshold(staff_sims, non_staff_sims)
            all_results.append(best_threshold)

            # 結果出力とヒストグラム保存
            print(f"{name}: Staff {np.mean(staff_sims):.3f} / Non-staff {np.mean(non_staff_sims):.3f}")
            print(f"  Best threshold: {best_threshold:.3f}")
            self.save_histogram(name, staff_sims, non_staff_sims, output_dir, best_threshold)

        # 全体の推奨閾値
        print(f"\nRecommended threshold (average): {np.mean(all_results):.3f}")

    def find_best_threshold(self, staff_sims, non_staff_sims):
        """最適な閾値を探索"""
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_accuracy = 0
        best_threshold = 0.5

        for threshold in thresholds:
            # 正解数を計算
            tp = sum(s >= threshold for s in staff_sims)  # True Positive
            tn = sum(s < threshold for s in non_staff_sims)  # True Negative

            # 精度計算
            accuracy = (tp + tn) / (len(staff_sims) + len(non_staff_sims))

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold

        return best_threshold

    def save_histogram(self, name, staff_sims, non_staff_sims, output_dir, best_threshold):
        # 既存のfigureをクリア
        plt.close("all")

        fig, ax = plt.subplots(figsize=(10, 6))

        # axオブジェクトを使用
        bins = np.linspace(0, 1, 25)
        ax.hist(staff_sims, bins, alpha=0.7, label="Staff", color="red")
        ax.hist(non_staff_sims, bins, alpha=0.7, label="Non-staff", color="blue")

        # 最適閾値を表示
        ax.axvline(
            x=best_threshold, color="green", linestyle="--", linewidth=2, label=f"Best threshold: {best_threshold:.3f}"
        )

        ax.set_xlabel("Similarity")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Registered: {name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        output_path = os.path.join(output_dir, f"histogram_{os.path.splitext(name)[0]}.png")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    analyzer = StaffSimilarityAnalyzer("onnx_models/clip_RN50.onnx")
    analyzer.analyze_and_save(
        "data/registered_staff", "data/staff_images", "data/non_staff_images", "results/analysis"
    )
