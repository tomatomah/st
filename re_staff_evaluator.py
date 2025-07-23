import glob
import os
from datetime import datetime

import numpy as np
from clip_image_similarity import CLIPSimilarityDetector


class StaffEvaluator:
    """スタッフ判定システムの評価ツール"""

    def __init__(self, model_path, model_name="Unknown", threshold=0.6):
        self.detector = CLIPSimilarityDetector(model_path)
        self.model_name = model_name
        self.threshold = threshold

    def evaluate_multiple(self, registered_dir, staff_dir, non_staff_dir, output_dir="results/evaluation"):
        """複数の登録画像で個別評価とサマリを作成"""
        os.makedirs(output_dir, exist_ok=True)

        # 画像パス取得
        registered_images = glob.glob(os.path.join(registered_dir, "*.jpg"))
        staff_images = glob.glob(os.path.join(staff_dir, "*.jpg"))
        non_staff_images = glob.glob(os.path.join(non_staff_dir, "*.jpg"))

        all_results = []

        # 各登録画像での評価
        for reg_img in registered_images:
            reg_name = os.path.basename(reg_img)
            print(f"\n{reg_name} で評価中...")

            result = self._evaluate_single(reg_img, staff_images, non_staff_images)
            result["registered_image"] = reg_name
            all_results.append(result)

            # 個別結果を表示
            print(f"  スタッフ検出率: {result['staff_detection_rate']:.1f}%")
            print(f"  非スタッフ検出率: {result['non_staff_detection_rate']:.1f}%")
            print(f"  誤検出率: {result['false_positive_rate']:.1f}%")
            print(f"  未検出率: {result['false_negative_rate']:.1f}%")

        # サマリレポート作成
        self._save_summary_report(all_results, output_dir)

        return all_results

    def _evaluate_single(self, registered_img, staff_images, non_staff_images):
        """単一登録画像での評価"""
        # 登録画像の特徴量
        reg_features = self.detector.extract_features(registered_img)

        # 評価カウンタ
        tp = sum(1 for img in staff_images if self._is_similar(img, reg_features))
        fn = len(staff_images) - tp

        fp = sum(1 for img in non_staff_images if self._is_similar(img, reg_features))
        tn = len(non_staff_images) - fp

        # 指標計算
        staff_detection_rate = (tp / len(staff_images)) * 100 if staff_images else 0
        non_staff_detection_rate = (tn / len(non_staff_images)) * 100 if non_staff_images else 0
        false_positive_rate = (fp / len(non_staff_images)) * 100 if non_staff_images else 0
        false_negative_rate = (fn / len(staff_images)) * 100 if staff_images else 0

        return {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "staff_detection_rate": staff_detection_rate,
            "non_staff_detection_rate": non_staff_detection_rate,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
        }

    def _is_similar(self, img_path, reg_features):
        """類似度判定"""
        test_features = self.detector.extract_features(img_path)
        similarity = float(np.dot(test_features, reg_features))
        return similarity >= self.threshold

    def _save_summary_report(self, results, output_dir):
        """サマリレポートを保存"""
        # 統計計算
        staff_rates = [r["staff_detection_rate"] for r in results]
        non_staff_rates = [r["non_staff_detection_rate"] for r in results]
        fp_rates = [r["false_positive_rate"] for r in results]
        fn_rates = [r["false_negative_rate"] for r in results]

        avg_staff_rate = np.mean(staff_rates)
        std_staff_rate = np.std(staff_rates)
        avg_non_staff_rate = np.mean(non_staff_rates)
        std_non_staff_rate = np.std(non_staff_rates)
        avg_fp_rate = np.mean(fp_rates)
        std_fp_rate = np.std(fp_rates)
        avg_fn_rate = np.mean(fn_rates)
        std_fn_rate = np.std(fn_rates)

        # 全体の混同行列
        total_tp = sum(r["tp"] for r in results)
        total_tn = sum(r["tn"] for r in results)
        total_fp = sum(r["fp"] for r in results)
        total_fn = sum(r["fn"] for r in results)

        # レポート作成
        report = f"""CLIPモデル性能評価サマリ
生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
モデル: {self.model_name}
しきい値: {self.threshold}

=== 登録画像別の評価結果 ===
{'登録画像':<20} {'スタッフ':<10} {'非スタッフ':<10} {'誤検出率':<10} {'未検出率':<10}
                   検出率     検出率
{'-'*75}
"""

        for result in results:
            report += f"{result['registered_image']:<20} "
            report += f"{result['staff_detection_rate']:>6.1f}%   "
            report += f"{result['non_staff_detection_rate']:>6.1f}%   "
            report += f"{result['false_positive_rate']:>6.1f}%   "
            report += f"{result['false_negative_rate']:>6.1f}%\n"

        report += f"""
=== 統計サマリ ===
スタッフ検出率:   {avg_staff_rate:.1f}% ± {std_staff_rate:.1f}%
非スタッフ検出率: {avg_non_staff_rate:.1f}% ± {std_non_staff_rate:.1f}%
誤検出率:         {avg_fp_rate:.1f}% ± {std_fp_rate:.1f}%
未検出率:         {avg_fn_rate:.1f}% ± {std_fn_rate:.1f}%

=== 全体の混同行列（全登録画像合計） ===
              予測
              スタッフ  非スタッフ
実際
スタッフ        {total_tp:4d}       {total_fn:4d}
非スタッフ      {total_fp:4d}       {total_tn:4d}

全体精度: {(total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn) * 100:.1f}%
スタッフ検出率: {(total_tp / (total_tp + total_fn)) * 100:.1f}%
非スタッフ検出率: {(total_tn / (total_tn + total_fp)) * 100:.1f}%
"""

        # 保存
        report_path = os.path.join(output_dir, f"サマリ_{self.model_name}_閾値{self.threshold}.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"\nサマリ保存先: {report_path}")


# 使用例：複数モデルの評価
if __name__ == "__main__":
    models = [
        ("onnx_models/clip_RN50.onnx", "RN50"),
        ("onnx_models/clip_ViT-B_32.onnx", "ViT-B_32"),
        # 他のモデルを追加
    ]

    for model_path, model_name in models:
        if os.path.exists(model_path):
            print(f"\n{'='*60}")
            print(f"{model_name} を評価中")
            print(f"{'='*60}")

            evaluator = StaffEvaluator(model_path, model_name, threshold=0.6)
            evaluator.evaluate_multiple(
                "data/registered_staff",
                "data/staff_images",
                "data/non_staff_images",
                f"results/evaluation_{model_name}",
            )
