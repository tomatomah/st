import glob
import os
import shutil
from datetime import datetime

import numpy as np
from clip_image_similarity import CLIPSimilarityDetector


class StaffEvaluator:
    """スタッフ判定システムの評価ツール"""

    def __init__(self, model_path, threshold=0.6):
        self.detector = CLIPSimilarityDetector(model_path)
        self.threshold = threshold

    def evaluate_and_sort(self, registered_path, staff_dir, non_staff_dir, output_dir="results/evaluation"):
        """判定精度を評価し、画像を仕分け

        Args:
            registered_path: 登録画像のファイルパスまたはディレクトリパス
            staff_dir: スタッフ画像のディレクトリ
            non_staff_dir: 非スタッフ画像のディレクトリ
            output_dir: 出力ディレクトリ
        """
        # 出力フォルダ作成
        folders = {
            "staff_correct": os.path.join(output_dir, "staff_correct"),
            "staff_incorrect": os.path.join(output_dir, "staff_incorrect"),
            "non_staff_correct": os.path.join(output_dir, "non_staff_correct"),
            "non_staff_incorrect": os.path.join(output_dir, "non_staff_incorrect"),
        }
        for folder in folders.values():
            os.makedirs(folder, exist_ok=True)

        # 登録画像の取得（ファイルまたはフォルダ対応）
        if os.path.isfile(registered_path):
            # 単一ファイルの場合
            registered = [registered_path]
            print(f"Using single registered image: {os.path.basename(registered_path)}")
        else:
            # フォルダの場合
            registered = glob.glob(os.path.join(registered_path, "*.jpg"))
            print(f"Using {len(registered)} registered images from folder")

        # テスト画像パス取得
        get_images = lambda d: glob.glob(os.path.join(d, "*.jpg"))
        staff_images = get_images(staff_dir)
        non_staff_images = get_images(non_staff_dir)

        # 登録画像の特徴量を事前計算
        registered_features = {img: self.detector.extract_features(img) for img in registered}

        # 評価結果
        results = {
            "tp": 0,  # True Positive (スタッフを正しくスタッフと判定)
            "tn": 0,  # True Negative (非スタッフを正しく非スタッフと判定)
            "fp": 0,  # False Positive (非スタッフを誤ってスタッフと判定)
            "fn": 0,  # False Negative (スタッフを誤って非スタッフと判定)
        }

        # スタッフ画像を評価
        print("Evaluating staff images...")
        for img_path in staff_images:
            is_staff = self._predict(img_path, registered_features)
            if is_staff:
                results["tp"] += 1
                self._copy_image(img_path, folders["staff_correct"])
            else:
                results["fn"] += 1
                self._copy_image(img_path, folders["staff_incorrect"])

        # 非スタッフ画像を評価
        print("Evaluating non-staff images...")
        for img_path in non_staff_images:
            is_staff = self._predict(img_path, registered_features)
            if not is_staff:
                results["tn"] += 1
                self._copy_image(img_path, folders["non_staff_correct"])
            else:
                results["fp"] += 1
                self._copy_image(img_path, folders["non_staff_incorrect"])

        # 評価レポート作成
        self._save_report(results, output_dir, registered)

        return results

    def _predict(self, img_path, registered_features):
        """画像がスタッフかどうか判定"""
        test_features = self.detector.extract_features(img_path)

        # 各登録画像との類似度を計算
        max_similarity = 0
        for reg_features in registered_features.values():
            similarity = float(np.dot(test_features, reg_features))
            max_similarity = max(max_similarity, similarity)

        return max_similarity >= self.threshold

    def _copy_image(self, src_path, dst_dir):
        """画像を指定フォルダにコピー"""
        filename = os.path.basename(src_path)
        dst_path = os.path.join(dst_dir, filename)
        shutil.copy2(src_path, dst_path)

    def _save_report(self, results, output_dir, registered_images):
        """評価レポートを保存"""
        tp, tn, fp, fn = results["tp"], results["tn"], results["fp"], results["fn"]

        # 評価指標計算
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # 登録画像情報
        registered_info = "\n".join([f"  - {os.path.basename(img)}" for img in registered_images])

        # レポート作成
        report = f"""Staff Detection Evaluation Report
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Threshold: {self.threshold}

        === Registered Images ({len(registered_images)}) ===
        {registered_info}

        === Results ===
        Staff images:
        - Correctly detected: {tp}
        - Incorrectly rejected: {fn}
        
        Non-staff images:
        - Correctly rejected: {tn}
        - Incorrectly detected: {fp}

        === Metrics ===
        Accuracy: {accuracy:.3f}
        Precision: {precision:.3f}
        Recall: {recall:.3f}
        F1-Score: {f1:.3f}

        === Confusion Matrix ===
                    Predicted
                    Staff  Non-Staff
        Actual Staff    {tp:4d}     {fn:4d}
        Non-Staff       {fp:4d}     {tn:4d}
        """

        # レポート保存
        report_path = os.path.join(output_dir, "evaluation_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"\n{report}")
        print(f"Report saved: {report_path}")


if __name__ == "__main__":
    # 単一ファイルの場合
    evaluator = StaffEvaluator("onnx_models/clip_RN50.onnx", threshold=0.6)
    evaluator.evaluate_and_sort(
        "data/registered_staff/img_001_crop_0.jpg", "data/staff_images", "data/non_staff_images"
    )  # 単一の登録画像

    # フォルダの場合（従来通り）
    # evaluator.evaluate_and_sort(
    #     "data/registered_staff",  # フォルダ内の複数画像
    #     "data/staff_images",
    #     "data/non_staff_images"
    # )
