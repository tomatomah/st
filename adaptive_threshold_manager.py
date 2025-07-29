import os

import numpy as np
from clip_image_similarity import CLIPSimilarityDetector


class AdaptiveThresholdManager:
    """本番環境での閾値自動調整システム"""

    def __init__(self, model_path):
        self.detector = CLIPSimilarityDetector(model_path)
        self.threshold = 0.6  # 初期値
        self.confidence_margin = 0.05  # 信頼度マージン

    def calibrate_threshold(self, registered_images, calibration_samples=None):
        """初期閾値の自動設定

        Args:
            registered_images: 登録画像リスト
            calibration_samples: (staff_samples, non_staff_samples)のタプル（オプション）
        """
        print("閾値の自動調整を開始...")

        # 方法1: 登録画像間の類似度から推定
        inter_similarity = self._calculate_inter_similarity(registered_images)

        if calibration_samples:
            # 方法2: キャリブレーションサンプルがある場合
            staff_samples, non_staff_samples = calibration_samples
            optimal_threshold = self._find_optimal_threshold_from_samples(
                registered_images, staff_samples, non_staff_samples
            )
            self.threshold = optimal_threshold
        else:
            # 登録画像間の類似度を基準に設定
            # 同じ制服画像同士は高い類似度を持つはず
            self.threshold = inter_similarity - self.confidence_margin

        print(f"自動設定された閾値: {self.threshold:.3f}")
        return self.threshold

    def _calculate_inter_similarity(self, registered_images):
        """登録画像間の平均類似度を計算"""
        if len(registered_images) < 2:
            return 0.65  # デフォルト値

        similarities = []
        for i in range(len(registered_images)):
            for j in range(i + 1, len(registered_images)):
                sim = self.detector.calculate_similarity(registered_images[i], registered_images[j])
                similarities.append(sim)

        # 平均類似度（同じ制服なので高いはず）
        avg_similarity = np.mean(similarities)
        print(f"登録画像間の平均類似度: {avg_similarity:.3f}")

        return avg_similarity

    def _find_optimal_threshold_from_samples(self, registered_images, staff_samples, non_staff_samples):
        """少数のサンプルから最適閾値を探索"""
        # 各サンプルの最大類似度を計算
        staff_scores = []
        for sample in staff_samples[:5]:  # 最大5枚
            scores = [self.detector.calculate_similarity(reg, sample) for reg in registered_images]
            staff_scores.append(max(scores))

        non_staff_scores = []
        for sample in non_staff_samples[:5]:  # 最大5枚
            scores = [self.detector.calculate_similarity(reg, sample) for reg in registered_images]
            non_staff_scores.append(max(scores))

        # 最適な分離点を探す
        staff_min = min(staff_scores) if staff_scores else 0.7
        non_staff_max = max(non_staff_scores) if non_staff_scores else 0.5

        # 中間点を閾値とする
        threshold = (staff_min + non_staff_max) / 2

        print(f"スタッフ最小スコア: {staff_min:.3f}")
        print(f"非スタッフ最大スコア: {non_staff_max:.3f}")

        return threshold

    def adaptive_predict(self, test_image, registered_images, return_confidence=False):
        """適応的な予測（信頼度付き）

        Args:
            test_image: テスト画像
            registered_images: 登録画像リスト
            return_confidence: 信頼度を返すか
        """
        # 各登録画像との類似度
        similarities = [self.detector.calculate_similarity(reg, test_image) for reg in registered_images]
        max_similarity = max(similarities)

        # 予測
        is_staff = max_similarity >= self.threshold

        # 信頼度計算（閾値からの距離）
        confidence = abs(max_similarity - self.threshold) / self.confidence_margin
        confidence = min(confidence, 1.0)

        if return_confidence:
            return is_staff, confidence, max_similarity
        return is_staff

    def update_threshold_online(self, feedback_data):
        """運用中のフィードバックから閾値を更新

        Args:
            feedback_data: [(similarity, is_correct), ...]
        """
        if len(feedback_data) < 10:
            return  # データ不足

        # 誤判定の境界値を分析
        false_positives = [sim for sim, correct in feedback_data if not correct and sim >= self.threshold]
        false_negatives = [sim for sim, correct in feedback_data if not correct and sim < self.threshold]

        if false_positives and false_negatives:
            # 新しい閾値候補
            new_threshold = (max(false_negatives) + min(false_positives)) / 2
            # 緩やかに更新（急激な変化を避ける）
            self.threshold = 0.9 * self.threshold + 0.1 * new_threshold
            print(f"閾値を更新: {self.threshold:.3f}")


# 使用例
if __name__ == "__main__":
    # 初期設定
    manager = AdaptiveThresholdManager("onnx_models/clip_RN50.onnx")

    registered_images = ["data/registered_staff/uniform1.jpg", "data/registered_staff/uniform2.jpg"]

    # 方法1: 登録画像のみから閾値を推定
    manager.calibrate_threshold(registered_images)

    # 方法2: 少数のサンプルがある場合
    calibration_samples = (
        ["data/calib/staff1.jpg", "data/calib/staff2.jpg"],
        ["data/calib/non_staff1.jpg", "data/calib/non_staff2.jpg"],
    )
    manager.calibrate_threshold(registered_images, calibration_samples)

    # 予測（信頼度付き）
    is_staff, confidence, score = manager.adaptive_predict("test_image.jpg", registered_images, return_confidence=True)
    print(f"判定: {'スタッフ' if is_staff else '非スタッフ'} (信頼度: {confidence:.2f})")
