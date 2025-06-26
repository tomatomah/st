import os

import clip
import cv2
import numpy as np
import torch
from PIL import Image


class SimpleStaffDetector:
    def __init__(self, device="cpu"):
        """スタッフ判定システムの初期化"""
        self.device = device
        print("Loading CLIP model...")
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        print(f"CLIP model loaded on {device}")

        # 制服の特徴量を保存する変数
        self.uniform_features = None
        self.uniform_name = None

        # 判定閾値（調整可能）
        self.similarity_threshold = 0.6

        # CPU最適化
        torch.set_num_threads(4)
        torch.set_grad_enabled(False)

    def register_uniform(self, uniform_image_path, uniform_name="staff_uniform"):
        """制服画像を登録する"""
        if not os.path.exists(uniform_image_path):
            raise FileNotFoundError(f"制服画像が見つかりません: {uniform_image_path}")

        print(f"制服画像を登録中: {uniform_image_path}")

        # 画像読み込み
        try:
            uniform_image = Image.open(uniform_image_path)
            if uniform_image.mode != "RGB":
                uniform_image = uniform_image.convert("RGB")
        except Exception as e:
            raise ValueError(f"画像の読み込みに失敗しました: {e}")

        # CLIP前処理
        image_input = self.preprocess(uniform_image).unsqueeze(0).to(self.device)

        # 特徴量抽出
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            # 正規化（コサイン類似度計算のため）
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        self.uniform_features = image_features
        self.uniform_name = uniform_name
        print(f"制服登録完了: {uniform_name}")

    def detect_staff(self, person_image_path):
        """人物画像からスタッフかどうかを判定する"""
        if self.uniform_features is None:
            raise ValueError("制服画像が登録されていません。先にregister_uniform()を実行してください。")

        if not os.path.exists(person_image_path):
            raise FileNotFoundError(f"人物画像が見つかりません: {person_image_path}")

        # 人物画像読み込み
        try:
            person_image = Image.open(person_image_path)
            if person_image.mode != "RGB":
                person_image = person_image.convert("RGB")
        except Exception as e:
            raise ValueError(f"人物画像の読み込みに失敗しました: {e}")

        # CLIP前処理
        image_input = self.preprocess(person_image).unsqueeze(0).to(self.device)

        # 特徴量抽出
        with torch.no_grad():
            person_features = self.model.encode_image(image_input)
            # 正規化
            person_features = person_features / person_features.norm(dim=-1, keepdim=True)

        # コサイン類似度計算
        similarity = torch.cosine_similarity(person_features, self.uniform_features).item()

        # 判定
        is_staff = similarity > self.similarity_threshold

        return {
            "is_staff": is_staff,
            "similarity": similarity,
            "uniform_name": self.uniform_name,
            "threshold": self.similarity_threshold,
        }

    def set_threshold(self, threshold):
        """判定閾値を設定する"""
        self.similarity_threshold = threshold
        print(f"判定閾値を {threshold} に設定しました")


def demo_usage():
    """使用例のデモンストレーション"""
    # スタッフ検出システムの初期化
    detector = SimpleStaffDetector(device="cpu")

    # 制服画像を登録（実際のパスに変更してください）
    uniform_image_path = "uniform.jpg"

    # サンプル画像が存在しない場合の処理
    if not os.path.exists(uniform_image_path):
        print(f"警告: {uniform_image_path} が見つかりません")
        print("実際の制服画像パスに変更してから実行してください")
        return

    try:
        # 制服登録
        detector.register_uniform(uniform_image_path, "company_uniform")

        # テスト用人物画像のパス（実際のパスに変更してください）
        test_images = ["staff1.jpg", "staff2.jpg"]

        print("\n=== スタッフ判定結果 ===")
        for person_image in test_images:
            if os.path.exists(person_image):
                result = detector.detect_staff(person_image)

                print(f"\n画像: {person_image}")
                print(f"スタッフ判定: {'はい' if result['is_staff'] else 'いいえ'}")
                print(f"類似度: {result['similarity']:.3f}")
                print(f"閾値: {result['threshold']}")
            else:
                print(f"\n画像が見つかりません: {person_image}")

    except Exception as e:
        print(f"エラーが発生しました: {e}")


def create_test_detection(detector, person_image_path, output_path=None):
    """検出結果を画像に描画して保存する"""
    try:
        # 判定実行
        result = detector.detect_staff(person_image_path)

        # 元画像読み込み（OpenCV用）
        image = cv2.imread(person_image_path)
        if image is None:
            raise ValueError("画像を読み込めませんでした")

        # 結果をテキストで描画
        text_lines = [
            f"Staff: {'YES' if result['is_staff'] else 'NO'}",
            f"Similarity: {result['similarity']:.3f}",
            f"Threshold: {result['threshold']}",
        ]

        # テキスト描画設定
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        color = (0, 255, 0) if result["is_staff"] else (0, 0, 255)

        # 各行を描画
        y_offset = 30
        for i, line in enumerate(text_lines):
            y_pos = y_offset + (i * 30)
            cv2.putText(image, line, (10, y_pos), font, font_scale, color, thickness)

        # 画像保存
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"結果画像を保存しました: {output_path}")

        return image, result

    except Exception as e:
        print(f"検出処理でエラーが発生しました: {e}")
        return None, None


if __name__ == "__main__":
    # 基本的な使用例
    print("=== Simple Staff Detection System ===")

    # システム初期化
    detector = SimpleStaffDetector()

    # 実際に使用する場合は、以下のパスを実際の画像ファイルパスに変更してください
    uniform_path = "uniform.jpg"
    person_path = "staff1.jpg"

    print("\n使用方法:")
    print("1. uniform_path と person_path を実際のファイルパスに変更")
    print("2. 以下のコードを実行")
    print()
    print("# 制服登録")
    print("detector.register_uniform(uniform_path, 'staff_uniform')")
    print()
    print("# スタッフ判定")
    print("result = detector.detect_staff(person_path)")
    print("print(result)")
    print()
    print("# 閾値調整（必要に応じて）")
    print("detector.set_threshold(0.6)  # より厳格な判定")
    print()

    # デモ実行（実際の画像がある場合）
    demo_usage()
