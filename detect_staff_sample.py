import os

import clip
import torch
from PIL import Image


class SimpleStaffDetector:
    def __init__(self, device="cpu", model_name="RN50", threshold=0.6):
        """スタッフ判定システムの初期化"""
        self.device = device
        self.threshold = threshold
        self.uniform_features = None
        self.uniform_name = None

        # CLIP設定
        print(f"Loading CLIP model: {model_name}")
        self.model, self.preprocess = clip.load(model_name, device=device)
        torch.set_num_threads(4)
        torch.set_grad_enabled(False)
        print(f"Model loaded on {device}")

    def _load_and_preprocess_image(self, image_path):
        """画像の読み込みと前処理"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"画像が見つかりません: {image_path}")

        image = Image.open(image_path).convert("RGB")
        return self.preprocess(image).unsqueeze(0).to(self.device)

    def _extract_features(self, image_tensor):
        """特徴量抽出と正規化"""
        with torch.no_grad():
            features = self.model.encode_image(image_tensor)
            return features / features.norm(dim=-1, keepdim=True)

    def register_uniform(self, uniform_image_path, uniform_name="staff_uniform"):
        """制服画像を登録"""
        print(f"制服登録中: {uniform_image_path}")

        image_tensor = self._load_and_preprocess_image(uniform_image_path)
        self.uniform_features = self._extract_features(image_tensor)
        self.uniform_name = uniform_name

        print(f"制服登録完了: {uniform_name}")

    def detect_staff(self, person_image_path):
        """スタッフ判定"""
        if self.uniform_features is None:
            raise ValueError("制服画像が未登録です。register_uniform()を先に実行してください。")

        image_tensor = self._load_and_preprocess_image(person_image_path)
        person_features = self._extract_features(image_tensor)

        similarity = torch.cosine_similarity(person_features, self.uniform_features).item()
        is_staff = similarity > self.threshold

        return {
            "is_staff": is_staff,
            "similarity": similarity,
            "uniform_name": self.uniform_name,
            "threshold": self.threshold,
        }

    def set_threshold(self, threshold):
        """判定閾値を設定"""
        self.threshold = threshold
        print(f"閾値を{threshold}に設定しました")


def demo():
    """使用例デモ"""
    # 初期化
    detector = SimpleStaffDetector()

    # 制服登録
    uniform_path = "uniform.jpg"
    if not os.path.exists(uniform_path):
        print(f"制服画像 '{uniform_path}' が見つかりません")
        return

    detector.register_uniform(uniform_path, "company_uniform")

    # テスト画像
    test_images = ["staff1.jpg", "staff2.jpg"]

    print("\n=== 判定結果 ===")
    for image_path in test_images:
        if os.path.exists(image_path):
            result = detector.detect_staff(image_path)
            status = "スタッフ" if result["is_staff"] else "非スタッフ"
            print(f"{image_path}: {status} (類似度: {result['similarity']:.3f})")
        else:
            print(f"{image_path}: ファイルが見つかりません")


if __name__ == "__main__":
    print("=== Simple Staff Detection System ===")
    print("\n使用方法:")
    print("detector = SimpleStaffDetector()")
    print("detector.register_uniform('uniform.jpg')")
    print("result = detector.detect_staff('person.jpg')")
    print()

    demo()
