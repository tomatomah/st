import json
import os
import time

import clip
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchvision.transforms as transforms
from PIL import Image


class BaseStaffDetector:
    """スタッフ判定システムの基底クラス"""

    def __init__(self, threshold=0.6):
        self.threshold = threshold
        self.uniform_features = None
        self.uniform_name = None

    def _load_image(self, image_path):
        """画像読み込み"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"画像が見つかりません: {image_path}")
        return Image.open(image_path).convert("RGB")

    def _calculate_similarity(self, features1, features2):
        """類似度計算（サブクラスで実装）"""
        raise NotImplementedError

    def register_uniform(self, image_path, name="staff_uniform"):
        """制服画像登録"""
        print(f"制服登録中: {image_path}")
        self.uniform_features = self._extract_features(image_path)
        self.uniform_name = name
        print(f"制服登録完了: {name}")

    def detect_staff(self, image_path):
        """スタッフ判定"""
        if self.uniform_features is None:
            raise ValueError("制服画像が未登録です")

        person_features = self._extract_features(image_path)
        similarity = self._calculate_similarity(person_features, self.uniform_features)

        return {
            "is_staff": similarity > self.threshold,
            "similarity": similarity,
            "uniform_name": self.uniform_name,
            "threshold": self.threshold,
        }


class SimpleStaffDetector(BaseStaffDetector):
    """PyTorch版スタッフ判定"""

    def __init__(self, model_name="RN50", device="cpu", threshold=0.6):
        super().__init__(threshold)
        print(f"Loading CLIP model: {model_name}")
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.device = device
        torch.set_num_threads(4)
        torch.set_grad_enabled(False)

    def _extract_features(self, image_path):
        """特徴量抽出"""
        image = self._load_image(image_path)
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model.encode_image(image_tensor)
            return features / features.norm(dim=-1, keepdim=True)

    def _calculate_similarity(self, features1, features2):
        """コサイン類似度計算"""
        return torch.cosine_similarity(features1, features2).item()


class OptimizedStaffDetector(BaseStaffDetector):
    """ONNX版高速スタッフ判定"""

    def __init__(self, onnx_path="clip_image_encoder.onnx", threshold=0.6):
        super().__init__(threshold)
        print("Loading ONNX model...")
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.preprocess = self._create_preprocess()
        print("ONNX model loaded")

    def _create_preprocess(self):
        """前処理パイプライン作成"""
        # デフォルトCLIP設定
        config = {
            "resize": 224,
            "center_crop": 224,
            "mean": [0.48145466, 0.4578275, 0.40821073],
            "std": [0.26862954, 0.26130258, 0.27577711],
        }

        # 設定ファイルがあれば読み込み
        if os.path.exists("clip_preprocess_config.json"):
            with open("clip_preprocess_config.json", "r") as f:
                config.update(json.load(f))

        return transforms.Compose(
            [
                transforms.Resize(config["resize"], interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(config["center_crop"]),
                transforms.ToTensor(),
                transforms.Normalize(mean=config["mean"], std=config["std"]),
            ]
        )

    def _extract_features(self, image_path):
        """ONNX特徴量抽出"""
        image = self._load_image(image_path)
        image_array = self.preprocess(image).unsqueeze(0).numpy()

        # ONNX推論
        input_name = self.session.get_inputs()[0].name
        features = self.session.run(None, {input_name: image_array})[0]

        # L2正規化
        norm = np.linalg.norm(features, axis=1, keepdims=True)
        return features / (norm + 1e-8)

    def _calculate_similarity(self, features1, features2):
        """NumPy類似度計算"""
        return np.dot(features1, features2.T).item()


class CLIPConverter:
    """CLIP→ONNX変換器"""

    @staticmethod
    def convert(model_name="RN50", output_path="clip_image_encoder.onnx"):
        """ONNX変換実行"""
        print(f"Converting {model_name} to ONNX...")

        try:
            # モデル読み込み
            model, _ = clip.load(model_name, device="cpu", jit=False)
            model.visual.eval()

            # ONNX変換
            dummy_input = torch.randn(1, 3, 224, 224)
            torch.onnx.export(
                model.visual,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=17,
                input_names=["image"],
                output_names=["features"],
                dynamic_axes={"image": {0: "batch_size"}, "features": {0: "batch_size"}},
            )

            # 検証
            onnx.checker.check_model(onnx.load(output_path))

            # 設定保存
            CLIPConverter._save_config()

            print(f"✅ 変換完了: {output_path}")
            return True

        except Exception as e:
            print(f"❌ 変換失敗: {e}")
            return False

    @staticmethod
    def _save_config():
        """前処理設定保存"""
        config = {
            "resize": 224,
            "center_crop": 224,
            "mean": [0.48145466, 0.4578275, 0.40821073],
            "std": [0.26862954, 0.26130258, 0.27577711],
        }
        with open("clip_preprocess_config.json", "w") as f:
            json.dump(config, f, indent=2)


def benchmark(uniform_path="uniform.jpg", iterations=10):
    """性能比較テスト"""
    if not os.path.exists(uniform_path):
        print(f"❌ テスト画像が見つかりません: {uniform_path}")
        return

    print("=== 性能比較テスト ===")

    # PyTorch版テスト
    print("\n1. PyTorch版")
    pytorch_detector = SimpleStaffDetector()
    pytorch_detector.register_uniform(uniform_path)

    pytorch_times = []
    for _ in range(iterations):
        start = time.time()
        pytorch_result = pytorch_detector.detect_staff(uniform_path)
        pytorch_times.append(time.time() - start)

    # ONNX版テスト
    print("\n2. ONNX版")
    if not os.path.exists("clip_image_encoder.onnx"):
        print("❌ ONNXモデルが見つかりません")
        return

    onnx_detector = OptimizedStaffDetector()
    onnx_detector.register_uniform(uniform_path)

    onnx_times = []
    for _ in range(iterations):
        start = time.time()
        onnx_result = onnx_detector.detect_staff(uniform_path)
        onnx_times.append(time.time() - start)

    # 結果表示
    pytorch_avg = np.mean(pytorch_times) * 1000
    onnx_avg = np.mean(onnx_times) * 1000
    speedup = pytorch_avg / onnx_avg

    print(f"\n📊 結果:")
    print(f"PyTorch: {pytorch_avg:.2f}ms")
    print(f"ONNX:    {onnx_avg:.2f}ms")
    print(f"高速化:  {speedup:.2f}x")
    print(f"精度差:  {abs(pytorch_result['similarity'] - onnx_result['similarity']):.6f}")


def demo():
    """簡単なデモ"""
    print("=== CLIP スタッフ判定システム ===")

    # ONNX変換
    if not os.path.exists("clip_image_encoder.onnx"):
        print("ONNXモデルを変換中...")
        if not CLIPConverter.convert():
            return

    # デモ実行
    detector = OptimizedStaffDetector()

    uniform_path = input("制服画像パス: ").strip()
    if uniform_path and os.path.exists(uniform_path):
        detector.register_uniform(uniform_path)

        person_path = input("判定画像パス: ").strip()
        if person_path and os.path.exists(person_path):
            result = detector.detect_staff(person_path)
            status = "スタッフ" if result["is_staff"] else "非スタッフ"
            print(f"\n結果: {status} (類似度: {result['similarity']:.4f})")
        else:
            print("❌ 判定画像が見つかりません")
    else:
        print("❌ 制服画像が見つかりません")


def main():
    """メイン関数"""
    commands = {
        "1": ("ONNX変換", lambda: CLIPConverter.convert()),
        "2": ("デモ実行", demo),
        "3": ("性能比較", benchmark),
        "4": ("終了", lambda: None),
    }

    while True:
        print("\n=== メニュー ===")
        for key, (desc, _) in commands.items():
            print(f"{key}. {desc}")

        choice = input("\n選択: ").strip()

        if choice in commands:
            if choice == "4":
                break
            commands[choice][1]()
        else:
            print("❌ 無効な選択")


if __name__ == "__main__":
    main()
