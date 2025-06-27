import json
import os
import shutil
import time
from typing import Any, Dict, Optional

import clip
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchvision.transforms as transforms
from PIL import Image

# 量子化サポート確認
try:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    QUANTIZATION_AVAILABLE = True
except ImportError:
    QUANTIZATION_AVAILABLE = False


class Config:
    """設定管理クラス"""

    DEFAULT = {
        "resize": 224,
        "center_crop": 224,
        "mean": [0.48145466, 0.4578275, 0.40821073],
        "std": [0.26862954, 0.26130258, 0.27577711],
    }

    @staticmethod
    def load(path: str = "clip_config.json") -> Dict[str, Any]:
        """設定読み込み"""
        config = Config.DEFAULT.copy()
        if os.path.exists(path):
            with open(path, "r") as f:
                config.update(json.load(f))
        return config

    @staticmethod
    def save(path: str = "clip_config.json") -> None:
        """設定保存"""
        with open(path, "w") as f:
            json.dump(Config.DEFAULT, f, indent=2)


class ModelConverter:
    """モデル変換器（シンプル版）"""

    @staticmethod
    def convert(model_name: str = "RN50") -> bool:
        """CLIP → ONNX変換（INT8のみ）"""
        print(f"=== {model_name} → INT8変換 ===")

        fp32_path = "clip_fp32.onnx"
        int8_path = "clip_int8.onnx"

        # Step 1: FP32変換
        if not ModelConverter._to_fp32(model_name, fp32_path):
            return False

        # Step 2: INT8量子化
        if ModelConverter._to_int8(fp32_path, int8_path):
            ModelConverter._show_comparison(fp32_path, int8_path)
        else:
            # フォールバック
            shutil.copy2(fp32_path, int8_path)
            print("⚠️ 量子化失敗 - FP32を使用")

        Config.save()
        print("✅ 変換完了")
        return True

    @staticmethod
    def _to_fp32(model_name: str, output_path: str) -> bool:
        """FP32変換"""
        try:
            model, _ = clip.load(model_name, device="cpu", jit=False)
            dummy_input = torch.randn(1, 3, 224, 224)

            torch.onnx.export(
                model.visual,
                dummy_input,
                output_path,
                input_names=["image"],
                output_names=["features"],
                dynamic_axes={"image": {0: "batch"}, "features": {0: "batch"}},
            )

            onnx.checker.check_model(onnx.load(output_path))
            print(f"FP32完了: {output_path}")
            return True
        except Exception as e:
            print(f"FP32失敗: {e}")
            return False

    @staticmethod
    def _to_int8(fp32_path: str, int8_path: str) -> bool:
        """INT8量子化"""
        if not QUANTIZATION_AVAILABLE:
            return False

        # 複数の量子化方式を試行
        configs = [{"weight_type": QuantType.QUInt8}, {"weight_type": QuantType.QInt8}]

        for config in configs:
            try:
                quantize_dynamic(fp32_path, int8_path, **config)
                if ModelConverter._test_model(int8_path):
                    print(f"INT8完了: {int8_path}")
                    return True
            except Exception:
                continue

        return False

    @staticmethod
    def _test_model(model_path: str) -> bool:
        """モデル動作テスト"""
        try:
            session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
            input_name = session.get_inputs()[0].name
            session.run(None, {input_name: dummy_input})
            return True
        except Exception:
            return False

    @staticmethod
    def _show_comparison(fp32_path: str, int8_path: str) -> None:
        """サイズ比較表示"""
        fp32_size = os.path.getsize(fp32_path) / (1024**2)
        int8_size = os.path.getsize(int8_path) / (1024**2)
        print(f"圧縮: {fp32_size:.1f}MB → {int8_size:.1f}MB ({fp32_size/int8_size:.1f}x)")


class StaffDetector:
    """スタッフ判定器（統合版）"""

    def __init__(self, model_path: str = "clip_int8.onnx", threshold: float = 0.6):
        self.threshold = threshold
        self.uniform_features = None
        self.uniform_name = None

        # モデルパス確定
        model_path = self._find_model(model_path)
        print(f"モデル読み込み: {model_path}")

        # セッション作成
        self.session = self._create_session(model_path)
        self.preprocess = self._create_preprocess()
        print("読み込み完了")

    def _find_model(self, preferred_path: str) -> str:
        """利用可能なモデルを検索"""
        candidates = [preferred_path, "clip_fp32.onnx", "clip_int8.onnx"]

        for path in candidates:
            if os.path.exists(path):
                return path

        raise FileNotFoundError("モデルファイルが見つかりません")

    def _create_session(self, model_path: str) -> ort.InferenceSession:
        """セッション作成（エラー対応）"""
        session_configs = [
            # 最適化ON
            {"providers": ["CPUExecutionProvider"], "sess_options": self._get_session_options(True)},
            # 最適化OFF
            {"providers": ["CPUExecutionProvider"], "sess_options": self._get_session_options(False)},
            # 最小設定
            {"providers": ["CPUExecutionProvider"]},
        ]

        for config in session_configs:
            try:
                if "sess_options" in config:
                    session = ort.InferenceSession(model_path, **config)
                else:
                    session = ort.InferenceSession(model_path, providers=config["providers"])

                # 動作テスト
                self._test_session(session)
                return session
            except Exception:
                continue

        raise RuntimeError("セッション作成に失敗")

    def _get_session_options(self, optimize: bool) -> ort.SessionOptions:
        """セッションオプション作成"""
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4

        if optimize:
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        else:
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        return options

    def _test_session(self, session: ort.InferenceSession) -> None:
        """セッション動作テスト"""
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        input_name = session.get_inputs()[0].name
        session.run(None, {input_name: dummy_input})

    def _create_preprocess(self) -> transforms.Compose:
        """前処理パイプライン作成"""
        config = Config.load()

        return transforms.Compose(
            [
                transforms.Resize(config["resize"], interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(config["center_crop"]),
                transforms.ToTensor(),
                transforms.Normalize(mean=config["mean"], std=config["std"]),
            ]
        )

    def _load_image(self, image_path: str) -> Image.Image:
        """画像読み込み"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"画像なし: {image_path}")
        return Image.open(image_path).convert("RGB")

    def _extract_features(self, image_path: str) -> np.ndarray:
        """特徴量抽出"""
        image = self._load_image(image_path)

        # 前処理
        tensor = self.preprocess(image).unsqueeze(0)
        array = tensor.numpy().astype(np.float32)

        # 推論
        input_name = self.session.get_inputs()[0].name
        features = self.session.run(None, {input_name: array})[0]

        # L2正規化
        norm = np.linalg.norm(features, axis=1, keepdims=True)
        return features / (norm + 1e-8)

    def register_uniform(self, image_path: str, name: str = "staff_uniform") -> None:
        """制服登録"""
        print(f"制服登録: {image_path}")
        self.uniform_features = self._extract_features(image_path)
        self.uniform_name = name
        print(f"登録完了: {name}")

    def detect_staff(self, image_path: str) -> Dict[str, Any]:
        """スタッフ判定"""
        if self.uniform_features is None:
            raise ValueError("制服未登録")

        features = self._extract_features(image_path)
        similarity = np.dot(features, self.uniform_features.T).item()

        return {
            "is_staff": similarity > self.threshold,
            "similarity": similarity,
            "uniform_name": self.uniform_name,
            "threshold": self.threshold,
        }

    def set_threshold(self, threshold: float) -> None:
        """閾値設定"""
        self.threshold = threshold
        print(f"閾値: {threshold}")


def benchmark(uniform_path: str = "uniform.jpg", iterations: int = 10) -> None:
    """性能比較"""
    if not os.path.exists(uniform_path):
        print(f"❌ テスト画像なし: {uniform_path}")
        return

    print("=== 性能比較 ===")

    models = [("FP32", "clip_fp32.onnx"), ("INT8", "clip_int8.onnx")]
    results = {}

    for name, path in models:
        if not os.path.exists(path):
            continue

        print(f"\n{name}テスト中...")
        detector = StaffDetector(path)
        detector.register_uniform(uniform_path)

        # 性能測定
        times = []
        for i in range(iterations):
            start = time.time()
            result = detector.detect_staff(uniform_path)
            times.append(time.time() - start)

            if i == 0:
                results[name] = result

        avg_time = np.mean(times) * 1000
        print(f"平均時間: {avg_time:.2f}ms")
        print(f"類似度: {result['similarity']:.6f}")

    # 比較表示
    if len(results) >= 2:
        fp32_sim = results.get("FP32", {}).get("similarity", 0)
        int8_sim = results.get("INT8", {}).get("similarity", 0)
        print(f"\n精度差: {abs(fp32_sim - int8_sim):.6f}")


def export_cpp_info() -> None:
    """C++移植情報出力"""
    info = {
        "model": {"path": "clip_int8.onnx", "input_shape": [1, 3, 224, 224], "output_shape": [1, 1024]},
        "preprocess": {"resize": 224, "normalize_mean": [0.481, 0.458, 0.408], "normalize_std": [0.269, 0.261, 0.276]},
        "inference": {"provider": "CPUExecutionProvider", "l2_normalize": True, "similarity": "cosine"},
    }

    with open("cpp_export_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print("C++情報出力: cpp_export_info.json")


def demo() -> None:
    """簡単デモ"""
    try:
        detector = StaffDetector()

        uniform_path = input("制服画像: ").strip()
        if uniform_path and os.path.exists(uniform_path):
            detector.register_uniform(uniform_path)

            person_path = input("判定画像: ").strip()
            if person_path and os.path.exists(person_path):
                result = detector.detect_staff(person_path)
                status = "スタッフ" if result["is_staff"] else "非スタッフ"
                print(f"結果: {status} (類似度: {result['similarity']:.4f})")
    except Exception as e:
        print(f"エラー: {e}")


def main():
    """メイン関数"""
    commands = {
        "1": ("INT8変換", lambda: ModelConverter.convert()),
        "2": ("デモ実行", demo),
        "3": ("性能比較", benchmark),
        "4": ("C++情報", export_cpp_info),
        "5": ("終了", lambda: None),
    }

    print("=== 量子化CLIP スタッフ判定 ===")

    while True:
        print("\nメニュー:")
        for key, (desc, _) in commands.items():
            print(f"{key}. {desc}")

        choice = input("選択: ").strip()

        if choice in commands:
            if choice == "5":
                break
            commands[choice][1]()
        else:
            print("❌ 無効な選択")


if __name__ == "__main__":
    main()
