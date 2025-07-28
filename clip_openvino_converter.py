import os

import cv2
import numpy as np
from openvino.runtime import Core
from openvino.tools import mo


class CLIPOpenVINOConverter:
    """CLIP ONNXモデルのOpenVINO変換・推論ツール"""

    def __init__(self):
        self.core = Core()
        self.model_configs = {
            "RN50": 224,
            "RN101": 224,
            "RN50x4": 288,
            "RN50x16": 384,
            "RN50x64": 448,
            "ViT-B_32": 224,
            "ViT-B_16": 224,
            "ViT-L_14": 224,
            "ViT-L_14_336px": 336,
        }

    def convert_to_openvino(self, onnx_path, output_dir="openvino_models"):
        """ONNXモデルをOpenVINO形式に変換"""
        os.makedirs(output_dir, exist_ok=True)

        model_name = os.path.basename(onnx_path).replace(".onnx", "")
        output_path = os.path.join(output_dir, model_name)

        # モデル名から入力サイズを推定
        input_size = 224
        for name, size in self.model_configs.items():
            if name.replace("/", "_").replace("@", "_") in model_name:
                input_size = size
                break

        try:
            # Model Optimizerで変換（入力形状を明示的に指定）
            mo_command = f"mo --input_model {onnx_path} --output_dir {output_dir} --model_name {model_name} --input_shape [1,3,{input_size},{input_size}]"
            os.system(mo_command)

            # 変換されたファイルの確認
            xml_path = f"{output_path}.xml"
            bin_path = f"{output_path}.bin"

            if os.path.exists(xml_path) and os.path.exists(bin_path):
                return {"success": True, "xml_path": xml_path, "bin_path": bin_path, "input_size": input_size}
            else:
                return {"success": False, "error": "Conversion failed"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def test_inference(self, xml_path, test_image_path=None):
        """OpenVINOモデルでテスト推論"""
        # モデル読み込み
        model = self.core.read_model(xml_path)
        compiled_model = self.core.compile_model(model, "CPU")

        # モデル名から入力サイズを推定
        model_name = os.path.basename(xml_path)
        input_size = 224
        for name, size in self.model_configs.items():
            if name.replace("/", "_").replace("@", "_") in model_name:
                input_size = size
                break

        # 固定の入力形状を使用
        n, c, h, w = 1, 3, input_size, input_size

        # テスト画像準備
        if test_image_path and os.path.exists(test_image_path):
            # 実画像を使用
            img = cv2.imread(test_image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
            img = img.astype(np.float32) / 255.0

            # ImageNet正規化
            mean = np.array([0.48145466, 0.4578275, 0.40821073])
            std = np.array([0.26862954, 0.26130258, 0.27577711])
            img = (img - mean) / std

            # CHW形式に変換
            img = img.transpose(2, 0, 1)
            img = np.expand_dims(img, axis=0)
        else:
            # ダミーデータ使用
            img = np.random.randn(n, c, h, w).astype(np.float32)

        # 推論実行
        infer_request = compiled_model.create_infer_request()
        result = infer_request.infer({0: img})

        # 出力取得
        output = list(result.values())[0]
        features = output[0]
        norm_features = features / (np.linalg.norm(features) + 1e-8)

        return {
            "input_shape": (n, c, h, w),
            "output_shape": output.shape,
            "feature_dim": len(features),
            "feature_norm": np.linalg.norm(features),
            "sample_features": norm_features[:5],  # 最初の5要素
        }

    def compare_with_original(self, original_onnx_path, xml_path, num_iterations=100):
        """元のONNXモデルとOpenVINOモデルの比較"""
        import time

        import onnxruntime as ort

        # 入力サイズ取得
        input_size = 224
        for name, size in self.model_configs.items():
            if name.replace("/", "_").replace("@", "_") in original_onnx_path:
                input_size = size
                break

        # ダミーデータ
        dummy_input = np.random.randn(1, 3, input_size, input_size).astype(np.float32)

        print(f"\n性能比較 (入力サイズ: {input_size}x{input_size}):")

        # 元のONNXモデル（量子化前）で推論
        if os.path.exists(original_onnx_path):
            ort_session = ort.InferenceSession(original_onnx_path)

            # ウォームアップ
            for _ in range(10):
                ort_session.run(None, {ort_session.get_inputs()[0].name: dummy_input})

            ort_times = []
            for _ in range(num_iterations):
                start = time.perf_counter()
                ort_output = ort_session.run(None, {ort_session.get_inputs()[0].name: dummy_input})[0]
                ort_times.append(time.perf_counter() - start)

            ort_avg = np.mean(ort_times) * 1000
            print(f"元のONNXモデル:     {ort_avg:.2f} ms")
        else:
            print(f"元のONNXモデルが見つかりません: {original_onnx_path}")
            ort_avg = None
            ort_output = None

        # OpenVINO INT8モデルで推論
        model = self.core.read_model(xml_path)
        compiled_model = self.core.compile_model(model, "CPU")
        infer_request = compiled_model.create_infer_request()

        # ウォームアップ
        for _ in range(10):
            infer_request.infer({0: dummy_input})

        ov_times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            result = infer_request.infer({0: dummy_input})
            ov_times.append(time.perf_counter() - start)

        ov_output = list(result.values())[0]
        ov_avg = np.mean(ov_times) * 1000
        print(f"OpenVINO INT8:      {ov_avg:.2f} ms")

        if ort_avg:
            speedup = ort_avg / ov_avg
            print(f"高速化率:           {speedup:.2f}x")

        # 精度比較（同じ入力での出力差）
        if ort_output is not None:
            ort_norm = ort_output[0] / (np.linalg.norm(ort_output[0]) + 1e-8)
            ov_norm = ov_output[0] / (np.linalg.norm(ov_output[0]) + 1e-8)
            similarity = np.dot(ort_norm, ov_norm)
            print(f"\n出力の類似度:       {similarity:.4f}")


if __name__ == "__main__":
    converter = CLIPOpenVINOConverter()

    # 量子化モデルをOpenVINOに変換
    quantized_model = "onnx_models_quantized/clip_RN50_int8.onnx"
    original_model = "onnx_models/clip_RN50.onnx"  # 元のモデルパス

    if os.path.exists(quantized_model):
        print("OpenVINO形式への変換を開始...")
        result = converter.convert_to_openvino(quantized_model)

        if result["success"]:
            print(f"変換成功: {result['xml_path']}")

            # テスト推論
            print("\nテスト推論を実行...")
            inference_result = converter.test_inference(result["xml_path"])

            print(f"入力形状: {inference_result['input_shape']}")
            print(f"出力形状: {inference_result['output_shape']}")
            print(f"特徴次元: {inference_result['feature_dim']}")
            print(f"特徴量サンプル: {inference_result['sample_features']}")

            # 元のモデルとの比較
            converter.compare_with_original(original_model, result["xml_path"])
        else:
            print(f"変換失敗: {result['error']}")
    else:
        print(f"モデルが見つかりません: {quantized_model}")
