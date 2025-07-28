import os

import numpy as np
from onnxruntime.quantization import QuantType, quantize_dynamic


class CLIPModelQuantizer:
    """CLIPモデルのINT8量子化ツール"""

    def __init__(self):
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

    def quantize_model(self, input_path, output_dir="onnx_models_quantized"):
        """ONNXモデルをINT8に量子化"""
        os.makedirs(output_dir, exist_ok=True)

        # 出力パス生成
        model_name = os.path.basename(input_path).replace(".onnx", "")
        output_path = os.path.join(output_dir, f"{model_name}_int8.onnx")

        try:
            # 動的量子化実行（optimize_modelパラメータを削除）
            quantize_dynamic(input_path, output_path, weight_type=QuantType.QInt8)

            # サイズ比較
            original_size = os.path.getsize(input_path) / (1024 * 1024)
            quantized_size = os.path.getsize(output_path) / (1024 * 1024)
            reduction = (1 - quantized_size / original_size) * 100

            return {
                "success": True,
                "output_path": output_path,
                "original_size_mb": original_size,
                "quantized_size_mb": quantized_size,
                "reduction_percent": reduction,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def quantize_all_models(self, model_dir="onnx_models"):
        """全CLIPモデルを量子化"""
        print(f"{'Model':<20} {'Original':<12} {'Quantized':<12} {'Reduction':<10}")
        print("-" * 60)

        quantized_models = []

        for model_name in self.model_configs.keys():
            safe_name = model_name.replace("/", "_").replace("@", "_")
            input_path = os.path.join(model_dir, f"clip_{safe_name}.onnx")

            if os.path.exists(input_path):
                result = self.quantize_model(input_path)

                if result["success"]:
                    print(
                        f"{model_name:<20} "
                        f"{result['original_size_mb']:>8.1f} MB  "
                        f"{result['quantized_size_mb']:>8.1f} MB  "
                        f"{result['reduction_percent']:>7.1f}%"
                    )
                    quantized_models.append((input_path, result["output_path"], model_name))
                else:
                    print(f"{model_name:<20} Error: {result['error']}")
            else:
                print(f"{model_name:<20} Model not found")

        return quantized_models

    def validate_quantized_model(self, original_path, quantized_path, model_name=None, num_samples=10):
        """量子化前後の出力差を検証"""
        import onnxruntime as ort

        # ファイル存在確認
        if not os.path.exists(quantized_path):
            print(f"量子化モデルが見つかりません: {quantized_path}")
            return None

        # 入力サイズを取得
        if model_name:
            input_size = self.model_configs.get(model_name, 224)
        else:
            # モデル名を推測
            for name, size in self.model_configs.items():
                safe_name = name.replace("/", "_").replace("@", "_")
                if safe_name in original_path:
                    input_size = size
                    break
            else:
                input_size = 224

        try:
            # セッション作成
            original_session = ort.InferenceSession(original_path)
            quantized_session = ort.InferenceSession(quantized_path)

            # 入力名を取得
            input_name = original_session.get_inputs()[0].name

            # ダミーデータで検証
            differences = []
            for _ in range(num_samples):
                dummy_input = np.random.randn(1, 3, input_size, input_size).astype(np.float32)

                # 推論実行
                orig_output = original_session.run(None, {input_name: dummy_input})[0]
                quant_output = quantized_session.run(None, {input_name: dummy_input})[0]

                # コサイン類似度計算
                orig_norm = orig_output / (np.linalg.norm(orig_output) + 1e-8)
                quant_norm = quant_output / (np.linalg.norm(quant_output) + 1e-8)
                similarity = np.dot(orig_norm.flatten(), quant_norm.flatten())
                differences.append(similarity)

            avg_similarity = np.mean(differences)
            print(f"量子化前後の平均類似度: {avg_similarity:.4f}")

            return avg_similarity

        except Exception as e:
            print(f"検証エラー: {e}")
            return None


if __name__ == "__main__":
    quantizer = CLIPModelQuantizer()

    # 全モデルを量子化
    print("CLIPモデルのINT8量子化を開始します...\n")
    quantized_models = quantizer.quantize_all_models()

    # 量子化されたモデルの検証
    if quantized_models:
        print("\n\n=== 量子化精度の検証 ===")
        for original_path, quantized_path, model_name in quantized_models[:1]:  # 最初の1つだけ検証
            print(f"\nモデル: {model_name}")
            quantizer.validate_quantized_model(original_path, quantized_path, model_name)
