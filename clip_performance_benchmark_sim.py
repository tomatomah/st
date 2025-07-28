import os
import time

import cv2
import numpy as np
import onnxruntime as ort
from clip_image_similarity import CLIPSimilarityDetector


class CLIPEndToEndBenchmark:
    """CLIP画像処理の全体性能測定ツール"""

    def __init__(self):
        self.model_configs = {
            "RN50": {"input_size": 224},
            "RN101": {"input_size": 224},
            "RN50x4": {"input_size": 288},
            "RN50x16": {"input_size": 384},
            "RN50x64": {"input_size": 448},
            "ViT-B_32": {"input_size": 224},
            "ViT-B_16": {"input_size": 224},
            "ViT-L_14": {"input_size": 224},
            "ViT-L_14_336px": {"input_size": 336},
        }

    def create_dummy_images(self, input_size):
        """テスト用のダミー画像を作成"""
        # ランダムなRGB画像を生成してファイルに保存
        dummy_ref = np.random.randint(0, 255, (input_size, input_size, 3), dtype=np.uint8)
        dummy_target = np.random.randint(0, 255, (input_size, input_size, 3), dtype=np.uint8)

        cv2.imwrite("temp_ref.jpg", dummy_ref)
        cv2.imwrite("temp_target.jpg", dummy_target)

        return "temp_ref.jpg", "temp_target.jpg"

    def benchmark_model(self, model_path, model_name, num_iterations=100, warmup=10):
        """単一モデルの全体処理時間を測定"""
        config = self.model_configs.get(model_name, {"input_size": 224})
        input_size = config["input_size"]

        # CLIPSimilarityDetector初期化
        detector = CLIPSimilarityDetector(model_path, input_size)

        # ダミー画像作成
        ref_path, target_path = self.create_dummy_images(input_size)

        # ウォームアップ
        for _ in range(warmup):
            detector.calculate_similarity(ref_path, target_path)

        # 各処理段階の時間測定
        preprocess_times = []
        extract_times = []
        similarity_times = []
        total_times = []

        for _ in range(num_iterations):
            # 全体時間
            total_start = time.perf_counter()

            # 前処理時間（登録画像）
            pre_start = time.perf_counter()
            ref_preprocessed = detector.preprocess(ref_path)
            pre_end = time.perf_counter()
            preprocess_times.append((pre_end - pre_start) * 1000)

            # 特徴抽出時間（登録画像）
            ext_start = time.perf_counter()
            ref_features = detector.session.run(None, {detector.input_name: ref_preprocessed})[0][0]
            ref_features = ref_features / np.linalg.norm(ref_features)
            ext_end = time.perf_counter()
            extract_times.append((ext_end - ext_start) * 1000)

            # 対象画像の処理と類似度計算
            sim_start = time.perf_counter()
            target_features = detector.extract_features(target_path)
            similarity = float(np.dot(ref_features, target_features))
            sim_end = time.perf_counter()
            similarity_times.append((sim_end - sim_start) * 1000)

            total_end = time.perf_counter()
            total_times.append((total_end - total_start) * 1000)

        # 一時ファイル削除
        os.remove(ref_path)
        os.remove(target_path)

        # 統計計算
        return {
            "preprocess_ms": np.mean(preprocess_times),
            "extract_ms": np.mean(extract_times),
            "similarity_ms": np.mean(similarity_times),
            "total_ms": np.mean(total_times),
            "fps": 1000 / np.mean(total_times),
            "input_size": input_size,
        }

    def benchmark_all_models(self, model_dir="onnx_models"):
        """全モデルの性能測定"""
        print(
            f"{'Model':<20} {'Input':<10} {'前処理':<10} {'特徴抽出':<10} {'類似度':<10} {'合計(ms)':<12} {'FPS':<8}"
        )
        print("-" * 90)

        results = {}

        for model_name, config in self.model_configs.items():
            # モデルパス生成
            safe_name = model_name.replace("/", "_").replace("@", "_")
            model_path = os.path.join(model_dir, f"clip_{safe_name}.onnx")

            if os.path.exists(model_path):
                try:
                    result = self.benchmark_model(model_path, model_name)

                    print(
                        f"{model_name:<20} "
                        f"{result['input_size']}x{result['input_size']:<6} "
                        f"{result['preprocess_ms']:>6.2f}    "
                        f"{result['extract_ms']:>8.2f}    "
                        f"{result['similarity_ms']:>7.2f}   "
                        f"{result['total_ms']:>9.2f}   "
                        f"{result['fps']:>6.1f}"
                    )

                    results[model_name] = result
                except Exception as e:
                    print(f"{model_name:<20} Error: {e}")
            else:
                print(f"{model_name:<20} Model not found")

        return results

    def save_results(self, results, output_file="benchmark_e2e_results.txt"):
        """結果をファイルに保存"""
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("CLIP画像処理性能ベンチマーク結果\n")
            f.write("（画像入力→類似度計算までの全処理時間）\n")
            f.write("=" * 90 + "\n\n")

            f.write(
                f"{'モデル':<20} {'入力サイズ':<12} {'前処理(ms)':<12} {'特徴抽出(ms)':<14} {'類似度(ms)':<12} {'合計(ms)':<10} {'FPS':<8}\n"
            )
            f.write("-" * 90 + "\n")

            for model, result in sorted(results.items()):
                f.write(
                    f"{model:<20} "
                    f"{result['input_size']}x{result['input_size']:<9} "
                    f"{result['preprocess_ms']:>8.2f}     "
                    f"{result['extract_ms']:>10.2f}     "
                    f"{result['similarity_ms']:>10.2f}    "
                    f"{result['total_ms']:>8.2f}   "
                    f"{result['fps']:>6.1f}\n"
                )

            # 処理段階別の割合
            f.write("\n" + "=" * 90 + "\n")
            f.write("処理段階別の時間割合\n")
            f.write("-" * 90 + "\n")

            for model, result in sorted(results.items()):
                total = result["total_ms"]
                pre_pct = (result["preprocess_ms"] / total) * 100
                ext_pct = (result["extract_ms"] / total) * 100
                sim_pct = (result["similarity_ms"] / total) * 100

                f.write(
                    f"{model:<20} "
                    f"前処理: {pre_pct:>5.1f}%  "
                    f"特徴抽出: {ext_pct:>5.1f}%  "
                    f"類似度計算: {sim_pct:>5.1f}%\n"
                )

        print(f"\n結果を保存: {output_file}")


if __name__ == "__main__":
    benchmark = CLIPEndToEndBenchmark()
    results = benchmark.benchmark_all_models()
    benchmark.save_results(results)
