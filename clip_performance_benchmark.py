import os
import time

import numpy as np
import onnxruntime as ort


class CLIPPerformanceBenchmark:
    """CLIPモデルの推論性能測定ツール"""

    def __init__(self):
        self.model_configs = {
            "RN50": {"input_size": 224, "features": 1024},
            "RN101": {"input_size": 224, "features": 512},
            "RN50x4": {"input_size": 288, "features": 640},
            "RN50x16": {"input_size": 384, "features": 768},
            "RN50x64": {"input_size": 448, "features": 1024},
            "ViT-B_32": {"input_size": 224, "features": 512},
            "ViT-B_16": {"input_size": 224, "features": 512},
            "ViT-L_14": {"input_size": 224, "features": 768},
            "ViT-L_14_336px": {"input_size": 336, "features": 768},
        }

    def benchmark_model(self, model_path, model_name, num_iterations=10, warmup=1):
        """単一モデルのベンチマーク実行"""
        config = self.model_configs.get(model_name, {"input_size": 224})
        input_size = config["input_size"]

        # セッション作成
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name

        # ダミー画像作成（正規化済み）
        dummy_image = np.random.randn(1, 3, input_size, input_size).astype(np.float32)

        # ウォームアップ
        for _ in range(warmup):
            session.run(None, {input_name: dummy_image})

        # 推論時間測定
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            session.run(None, {input_name: dummy_image})
            end = time.perf_counter()
            times.append(end - start)

        # 統計計算
        times = np.array(times) * 1000  # ms変換
        mean_time = np.mean(times)
        std_time = np.std(times)
        fps = 1000 / mean_time

        return {"mean_ms": mean_time, "std_ms": std_time, "fps": fps, "input_size": input_size}

    def benchmark_all_models(self, model_dir="onnx_models"):
        """全モデルのベンチマーク実行"""
        print(f"{'Model':<20} {'Input':<10} {'Time (ms)':<15} {'FPS':<10}")
        print("-" * 60)

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
                        f"{result['mean_ms']:.2f} ± {result['std_ms']:.2f}  "
                        f"{result['fps']:.1f}"
                    )

                    results[model_name] = result
                except Exception as e:
                    print(f"{model_name:<20} Error: {e}")
            else:
                print(f"{model_name:<20} Model not found")

        return results

    def save_results(self, results, output_file="benchmark_results.txt"):
        """結果をファイルに保存"""
        with open(output_file, "w") as f:
            f.write("CLIP Model Performance Benchmark Results\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"{'Model':<20} {'Input Size':<12} {'Mean (ms)':<12} {'FPS':<10}\n")
            f.write("-" * 60 + "\n")

            for model, result in sorted(results.items()):
                f.write(
                    f"{model:<20} "
                    f"{result['input_size']}x{result['input_size']:<9} "
                    f"{result['mean_ms']:>8.2f}     "
                    f"{result['fps']:>8.1f}\n"
                )

            # 最速/最遅モデル
            if results:
                fastest = min(results.items(), key=lambda x: x[1]["mean_ms"])
                slowest = max(results.items(), key=lambda x: x[1]["mean_ms"])

                f.write("\n" + "=" * 60 + "\n")
                f.write(f"Fastest: {fastest[0]} ({fastest[1]['mean_ms']:.2f} ms)\n")
                f.write(f"Slowest: {slowest[0]} ({slowest[1]['mean_ms']:.2f} ms)\n")

        print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    benchmark = CLIPPerformanceBenchmark()
    results = benchmark.benchmark_all_models()
    benchmark.save_results(results)
