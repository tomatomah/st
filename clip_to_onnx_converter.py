import os

import clip
import onnx
import torch

# モデル別の入力サイズ定義
INPUT_SIZES = {"RN50x4": 288, "RN50x16": 384, "RN50x64": 448, "@336px": 336, "336px": 336}


def get_input_size(model_name):
    """モデル名から適切な入力サイズを取得"""
    for key, size in INPUT_SIZES.items():
        if key in model_name:
            return size
    return 224


def convert_clip_to_onnx(model_name, output_dir="onnx_models"):
    """CLIPモデルをONNX形式に変換"""
    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)

    # モデル読み込み
    model, _ = clip.load(model_name, device="cpu", jit=False)
    model.visual.eval()

    # 入力設定
    input_size = get_input_size(model_name)
    dummy_input = torch.randn(1, 3, input_size, input_size)

    # 特徴量次元数を取得
    with torch.no_grad():
        features = model.visual(dummy_input)
        feature_dim = features.shape[1]

    # 出力パス生成
    safe_name = model_name.replace("/", "_").replace("@", "_")
    output_path = os.path.join(output_dir, f"clip_{safe_name}.onnx")

    # ONNX変換
    torch.onnx.export(
        model.visual,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        input_names=["image"],
        output_names=["features"],
        dynamic_axes={"image": {0: "batch_size"}, "features": {0: "batch_size"}},
        do_constant_folding=True,
    )

    # 検証
    onnx.checker.check_model(onnx.load(output_path))

    # ファイルサイズ取得
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    return {"path": output_path, "size_mb": file_size_mb, "input_size": input_size, "feature_dim": feature_dim}


def convert_all_models():
    """利用可能な全CLIPモデルを変換"""
    models = ["RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"]

    print(f"Converting {len(models)} CLIP models to ONNX format\n")
    print(f"{'Model':<15} {'Input':<10} {'Features':<10} {'Size':<12} {'Status':<13}")
    print("-" * 60)

    results = []

    for model_name in models:
        try:
            info = convert_clip_to_onnx(model_name)
            size_str = f"{info['size_mb']:.1f} MB"
            input_str = f"{info['input_size']}x{info['input_size']}"
            print(f"{model_name:<15} {input_str:<10} {info['feature_dim']:<10} {size_str:<12} {'OK':<13}")
            results.append((model_name, True, info))
        except Exception as e:
            print(f"{model_name:<15} {'--':<10} {'--':<10} {'--':<12} {'FAILED':<13}")
            results.append((model_name, False, None))

    # 成功数のみ表示
    success_count = sum(1 for _, status, _ in results if status)
    print(f"\nCompleted: {success_count}/{len(models)} models")

    return results


if __name__ == "__main__":
    convert_all_models()
