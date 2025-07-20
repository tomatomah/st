import cv2
import numpy as np
import onnxruntime as ort


class StaffDetector:
    """CLIPベースのスタッフ検出器（ONNX Runtime使用）"""

    def __init__(self, model_path, input_size=224):
        self.input_size = input_size
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, image_path):
        """CLIPモデル用の画像前処理（OpenCV版）"""
        # 1. 画像を読み込む（OpenCVはBGR形式で読み込む）
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"画像の読み込みに失敗しました: {image_path}")

        # 2. BGRからRGBに変換
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 3. バイキュービック補間でモデル入力サイズにリサイズ
        img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_CUBIC)

        # 4. float32型に変換し、[0, 1]に正規化
        arr = img.astype(np.float32) / 255.0

        # 5. 次元の順序をHWCからCHWに変換（高さ×幅×チャンネル → チャンネル×高さ×幅）
        arr = arr.transpose(2, 0, 1)

        # 6. ImageNet正規化を適用: (pixel - mean) / std
        # ブロードキャスト用に3次元配列として定義
        mean = np.array([[[0.48145466]], [[0.4578275]], [[0.40821073]]], dtype=np.float32)
        std = np.array([[[0.26862954]], [[0.26130258]], [[0.27577711]]], dtype=np.float32)
        arr = (arr - mean) / std

        # 7. バッチ次元を追加: CHW → NCHW（バッチ×チャンネル×高さ×幅）
        arr = np.expand_dims(arr, axis=0)

        return arr

    def get_features(self, image_path):
        """画像からL2正規化された特徴量を抽出"""
        preprocessed = self.preprocess(image_path)
        features = self.session.run(None, {self.input_name: preprocessed})[0][0]
        return features / (np.linalg.norm(features) + 1e-8)

    def compare(self, uniform_path, person_path):
        """制服画像と人物画像のコサイン類似度を計算"""
        uniform_feat = self.get_features(uniform_path)
        person_feat = self.get_features(person_path)
        return np.dot(uniform_feat, person_feat.T)


def main():
    # 設定
    MODEL_PATH = "onnx_models/clip_RN50.onnx"
    UNIFORM_IMAGE = "uniform.jpg"
    PERSON_IMAGE = "staff1.jpg"
    THRESHOLD = 0.6

    # 検出実行
    detector = StaffDetector(MODEL_PATH)
    similarity = detector.compare(UNIFORM_IMAGE, PERSON_IMAGE)
    is_staff = similarity > THRESHOLD

    # 結果出力
    print(f"Similarity: {similarity:.4f}")
    print(f"Result: {'STAFF' if is_staff else 'NOT STAFF'} (threshold: {THRESHOLD})")


if __name__ == "__main__":
    main()
