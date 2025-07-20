import cv2
import numpy as np
import onnxruntime as ort


class CLIPSimilarityDetector:
    """CLIP-ONNXを使用した画像類似度検出器"""

    def __init__(self, model_path, input_size=224):
        self.input_size = input_size
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

        # ImageNet正規化パラメータ（CHW形式用にブロードキャスト済み）
        self.mean = np.array([[[0.48145466]], [[0.4578275]], [[0.40821073]]], dtype=np.float32)
        self.std = np.array([[[0.26862954]], [[0.26130258]], [[0.27577711]]], dtype=np.float32)

    def preprocess(self, image_path):
        """画像を前処理してモデル入力形式に変換"""
        # 画像読み込み・変換
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_CUBIC)

        # 正規化とチャンネル順変更
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC → CHW
        img = (img - self.mean) / self.std
        img = np.expand_dims(img, axis=0)  # CHW → NCHW

        return img

    def extract_features(self, image_path):
        """画像から正規化済み特徴量を抽出"""
        img = self.preprocess(image_path)
        features = self.session.run(None, {self.input_name: img})[0][0]
        return features / np.linalg.norm(features)

    def calculate_similarity(self, reference_path, target_path):
        """登録画像と対象画像の類似度を計算"""
        reference_feat = self.extract_features(reference_path)
        target_feat = self.extract_features(target_path)
        return float(np.dot(reference_feat, target_feat))


if __name__ == "__main__":
    reference_img = "sample/staff1.jpg"
    target_img = "sample/staff2.jpg"
    detector = CLIPSimilarityDetector("onnx_models/clip_RN50.onnx")
    similarity = detector.calculate_similarity(reference_img, target_img)
    print(f"similarity: {similarity:.3f}")
