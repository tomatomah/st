import json
import os
import shutil
import zipfile
from urllib.parse import urlparse

import requests


class COCOExtractor:
    def __init__(self, data_dir="./coco_data", output_dir="./filtered_images"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.images_dir = os.path.join(data_dir, "images")
        self.annotations_dir = os.path.join(data_dir, "annotations")

        # ディレクトリ作成
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.annotations_dir, exist_ok=True)

    def download_file(self, url, save_path):
        """ファイルをダウンロード"""
        print(f"Downloading {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded: {save_path}")

    def extract_zip(self, zip_path, extract_to):
        """ZIPファイルを解凍"""
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extracted to: {extract_to}")

    def download_coco_data(self, dataset="val2017"):
        """MS COCOデータセットをダウンロード"""
        # 画像とアノテーションのURL
        base_url = "http://images.cocodataset.org"

        if dataset == "val2017":
            images_url = f"{base_url}/zips/val2017.zip"
            annotations_url = f"{base_url}/annotations/annotations_trainval2017.zip"
        elif dataset == "train2017":
            images_url = f"{base_url}/zips/train2017.zip"
            annotations_url = f"{base_url}/annotations/annotations_trainval2017.zip"
        else:
            raise ValueError("サポートされていないデータセット")

        # 画像ダウンロード
        images_zip = os.path.join(self.data_dir, f"{dataset}.zip")
        if not os.path.exists(images_zip):
            self.download_file(images_url, images_zip)

        # アノテーションダウンロード
        annotations_zip = os.path.join(self.data_dir, "annotations.zip")
        if not os.path.exists(annotations_zip):
            self.download_file(annotations_url, annotations_zip)

        # 解凍
        if not os.path.exists(os.path.join(self.images_dir, dataset)):
            self.extract_zip(images_zip, self.images_dir)

        if not os.path.exists(os.path.join(self.annotations_dir, "annotations")):
            self.extract_zip(annotations_zip, self.annotations_dir)
            # アノテーションファイルを移動
            annotations_subdir = os.path.join(self.annotations_dir, "annotations")
            if os.path.exists(annotations_subdir):
                for file in os.listdir(annotations_subdir):
                    shutil.move(os.path.join(annotations_subdir, file), os.path.join(self.annotations_dir, file))
                os.rmdir(annotations_subdir)

    def load_annotations(self, dataset="val2017"):
        """アノテーションファイルを読み込み"""
        annotation_file = os.path.join(self.annotations_dir, f"instances_{dataset}.json")

        if not os.path.exists(annotation_file):
            raise FileNotFoundError(f"アノテーションファイルが見つかりません: {annotation_file}")

        with open(annotation_file, "r") as f:
            return json.load(f)

    def get_category_id(self, annotations, target_label):
        """ラベル名からカテゴリIDを取得"""
        for category in annotations["categories"]:
            if category["name"].lower() == target_label.lower():
                return category["id"]

        # 利用可能なカテゴリを表示
        available_categories = [cat["name"] for cat in annotations["categories"]]
        raise ValueError(f"ラベル '{target_label}' が見つかりません。利用可能なカテゴリ: {available_categories}")

    def extract_images_with_label(self, target_label, dataset="val2017"):
        """指定されたラベルが含まれる画像を抽出"""
        print(f"ラベル '{target_label}' が含まれる画像を抽出中...")

        # アノテーション読み込み
        annotations = self.load_annotations(dataset)

        # カテゴリID取得
        category_id = self.get_category_id(annotations, target_label)
        print(f"カテゴリID {category_id} を検索")

        # 対象カテゴリのアノテーションを持つ画像IDを取得
        target_image_ids = set()
        for annotation in annotations["annotations"]:
            if annotation["category_id"] == category_id:
                target_image_ids.add(annotation["image_id"])

        print(f"対象画像数: {len(target_image_ids)}")

        # 画像ファイル名とIDのマッピングを作成
        image_id_to_filename = {}
        for image_info in annotations["images"]:
            image_id_to_filename[image_info["id"]] = image_info["file_name"]

        # ラベル専用の出力ディレクトリを作成
        label_output_dir = os.path.join(self.output_dir, target_label)
        os.makedirs(label_output_dir, exist_ok=True)

        # 対象画像をコピー
        copied_count = 0
        images_source_dir = os.path.join(self.images_dir, dataset)

        for image_id in target_image_ids:
            if image_id in image_id_to_filename:
                filename = image_id_to_filename[image_id]
                source_path = os.path.join(images_source_dir, filename)
                dest_path = os.path.join(label_output_dir, filename)

                if os.path.exists(source_path):
                    shutil.copy2(source_path, dest_path)
                    copied_count += 1
                    if copied_count % 100 == 0:
                        print(f"コピー済み: {copied_count}枚")

        print(f"抽出完了: {copied_count}枚の画像を {label_output_dir} に保存しました")
        return copied_count


def main():
    """メイン関数"""
    # 使用例
    extractor = COCOExtractor()

    # データセットをダウンロード（初回のみ）
    try:
        extractor.download_coco_data("val2017")  # val2017またはtrain2017
    except Exception as e:
        print(f"ダウンロードエラー: {e}")
        return

    # 特定のラベルが含まれる画像を抽出
    target_label = "person"  # ここで抽出したいラベルを指定

    try:
        count = extractor.extract_images_with_label(target_label, "val2017")
        print(f"\n抽出結果: {count}枚の'{target_label}'画像を取得しました")
    except Exception as e:
        print(f"抽出エラー: {e}")


if __name__ == "__main__":
    main()
