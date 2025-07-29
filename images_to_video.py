import glob
import os

import cv2


def create_video_from_images(image_dir, output_path="output_video.mp4", fps=30):
    """画像群から動画を作成

    Args:
        image_dir: 画像が格納されたディレクトリ
        output_path: 出力動画ファイルパス
        fps: フレームレート
    """
    # 画像ファイル取得（ソート済み）
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    if not image_files:
        print(f"画像が見つかりません: {image_dir}")
        return

    # 最初の画像から動画サイズを取得
    first_image = cv2.imread(image_files[0])
    height, width = first_image.shape[:2]

    # 動画ライター設定
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 画像を動画に書き込み
    for i, image_file in enumerate(image_files):
        img = cv2.imread(image_file)
        if img is None:
            print(f"読み込みエラー: {image_file}")
            continue

        # サイズが異なる場合はリサイズ
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))

        out.write(img)
        print(f"処理中: {i+1}/{len(image_files)}", end="\r")

    # 終了処理
    out.release()
    print(f"\n完了: {output_path} ({len(image_files)}フレーム)")


if __name__ == "__main__":
    # 使用例
    create_video_from_images("data/output_frames", "result_video.mp4", fps=30)
