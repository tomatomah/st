import os

import cv2


def extract_frames(video_path, output_dir="frames", interval=30):
    """動画からフレームを抽出して保存

    Args:
        video_path: 動画ファイルのパス
        output_dir: 出力ディレクトリ
        interval: 抽出間隔（フレーム数）
    """
    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)

    # 動画読み込み
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"動画を開けません: {video_path}")
        return

    # 動画情報取得
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"FPS: {fps}, 総フレーム数: {total_frames}")

    # フレーム抽出
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 指定間隔でフレーム保存
        if frame_count % interval == 0:
            filename = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1
            print(f"保存: {filename}")

        frame_count += 1

    # 終了処理
    cap.release()
    print(f"\n完了: {saved_count}枚のフレームを保存しました")


if __name__ == "__main__":
    # 使用例
    extract_frames("data/sample.mp4", "data/output_frames", interval=30)  # 30フレームごとに保存
