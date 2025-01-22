"""音声/動画ファイルの文字起こしと話者分離を行うStreamlitアプリ（Markdown表示版）"""

import datetime
import logging
from pathlib import Path
from typing import Final

import streamlit as st

from libs.transcription import Transcription

# 環境設定
logging.basicConfig(level=logging.INFO)

# 定数
HUGGING_FACE_TOKEN: Final[str] = st.secrets["HUGGING_FACE_TOKEN"]
MEDIA_DIR = Path("./assets/media")
EXPORT_DIR = Path("./assets/texts")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# ページ設定
st.set_page_config(
    page_title="AI文字起こしツール",
    page_icon="🎙️",
    layout="centered",
)


def save_uploaded_file(
    uploaded_file: st.runtime.uploaded_file_manager.UploadedFile,
) -> Path | None:
    """アップロードされたファイルを保存"""
    media_path = MEDIA_DIR / uploaded_file.name
    if not media_path.suffix:
        return None
    with media_path.open("wb") as f:
        f.write(uploaded_file.getbuffer())
    return media_path if media_path.exists() else None


def process_file(media_path: Path) -> Transcription:
    """ファイル処理のメインロジック"""
    transcription = Transcription(str(media_path))

    with st.status("🚀 ファイル処理を開始...", expanded=True) as status:
        # 動画→音声変換
        if transcription.is_video():
            st.write("🎥 動画を音声に変換中...")
            transcription.convert_video_to_audio()

        # 音声→mp3変換
        if not transcription.is_mp3():
            st.write("🔁 音声をMP3に変換中...")
            transcription.convert_audio_to_mp3()

        # 文字起こし
        st.write("📝 文字起こしを実行中...")
        transcription.transcribe_audio("large-v3-turbo")
        # 話者分離
        st.write("👥 話者分離を実行中...")
        transcription.diarize_audio(HUGGING_FACE_TOKEN)

        # 結果マージ
        st.write("🔄 結果を統合中...")
        transcription.merge_results()

        status.update(label="✅ 処理完了!", state="complete", expanded=False)

    return transcription


def display_results(transcription: Transcription, prefix: str):
    """結果をMarkdownで表示・ダウンロード"""
    md_file = EXPORT_DIR / f"{prefix}_result.md"
    transcription.export_results_to_md(md_file)

    st.subheader("📝 文字起こし結果")
    with md_file.open(encoding="utf-8") as f:
        st.markdown(f.read())

    st.download_button(
        label="📥 Markdownをダウンロード",
        data=md_file.read_text(),
        file_name=md_file.name,
        mime="text/markdown",
    )


def main():
    """メインのUIレイアウト"""
    st.title("🎙️ AI文字起こしツール")
    st.markdown("""
    ### 使い方
    1. 音声/動画ファイルをアップロード
    2. 処理開始ボタンをクリック
    3. 結果を画面で確認・ダウンロード
    """)

    uploaded_file = st.file_uploader(
        "音声/動画ファイルをアップロード",
        type=["mp3", "wav", "mp4", "mov", "avi", "m4a"],
        help="対応形式: mp3, wav, mp4, mov, avi, m4a",
    )

    if uploaded_file and st.button("▶️ 処理を開始"):
        try:
            media_path = save_uploaded_file(uploaded_file)
            if media_path is None:
                st.error("無効なファイル形式です。")
                return
            transcription = process_file(media_path)

            prefix = datetime.datetime.now(tz=datetime.timezone.utc).strftime(  # noqa: UP017
                "%Y%m%d-%H%M%S",
            )
            display_results(transcription, prefix)

            if transcription.is_video():
                Path(transcription.media_file_path).unlink(missing_ok=True)

        except Exception as e:
            st.error(f"エラーが発生しました: {e!s}")
            logging.exception("Error occurred")


if __name__ == "__main__":
    main()
