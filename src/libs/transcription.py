"""メディアファイル（音声または動画）から文字起こしや話者分離を行うクラス。

このモジュールは、メディアファイル（音声または動画）から文字起こしや話者分離を行うクラスを提供します。

Example:
    以下のようにして、文字起こしや話者分離を行うことができます::

        from libs.transcription import Transcription

        # メディアファイルを指定
        transcription = Transcription("sample.mp3")

        # 必要に応じて文字起こし: スキップしたい場合は呼ばない
        # transcription.transcribe_audio("large-v3-turbo")

        # 話者分離だけ行う
        transcription.diarize_audio("HUGGING_FACE_TOKEN")

        # 結果をマージ
        transcription.merge_results()

        # 結果を様々な形式で出力(merged_resultsを使う場合は use_merge=True)
        transcription.export_results_to_csv("result.csv", use_merge=True)
        transcription.export_results_to_json("result.json", use_merge=True)
        transcription.export_results_to_md("result.md", use_merge=True)

"""

import json
import mimetypes
import subprocess
from collections.abc import Iterable
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torchaudio
from pyannote.audio import Pipeline
from pyannote.core import Annotation

from pywhispercpp.model import Model, Segment


class Transcription:
    """メディアファイル（音声または動画）から文字起こしや話者分離を行うクラス。

    Attributes:
        media_file_path (str): メディアファイルのパス
        transcriptions (list[dict[str, Any]]): 文字起こし結果を格納するリスト
        speaker_segments (list[dict[str, Any]]): 話者分離の結果を格納するリスト
        merged_results (list[dict[str, Any]]): 文字起こしと話者分離の結果を結合したリスト

    """

    def __init__(self, media_file_path: str) -> None:
        """クラスの初期化メソッド。

        Args:
            media_file_path (str): メディアファイルのパス

        """
        self.media_file_path: str = media_file_path
        self.transcriptions: list[dict[str, Any]] = []
        self.speaker_segments: list[dict[str, Any]] = []
        self.merged_results: list[dict[str, Any]] = []

    def is_video(self) -> bool:
        """メディアファイルが動画か否かを判定する。

        Returns:
            bool: 動画ファイルであればTrue、そうでなければFalse。

        """
        mime_type = mimetypes.guess_type(self.media_file_path)[0]
        if mime_type is None:
            return False
        return mime_type.startswith("video")

    def is_mp3(self) -> bool:
        """メディアファイルがmp3形式か否かを判定する。

        Returns:
            bool: mp3形式であればTrue、そうでなければFalse。

        """
        mime_type = mimetypes.guess_type(self.media_file_path)[0]
        if mime_type is None:
            return False
        return mime_type == "audio/mpeg"

    def convert_video_to_audio(self) -> None:
        """動画ファイルを音声ファイルに変換し、mp3形式で保存する。"""
        media_path = Path(self.media_file_path)
        output_path = str(media_path.with_suffix(".mp3"))

        command = [
            "ffmpeg",
            "-i",
            str(media_path),
            "-q:a",
            "0",
            "-map",
            "a",
            output_path,
        ]

        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            error_message = f"ffmpeg error: {result.stderr}"
            raise RuntimeError(error_message)

        self.media_file_path = output_path

    def convert_audio_to_mp3(self) -> None:
        """音声ファイルをmp3形式に変換する。"""
        audio_path = Path(self.media_file_path)
        output_path = str(audio_path.with_suffix(".mp3"))

        waveform, sample_rate = torchaudio.load(self.media_file_path)
        torchaudio.save(output_path, waveform, sample_rate)
        self.media_file_path = output_path

    def transcribe_audio(self, whisper_model_size: str = "large-v3-turbo") -> None:
        """音声ファイルを文字起こしする。

        Args:
            whisper_model_size (str):
                Whisperモデルのサイズ（"tiny", "base", "small", "medium", "large", "large-v3-turbo" など）

        """
        model = Model(whisper_model_size, print_realtime=True, print_progress=True)
        segments: Iterable[Segment] = model.transcribe(
            self.media_file_path,
            language="ja",
        )
        self.transcriptions.clear()

        for segment in segments:
            self.transcriptions.append(
                {
                    "start_time": segment.t0 / 100,
                    "end_time": segment.t1 / 100,
                    "text": segment.text,
                },
            )

    def diarize_audio(self, hugging_face_token: str) -> None:
        """音声ファイルを話者分離する。

        モデル利用時にHugging Faceのトークンが必要。
        1. HuggingFace（https://huggingface.co/）のアカウントを作成
        2. `pyannote/speaker-diarization-3.1` への利用申請
        3. アクセストークンを発行（https://huggingface.co/settings/tokens）

        Args:
            hugging_face_token (str): HuggingFaceのアクセストークン

        """
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hugging_face_token,
        )

        diarization: Annotation
        if torch.cuda.is_available():
            waveform, sample_rate = torchaudio.load(self.media_file_path)
            pipeline.to(torch.device("cuda"))
            diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})
        elif torch.backends.mps.is_available():
            waveform, sample_rate = torchaudio.load(self.media_file_path)
            pipeline.to(torch.device("mps"))
            diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})
        else:
            # CPU fallback
            diarization = pipeline(self.media_file_path)

        self.speaker_segments.clear()
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            self.speaker_segments.append(
                {
                    "start_time": segment.start,
                    "end_time": segment.end,
                    "speaker": speaker,
                },
            )

    def merge_results(self) -> None:
        """文字起こしと話者分離の結果を時間軸に基づいて結合する。

        - 文字起こしが行われた場合: 文字起こし + 話者分離を時間軸でマージ
        - 文字起こしをスキップした場合: 話者分離の結果のみを `merged_results` に格納
        """
        self.merged_results.clear()

        # もし transcriptions が空 = 文字起こしがスキップされた(または何も検出されなかった)場合は、
        # 話者分離の結果のみを merged_results に格納
        if not self.transcriptions:
            for sp in self.speaker_segments:
                # 話者分離のみで出力したい形式を定義
                self.merged_results.append(
                    {
                        "start_time": self._format_seconds_to_hhmmss(sp["start_time"]),
                        "end_time": self._format_seconds_to_hhmmss(sp["end_time"]),
                        "speaker": sp["speaker"],
                        "text": "",  # 話者分離だけなので空テキスト
                    },
                )
            return

        # 話者セグメントを開始時間でソート
        sorted_speakers = sorted(self.speaker_segments, key=lambda x: x["start_time"])
        speaker_idx = 0

        for tr in self.transcriptions:
            tr_start = tr["start_time"]
            tr_end = tr["end_time"]
            best_speaker = None
            max_overlap = 0

            # 現在の位置から検索を開始（パフォーマンス改善）
            while speaker_idx < len(sorted_speakers):
                sp = sorted_speakers[speaker_idx]
                sp_start = sp["start_time"]
                sp_end = sp["end_time"]

                # 時間範囲が完全に過去のセグメントはスキップ
                if sp_end <= tr_start:
                    speaker_idx += 1
                    continue

                # 未来のセグメントに到達したら終了
                if sp_start >= tr_end:
                    break

                # オーバーラップ計算
                overlap_start = max(tr_start, sp_start)
                overlap_end = min(tr_end, sp_end)
                overlap = overlap_end - overlap_start

                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = sp["speaker"]

                speaker_idx += 1

            # 検索位置をリセット（次の文字起こしセグメント用）
            speaker_idx = max(0, speaker_idx - 3)  # バッファを持たせて再検索

            self.merged_results.append(
                {
                    "start_time": self._format_seconds_to_hhmmss(tr_start),
                    "end_time": self._format_seconds_to_hhmmss(tr_end),
                    "speaker": best_speaker or "UNKNOWN",
                    "text": tr["text"],
                },
            )

    def _format_seconds_to_hhmmss(self, seconds: float) -> str:
        """秒数をhh:mm:ss形式に変換（ミリ秒切り捨て）"""
        return str(timedelta(seconds=seconds)).split(".")[0]  # ミリ秒部分を除去

    def export_results_to_csv(
        self,
        file_path: str,
        encoding: str = "utf-8",
    ) -> None:
        """結果をCSV形式で保存する。

        空のリストであれば空ファイルを作成する。

        Args:
            file_path (str): 保存先のファイルパス
            encoding (str): 保存時の文字コード

        """
        if not self.merged_results:
            pd.DataFrame(self.transcriptions).to_csv(
                file_path,
                index=False,
                encoding=encoding,
            )
            return

        df = pd.DataFrame(self.merged_results)
        df.to_csv(file_path, index=False, encoding=encoding)

    def export_results_to_json(
        self,
        file_path: str,
        encoding: str = "utf-8",
    ) -> None:
        """結果をJSON形式で保存する。

        空のリストであれば空ファイルを作成する。

        Args:
            file_path (str): 保存先のファイルパス
            encoding (str): 保存時の文字コード

        """
        if not self.transcriptions:
            with Path(file_path).open("w", encoding=encoding) as file:
                json.dump(self.transcriptions, file)
            return

        with Path(file_path).open("w", encoding=encoding) as file:
            json.dump(self.merged_results, file, ensure_ascii=False, indent=4)

    def export_results_to_md(
        self,
        file_path: str,
        encoding: str = "utf-8",
    ) -> None:
        """結果をMarkdown形式で保存する。

        空のリストであれば空ファイルを作成する。

        Args:
            file_path (str): 保存先のファイルパス
            encoding (str): 保存時の文字コード

        """
        if not self.merged_results:
            with Path(file_path).open("w", encoding=encoding) as file:
                file.write("No results found.")
            return

        col_names = list(self.merged_results[0].keys())
        separators = ["---"] * len(col_names)

        header_row = self._format_values_to_md_table_row(col_names)
        separator_row = self._format_values_to_md_table_row(separators)

        rows = [header_row, separator_row]
        for item in self.merged_results:
            values = [str(item[col]) for col in col_names]
            row = self._format_values_to_md_table_row(values)
            rows.append(row)

        with Path(file_path).open("w", encoding=encoding) as file:
            file.write("\n".join(rows))

    def _format_values_to_md_table_row(self, values: list[str]) -> str:
        """内部利用: リストの値をMarkdownテーブル行の形式に変換する。"""
        return f"| {' | '.join(values)} |"
