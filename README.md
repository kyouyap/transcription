# Transcription

## 概要
Transcriptionは、音声認識とテキスト変換を行うためのプロジェクトです。

## バージョン
0.1.0

## 必要な環境
- Python >= 3.12.0
- Apple Silicon Mac

## インストール
以下のコマンドを実行して依存関係をインストールします。

```bash
uv sync
```

## 依存関係
- platformdirs >= 4.3.6
- pyannote-audio >= 3.3.2
- python-dotenv >= 1.0.1
- pywhispercpp
- streamlit >= 1.41.1
- torch >= 2.5.1
- torchaudio >= 2.5.1
- torchvision >= 0.20.1
- yarl == 1.13.1

## 開発用依存関係
- mypy >= 1.14.1
- ruff >= 0.9.2

## 使用方法
以下のコマンドを実行してアプリケーションを起動します。

```bash
streamlit run src/ui.py
```

## プロジェクト構成
```
.transcription/
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
├── .github/
├── .streamlit/
├── assets/
│   ├── media/
│   │   └── memo.txt
│   └── texts/
│       └── memo.txt
├── src/
│   ├── __init__.py
│   ├── ui.py
│   └── libs/
│       ├── __init__.py
│       └── transcription.py
```

## ライセンス
このプロジェクトはMITライセンスの下で提供されています。
