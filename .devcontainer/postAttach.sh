#!/bin/sh

uv lock
uv venv
uv sync --frozen

# 自作パッケージのインストール。下記参考:
# https://docs.astral.sh/uv/pip/packages/#editable-packages
uv pip install -e .
