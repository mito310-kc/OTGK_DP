# 下記から任意のイメージを選んで利用
# https://docs.astral.sh/uv/guides/integration/docker/#available-images
# TODO: 複数のバージョンのPythonを導入
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

SHELL ["/bin/bash", "-c"]

ENV TZ=JST-9

RUN apt-get update \
    && apt-get install -y --no-install-recommends --no-install-suggests \
    openssh-server \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ユーザー設定
ARG USERID
ARG USERNAME
ARG GROUPID
ARG GROUPNAME
RUN groupadd -g ${GROUPID} ${GROUPNAME} \
    && useradd -g ${GROUPID} -u ${USERID} -s /bin/bash -m ${USERNAME} \
    && usermod -aG root ${USERNAME}
USER ${USERNAME}

ENV PATH="/home/${USERNAME}/.local/bin:$PATH"

# キャッシュ設定は、既定値が Docker だと失敗するため公式の推奨に従って変更。下記参照:
# https://docs.astral.sh/uv/guides/integration/docker/#caching
ENV UV_LINK_MODE=copy
