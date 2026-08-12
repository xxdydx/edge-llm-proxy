FROM ghcr.io/mlsys-io/flowmesh_ssh:v0.1.7-gpu

# Stay root for the whole build, and don't switch USER at the end either. The
# base image's own entrypoint (/flowmesh-ssh-session.sh) creates the session
# user at container *start*, not build time — it useradd's "flowmesh" at
# SSH_UID/SSH_GID (default 10001), writes authorized_keys, then execs sshd.
# All of that needs root, so the image must still be root when the container
# starts, or the entrypoint itself fails.
USER root

# Everything that was missing on the stock image, plus the basics for working
# in a disposable container. build-essential (gcc/g++/make/libc headers) is
# required at *runtime*, not just for pip builds: Triton JIT-compiles a small
# C extension the first time any Triton kernel runs, even with vLLM's
# --enforce-eager, and the session user has no sudo/apt to install it later.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        less \
        rsync \
        tmux \
        vim-tiny \
    && rm -rf /var/lib/apt/lists/*

# uv ships as a static binary — copying it from the official image avoids
# needing curl or a package manager. Pin a tag once a version is known good.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Scratch is wiped between sessions; these live in the image instead.
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/bin:$PATH

# Python plus the proxy's own dependencies. These are small, stable, and needed
# every session — unlike vLLM, they are worth baking in.
RUN uv python install 3.12 \
    && uv venv /opt/venv --python 3.12 \
    && uv pip install --python /opt/venv/bin/python \
        fastapi 'uvicorn[standard]' httpx

# Claude Code is NOT baked in. Its installer runs the downloaded amd64 binary
# to verify itself, which segfaults (exit 139) under the QEMU emulation this
# cross-build uses. bootstrap.sh installs it at runtime instead, where it runs
# on real amd64 hardware. Put the eventual location on PATH regardless.
ENV PATH=/home/flowmesh/.local/bin:$PATH

# The VS Code CLI, so tunnels do not re-download it every session. Remote-SSH
# cannot work here (the entrypoint sets AllowTcpForwarding no at container
# start), so tunnels are the only route in.
RUN curl -fsSL "https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64" \
      -o /tmp/vscode-cli.tgz \
    && tar -xf /tmp/vscode-cli.tgz -C /usr/local/bin \
    && chmod a+rx /usr/local/bin/code \
    && rm -f /tmp/vscode-cli.tgz

# The session user (created by the entrypoint at container start, UID unknown
# at build time — see above) needs to write here: bootstrap.sh installs vLLM
# into this venv at runtime. World-writable is fine — this is a single-tenant
# disposable dev container, not a shared multi-user image.
RUN chmod -R a+rwX /opt/venv /opt/python

# Do NOT set ENTRYPOINT, CMD, or a non-root USER — the base image's sshd
# entrypoint must run as root to provision the session user and bind sshd, or
# the session will never come up.
