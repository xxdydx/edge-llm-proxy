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
#
# python3 and jq are here because agents reach for them constantly and cannot
# install anything themselves. A system python3 is deliberate even though
# /opt/venv has one: the venv belongs to the proxy (and later vLLM), so an
# agent scripting a one-liner should not be able to disturb it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        jq \
        less \
        python3 \
        rsync \
        tmux \
        vim-tiny \
        xz-utils \
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

# Node, from the official tarball rather than apt — Debian's package trails
# several major versions, and agents that shell out to node/npx expect current
# behaviour. Resolves the newest LTS at build time; jq is available by now.
RUN v="$(curl -fsSL https://nodejs.org/dist/index.json \
           | jq -r '[.[] | select(.lts != false)][0].version')" \
    && echo "node $v" \
    && curl -fsSL "https://nodejs.org/dist/$v/node-$v-linux-x64.tar.xz" \
       | tar -xJ -C /usr/local --strip-components=1 \
             --exclude=CHANGELOG.md --exclude=LICENSE --exclude=README.md \
    && node --version && npm --version

# Claude Code — the harness under test.
#
# Fetched directly rather than via claude.ai/install.sh. That installer
# downloads a Bun standalone executable and then *runs* it to self-install,
# which segfaults under the QEMU emulation this cross-build uses ("qemu:
# uncaught target signal 11") — and it cleans up the download on failure, so
# there is nothing to salvage. The binary needs no installation beyond being
# on PATH and executable, so download it and skip the self-install entirely.
RUN v="$(curl -fsSL https://downloads.claude.ai/claude-code-releases/latest)" \
    && echo "claude code $v" \
    && curl -fsSL -o /usr/local/bin/claude \
         "https://downloads.claude.ai/claude-code-releases/$v/linux-x64/claude" \
    && chmod a+rx /usr/local/bin/claude \
    && test -s /usr/local/bin/claude

ENV PATH=/home/flowmesh/.local/bin:$PATH

# The VS Code CLI, so tunnels do not re-download it every session. Remote-SSH
# cannot work here (the entrypoint sets AllowTcpForwarding no at container
# start), so tunnels are the only route in.
RUN curl -fsSL "https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64" \
      -o /tmp/vscode-cli.tgz \
    && tar -xf /tmp/vscode-cli.tgz -C /usr/local/bin \
    && chmod a+rx /usr/local/bin/code \
    && rm -f /tmp/vscode-cli.tgz

# Dockerfile ENV does not reach an SSH session: sshd builds a fresh PATH for
# login shells and PAM reads /etc/environment, neither of which sees the ENV
# above. Without this, /opt/venv/bin is invisible and `python3` resolves only
# to the apt one — the symptom that showed up as "python3: command not found"
# before python3 was installed at all.
RUN touch /etc/environment \
    && sed -i '/^PATH=/d' /etc/environment \
    && printf 'PATH="%s"\n' "$PATH" >> /etc/environment \
    && printf 'export PATH="%s"\n' "$PATH" > /etc/profile.d/10-edge-llm.sh \
    && chmod a+rx /etc/profile.d/10-edge-llm.sh

# The session user (created by the entrypoint at container start, UID unknown
# at build time — see above) needs to write here: bootstrap.sh installs vLLM
# into this venv at runtime. World-writable is fine — this is a single-tenant
# disposable dev container, not a shared multi-user image.
RUN chmod -R a+rwX /opt/venv /opt/python

# Do NOT set ENTRYPOINT, CMD, or a non-root USER — the base image's sshd
# entrypoint must run as root to provision the session user and bind sshd, or
# the session will never come up.
