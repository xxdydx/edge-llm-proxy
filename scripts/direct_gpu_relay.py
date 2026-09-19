#!/usr/bin/env python3
"""Expose remote vLLM locally through SSH command execution.

FlowMesh's sshd disables TCP forwarding, and Lumid's gateway may not provide an
SCP subsystem. Each accepted TCP connection therefore runs the repository's
remote ``stdio_relay.py`` through a normal SSH command. OpenSSH multiplexing
reuses one authenticated transport across HTTP connections.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import threading
from pathlib import Path


def pump_socket_to_process(conn: socket.socket, proc: subprocess.Popen[bytes]) -> None:
    assert proc.stdin is not None
    try:
        while data := conn.recv(65536):
            proc.stdin.write(data)
            proc.stdin.flush()
    except (BrokenPipeError, ConnectionError, OSError):
        pass
    finally:
        try:
            proc.stdin.close()
        except OSError:
            pass


def pump_process_to_socket(conn: socket.socket, proc: subprocess.Popen[bytes]) -> None:
    assert proc.stdout is not None
    try:
        # BufferedIOReader.read(size) may wait for ``size`` bytes or EOF. HTTP
        # keep-alive deliberately provides neither after a small response, so
        # use the descriptor directly and forward each available chunk.
        while data := os.read(proc.stdout.fileno(), 65536):
            conn.sendall(data)
    except (BrokenPipeError, ConnectionError, OSError):
        pass


def handle_connection(
    conn: socket.socket,
    ssh_command: list[str],
    error_log: Path,
) -> None:
    with conn, error_log.open("ab", buffering=0) as stderr:
        proc = subprocess.Popen(
            ssh_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
        )
        inbound = threading.Thread(
            target=pump_socket_to_process, args=(conn, proc), daemon=True
        )
        inbound.start()
        pump_process_to_socket(conn, proc)
        inbound.join(timeout=2)
        if proc.poll() is None:
            proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", required=True, type=int)
    parser.add_argument("--ssh-target", required=True)
    parser.add_argument("--ssh-port", required=True, type=int)
    parser.add_argument("--remote-relay", required=True)
    parser.add_argument("--remote-port", default=8001, type=int)
    parser.add_argument(
        "--error-log", default="/tmp/flowmesh-direct-gpu-relay-ssh.log"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    control_path = f"/tmp/flowmesh-direct-{os.getuid()}-{args.remote_port}-%C"
    ssh_command = [
        "ssh",
        "-T",
        "-p",
        str(args.ssh_port),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=12",
        "-o",
        "ControlMaster=auto",
        "-o",
        "ControlPersist=600",
        "-o",
        f"ControlPath={control_path}",
        args.ssh_target,
        "python3",
        args.remote_relay,
        str(args.remote_port),
    ]
    error_log = Path(args.error_log)
    stop = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((args.listen_host, args.listen_port))
        listener.listen(64)
        listener.settimeout(1)
        print(
            f"listening on {args.listen_host}:{args.listen_port} -> "
            f"{args.ssh_target}:localhost:{args.remote_port}",
            flush=True,
        )
        while not stop.is_set():
            try:
                conn, _ = listener.accept()
            except socket.timeout:
                continue
            threading.Thread(
                target=handle_connection,
                args=(conn, ssh_command, error_log),
                daemon=True,
            ).start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
