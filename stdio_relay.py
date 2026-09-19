#!/usr/bin/env python3
"""Relay stdin/stdout <-> a local TCP socket, select-based. Used as the
remote end of an SSH-command-exec-based tunnel (no AllowTcpForwarding
needed, since this is just a normal remote command, not a forwarded
channel).

Keeps relaying socket->stdout after stdin hits EOF (our write side closing
doesn't mean the server is done responding) - only exits once the socket
itself closes.

See claude-memory/wiki/system/SSH command-exec relay bridges Docker on Mac
to remote vLLM.md for the local-side `socat TCP-LISTEN:18001,fork` invocation.
"""
import os
import select
import socket
import sys

HOST, PORT = "localhost", int(sys.argv[1]) if len(sys.argv) > 1 else 8001

with open("/tmp/stdio_relay.log", "a") as log:
    sock = socket.create_connection((HOST, PORT))
    stdin_fd = sys.stdin.fileno()
    stdout_fd = sys.stdout.fileno()
    stdin_open = True
    while True:
        read_fds = [sock] + ([stdin_fd] if stdin_open else [])
        # A large local model can legitimately spend more than 60 seconds in
        # prefill/thinking without emitting a byte. Do not interpret silence
        # as a dead connection; socket EOF remains the authoritative exit.
        r, _, _ = select.select(read_fds, [], [])
        if stdin_fd in r:
            data = os.read(stdin_fd, 65536)
            if not data:
                stdin_open = False
            else:
                sock.sendall(data)
        if sock in r:
            data = sock.recv(65536)
            if not data:
                break
            os.write(stdout_fd, data)
    sock.close()
