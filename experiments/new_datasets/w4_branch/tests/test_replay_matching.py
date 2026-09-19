"""Offline regression tests for the w4_branch mitm proxy's replay matching.

Root cause (see AUDIT.md section 9): the proxy used to decide whether to
serve a cached title-generation reply purely by arrival order (the first
HTTP request received), not by request content. On a branch launch the
real main-task request often arrives first (title-gen does not always
fire on a reused session_id) and was incorrectly served the cached title
reply, causing Claude Code's engine to treat the title text as the whole
task's answer and stop after one turn.

These tests exercise the real Proxy/Handler classes over real local TCP
sockets (no Docker, no network egress, no experimental datasets touched).
A tiny local HTTP server stands in for the downstream relay so "live
forward" requests have somewhere deterministic to land.
"""
from __future__ import annotations

import http.client
import json
import socketserver
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler
from pathlib import Path

W4 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(W4))
import run_prospective_dataset_b as m  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TITLE_BODY = json.loads((FIXTURES / "title_gen_request.json").read_text())
MAIN_BODY = json.loads((FIXTURES / "main_task_request.json").read_text())


class _FakeRelay(BaseHTTPRequestHandler):
    """Stand-in for the real 18010/18011 relay: records what it receives."""

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("content-length", 0))
        body = self.rfile.read(length)
        self.server.received.append(json.loads(body))  # type: ignore[attr-defined]
        payload = b'{"type":"message","stop_reason":"end_turn","content":[]}'
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # silence
        pass


def _start_fake_relay() -> tuple[socketserver.TCPServer, int]:
    srv = socketserver.TCPServer(("127.0.0.1", 0), _FakeRelay)
    srv.received = []  # type: ignore[attr-defined]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def _post(port: int, body: dict) -> tuple[int, bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    payload = json.dumps(body).encode()
    conn.request("POST", "/v1/messages?beta=true", body=payload,
                 headers={"content-type": "application/json", "content-length": str(len(payload))})
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, data


class TitleGenMatcherUnitTests(unittest.TestCase):
    def test_title_gen_request_detected(self):
        self.assertTrue(m._is_title_gen_request(TITLE_BODY))

    def test_main_task_request_not_detected_as_title(self):
        self.assertFalse(m._is_title_gen_request(MAIN_BODY))

    def test_missing_system_field_does_not_crash(self):
        self.assertFalse(m._is_title_gen_request({"messages": []}))

    def test_string_system_field_handled(self):
        self.assertTrue(m._is_title_gen_request({"system": "You are naming a coding session, go."}))
        self.assertFalse(m._is_title_gen_request({"system": "You are a helpful assistant."}))


class ReplayRoutingIntegrationTests(unittest.TestCase):
    """Drives the real Proxy/Handler over real sockets for the three cases
    called out explicitly: title-before-main, main-before-title, no-title.
    """

    def _make_proxy(self, relay_port: int, replay_bytes: bytes) -> m.Proxy:
        bp = m.Proxy(("127.0.0.1", 0), m.Handler)
        bp.relay = relay_port
        bp.requests = []
        bp.boundary = threading.Event()
        bp.release = threading.Event()
        bp.held = threading.Event()
        bp.hold_boundary = False
        bp.replay_diagnostics = []
        bp.replay_response = replay_bytes
        threading.Thread(target=bp.serve_forever, daemon=True).start()
        return bp

    def setUp(self):
        self.relay, self.relay_port = _start_fake_relay()
        cached_title_reply = b'{"type":"message","stop_reason":"end_turn","content":[{"type":"text","text":"cached title"}]}'
        self.bp = self._make_proxy(self.relay_port, cached_title_reply)
        self.cached_title_reply = cached_title_reply

    def tearDown(self):
        self.bp.shutdown(); self.bp.server_close()
        self.relay.shutdown(); self.relay.server_close()

    def test_title_before_main(self):
        # Title-gen arrives first: gets the cached reply, never reaches the relay.
        status, data = _post(self.bp.server_address[1], TITLE_BODY)
        self.assertEqual(status, 200)
        self.assertEqual(data, self.cached_title_reply)
        self.assertEqual(len(self.relay.received), 0)
        self.assertTrue(self.bp.requests[0].get("replay_matched"))

        # Main task arrives second: does NOT get the cached title reply,
        # gets forwarded live to the relay instead.
        status2, data2 = _post(self.bp.server_address[1], MAIN_BODY)
        self.assertEqual(status2, 200)
        self.assertNotEqual(data2, self.cached_title_reply)
        self.assertEqual(len(self.relay.received), 1)
        self.assertEqual(self.relay.received[0]["messages"][0]["content"][0]["text"][:20],
                          MAIN_BODY["messages"][0]["content"][0]["text"][:20])

    def test_main_before_title(self):
        # This is the exact regression: the real main-task request arrives
        # FIRST (no preceding title-gen call -- confirmed to happen on
        # reused branch session_ids). It must NOT be served the cached
        # title reply.
        status, data = _post(self.bp.server_address[1], MAIN_BODY)
        self.assertEqual(status, 200)
        self.assertNotEqual(data, self.cached_title_reply,
                             "regression: main-task request was served the cached title reply")
        self.assertEqual(len(self.relay.received), 1,
                          "main-task request must be forwarded live, not swallowed by replay")
        self.assertEqual(len(self.bp.replay_diagnostics), 1)
        self.assertIn("not title-gen-shaped", self.bp.replay_diagnostics[0]["reason"])

        # A genuine title-gen call arriving afterward still gets served
        # correctly (replay is one-shot on content match, not position).
        status2, data2 = _post(self.bp.server_address[1], TITLE_BODY)
        self.assertEqual(status2, 200)
        self.assertEqual(data2, self.cached_title_reply)
        self.assertEqual(len(self.relay.received), 1)  # unchanged: title-gen never forwarded

    def test_no_title_ever(self):
        # Only the main task ever arrives; no title-gen call happens at all
        # for this session. Must be forwarded live on the very first request.
        status, data = _post(self.bp.server_address[1], MAIN_BODY)
        self.assertEqual(status, 200)
        self.assertNotEqual(data, self.cached_title_reply)
        self.assertEqual(len(self.relay.received), 1)


if __name__ == "__main__":
    unittest.main()
