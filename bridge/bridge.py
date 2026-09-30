#!/usr/bin/env python3
"""Muse bridge — Railway variant.

Same OpenAI-compatible endpoint as the local bridge
(/v1/models, /v1/chat/completions, /health), but built to run as a
standalone Railway service:

- Queue dir comes from QUEUE_DIR env (default ./queue).
- Listens on 0.0.0.0:$PORT (Railway injects PORT).
- Extra INTERNAL endpoints so the Muse poller (running elsewhere) can
  drain the queue over HTTPS without filesystem access:
    GET  /internal/pending          -> {"pending": [{id, received_at, request}, ...]}
    POST /internal/answer           -> {"id": rid, "content": "..."} writes done/<rid>.json
                                       and removes pending/<rid>.json (mirrors the
                                       local poller's done-then-delete order)
    DELETE /internal/pending/<rid>  -> drop one stale pending request
  All /internal/* endpoints require:  Authorization: Bearer <BRIDGE_TOKEN>
  (compared with hmac.compare_digest; fail-closed when BRIDGE_TOKEN is unset).

v3 base: threaded server, per-connection socket timeout, BrokenPipe-safe
sends, immediate SSE headers with keepalive comments.
"""
import hmac
import json
import os
import re
import time
import uuid
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BASE = os.environ.get("QUEUE_DIR", "./queue")
PENDING = os.path.join(BASE, "pending")
DONE = os.path.join(BASE, "done")
TOKEN = os.environ.get("BRIDGE_TOKEN", "")
PORT = int(os.environ.get("PORT", "8765"))
MAX_PENDING = 5
WAIT_SECS = 240
KEEPALIVE_SECS = 15
MAX_BODY = 8 * 1024 * 1024  # internal answer payload cap

RID_RE = re.compile(r"^[0-9a-f]{32}$")


def _is_dashboard_probe(req):
    """Detect 9Router dashboard's 'Test Connection' probe.

    The dashboard sends {max_tokens:1024, stream:false,
    messages:[..., {role:"user", content:"hi"}]} with a 15s client timeout,
    which the 60s+ poller loop can never meet. Matching requests are answered
    instantly below; everything else goes through the normal queue.
    """
    try:
        if not isinstance(req, dict) or req.get("stream"):
            return False
        if req.get("max_tokens") != 1024:
            return False
        msgs = req.get("messages") or []
        if not msgs:
            return False
        last = msgs[-1]
        return (isinstance(last, dict) and last.get("role") == "user"
                and str(last.get("content", "")).strip().lower() == "hi")
    except Exception:
        return False


def _probe_completion():
    cid = "chatcmpl-" + uuid.uuid4().hex[:12]
    return {"id": cid, "object": "chat.completion", "created": int(time.time()),
            "model": "muse",
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content":
                                     "Halo! Muse online — bridge 9Router aktif dan siap."},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}


def _auth_ok(handler):
    if not TOKEN:
        return False
    got = handler.headers.get("Authorization", "")
    if not got.startswith("Bearer "):
        return False
    return hmac.compare_digest(got[7:].strip(), TOKEN)


class H(BaseHTTPRequestHandler):
    timeout = 120  # per-socket-op timeout: slow-loris bodies can't wedge a thread forever

    def log_message(self, *a):
        pass

    def _send(self, code, obj, ctype="application/json"):
        try:
            body = obj.encode() if isinstance(obj, str) else json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            pass  # client went away; nothing to do

    def _sse_write(self, payload: str) -> bool:
        """Write one SSE payload, flushed. Returns False if the client is gone."""
        try:
            self.wfile.write(payload.encode())
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            return False

    def _cleanup(self, rid: str):
        for d in (os.path.join(PENDING, rid + ".json"), os.path.join(DONE, rid + ".json")):
            try:
                os.remove(d)
            except Exception:
                pass

    def _read_json_body(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            return None
        if length <= 0 or length > MAX_BODY:
            return None
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            return None

    # ---- internal (poller) endpoints ----

    def _handle_internal_get(self):
        if self.path.rstrip("/") == "/internal/pending":
            items = []
            try:
                for fn in sorted(os.listdir(PENDING)):
                    if not fn.endswith(".json"):
                        continue
                    try:
                        with open(os.path.join(PENDING, fn)) as f:
                            items.append(json.load(f))
                    except Exception:
                        continue
            except Exception:
                pass
            # oldest first
            items.sort(key=lambda x: x.get("received_at", 0))
            return self._send(200, {"pending": items})
        return self._send(404, {"error": "not found"})

    def _handle_internal_post(self):
        if self.path.rstrip("/") == "/internal/answer":
            body = self._read_json_body()
            if not isinstance(body, dict):
                return self._send(400, {"error": {"message": "bad json"}})
            rid = body.get("id", "")
            content = body.get("content", "")
            if not isinstance(rid, str) or not RID_RE.match(rid):
                return self._send(400, {"error": {"message": "bad id"}})
            if not isinstance(content, str):
                return self._send(400, {"error": {"message": "bad content"}})
            try:
                with open(os.path.join(DONE, rid + ".json"), "w") as f:
                    json.dump({"content": content}, f, ensure_ascii=False)
                try:
                    os.remove(os.path.join(PENDING, rid + ".json"))
                except Exception:
                    pass
            except Exception as e:
                return self._send(500, {"error": {"message": str(e)[:200]}})
            return self._send(200, {"ok": True})
        return self._send(404, {"error": "not found"})

    def _handle_internal_delete(self):
        # DELETE /internal/pending/<rid>
        parts = self.path.rstrip("/").split("/")
        if len(parts) == 4 and parts[1] == "internal" and parts[2] == "pending" \
                and RID_RE.match(parts[3] or ""):
            try:
                os.remove(os.path.join(PENDING, parts[3] + ".json"))
            except Exception:
                pass
            return self._send(200, {"ok": True})
        return self._send(404, {"error": "not found"})

    # ---- public endpoints ----

    def do_GET(self):
        try:
            if self.path.startswith("/internal/"):
                if not _auth_ok(self):
                    return self._send(403, {"error": "forbidden"})
                return self._handle_internal_get()
            if self.path.rstrip("/") == "/v1/models":
                self._send(200, {"object": "list", "data": [
                    {"id": "muse", "object": "model", "created": 0, "owned_by": "muse"},
                    {"id": "muse-fast", "object": "model", "created": 0, "owned_by": "muse"}]})
            elif self.path == "/health":
                self._send(200, {"ok": True})
            else:
                self._send(404, {"error": "not found"})
        except Exception as e:
            try:
                self._send(500, {"error": {"message": str(e)[:200]}})
            except Exception:
                pass

    def do_POST(self):
        try:
            if self.path.startswith("/internal/"):
                if not _auth_ok(self):
                    return self._send(403, {"error": "forbidden"})
                return self._handle_internal_post()
            if self.path.rstrip("/") != "/v1/chat/completions":
                return self._send(404, {"error": "not found"})
            length = int(self.headers.get("Content-Length", 0))
            try:
                req = json.loads(self.rfile.read(length) or b"{}")
            except Exception:
                return self._send(400, {"error": {"message": "bad json"}})
            if _is_dashboard_probe(req):
                return self._send(200, _probe_completion())
            try:
                npend = len(os.listdir(PENDING))
            except Exception:
                npend = 0
            if npend >= MAX_PENDING:
                return self._send(429, {"error": {"message": "Muse bridge busy, try again in a bit"}})
            rid = uuid.uuid4().hex
            with open(os.path.join(PENDING, rid + ".json"), "w") as f:
                json.dump({"id": rid, "received_at": time.time(), "request": req}, f)

            if req.get("stream"):
                return self._handle_stream(req, rid)
            answer = self._wait_answer(rid)
            self._cleanup(rid)
            if answer is None:
                return self._send(504, {"error": {"message": "Muse did not answer in time"}})
            cid = "chatcmpl-" + uuid.uuid4().hex[:12]
            created = int(time.time())
            resp = {"id": cid, "object": "chat.completion", "created": created, "model": "muse",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": answer},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
            self._send(200, resp)
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            pass
        except Exception as e:
            try:
                self._send(500, {"error": {"message": str(e)[:200]}})
            except Exception:
                pass

    def do_DELETE(self):
        try:
            if self.path.startswith("/internal/"):
                if not _auth_ok(self):
                    return self._send(403, {"error": "forbidden"})
                return self._handle_internal_delete()
            return self._send(404, {"error": "not found"})
        except Exception as e:
            try:
                self._send(500, {"error": {"message": str(e)[:200]}})
            except Exception:
                pass

    def _wait_answer(self, rid: str, keepalive_cb=None):
        """Wait up to WAIT_SECS for done/<rid>.json. keepalive_cb() is called
        every KEEPALIVE_SECS; if it returns False the client is gone."""
        deadline = time.time() + WAIT_SECS
        answer = None
        last_ping = time.time()
        while time.time() < deadline:
            dp = os.path.join(DONE, rid + ".json")
            if os.path.exists(dp):
                try:
                    with open(dp) as f:
                        answer = json.load(f).get("content", "")
                except Exception:
                    answer = ""
                try:
                    os.remove(dp)
                except Exception:
                    pass
                break
            if keepalive_cb and time.time() - last_ping >= KEEPALIVE_SECS:
                if not keepalive_cb():
                    break  # client disconnected
                last_ping = time.time()
            time.sleep(1)
        return answer

    def _handle_stream(self, req, rid: str):
        # Headers go out immediately; body chunks follow when the answer lands.
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            self._cleanup(rid)
            return
        if not self._sse_write(": connected\n\n"):
            self._cleanup(rid)
            return
        answer = self._wait_answer(rid, keepalive_cb=lambda: self._sse_write(": ping\n\n"))
        self._cleanup(rid)
        cid = "chatcmpl-" + uuid.uuid4().hex[:12]
        created = int(time.time())
        if answer is None:
            err = {"error": {"message": "Muse did not answer in time", "type": "timeout"}}
            self._sse_write("data: " + json.dumps(err) + "\n\ndata: [DONE]\n\n")
            return
        chunk1 = {"id": cid, "object": "chat.completion.chunk", "created": created,
                  "model": "muse", "choices": [{"index": 0,
                  "delta": {"role": "assistant", "content": answer}, "finish_reason": None}]}
        chunk2 = {"id": cid, "object": "chat.completion.chunk", "created": created,
                  "model": "muse", "choices": [{"index": 0, "delta": {},
                  "finish_reason": "stop"}]}
        self._sse_write("data: " + json.dumps(chunk1) + "\n\n")
        self._sse_write("data: " + json.dumps(chunk2) + "\n\ndata: [DONE]\n\n")


if __name__ == "__main__":
    os.makedirs(PENDING, exist_ok=True)
    os.makedirs(DONE, exist_ok=True)
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), H)
    srv.daemon_threads = True
    srv.allow_reuse_address = True
    print(f"muse-bridge (railway) listening on 0.0.0.0:{PORT}, queue={BASE}, "
          f"internal-auth={'on' if TOKEN else 'OFF (set BRIDGE_TOKEN!)'}", flush=True)
    srv.serve_forever()
