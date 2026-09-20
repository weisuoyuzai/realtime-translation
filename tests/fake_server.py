"""A tiny OpenAI-compatible server for tests: chat completions (SSE) and audio transcriptions."""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeOpenAI:
    def __init__(self):
        self.chat_requests: list[dict] = []
        self.asr_requests: list[bytes] = []
        # behaviours ---------------------------------------------------------
        self.reject_temperature = False          # 400 when "temperature" is sent
        self.reject_extra = False                # 400 when any unknown param is sent
        self.reject_verbose_json = False         # ASR: 400 unless response_format=json
        self.chat_status = 200
        self.asr_status = 200
        self.reply = lambda user_text, n: f"译:{user_text}"     # n = request counter
        self.chunk_delay = 0.0
        self.models = ['m-b', 'm-a']
        self.models_status = 200
        self.models_queries: list[str] = []
        self.asr_reply = {"text": "hello world", "language": "english", "segments": [
            {"text": "hello world", "avg_logprob": -0.2, "no_speech_prob": 0.01, "compression_ratio": 1.1}]}
        self._httpd: ThreadingHTTPServer | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_address[1]}/v1"

    def start(self) -> "FakeOpenAI":
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path.split("?")[0].endswith("/models"):
                    outer.models_queries.append(self.path.partition("?")[2])
                    if outer.models_status != 200:
                        return self._json(outer.models_status, {"error": "nope"})
                    self._json(200, {"data": [{"id": m} for m in outer.models]})
                else:
                    self._json(404, {})

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                if self.path.endswith("/chat/completions"):
                    return self._chat(json.loads(body))
                if self.path.endswith("/audio/transcriptions"):
                    return self._asr(body)
                self._json(404, {})

            def _chat(self, req):
                outer.chat_requests.append(req)
                if outer.chat_status != 200:
                    return self._json(outer.chat_status, {"error": {"message": "boom"}})
                known = {"model", "messages", "stream", "temperature", "max_tokens", "max_completion_tokens"}
                if outer.reject_temperature and "temperature" in req:
                    return self._json(400, {"error": {"message": "Unsupported value: 'temperature' does not support 0"}})
                if outer.reject_extra and set(req) - known:
                    return self._json(400, {"error": {"message": f"Unrecognized request argument: {sorted(set(req) - known)}"}})
                text = outer.reply(req["messages"][-1]["content"], len(outer.chat_requests))
                if not req.get("stream"):
                    return self._json(200, {"choices": [{"message": {"content": text}}]})
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                try:
                    step = max(1, len(text) // 3)
                    for i in range(0, len(text), step):
                        chunk = {"choices": [{"delta": {"content": text[i:i + step]}}]}
                        self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
                        self.wfile.flush()
                        if outer.chunk_delay:
                            time.sleep(outer.chunk_delay)
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass                                    # client cancelled mid-stream

            def _asr(self, body):
                outer.asr_requests.append(body)
                if outer.asr_status != 200:
                    return self._json(outer.asr_status, {"error": {"message": "nope"}})
                if outer.reject_verbose_json and b"verbose_json" in body:
                    return self._json(400, {"error": {"message": "response_format 'verbose_json' is not compatible"}})
                self._json(200, outer.asr_reply)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()
