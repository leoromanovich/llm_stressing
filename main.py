import json
import os
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from locust import HttpUser, task, events, constant


# ---- CLI options -----------------------------------------------------------
@events.init_command_line_parser.add_listener
def _(parser):
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/chat/completions",
        help="OpenAI-compatible endpoint path (e.g., /v1/chat/completions or /v1/completions)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("MODEL", "gpt-3.5-turbo"),
        help="Model name to send in the request payload",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use stream=true and measure full-stream duration (from first byte until [DONE])",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.getenv("MAX_TOKENS", 64)),
        help="max_tokens for generation (keep small for autocomplete)",
    )
    parser.add_argument(
        "--temperature", type=float, default=float(os.getenv("TEMPERATURE", 0.0)), help="temperature for generation"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Optional deterministic seed if backend supports it (e.g., vLLM)"
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Path to prompts file: .txt (one prompt per line) or .jsonl with keys like 'prompt','text','query','user'",
    )
    parser.add_argument(
        "--timeout", type=float, default=float(os.getenv("TIMEOUT", 120.0)), help="Request timeout in seconds"
    )
    parser.add_argument(
        "--auth", type=str, default=os.getenv("OPENAI_API_KEY"), help="Bearer token for Authorization header (optional)"
    )
    parser.add_argument(
        "--input-type",
        type=str,
        default="chat",
        choices=["chat", "completion"],
        help="Use chat (OpenAI Chat Completions) or completion (legacy Completions)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Custom name for the metric in Locust stats (otherwise derived from endpoint/model)",
    )


# ---- Helpers ---------------------------------------------------------------
DEFAULT_PROMPTS: List[str] = [
    "Write a short title for a blog post about streaming LLMs.",
    "Suggest a function name to parse JSON safely.",
    "Autocomplete: 'SELECT name, email FROM users WHERE'",
    "Complete a commit message: 'Fix race condition in'",
    "Autocomplete a Python docstring for a function that computes cosine similarity.",
    "Autocomplete a log message for retrying a failed HTTP call.",
    "Suggest a short variable name for an HTTP retry counter.",
    "Write a concise subject line for a release announcement.",
    "Autocomplete a shell one-liner to count lines in all .py files.",
    "Suggest a concise error explanation for a 502 from upstream.",
    "Autocomplete a short comment describing a binary search implementation.",
    "Continue: 'When using PostgreSQL indexes, remember to'",
    "Autocomplete a short title for a PR that refactors configuration loading.",
    "Continue: 'In Kubernetes, a liveness probe can be used to'",
    "Suggest a short name for a health-check endpoint.",
    "Autocomplete: 'def cache_key(user_id: int, feature: str) -> str:'",
    "Continue: 'Feature flags should be'",
    "Continue: 'To avoid deadlocks, the transaction order should'",
    "Autocomplete: 'from typing import'",
    "Autocomplete a brief changelog entry about improving latency by batching requests.",
]


def load_prompts(path: Optional[str]) -> List[str]:
    if not path:
        return DEFAULT_PROMPTS
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")
    prompts: List[str] = []
    if path.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    prompts.append(s)
    elif path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                for k in ("prompt", "text", "query", "user", "input"):
                    if k in obj and isinstance(obj[k], str) and obj[k].strip():
                        prompts.append(obj[k].strip())
                        break
    else:
        raise ValueError("Unsupported prompt file extension. Use .txt or .jsonl")
    if not prompts:
        raise ValueError("Prompt file is empty or has no usable fields")
    return prompts


@dataclass
class Opts:
    endpoint: str
    model: str
    stream: bool
    max_tokens: int
    temperature: float
    seed: Optional[int]
    prompt_file: Optional[str]
    timeout: float
    auth: Optional[str]
    input_type: str  # 'chat' or 'completion'
    name: Optional[str]


def build_headers(auth: Optional[str]) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if auth:
        headers["Authorization"] = f"Bearer {auth}"
    return headers


def to_chat_messages(prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You are a concise autocomplete assistant."},
        {"role": "user", "content": prompt},
    ]


def sse_events(response):
    """Yield parsed JSON objects from an SSE stream of OpenAI chat completions.
    Stops on [DONE].
    """
    for raw in response.iter_lines(decode_unicode=True):
        if not raw:
            continue
        if raw.startswith("data:"):
            data = raw[5:].strip()
            if not data:
                continue
            if data == "[DONE]":
                break
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                # Best-effort: skip malformed lines
                continue


# ---- Locust user -----------------------------------------------------------
class LLMUser(HttpUser):
    # Set to 0 for maximum throughput; tune if you want user think-time
    wait_time = constant(0)

    def on_start(self):
        po = self.environment.parsed_options
        self.opts = Opts(
            endpoint=po.endpoint,
            model=po.model,
            stream=bool(po.stream),
            max_tokens=int(po.max_tokens),
            temperature=float(po.temperature),
            seed=po.seed,
            prompt_file=po.prompt_file,
            timeout=float(po.timeout),
            auth=po.auth,
            input_type=po.input_type,
            name=po.name,
        )
        self.prompts = load_prompts(self.opts.prompt_file)
        self.headers = build_headers(self.opts.auth)

        # A stable metric name helps when comparing runs
        self.metric_name = self.opts.name or (
            f"{self.opts.endpoint} - {self.opts.input_type} - "
            f"{'stream' if self.opts.stream else 'nonstream'} - {self.opts.model}"
        )

    @task
    def generate(self):
        prompt = random.choice(self.prompts)
        if self.opts.input_type == "chat":
            payload: Dict[str, Any] = {
                "model": self.opts.model,
                "messages": to_chat_messages(prompt),
                "max_tokens": self.opts.max_tokens,
                "temperature": self.opts.temperature,
            }
            if self.opts.seed is not None:
                payload["seed"] = self.opts.seed
            if self.opts.stream:
                payload["stream"] = True
                self._post_chat_stream(payload)
            else:
                self._post_json(payload)
        else:  # completion (legacy /v1/completions)
            payload = {
                "model": self.opts.model,
                "prompt": prompt,
                "max_tokens": self.opts.max_tokens,
                "temperature": self.opts.temperature,
            }
            if self.opts.seed is not None:
                payload["seed"] = self.opts.seed
            if self.opts.stream:
                payload["stream"] = True
                self._post_stream_completions(payload)
            else:
                self._post_json(payload)

    # --- Non-stream: let Locust measure full response time -----------------
    def _post_json(self, payload: Dict[str, Any]):
        # For non-streaming, Locust's HttpSession measures until body is fully read.
        with self.client.post(
            self.opts.endpoint,
            headers=self.headers,
            json=payload,
            timeout=self.opts.timeout,
            name=self.metric_name,
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                # Surface a concise failure reason
                try:
                    err = resp.json()
                    msg = err.get("error", {}).get("message") or str(err)[:300]
                except Exception:
                    msg = resp.text[:300]
                resp.failure(f"HTTP {resp.status_code}: {msg}")
            else:
                # Optionally sanity-check schema
                try:
                    _ = resp.json()
                except Exception:
                    resp.failure("Invalid JSON in response")
                    return
                resp.success()

    # --- Streaming for Chat Completions (manual timing of the full stream) --
    def _post_chat_stream(self, payload: Dict[str, Any]):
        url = self.client.base_url + self.opts.endpoint.lstrip("/")
        start = time.perf_counter()
        exc = None
        resp = None
        response_len = 0
        try:
            resp = self.client.session.post(
                url,
                headers=self.headers,
                data=json.dumps(payload),
                stream=True,
                timeout=self.opts.timeout,
            )
            status = resp.status_code
            if status != 200:
                # Read a bit of body for diagnostics
                body = resp.text[:500]
                exc = Exception(f"HTTP {status}: {body}")
            else:
                full = []
                for ev in sse_events(resp):
                    if not ev:
                        continue
                    delta = ev.get("choices", [{}])[0].get("delta", {})
                    piece = delta.get("content")
                    if piece:
                        full.append(piece)
                        response_len += len(piece)
        except Exception as e:
            exc = e
        finally:
            total_ms = (time.perf_counter() - start) * 1000.0
            # Manually fire a Locust event so response_time reflects full-stream duration
            events.request.fire(
                request_type="SSE",
                name=self.metric_name,
                response_time=total_ms,
                response_length=response_len,
                exception=exc,
                context={"user": self},
                url=url,
                response=None,
            )
            try:
                if resp is not None:
                    resp.close()
            except Exception:
                pass

    # --- Streaming for legacy Completions (if you use /v1/completions) -----
    def _post_stream_completions(self, payload: Dict[str, Any]):
        url = self.client.base_url + self.opts.endpoint.lstrip("/")
        start = time.perf_counter()
        exc = None
        resp = None
        response_len = 0
        try:
            resp = self.client.session.post(
                url,
                headers=self.headers,
                data=json.dumps(payload),
                stream=True,
                timeout=self.opts.timeout,
            )
            status = resp.status_code
            if status != 200:
                body = resp.text[:500]
                exc = Exception(f"HTTP {status}: {body}")
            else:
                # For legacy completions, stream chunks contain {choices: [{text: "..."}]} under data:
                for raw in resp.iter_lines(decode_unicode=True):
                    if not raw:
                        continue
                    if raw.startswith("data:"):
                        data = raw[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            ev = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        text_piece = ev.get("choices", [{}])[0].get("text", "")
                        if text_piece:
                            response_len += len(text_piece)
        except Exception as e:
            exc = e
        finally:
            total_ms = (time.perf_counter() - start) * 1000.0
            events.request.fire(
                request_type="SSE",
                name=self.metric_name,
                response_time=total_ms,
                response_length=response_len,
                exception=exc,
                context={"user": self},
                url=url,
                response=None,
            )
            try:
                if resp is not None:
                    resp.close()
            except Exception:
                pass
