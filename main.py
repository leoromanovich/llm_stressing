import json
import os
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from locust import HttpUser, task, events, constant, LoadTestShape

# =========================
#   STAGES: последовательно
#   duration — в секундах
# =========================
STAGES = [
    {"label": "N=5 | t=32", "duration": 60, "users": 5, "spawn_rate": 2, "overrides": {"max_tokens": 32}},
    {"label": "N=10 | t=32", "duration": 60, "users": 10, "spawn_rate": 2},
    {"label": "N=20 | t=32", "duration": 60, "users": 20, "spawn_rate": 4},
]
_CURRENT_STAGE_IDX = 0  # обновляется shape'ом


def current_stage() -> Dict[str, Any]:
    return STAGES[_CURRENT_STAGE_IDX]


class StagesShape(LoadTestShape):
    """Последовательные окна нагрузки. Суммарная длительность = сумма duration."""

    stages = STAGES

    def tick(self):
        global _CURRENT_STAGE_IDX
        run_time = self.get_run_time()
        acc = 0
        for i, s in enumerate(self.stages):
            acc += s["duration"]
            if run_time < acc:
                _CURRENT_STAGE_IDX = i
                return (s["users"], s["spawn_rate"])
        return None  # завершение теста после последнего стейджа


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
        "--stream", action="store_true", help="Use stream=true and measure full-stream duration (until [DONE])"
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
        help="Path to prompts file: .txt (one per line) or .jsonl with 'prompt'/'text'/'query'/'user'",
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
    """Yield parsed JSON objects from an SSE stream of OpenAI chat completions. Stops on [DONE]."""
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
                continue


# ---- Locust user -----------------------------------------------------------
class LLMUser(HttpUser):
    wait_time = constant(0)  # максимум throughput

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

        # базовое имя; к нему добавим label стейджа
        self.metric_base = self.opts.name or (
            f"{self.opts.endpoint} - {self.opts.input_type} - "
            f"{'stream' if self.opts.stream else 'nonstream'} - {self.opts.model}"
        )

    @task
    def generate(self):
        prompt = random.choice(self.prompts)
        stage = current_stage()
        metric_name = f"[{stage['label']}] {self.metric_base}"

        # базовый payload
        if self.opts.input_type == "chat":
            payload: Dict[str, Any] = {
                "model": self.opts.model,
                "messages": to_chat_messages(prompt),
                "max_tokens": self.opts.max_tokens,
                "temperature": self.opts.temperature,
            }
        else:
            payload = {
                "model": self.opts.model,
                "prompt": prompt,
                "max_tokens": self.opts.max_tokens,
                "temperature": self.opts.temperature,
            }
        if self.opts.seed is not None:
            payload["seed"] = self.opts.seed

        # overrides стейджа
        ov = stage.get("overrides", {})
        if "model" in ov:
            payload["model"] = ov["model"]
        if "max_tokens" in ov:
            payload["max_tokens"] = ov["max_tokens"]
        if "temperature" in ov:
            payload["temperature"] = ov["temperature"]
        streaming = self.opts.stream
        if "stream" in ov:
            streaming = bool(ov["stream"])

        # отправка
        if self.opts.input_type == "chat":
            if streaming:
                payload["stream"] = True
                self._post_chat_stream(payload, metric_name)
            else:
                self._post_json(payload, metric_name)
        else:
            if streaming:
                payload["stream"] = True
                self._post_stream_completions(payload, metric_name)
            else:
                self._post_json(payload, metric_name)

    # --- Non-stream: Locust сам меряет до полной выдачи тела ---------------
    def _post_json(self, payload: Dict[str, Any], metric_name: str):
        with self.client.post(
            self.opts.endpoint,
            headers=self.headers,
            json=payload,
            timeout=self.opts.timeout,
            name=metric_name,
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                try:
                    err = resp.json()
                    msg = err.get("error", {}).get("message") or str(err)[:300]
                except Exception:
                    msg = resp.text[:300]
                resp.failure(f"HTTP {resp.status_code}: {msg}")
            else:
                try:
                    _ = resp.json()
                except Exception:
                    resp.failure("Invalid JSON in response")
                    return
                resp.success()

    # --- Streaming for Chat Completions: TTFT + полный стрим ----------------
    def _post_chat_stream(self, payload: Dict[str, Any], metric_name: str):
        t0 = time.perf_counter()
        ttft_ms = None
        with self.client.post(
            self.opts.endpoint,
            headers=self.headers,
            data=json.dumps(payload),
            stream=True,
            timeout=self.opts.timeout,
            name=metric_name,  # «полный стрим» попадёт в эту метрику (тип=POST)
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                body = resp.text[:500]
                resp.failure(f"HTTP {resp.status_code}: {body}")
                return
            try:
                for ev in sse_events(resp):
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t0) * 1000.0
                        # отдельная метрика TTFT
                        events.request.fire(
                            request_type="TTFT",
                            name=metric_name,
                            response_time=ttft_ms,
                            response_length=0,
                            exception=None,
                            context={"user": self},
                            url=self.client.base_url + self.opts.endpoint.lstrip("/"),
                            response=None,
                        )
                    # можно суммировать длину, если нужно
                resp.success()
            except Exception as e:
                resp.failure(str(e))

    # --- Streaming для legacy /v1/completions --------------------------------
    def _post_stream_completions(self, payload: Dict[str, Any], metric_name: str):
        t0 = time.perf_counter()
        ttft_ms = None
        with self.client.post(
            self.opts.endpoint,
            headers=self.headers,
            data=json.dumps(payload),
            stream=True,
            timeout=self.opts.timeout,
            name=metric_name,
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                body = resp.text[:500]
                resp.failure(f"HTTP {resp.status_code}: {body}")
                return
            try:
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
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - t0) * 1000.0
                            events.request.fire(
                                request_type="TTFT",
                                name=metric_name,
                                response_time=ttft_ms,
                                response_length=0,
                                exception=None,
                                context={"user": self},
                                url=self.client.base_url + self.opts.endpoint.lstrip("/"),
                                response=None,
                            )
                resp.success()
            except Exception as e:
                resp.failure(str(e))
