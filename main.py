# locustfile.py
import json
import os
import random
from typing import Dict, Any, List

from locust import HttpUser, task, constant, LoadTestShape, events

# =========================
#   3 СТАДИИ: последовательно
#   duration — сек.
#   Здесь меняем параметры модели (max_tokens / temperature),
#   нагрузку оставляем одинаковой, чтобы сравнение было чище.
# =========================
STAGES = [
    {
        "label": "t=16 | temp=0.0",
        "duration": 60,
        "users": 10,
        "spawn_rate": 5,
        "overrides": {"max_tokens": 16, "temperature": 0.0},
    },
    {
        "label": "t=32 | temp=0.2",
        "duration": 60,
        "users": 10,
        "spawn_rate": 5,
        "overrides": {"max_tokens": 32, "temperature": 0.2},
    },
    {
        "label": "t=64 | temp=0.7",
        "duration": 60,
        "users": 10,
        "spawn_rate": 5,
        "overrides": {"max_tokens": 64, "temperature": 0.7},
    },
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
        return None  # завершение после последней стадии


# ---- CLI options -----------------------------------------------------------
@events.init_command_line_parser.add_listener
def _(parser):
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/chat/completions",
        help="OpenAI-совместимый путь (например, /v1/chat/completions)",
    )
    parser.add_argument("--model", type=str, default=os.getenv("MODEL", "gpt-3.5-turbo"), help="Имя модели для payload")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.getenv("MAX_TOKENS", 32)),
        help="max_tokens по умолчанию (может быть переопределён стадией)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("TEMPERATURE", 0.0)),
        help="temperature по умолчанию (может быть переопределён стадией)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Детерминизм, если бэкенд поддерживает")
    parser.add_argument(
        "--timeout", type=float, default=float(os.getenv("TIMEOUT", 120.0)), help="Таймаут запроса, сек"
    )
    parser.add_argument(
        "--auth", type=str, default=os.getenv("OPENAI_API_KEY"), help="Bearer-токен для Authorization (необязательно)"
    )
    parser.add_argument("--name", type=str, default=None, help="Кастомное имя метрики (иначе из endpoint/model)")


# ---- Статический набор коротких промптов -----------------------------------
PROMPTS: List[str] = [
    "Write a short title for a blog post about streaming LLMs.",
    "Suggest a function name to parse JSON safely.",
    "Autocomplete: 'SELECT name, email FROM users WHERE'",
    "Complete a commit message: 'Fix race condition in'",
    "Continue: 'In Kubernetes, a liveness probe can be used to'",
]


def build_headers(auth: str | None) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if auth:
        headers["Authorization"] = f"Bearer {auth}"
    return headers


def to_chat_messages(prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You are a concise autocomplete assistant."},
        {"role": "user", "content": prompt},
    ]


# ---- Locust user -----------------------------------------------------------
class LLMUser(HttpUser):
    wait_time = constant(0)  # максимум throughput

    def on_start(self):
        po = self.environment.parsed_options
        self.endpoint = po.endpoint
        self.model = po.model
        self.max_tokens = int(po.max_tokens)
        self.temperature = float(po.temperature)
        self.seed = po.seed
        self.timeout = float(po.timeout)
        self.headers = build_headers(po.auth)
        self.metric_base = po.name or f"{self.endpoint} - chat - nonstream - {self.model}"

    @task
    def generate(self):
        prompt = random.choice(PROMPTS)
        stage = current_stage()

        # Базовый payload (Chat Completions, без стрима)
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": to_chat_messages(prompt),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.seed is not None:
            payload["seed"] = self.seed

        # Переопределения для текущей стадии
        ov = stage.get("overrides", {})
        for k in ("model", "max_tokens", "temperature"):
            if k in ov:
                payload[k] = ov[k]

        metric_name = f"[{stage['label']}] {self.metric_base}"

        # Один простой POST: Locust сам меряет время до полного JSON
        with self.client.post(
            self.endpoint,
            headers=self.headers,
            json=payload,
            timeout=self.timeout,
            name=metric_name,
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                # короткий текст ошибки
                msg = ""
                try:
                    msg = resp.json()
                except Exception:
                    msg = resp.text
                resp.failure(f"HTTP {resp.status_code}: {str(msg)[:300]}")
                return
            try:
                _ = resp.json()
            except Exception:
                resp.failure("Invalid JSON in response")
                return
            resp.success()
