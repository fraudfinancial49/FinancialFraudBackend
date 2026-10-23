import logging
import os
from typing import Dict, List, Tuple

import requests

from app.core.config import settings

logger = logging.getLogger("xai_narrative_service")

# Hugging Face is now a last resort (it needs paid credits). Keep the list short.
_HF_FALLBACK_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-3-4b-it",
]

# HTTP statuses that mean "this provider won't work for any model right now"
_FATAL_STATUSES = {401, 402, 403, 429}

_SYSTEM_PROMPT = (
    "You are a fraud-analytics assistant explaining a transaction risk score to a bank "
    "fraud analyst. You are given the model's SHAP feature contributions (positive values "
    "push the fraud-risk score up, negative values push it down) and the router's final "
    "decision. Write a concise, plain-English explanation of WHY the model flagged or "
    "cleared this transaction, naming the 2-4 most influential features. Explain that "
    "SHAP contribution indicates model influence, not proof of fraud. Do not invent facts "
    "or claim causation not present in the data. Write 3-5 sentences, no heading or bullets."
)


class XAINarrativeError(Exception):
    pass


class ProviderError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def _top_features(contributions: Dict[str, float], n: int = 8) -> List[Tuple[str, float]]:
    return sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:n]


def _build_prompt(final_risk_score, routing_decision, contributions):
    feature_lines = "\n".join(
        f"- {name}: {value:+.4f}" for name, value in _top_features(contributions)
    )
    return (
        f"Final risk score: {final_risk_score:.1f}/100\n"
        f"Routing decision: {routing_decision}\n"
        f"Top SHAP feature contributions (feature: signed impact):\n"
        f"{feature_lines}\n"
        "Explain the strongest risk-increasing and risk-reducing factors."
    )


def _post(name: str, url: str, headers: dict, payload: dict, timeout: int) -> dict:
    """POST and raise ProviderError that includes status and response body."""
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise ProviderError(f"{name} request failed: {exc}") from exc

    if not resp.ok:
        body = resp.text[:300].replace("\n", " ")
        raise ProviderError(f"{name} HTTP {resp.status_code}: {body}", resp.status_code)
    try:
        return resp.json()
    except ValueError as exc:
        raise ProviderError(f"{name} returned non-JSON response") from exc


def _try_groq(user_prompt: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ProviderError("GROQ_API_KEY is not configured")

    data = _post(
        "Groq",
        "https://api.groq.com/openai/v1/chat/completions",
        {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        {
            # llama-3.3-70b-versatile was shut down on 2026-08-16.
            # gpt-oss-20b is a reasoning model: keep effort low and the token cap
            # generous so hidden reasoning doesn't leave an empty answer.
            "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_completion_tokens": 1024,
            "reasoning_effort": "low",
        },
        timeout=15,
    )
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError(f"Groq response had unexpected shape: {str(data)[:200]}") from exc
    if not text or not text.strip():
        raise ProviderError("Groq returned an empty explanation")
    return text.strip()


def _try_gemini(user_prompt: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ProviderError("GEMINI_API_KEY is not configured")

    # Pinned to a stable GA model. 2.5 models are restricted to existing users, so new
    # keys may get a 404 there. If this 404s for your key, set GEMINI_MODEL to a model
    # listed for your account in Google AI Studio.
    model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    data = _post(
        "Gemini",
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        {"x-goog-api-key": api_key, "Content-Type": "application/json"},
        {
            "systemInstruction": {"parts": [{"text": _SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            # Generous cap: thinking tokens can count against this limit
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1024},
        },
        timeout=20,
    )
    try:
        parts = data["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts)
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError(f"Gemini response had unexpected shape: {str(data)[:200]}") from exc
    if not text.strip():
        raise ProviderError("Gemini returned an empty explanation")
    return text.strip()


def _try_huggingface(user_prompt: str, model_id: str) -> str:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ProviderError("HF_TOKEN is not configured")

    data = _post(
        "HuggingFace",
        "https://router.huggingface.co/v1/chat/completions",
        {"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        {
            "model": model_id,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 400,
        },
        timeout=10,
    )
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError("HuggingFace response had unexpected shape") from exc
    if not text or not text.strip():
        raise ProviderError("HuggingFace returned an empty explanation")
    return text.strip()


def _template_narrative(final_risk_score, routing_decision, contributions) -> str:
    """Deterministic fallback built only from the SHAP values. No external calls."""
    top = _top_features(contributions, 4)
    up = [(n, v) for n, v in top if v > 0]
    down = [(n, v) for n, v in top if v < 0]

    parts = [
        f"The model assigned a final risk score of {final_risk_score:.1f}/100 "
        f"and the routing decision was '{routing_decision}'."
    ]
    if up:
        names = ", ".join(f"{n} ({v:+.3f})" for n, v in up)
        parts.append(f"The factors pushing the risk score up the most were {names}.")
    if down:
        names = ", ".join(f"{n} ({v:+.3f})" for n, v in down)
        parts.append(f"The factors pulling the risk score down the most were {names}.")
    parts.append(
        "SHAP values show how much each feature influenced the model's output; "
        "they are not proof of fraud. (Automated summary: the AI explanation "
        "service was unavailable.)"
    )
    return " ".join(parts)


def generate_narrative(
    final_risk_score: float,
    routing_decision: str,
    contributions: Dict[str, float],
) -> str:
    if not contributions:
        raise XAINarrativeError("No SHAP contributions provided to explain.")

    user_prompt = _build_prompt(final_risk_score, routing_decision, contributions)
    errors: List[str] = []

    providers = [
        ("Groq", lambda: _try_groq(user_prompt)),
        ("Gemini", lambda: _try_gemini(user_prompt)),
    ]
    for name, fn in providers:
        try:
            result = fn()
            logger.info("AI explanation generated using %s", name)
            return result
        except Exception as exc:
            logger.warning("%s explanation failed: %s", name, exc)
            errors.append(str(exc))

    # Hugging Face last: only if a token exists, and stop at the first fatal status
    if os.getenv("HF_TOKEN"):
        configured = getattr(settings, "XAI_LLM_MODEL", None)
        candidates = list(dict.fromkeys(
            ([configured] if configured else []) + _HF_FALLBACK_MODELS
        ))
        for model_id in candidates:
            try:
                result = _try_huggingface(user_prompt, model_id)
                logger.info("AI explanation generated using HF model %s", model_id)
                return result
            except ProviderError as exc:
                logger.warning("HF model %s failed: %s", model_id, exc)
                errors.append(str(exc))
                if exc.status in _FATAL_STATUSES:
                    break  # same failure would repeat for every model
            except Exception as exc:
                logger.warning("HF model %s failed: %s", model_id, exc)
                errors.append(str(exc))

    # Full detail goes to the server log; the user gets a usable summary
    logger.error("All AI providers failed: %s", " | ".join(errors))
    return _template_narrative(final_risk_score, routing_decision, contributions)
