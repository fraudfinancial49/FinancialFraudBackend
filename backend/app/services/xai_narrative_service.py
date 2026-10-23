
import logging
import os
import requests
from typing import Dict

from app.core.config import settings

logger = logging.getLogger("xai_narrative_service")

_FALLBACK_MODELS = [
    "google/gemma-3-4b-it",
    "ibm-granite/granite-4.2-3b",
    "ibm-granite/granite-4.2-8b",
    "meta-llama/Llama-3.1-8B-Instruct",
]

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


def _build_prompt(final_risk_score, routing_decision, contributions):
    top_features = sorted(
        contributions.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:8]

    feature_lines = "\n".join(
        f"- {name}: {value:+.4f}" for name, value in top_features
    )

    return (
        f"Final risk score: {final_risk_score:.1f}/100\n"
        f"Routing decision: {routing_decision}\n"
        f"Top SHAP feature contributions (feature: signed impact):\n"
        f"{feature_lines}\n"
        "Explain the strongest risk-increasing and risk-reducing factors."
    )


def _try_groq(user_prompt):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured")

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 300,
        },
        timeout=8,
    )
    response.raise_for_status()
    text = response.json()["choices"][0]["message"]["content"]
    if not text or not text.strip():
        raise RuntimeError("Groq returned an empty explanation")
    return text.strip()


def _try_gemini(user_prompt):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
        json={
            "systemInstruction": {
                "parts": [{"text": _SYSTEM_PROMPT}]
            },
            "contents": [
                {"role": "user", "parts": [{"text": user_prompt}]}
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 300,
            },
        },
        timeout=8,
    )
    response.raise_for_status()
    data = response.json()
    text = "".join(
        part.get("text", "")
        for part in data["candidates"][0]["content"]["parts"]
    )
    if not text.strip():
        raise RuntimeError("Gemini returned an empty explanation")
    return text.strip()


def _try_huggingface(user_prompt, model_id):
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not configured")

    response = requests.post(
        "https://router.huggingface.co/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json={
            "model": model_id,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 300,
        },
        timeout=4,
    )
    response.raise_for_status()
    text = response.json()["choices"][0]["message"]["content"]
    if not text or not text.strip():
        raise RuntimeError("Hugging Face returned an empty explanation")
    return text.strip()


def generate_narrative(
    final_risk_score: float,
    routing_decision: str,
    contributions: Dict[str, float],
) -> str:
    if not contributions:
        raise XAINarrativeError("No SHAP contributions provided to explain.")

    user_prompt = _build_prompt(
        final_risk_score,
        routing_decision,
        contributions,
    )

    providers = [
        ("Groq", lambda: _try_groq(user_prompt)),
        ("Gemini", lambda: _try_gemini(user_prompt)),
    ]

    for provider_name, generate in providers:
        try:
            result = generate()
            logger.info("AI explanation generated successfully using %s", provider_name)
            return result
        except Exception as exc:
            logger.warning("%s explanation generation failed: %s", provider_name, exc)

    configured_model = getattr(settings, "XAI_LLM_MODEL", None)
    candidates = list(dict.fromkeys(
        ([configured_model] if configured_model else []) + _FALLBACK_MODELS
    ))

    last_exc = None

    for model_id in candidates:
        try:
            result = _try_huggingface(user_prompt, model_id)
            logger.info("AI explanation generated using Hugging Face model %s", model_id)
            return result
        except Exception as exc:
            logger.warning(
                "Hugging Face explanation failed for model %s: %s",
                model_id,
                exc,
            )
            last_exc = exc

    logger.error("All AI explanation providers failed")
    raise XAINarrativeError(
        f"AI narrative generation failed for all providers: {last_exc}"
    ) from last_exc
