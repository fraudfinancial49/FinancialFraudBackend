"""Natural-language SHAP explanations via Hugging Face's free Inference
Providers API (huggingface_hub.InferenceClient) -- the same HF credential
(`HF_TOKEN`) already used elsewhere in this codebase to pull model artifacts.
No paid LLM API required."""
import logging
import os
from typing import Dict

from huggingface_hub import InferenceClient

from app.core.config import settings

logger = logging.getLogger("xai_narrative_service")

# Which models are actually "warm" on a free HF account's enabled providers
# drifts over time and can't be verified from this codebase -- rather than
# hardcode one guess that silently breaks again, try `settings.XAI_LLM_MODEL`
# first (so it stays operator-configurable) then fall through this list of
# other instruct models on "auto" routing (tries every provider the account
# has access to) until one actually answers. This specific list was confirmed
# present in the account's own huggingface.co/settings/inference-providers
# model breakdown -- ungated ones first, since the one gated entry
# (Llama-3.1-8B) can additionally fail if the account hasn't clicked through
# Meta's license on its model page, independent of provider availability.
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
    "decision. Write a concise, plain-English explanation (3-5 sentences) of WHY the model "
    "flagged or cleared this transaction, naming the 2-4 most influential features. Do not "
    "invent facts not present in the data. No preamble, no markdown headers, no bullet lists "
    "-- a short explanatory paragraph only."
)


class XAINarrativeError(Exception):
    pass


def generate_narrative(
    final_risk_score: float, routing_decision: str, contributions: Dict[str, float],
) -> str:
    if not contributions:
        raise XAINarrativeError("No SHAP contributions provided to explain.")

    top_features = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:8]
    feature_lines = "\n".join(f"- {name}: {value:+.4f}" for name, value in top_features)
    user_prompt = (
        f"Final risk score: {final_risk_score:.1f}/100\n"
        f"Routing decision: {routing_decision}\n"
        f"Top SHAP feature contributions (feature: signed impact):\n{feature_lines}\n"
    )

    token = os.getenv("HF_TOKEN") or None
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # Configured model first, then the fallback list -- de-duplicated, order
    # preserved.
    candidates = list(dict.fromkeys([settings.XAI_LLM_MODEL, *_FALLBACK_MODELS]))

    last_exc: Exception | None = None
    for model_id in candidates:
        try:
            client = InferenceClient(model=model_id, token=token, provider="auto")
            response = client.chat_completion(messages=messages, max_tokens=300, temperature=0.3)
            text = response.choices[0].message.content
            if not text:
                raise XAINarrativeError("Empty response from the language model.")
            if model_id != settings.XAI_LLM_MODEL:
                logger.warning(
                    "XAI_LLM_MODEL '%s' unavailable -- served this explanation from fallback model '%s' instead.",
                    settings.XAI_LLM_MODEL, model_id,
                )
            return text.strip()
        except Exception as exc:
            logger.warning("HF Inference narrative generation failed for model '%s': %s", model_id, exc)
            last_exc = exc
            continue

    logger.error("HF Inference narrative generation failed for every candidate model: %s", candidates)
    raise XAINarrativeError(f"AI narrative generation failed for all candidate models: {last_exc}") from last_exc
