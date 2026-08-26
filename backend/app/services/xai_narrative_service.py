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
    client = InferenceClient(model=settings.XAI_LLM_MODEL, token=token)
    try:
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=300,
            temperature=0.3,
        )
        text = response.choices[0].message.content
        if not text:
            raise XAINarrativeError("Empty response from the language model.")
        return text.strip()
    except XAINarrativeError:
        raise
    except Exception as exc:
        logger.exception("HF Inference narrative generation failed (model=%s).", settings.XAI_LLM_MODEL)
        raise XAINarrativeError(f"AI narrative generation failed: {exc}") from exc
