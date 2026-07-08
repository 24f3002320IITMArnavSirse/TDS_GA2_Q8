from __future__ import annotations

import json
import os
import re
from typing import Optional

import httpx

from models import InvoiceExtraction

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "5"))
OLLAMA_ENABLED = os.getenv("OLLAMA_ENABLED", "false").lower() in ("1", "true", "yes")

EXTRACTION_PROMPT = (
    "You are an invoice extraction engine. Extract only facts explicitly present "
    "in the invoice text. Return one JSON object with exactly vendor, amount, "
    "currency, and date. amount must be numeric. currency must be an uppercase "
    "ISO 3-letter code. date must be YYYY-MM-DD. Do not invent values. "
    "Prefer total due/amount due over subtotal or tax. Prefer invoice date over due date.\n\n"
    "Invoice text:\n{text}\n\n"
    "Respond with JSON only."
)


def _extract_json_object(raw: str) -> Optional[dict]:
    raw = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence_match:
        raw = fence_match.group(1)

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None


def _coerce_extraction(data: dict) -> Optional[InvoiceExtraction]:
    if not isinstance(data, dict):
        return None

    required = {"vendor", "amount", "currency", "date"}
    if not required.issubset(data.keys()):
        return None

    amount = data["amount"]
    if isinstance(amount, str):
        cleaned = amount.replace(",", "").strip()
        cleaned = re.sub(r"[^0-9.\-]", "", cleaned)
        try:
            amount = float(cleaned)
        except ValueError:
            return None

    try:
        return InvoiceExtraction(
            vendor=str(data["vendor"]),
            amount=float(amount),
            currency=str(data["currency"]),
            date=str(data["date"]),
        )
    except Exception:
        return None


async def extract_with_llm(text: str) -> Optional[InvoiceExtraction]:
    if not OLLAMA_ENABLED:
        return None

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": EXTRACTION_PROMPT.format(text=text),
        "stream": False,
        "options": {"temperature": 0},
    }

    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate",
                json=payload,
            )
            response.raise_for_status()
            body = response.json()
    except Exception:
        return None

    raw_output = body.get("response", "")
    data = _extract_json_object(raw_output)
    if data is None:
        return None

    return _coerce_extraction(data)
