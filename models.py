from __future__ import annotations

import re
from datetime import date, datetime

from pydantic import BaseModel, Field, field_validator


class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=1)

    @field_validator("text")
    @classmethod
    def text_not_whitespace_only(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty or whitespace-only")
        return v


class InvoiceExtraction(BaseModel):
    vendor: str
    amount: float
    currency: str
    date: str

    @field_validator("vendor")
    @classmethod
    def validate_vendor(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("vendor must be a non-empty string")
        return v

    @field_validator("amount")
    @classmethod
    def validate_amount(cls, v: float) -> float:
        if not isinstance(v, (int, float)):
            raise ValueError("amount must be numeric")
        v = float(v)
        if not (v == v and abs(v) != float("inf")):
            raise ValueError("amount must be finite")
        return v

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, v: str) -> str:
        v = v.strip().upper()
        if not re.fullmatch(r"[A-Z]{3}", v):
            raise ValueError("currency must be a 3-letter uppercase code")
        return v

    @field_validator("date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        v = v.strip()
        match = re.search(r"(\d{4}-\d{2}-\d{2})", v)
        if not match:
            raise ValueError("date must contain YYYY-MM-DD")
        normalized = match.group(1)
        try:
            datetime.strptime(normalized, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError("date must be a valid YYYY-MM-DD") from exc
        return normalized
