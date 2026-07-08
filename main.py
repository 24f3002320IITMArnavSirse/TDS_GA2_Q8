from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from llm_service import extract_with_llm
from models import ExtractRequest, InvoiceExtraction
from parser import parse_invoice

app = FastAPI(title="Local LLM Structured-Output Service")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": jsonable_encoder(exc.errors())},
    )


@app.get("/")
async def health():
    return {"status": "ok"}


@app.get("/health")
async def health_check():
    return {"status": "ok"}


def _reconcile(
    llm_result: Optional[InvoiceExtraction],
    parser_result: Optional[InvoiceExtraction],
    text: str,
) -> Optional[InvoiceExtraction]:
    if parser_result is None and llm_result is None:
        return None

    if parser_result is None:
        if llm_result and _llm_vendor_in_text(llm_result.vendor, text):
            return llm_result
        return None

    if llm_result is None:
        return parser_result

    vendor = parser_result.vendor
    if _llm_vendor_in_text(llm_result.vendor, text) and len(llm_result.vendor) > len(vendor):
        vendor = llm_result.vendor

    amount = parser_result.amount
    if abs(llm_result.amount - parser_result.amount) <= 0.01:
        amount = llm_result.amount

    currency = parser_result.currency
    if llm_result.currency == parser_result.currency:
        currency = llm_result.currency
    elif llm_result.currency in text.upper() or _currency_symbol_in_text(llm_result.currency, text):
        currency = llm_result.currency

    date_val = parser_result.date
    if llm_result.date in text or llm_result.date == parser_result.date:
        date_val = llm_result.date

    try:
        return InvoiceExtraction(
            vendor=vendor,
            amount=amount,
            currency=currency,
            date=date_val,
        )
    except Exception:
        return parser_result


def _llm_vendor_in_text(vendor: str, text: str) -> bool:
    return vendor.strip().lower() in text.lower()


def _currency_symbol_in_text(currency: str, text: str) -> bool:
    symbols = {"USD": "$", "EUR": "€", "GBP": "£"}
    sym = symbols.get(currency)
    return sym is not None and sym in text


async def extract_invoice(text: str) -> Optional[InvoiceExtraction]:
    parser_result = parse_invoice(text)
    try:
        llm_result = await extract_with_llm(text)
    except Exception:
        llm_result = None
    return _reconcile(llm_result, parser_result, text)


@app.post("/extract", response_model=InvoiceExtraction)
async def extract(request: ExtractRequest):
    result = await extract_invoice(request.text)
    if result is None:
        raise HTTPException(
            status_code=422,
            detail="Could not extract required invoice fields from the provided text",
        )
    return result
