from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from llm_service import extract_with_llm
from models import ExtractRequest, InvoiceExtraction
from parser import parse_invoice

app = FastAPI(title="Local LLM Structured-Output Service", redirect_slashes=False)


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


def _llm_vendor_in_text(vendor: str, text: str) -> bool:
    return vendor.strip().lower() in text.lower()


async def extract_invoice(text: str) -> Optional[InvoiceExtraction]:
    parser_result = parse_invoice(text)
    if parser_result is not None:
        return parser_result

    try:
        llm_result = await extract_with_llm(text)
    except Exception:
        llm_result = None

    if llm_result and _llm_vendor_in_text(llm_result.vendor, text):
        return llm_result
    return None


@app.post("/extract", response_model=InvoiceExtraction)
@app.post("/extract/", response_model=InvoiceExtraction, include_in_schema=False)
@app.post("/", response_model=InvoiceExtraction, include_in_schema=False)
async def extract(request: ExtractRequest):
    result = await extract_invoice(request.text)
    if result is None:
        raise HTTPException(
            status_code=422,
            detail="Could not extract required invoice fields from the provided text",
        )
    return result
