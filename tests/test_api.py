from __future__ import annotations

import random
import string
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from main import app
from parser import parse_invoice

client = TestClient(app)

EXPECTED_KEYS = {"vendor", "amount", "currency", "date"}


def test_case_1_acme_total_due():
    text = (
        "Vendor: Acme-X7Q9 Industries Ltd.\n"
        "Invoice Date: 2026-04-19\n"
        "Total Due: $1,234.56"
    )
    response = client.post("/extract", json={"text": text})
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == EXPECTED_KEYS
    assert "Acme-X7Q9 Industries Ltd." in data["vendor"]
    assert data["amount"] == 1234.56
    assert isinstance(data["amount"], (int, float))
    assert data["currency"] == "USD"
    assert data["date"] == "2026-04-19"


def test_case_2_seller_amount_due():
    text = (
        "INVOICE\n"
        "Seller: Nova-8821 Solutions LLC\n"
        "Date Issued: 2026-11-03\n"
        "Amount Due: EUR 582.25"
    )
    response = client.post("/extract", json={"text": text})
    assert response.status_code == 200
    data = response.json()
    assert "Nova-8821 Solutions LLC" in data["vendor"]
    assert abs(data["amount"] - 582.25) < 0.01
    assert data["currency"] == "EUR"
    assert data["date"] == "2026-11-03"


def test_case_3_balance_due_not_subtotal():
    text = (
        "Supplier: Bright-Q1 Services Ltd.\n"
        "Invoice Date: 2026-01-08\n"
        "Subtotal: GBP 700.00\n"
        "Tax: GBP 140.00\n"
        "Balance Due: GBP 840.00"
    )
    response = client.post("/extract", json={"text": text})
    assert response.status_code == 200
    data = response.json()
    assert abs(data["amount"] - 840.00) < 0.01
    assert data["currency"] == "GBP"


def test_case_4_grand_total_next_line():
    text = (
        "Vertex-992 Industries Ltd.\n"
        "INVOICE\n"
        "Invoice Date: 2026-06-25\n"
        "Invoice Number: 883721\n"
        "Grand Total\n"
        "£9,050.00"
    )
    response = client.post("/extract", json={"text": text})
    assert response.status_code == 200
    data = response.json()
    assert "Vertex-992 Industries Ltd." in data["vendor"]
    assert abs(data["amount"] - 9050.0) < 0.01
    assert data["currency"] == "GBP"
    assert data["date"] == "2026-06-25"


def test_empty_text_422():
    response = client.post("/extract", json={"text": ""})
    assert response.status_code == 422


def test_whitespace_text_422():
    response = client.post("/extract", json={"text": "   \n\t  "})
    assert response.status_code == 422


def test_missing_text_422():
    response = client.post("/extract", json={})
    assert response.status_code == 422


def test_garbage_no_500():
    response = client.post("/extract", json={"text": "asdfghjkl qwerty zxcvbn"})
    assert response.status_code != 500
    assert response.status_code in (200, 422)


def test_response_keys_exactly_four():
    text = "Vendor: Test Co.\nInvoice Date: 2026-01-01\nTotal Due: $100.00"
    response = client.post("/extract", json={"text": text})
    assert response.status_code == 200
    assert set(response.json().keys()) == EXPECTED_KEYS


def test_ollama_failure_fallback():
    text = (
        "Vendor: Fallback-123 Industries Ltd.\n"
        "Invoice Date: 2026-03-15\n"
        "Total Due: $500.00"
    )
    with patch("main.extract_with_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = ConnectionError("Ollama unavailable")
        response = client.post("/extract", json={"text": text})
    assert response.status_code == 200
    data = response.json()
    assert "Fallback-123" in data["vendor"]
    assert data["amount"] == 500.0


def test_ollama_returns_none_fallback():
    text = (
        "Seller: MockFail Corp.\n"
        "Date Issued: 2026-08-20\n"
        "Amount Due: EUR 250.50"
    )
    with patch("main.extract_with_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = None
        response = client.post("/extract", json={"text": text})
    assert response.status_code == 200
    assert response.json()["currency"] == "EUR"


def _random_id(length: int = 4) -> str:
    chars = string.ascii_uppercase + string.digits
    return "".join(random.choice(chars) for _ in range(length))


def _random_amount() -> float:
    return round(random.uniform(50, 9050), 2)


def _random_currency() -> str:
    return random.choice(["USD", "EUR", "GBP"])


def _currency_display(currency: str, amount: float) -> str:
    symbols = {"USD": "$", "EUR": "EUR ", "GBP": "£"}
    if currency == "USD":
        return f"${amount:,.2f}"
    if currency == "GBP":
        return f"£{amount:,.2f}"
    return f"EUR {amount:.2f}"


def _random_date_2026() -> str:
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    return f"2026-{month:02d}-{day:02d}"


LAYOUTS = [
    lambda v, d, a, c, disp: (
        f"Vendor: {v}\nInvoice Date: {d}\nTotal Due: {disp}"
    ),
    lambda v, d, a, c, disp: (
        f"INVOICE\nSeller: {v}\nDate Issued: {d}\nAmount Due: {disp}"
    ),
    lambda v, d, a, c, disp: (
        f"{v}\nINVOICE\nInvoice Date: {d}\nGrand Total\n{disp}"
    ),
    lambda v, d, a, c, disp: (
        f"Supplier: {v}\nBilling Date: {d}\nSubtotal: {disp}\n"
        f"Tax: {_currency_display(c, round(a * 0.2, 2))}\n"
        f"Balance Due: {_currency_display(c, round(a * 1.2, 2))}"
    ),
    lambda v, d, a, c, disp: (
        f"From: {v}\nIssued on: {d}\nFinal Total - {disp}"
    ),
    lambda v, d, a, c, disp: (
        f"Merchant: {v}\nDate: {d}\nPayment Due\n{disp}"
    ),
]


@pytest.mark.parametrize("seed", range(50))
def test_randomized_parser_cases(seed: int):
    random.seed(seed)
    vendor_id = _random_id()
    vendor = f"Acme-{vendor_id} Industries Ltd."
    amount = _random_amount()
    currency = _random_currency()
    date_str = _random_date_2026()
    disp = _currency_display(currency, amount)
    layout = random.choice(LAYOUTS)
    text = layout(vendor, date_str, amount, currency, disp)

    if "Balance Due" in text and "Subtotal" in text:
        expected_amount = round(amount * 1.2, 2)
    else:
        expected_amount = amount

    result = parse_invoice(text)
    assert result is not None, f"Parser failed for seed {seed}"
    assert vendor.lower() in result.vendor.lower()
    assert abs(result.amount - expected_amount) < 0.01
    assert result.currency == currency
    assert result.date == date_str


@pytest.mark.parametrize("seed", range(20))
def test_randomized_api_cases(seed: int):
    random.seed(1000 + seed)
    vendor_id = _random_id()
    vendor = f"Test-{vendor_id} Solutions LLC"
    amount = _random_amount()
    currency = _random_currency()
    date_str = _random_date_2026()
    disp = _currency_display(currency, amount)
    text = (
        f"Vendor: {vendor}\n"
        f"Invoice Date: {date_str}\n"
        f"Total Due: {disp}"
    )

    with patch("main.extract_with_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = None
        response = client.post("/extract", json={"text": text})

    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == EXPECTED_KEYS
    assert vendor.lower() in data["vendor"].lower()
    assert abs(data["amount"] - amount) < 0.01
    assert data["currency"] == currency
    assert data["date"] == date_str


def test_health_endpoint():
    response = client.get("/")
    assert response.status_code == 200


def test_post_root_extract_alias():
    text = (
        "Vendor: Root-Alias Corp.\n"
        "Invoice Date: 2026-05-10\n"
        "Total Due: $250.00"
    )
    response = client.post("/", json={"text": text})
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == EXPECTED_KEYS
    assert "Root-Alias Corp." in data["vendor"]


def test_post_extract_trailing_slash():
    text = (
        "Vendor: Slash-Test Ltd.\n"
        "Invoice Date: 2026-05-11\n"
        "Total Due: $99.00"
    )
    response = client.post("/extract/", json={"text": text})
    assert response.status_code == 200
    assert set(response.json().keys()) == EXPECTED_KEYS


def test_amount_is_numeric_not_string():
    text = "Vendor: Co.\nInvoice Date: 2026-02-02\nTotal: $99.99"
    response = client.post("/extract", json={"text": text})
    assert response.status_code == 200
    data = response.json()
    assert not isinstance(data["amount"], str)


def test_various_date_formats():
    formats = [
        ("Invoice Date: 15/07/2026", "2026-07-15"),
        ("Date Issued: 07/15/2026", "2026-07-15"),
        ("Billing Date: July 15, 2026", "2026-07-15"),
        ("Date: 15 July 2026", "2026-07-15"),
    ]
    for date_line, expected in formats:
        text = f"Vendor: DateTest Ltd.\n{date_line}\nTotal Due: $100.00"
        result = parse_invoice(text)
        assert result is not None
        assert result.date == expected
