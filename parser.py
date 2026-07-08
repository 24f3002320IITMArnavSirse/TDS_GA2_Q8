from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from dateutil import parser as date_parser

from models import InvoiceExtraction

CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP"}
CURRENCY_CODES = {"USD", "EUR", "GBP"}

AMOUNT_STRONG_LABELS = [
  "total due",
  "amount due",
  "balance due",
  "payment due",
  "grand total",
  "final total",
  "invoice total",
  "total amount",
  "total",
]

AMOUNT_REJECT_LABELS = [
  "subtotal",
  "tax",
  "vat",
  "discount",
  "quantity",
  "invoice number",
  "invoice #",
  "invoice no",
  "po number",
  "p.o.",
]

VENDOR_LABELS = [
  "vendor",
  "seller",
  "from",
  "supplier",
  "merchant",
  "billed by",
  "issued by",
]

DATE_PRIORITY_LABELS = [
  "invoice date",
  "date issued",
  "issued on",
  "billing date",
  "date",
]

DATE_LOW_PRIORITY_LABELS = [
  "due date",
  "payment due date",
]

HEADER_REJECT = re.compile(
  r"^(invoice|tax invoice|receipt|bill|statement)$",
  re.IGNORECASE,
)

COMPANY_SUFFIX = re.compile(
  r"\b("
  r"Ltd\.?|Limited|LLC|Inc\.?|Incorporated|Corp\.?|Corporation|"
  r"Industries|Solutions|Services|Technologies|Enterprises|"
  r"Co\.?|Company|PLC|GmbH"
  r")\b",
  re.IGNORECASE,
)

AMOUNT_PATTERN = re.compile(
    r"(?P<symbol>[$€£])?\s*"
    r"(?P<amount>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?)"
    r"\s*(?P<code>USD|EUR|GBP)?",
    re.IGNORECASE,
)

CURRENCY_CODE_PATTERN = re.compile(r"\b(USD|EUR|GBP)\b", re.IGNORECASE)


@dataclass
class AmountCandidate:
    value: float
    score: int
    currency_hint: Optional[str]
    position: int


def _normalize_amount(raw: str) -> float:
    cleaned = raw.replace(",", "")
    return float(cleaned)


def _detect_currency_near(text: str, start: int, end: int) -> Optional[str]:
    window = text[max(0, start - 30): min(len(text), end + 30)]
    for symbol, code in CURRENCY_SYMBOLS.items():
        if symbol in window:
            return code
    match = CURRENCY_CODE_PATTERN.search(window)
    if match:
        return match.group(1).upper()
    return None


def _line_has_reject_label(line: str) -> bool:
    lower = line.lower()
    return any(label in lower for label in AMOUNT_REJECT_LABELS)


def _score_amount_line(line: str, label: str, same_line: bool) -> int:
    score = 0
    lower_line = line.lower()
    lower_label = label.lower()

    priority_index = next(
        (i for i, l in enumerate(AMOUNT_STRONG_LABELS) if l in lower_label),
        len(AMOUNT_STRONG_LABELS),
    )
    score += (len(AMOUNT_STRONG_LABELS) - priority_index) * 20

    if same_line:
        score += 30
    if _line_has_reject_label(line):
        score -= 100

    for symbol in CURRENCY_SYMBOLS:
        if symbol in line:
            score += 10
            break
    if CURRENCY_CODE_PATTERN.search(line):
        score += 10

    return score


def extract_amount(text: str) -> tuple[Optional[float], Optional[str]]:
    lines = text.splitlines()
    candidates: list[AmountCandidate] = []

    for i, line in enumerate(lines):
        lower = line.lower()

        for label in AMOUNT_STRONG_LABELS:
            if label not in lower:
                continue

            label_pos = lower.find(label)
            after_label = line[label_pos + len(label):]
            after_label = re.sub(r"^[\s:\-–—]+", "", after_label)

            for match in AMOUNT_PATTERN.finditer(after_label):
                value = _normalize_amount(match.group("amount"))
                symbol = match.group("symbol")
                code = match.group("code")
                currency_hint = None
                if code:
                    currency_hint = code.upper()
                elif symbol:
                    currency_hint = CURRENCY_SYMBOLS.get(symbol)

                pos = text.find(line) + label_pos + len(label) + match.start()
                score = _score_amount_line(line, label, same_line=True)
                candidates.append(AmountCandidate(value, score, currency_hint, pos))

            if i + 1 < len(lines):
                next_line = lines[i + 1]
                if not _line_has_reject_label(next_line):
                    for match in AMOUNT_PATTERN.finditer(next_line):
                        value = _normalize_amount(match.group("amount"))
                        symbol = match.group("symbol")
                        code = match.group("code")
                        currency_hint = None
                        if code:
                            currency_hint = code.upper()
                        elif symbol:
                            currency_hint = CURRENCY_SYMBOLS.get(symbol)
                        pos = text.find(next_line) + match.start()
                        score = _score_amount_line(line, label, same_line=False) + 5
                        candidates.append(
                            AmountCandidate(value, score, currency_hint, pos)
                        )

    if not candidates:
        for i, line in enumerate(lines):
            if _line_has_reject_label(line):
                continue
            for match in AMOUNT_PATTERN.finditer(line):
                value = _normalize_amount(match.group("amount"))
                if value < 1:
                    continue
                symbol = match.group("symbol")
                code = match.group("code")
                currency_hint = None
                if code:
                    currency_hint = code.upper()
                elif symbol:
                    currency_hint = CURRENCY_SYMBOLS.get(symbol)
                pos = text.find(line) + match.start()
                score = 1
                if symbol or code:
                    score += 5
                candidates.append(AmountCandidate(value, score, currency_hint, pos))

    if not candidates:
        return None, None

    best = max(candidates, key=lambda c: (c.score, len(str(c.value)), -c.position))
    currency = best.currency_hint or _detect_currency_near(
        text, best.position, best.position + 20
    )
    return best.value, currency


def extract_currency(text: str, amount: Optional[float], amount_pos_hint: int = 0) -> Optional[str]:
    if amount is not None:
        near = _detect_currency_near(text, amount_pos_hint, amount_pos_hint + 50)
        if near:
            return near

    codes_found = CURRENCY_CODE_PATTERN.findall(text)
    if codes_found:
        return codes_found[-1].upper()

    for symbol, code in CURRENCY_SYMBOLS.items():
        if symbol in text:
            return code

    return None


def _clean_vendor_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^[\-\*\#]+\s*", "", line)
    for label in VENDOR_LABELS:
        pattern = re.compile(rf"^{label}\s*[:\-–—]\s*", re.IGNORECASE)
        if pattern.match(line):
            return pattern.sub("", line).strip()
    return line


def _is_reject_vendor_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if HEADER_REJECT.match(stripped):
        return True
    lower = stripped.lower()
    if any(label in lower for label in ["invoice number", "invoice #", "invoice no", "po number"]):
        return True
    if re.search(r"\b(total|subtotal|tax|vat|amount due|balance due)\b", lower):
        return True
    if re.search(r"\b(tel|phone|email|fax|www\.|http)\b", lower):
        return True
    if re.search(r"\b\d{1,5}\s+[a-z]+\s+(st|street|ave|avenue|road|rd|blvd)\b", lower):
        return True
    if re.fullmatch(r"[\d\-/\.]+", stripped):
        return True
    return False


def extract_vendor(text: str) -> Optional[str]:
    lines = text.splitlines()

    for line in lines:
        for label in VENDOR_LABELS:
            pattern = re.compile(
                rf"^{label}\s*[:\-–—]\s*(.+)$",
                re.IGNORECASE,
            )
            match = pattern.match(line.strip())
            if match:
                vendor = match.group(1).strip()
                if vendor:
                    return vendor

    for line in lines:
        cleaned = _clean_vendor_line(line)
        if _is_reject_vendor_line(cleaned):
            continue
        if len(cleaned) < 3:
            continue
        if COMPANY_SUFFIX.search(cleaned) or re.search(r"[A-Za-z].*\d", cleaned):
            return cleaned

    for line in lines:
        cleaned = _clean_vendor_line(line)
        if not _is_reject_vendor_line(cleaned) and len(cleaned) >= 3:
            if re.search(r"[A-Za-z]{2,}", cleaned):
                return cleaned

    return None


def _try_parse_date_string(raw: str) -> Optional[str]:
    raw = raw.strip()
    raw = re.sub(r"^[\s:\-–—]+", "", raw)

    iso_match = re.search(r"(\d{4})[-/](\d{2})[-/](\d{2})", raw)
    if iso_match:
        try:
            dt = datetime(int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3)))
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    for fmt in ("%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y"):
        match = re.search(
            rf"(\d{{1,2}})[/\-](\d{{1,2}})[/\-](\d{{4}})",
            raw,
        )
        if match:
            d, m, y = int(match.group(1)), int(match.group(2)), int(match.group(3))
            candidates = []
            for day, month in ((d, m), (m, d)):
                try:
                    dt = datetime(y, month, day)
                    candidates.append(dt)
                except ValueError:
                    continue
            if len(candidates) == 1:
                return candidates[0].strftime("%Y-%m-%d")
            if len(candidates) == 2:
                if y == 2026:
                    for dt in candidates:
                        if dt.month <= 12 and dt.day <= 31:
                            pass
                return candidates[0].strftime("%Y-%m-%d")

    month_name = re.search(
        r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})|"
        r"([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})",
        raw,
    )
    if month_name:
        try:
            if month_name.group(1):
                dt = date_parser.parse(
                    f"{month_name.group(1)} {month_name.group(2)} {month_name.group(3)}",
                    dayfirst=True,
                )
            else:
                dt = date_parser.parse(
                    f"{month_name.group(4)} {month_name.group(5)} {month_name.group(6)}",
                    dayfirst=False,
                )
            return dt.strftime("%Y-%m-%d")
        except (ValueError, OverflowError):
            pass

    try:
        dt = date_parser.parse(raw, dayfirst=True, fuzzy=True)
        return dt.strftime("%Y-%m-%d")
    except (ValueError, OverflowError):
        return None


def extract_date(text: str) -> Optional[str]:
    lines = text.splitlines()
    priority_dates: list[str] = []
    low_priority_dates: list[str] = []
    all_dates: list[str] = []

    for line in lines:
        lower = line.lower()
        date_part = re.sub(
            r"^.*?(invoice date|date issued|issued on|billing date|due date|payment due date|date)\s*[:\-–—]\s*",
            "",
            line,
            flags=re.IGNORECASE,
        )
        if date_part == line:
            date_part = line

        parsed = _try_parse_date_string(date_part)
        if not parsed:
            for match in re.finditer(
                r"\d{4}[-/]\d{2}[-/]\d{2}|\d{1,2}[/\-]\d{1,2}[/\-]\d{4}|"
                r"\d{1,2}\s+[A-Za-z]+\s+\d{4}|[A-Za-z]+\s+\d{1,2},?\s+\d{4}",
                line,
            ):
                parsed = _try_parse_date_string(match.group(0))
                if parsed:
                    break

        if not parsed:
            continue

        all_dates.append(parsed)
        is_low = any(lbl in lower for lbl in DATE_LOW_PRIORITY_LABELS)
        is_priority = any(lbl in lower for lbl in DATE_PRIORITY_LABELS)

        if is_priority and not is_low:
            priority_dates.append(parsed)
        elif is_low:
            low_priority_dates.append(parsed)

    if priority_dates:
        return priority_dates[0]
    if all_dates:
        dates_2026 = [d for d in all_dates if d.startswith("2026")]
        if dates_2026:
            return dates_2026[0]
        return all_dates[0]
    if low_priority_dates:
        return low_priority_dates[0]
    return None


def parse_invoice(text: str) -> Optional[InvoiceExtraction]:
    if not text or not text.strip():
        return None

    amount, currency_hint = extract_amount(text)
    vendor = extract_vendor(text)
    date_val = extract_date(text)
    currency = currency_hint or extract_currency(text, amount)

    if vendor is None or amount is None or currency is None or date_val is None:
        return None

    try:
        return InvoiceExtraction(
            vendor=vendor,
            amount=amount,
            currency=currency,
            date=date_val,
        )
    except Exception:
        return None
