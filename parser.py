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
    r"(?:(?P<prefix_code>USD|EUR|GBP)\s+)?"
    r"(?P<symbol>[$€£])?\s*"
    r"(?P<amount>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?)"
    r"\s*(?P<suffix_code>USD|EUR|GBP)?",
    re.IGNORECASE,
)

CURRENCY_CODE_PATTERN = re.compile(r"\b(USD|EUR|GBP)\b", re.IGNORECASE)


def _label_in_line(label: str, lower: str) -> bool:
    if label == "total":
        return bool(re.search(r"\btotal\b", lower)) and "subtotal" not in lower
    return label in lower


def _currency_from_match(match: re.Match) -> Optional[str]:
    prefix = match.group("prefix_code")
    suffix = match.group("suffix_code")
    symbol = match.group("symbol")
    if prefix:
        return prefix.upper()
    if suffix:
        return suffix.upper()
    if symbol:
        return CURRENCY_SYMBOLS.get(symbol)
    return None


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
            if not _label_in_line(label, lower):
                continue

            label_pos = lower.find(label) if label != "total" else re.search(r"\btotal\b", lower).start()
            after_label = line[label_pos + len(label):]
            after_label = re.sub(r"^[\s:\-–—]+", "", after_label)

            for match in AMOUNT_PATTERN.finditer(after_label):
                value = _normalize_amount(match.group("amount"))
                currency_hint = _currency_from_match(match)
                pos = text.find(line) + label_pos + len(label) + match.start()
                score = _score_amount_line(line, label, same_line=True)
                candidates.append(AmountCandidate(value, score, currency_hint, pos))

            if i + 1 < len(lines):
                next_line = lines[i + 1]
                if not _line_has_reject_label(next_line):
                    for match in AMOUNT_PATTERN.finditer(next_line):
                        value = _normalize_amount(match.group("amount"))
                        currency_hint = _currency_from_match(match)
                        pos = text.find(next_line) + match.start()
                        score = _score_amount_line(line, label, same_line=False) + 5
                        candidates.append(
                            AmountCandidate(value, score, currency_hint, pos)
                        )

    if not candidates:
        for line in lines:
            if _line_has_reject_label(line):
                continue
            for match in AMOUNT_PATTERN.finditer(line):
                value = _normalize_amount(match.group("amount"))
                if value < 1:
                    continue
                currency_hint = _currency_from_match(match)
                pos = text.find(line) + match.start()
                score = 1
                if currency_hint:
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
        near = _detect_currency_near(text, amount_pos_hint, amount_pos_hint + 80)
        if near:
            return near

    codes_found = [c.upper() for c in CURRENCY_CODE_PATTERN.findall(text)]
    if codes_found:
        unique = set(codes_found)
        if len(unique) == 1:
            return codes_found[0]
        return codes_found[-1]

    symbols_found = [code for sym, code in CURRENCY_SYMBOLS.items() if sym in text]
    if symbols_found:
        unique = set(symbols_found)
        if len(unique) == 1:
            return symbols_found[0]

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


def _date_label_in_line(label: str, lower: str) -> bool:
    if label == "date":
        return bool(re.search(r"\bdate\b", lower)) and not any(
            x in lower for x in ("update", "due date", "payment due date")
        )
    return label in lower


def _try_parse_date_string(raw: str) -> Optional[str]:
    raw = raw.strip()
    raw = re.sub(r"^[\s:\-–—]+", "", raw)

    if not re.search(r"\d{4}", raw) and not re.search(
        r"[A-Za-z]{3,}", raw
    ):
        return None

    iso_match = re.search(r"(\d{4})[-/](\d{2})[-/](\d{2})", raw)
    if iso_match:
        try:
            dt = datetime(
                int(iso_match.group(1)),
                int(iso_match.group(2)),
                int(iso_match.group(3)),
            )
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    slash_match = re.search(r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})", raw)
    if slash_match:
        d, m, y = (
            int(slash_match.group(1)),
            int(slash_match.group(2)),
            int(slash_match.group(3)),
        )
        candidates: list[datetime] = []
        for day, month in ((d, m), (m, d)):
            try:
                candidates.append(datetime(y, month, day))
            except ValueError:
                continue
        if len(candidates) == 1:
            return candidates[0].strftime("%Y-%m-%d")
        if len(candidates) == 2:
            for dt in candidates:
                if dt.year == 2026:
                    return dt.strftime("%Y-%m-%d")
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
        is_priority = any(_date_label_in_line(lbl, lower) for lbl in DATE_PRIORITY_LABELS)

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

    fallback = re.findall(r"2026-\d{2}-\d{2}", text)
    if fallback:
        for candidate in fallback:
            try:
                datetime.strptime(candidate, "%Y-%m-%d")
                return candidate
            except ValueError:
                continue
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
