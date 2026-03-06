"""
DMC River Water Level Scraper
Scrapes PDF reports from Sri Lanka Disaster Management Centre and extracts
river water level data into CSV.
"""

import argparse
import io
import re
import sys
import tempfile
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
import pdfplumber
import pandas as pd

BASE_URL = "https://www.dmc.gov.lk"
LISTING_URL = (
    "https://www.dmc.gov.lk/index.php"
    "?option=com_dmcreports&view=reports&report_type_id=6"
    "&Itemid=277&lang=en"
)


def fetch_listing(target_date: str, limit: int = 20) -> list[dict]:
    """Scrape the report listing page and return rows matching target_date."""
    matched = []
    start = 0

    while True:
        url = f"{LISTING_URL}&limit={limit}&limitstart={start}"
        print(f"  Fetching listing page: limitstart={start}")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # The page has multiple tables; find the report listing table
        # (the one that contains PDF download links, not the search form)
        table = None
        for t in soup.find_all("table"):
            if t.find("a", href=re.compile(r"\.pdf", re.I)):
                table = t
                break

        if not table:
            # Fallback: try a table with enough columns
            for t in soup.find_all("table"):
                rows_check = t.find_all("tr")
                if any(len(r.find_all(["td", "th"])) >= 4 for r in rows_check):
                    table = t
                    break

        if not table:
            print("  No report listing table found on page.")
            break

        rows = table.find_all("tr")
        found_any = False
        date_too_old = False

        data_rows = [r for r in rows if len(r.find_all("td")) >= 4]

        for row in data_rows:
            cols = row.find_all("td")

            title = cols[0].get_text(strip=True)
            date_text = cols[1].get_text(strip=True)
            time_text = cols[2].get_text(strip=True)

            link_tag = cols[3].find("a", href=True)
            if not link_tag:
                # Try other columns for the link
                link_tag = row.find("a", href=re.compile(r"\.pdf", re.I))
            if not link_tag:
                continue

            href = link_tag["href"]

            if date_text == target_date:
                found_any = True
                matched.append({
                    "title": title,
                    "date": date_text,
                    "time": time_text,
                    "pdf_url": urljoin(BASE_URL, href),
                })

            # Dates are descending; once we pass our target date, no need to go further
            if date_text < target_date:
                date_too_old = True

        # After scanning all rows on this page:
        if date_too_old:
            # We've gone past the target date on this page — stop regardless
            break

        # No data rows returned means we've exhausted all pages
        if not data_rows:
            break

        start += limit

    return matched


def download_pdf(url: str) -> bytes:
    """Download a PDF and return its bytes."""
    print(f"  Downloading: {url}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.content


# ---------------------------------------------------------------------------
# Column normalisation helpers
# ---------------------------------------------------------------------------

# Positional columns that pdfplumber often fails to label (multi-line headers).
# These positions are stable across all DMC water-level PDF reports.
POSITIONAL_COLUMN_MAP: dict[int, str] = {
    0: "River Basin",
    1: "Tributory/River",
    2: "Gauging Station",
    3: "Unit",
    4: "Alert Level",
    5: "Minor Flood Level",
    6: "Major Flood Level",
}

# Regex patterns for the time-varying water-level / rainfall columns.
# We normalise them into generic names so every report produces the same schema.
_WL_RE = re.compile(r"water\s*level", re.I)
_RF_RE = re.compile(r"(rf|rainfall)\s*in\s*mm", re.I)


def _normalise_headers(raw_headers: list[str], num_cols: int) -> list[str]:
    """
    Build a clean, consistent header list.

    Strategy:
    1. Use the positional map for columns 0-6 (always the same in DMC PDFs).
    2. Scan the remaining columns for water-level / rainfall patterns and
       assign generic names: Water Level 1, Water Level 2, Rainfall.
    3. Keep Remarks and Rising/Falling as-is.
    4. Mark anything left over as _drop (to be removed later).
    """
    headers: list[str] = []
    wl_counter = 0
    rf_counter = 0

    for i in range(num_cols):
        # --- positional columns (0-6) ---
        if i in POSITIONAL_COLUMN_MAP:
            headers.append(POSITIONAL_COLUMN_MAP[i])
            continue

        raw = raw_headers[i] if i < len(raw_headers) else ""
        raw_clean = re.sub(r"\s+", " ", (raw or "")).strip()
        raw_lower = raw_clean.lower()

        if "remarks" in raw_lower:
            headers.append("Remarks")
        elif "rising" in raw_lower or "falling" in raw_lower:
            headers.append("Water Level Rising or Falling")
        elif _WL_RE.search(raw_clean):
            wl_counter += 1
            headers.append(f"Water Level {wl_counter}")
        elif _RF_RE.search(raw_clean):
            rf_counter += 1
            headers.append(f"Rainfall {rf_counter}" if rf_counter > 1 else "Rainfall")
        elif raw_clean:
            # Keep any other legitimately named column
            headers.append(raw_clean)
        else:
            headers.append(f"_drop_{i}")

    return headers


def _is_subheader_row(row: list[str]) -> bool:
    """Return True if the row looks like a repeated sub-header (Unit, Alert Level …)."""
    text = " ".join((c or "") for c in row).lower()
    return "unit" in text and ("alert" in text or "flood" in text)


def extract_table_from_pdf(pdf_bytes: bytes) -> list[dict]:
    """Extract water level table data from a PDF."""
    records = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                if not table or len(table) < 2:
                    continue

                # Find the header row
                header_row = None
                header_idx = None
                for i, row in enumerate(table):
                    row_text = " ".join((cell or "").strip().lower() for cell in row)
                    if "river basin" in row_text or "station" in row_text:
                        header_row = row
                        header_idx = i
                        break

                if header_row is None:
                    header_row = table[0]
                    header_idx = 0

                # Clean raw header strings
                raw_headers = [
                    re.sub(r"\s+", " ", (h or "")).strip() for h in header_row
                ]

                # Determine actual column count from widest row
                num_cols = max(len(r) for r in table)

                # Build normalised headers
                headers = _normalise_headers(raw_headers, num_cols)

                # Process data rows
                for row in table[header_idx + 1 :]:
                    if not row or all(not (cell or "").strip() for cell in row):
                        continue

                    if _is_subheader_row(row):
                        continue

                    record = {}
                    for j, cell in enumerate(row):
                        if j < len(headers):
                            key = headers[j]
                            if key.startswith("_drop_"):
                                continue
                            record[key] = (cell or "").strip()

                    # Skip rows with too few values
                    values = [v for v in record.values() if v]
                    if len(values) < 3:
                        continue

                    records.append(record)

    return records


def main():
    parser = argparse.ArgumentParser(
        description="Scrape DMC River Water Level reports for a given date."
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Target date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--output",
        help="Output CSV filename (default: water_level_YYYY-MM-DD.csv)",
    )
    args = parser.parse_args()

    target_date = args.date
    # Validate date format
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", target_date):
        print(f"Error: Invalid date format '{target_date}'. Use YYYY-MM-DD.")
        sys.exit(1)

    output_file = args.output or f"water_level_{target_date}.csv"

    print(f"Scraping DMC reports for date: {target_date}")

    # Step 1: Find matching PDFs
    print("\n[1/3] Searching report listings...")
    listings = fetch_listing(target_date)

    if not listings:
        print(f"No reports found for {target_date}.")
        sys.exit(0)

    print(f"  Found {len(listings)} report(s) for {target_date}:")
    for entry in listings:
        print(f"    - {entry['title']} ({entry['time']})")

    # Step 2: Download and extract data from each PDF
    print("\n[2/3] Downloading and extracting PDF data...")
    all_records = []

    for entry in listings:
        try:
            pdf_bytes = download_pdf(entry["pdf_url"])
            records = extract_table_from_pdf(pdf_bytes)
            # Add metadata columns
            for rec in records:
                rec["report_title"] = entry["title"]
                rec["report_date"] = entry["date"]
                rec["report_time"] = entry["time"]
            all_records.extend(records)
            print(f"    Extracted {len(records)} rows from {entry['title']} ({entry['time']})")
        except Exception as e:
            print(f"    Error processing {entry['pdf_url']}: {e}")

    if not all_records:
        print("No data extracted from PDFs.")
        sys.exit(0)

    # Step 3: Save to CSV
    print(f"\n[3/3] Saving {len(all_records)} rows to {output_file}...")
    df = pd.DataFrame(all_records)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Done! Output saved to {output_file}")

    # Show preview
    print(f"\nPreview (first 5 rows):\n")
    print(df.head().to_string())


if __name__ == "__main__":
    main()
