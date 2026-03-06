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
                    # Try using the first row as header
                    header_row = table[0]
                    header_idx = 0

                # Clean header names
                headers = []
                for h in header_row:
                    h = (h or "").strip()
                    # Collapse whitespace
                    h = re.sub(r"\s+", " ", h)
                    headers.append(h)

                # Process data rows
                for row in table[header_idx + 1 :]:
                    if not row or all(not (cell or "").strip() for cell in row):
                        continue

                    record = {}
                    for j, cell in enumerate(row):
                        if j < len(headers):
                            key = headers[j] if headers[j] else f"col_{j}"
                            record[key] = (cell or "").strip()

                    # Skip rows that look like sub-headers or empty
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
