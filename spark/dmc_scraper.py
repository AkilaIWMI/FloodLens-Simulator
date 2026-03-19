import argparse
import io
import logging
import re
import sys
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
import pdfplumber
import pandas as pd
from google.cloud import storage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (overridable via CLI args)
# ---------------------------------------------------------------------------
DEFAULT_BASE_URL = "https://www.dmc.gov.lk"
DEFAULT_LISTING_URL = (
    "https://www.dmc.gov.lk/index.php"
    "?option=com_dmcreports&view=reports&report_type_id=6"
    "&Itemid=277&lang=en"
)

POSITIONAL_COLUMN_MAP: dict[int, str] = {
    0: "River Basin",
    1: "Tributory/River",
    2: "Gauging Station",
    3: "Unit",
    4: "Alert Level",
    5: "Minor Flood Level",
    6: "Major Flood Level",
}

_WL_RE = re.compile(r"water\s*level", re.I)
_RF_RE = re.compile(r"(rf|rainfall)\s*in\s*mm", re.I)


# ---------------------------------------------------------------------------
# Listing scraper
# ---------------------------------------------------------------------------

def fetch_listing(target_date: str, base_url: str, listing_url: str, limit: int = 20) -> list[dict]:
    """Scrape the report listing page and return rows matching target_date."""
    matched = []
    start = 0

    while True:
        url = f"{listing_url}&limit={limit}&limitstart={start}"
        logger.info("Fetching listing page: limitstart=%d", start)
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        table = None
        for t in soup.find_all("table"):
            if t.find("a", href=re.compile(r"\.pdf", re.I)):
                table = t
                break

        if not table:
            for t in soup.find_all("table"):
                rows_check = t.find_all("tr")
                if any(len(r.find_all(["td", "th"])) >= 4 for r in rows_check):
                    table = t
                    break

        if not table:
            logger.warning("No report listing table found on page.")
            break

        data_rows = [r for r in table.find_all("tr") if len(r.find_all("td")) >= 4]
        date_too_old = False

        for row in data_rows:
            cols = row.find_all("td")
            title     = cols[0].get_text(strip=True)
            date_text = cols[1].get_text(strip=True)
            time_text = cols[2].get_text(strip=True)

            link_tag = cols[3].find("a", href=True) or row.find("a", href=re.compile(r"\.pdf", re.I))
            if not link_tag:
                continue

            if date_text == target_date:
                matched.append({
                    "title":   title,
                    "date":    date_text,
                    "time":    time_text,
                    "pdf_url": urljoin(base_url, link_tag["href"]),
                })

            if date_text < target_date:
                date_too_old = True

        if date_too_old or not data_rows:
            break

        start += limit

    return matched


# ---------------------------------------------------------------------------
# PDF downloader
# ---------------------------------------------------------------------------

def download_pdf(url: str) -> bytes:
    logger.info("Downloading: %s", url)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.content


# ---------------------------------------------------------------------------
# Header normalisation
# ---------------------------------------------------------------------------

def _normalise_headers(raw_headers: list[str], num_cols: int) -> list[str]:
    headers: list[str] = []
    wl_counter = rf_counter = 0

    for i in range(num_cols):
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
            headers.append(raw_clean)
        else:
            headers.append(f"_drop_{i}")

    return headers


def _is_subheader_row(row: list[str]) -> bool:
    text = " ".join((c or "") for c in row).lower()
    return "unit" in text and ("alert" in text or "flood" in text)


# ---------------------------------------------------------------------------
# PDF table extractor
# ---------------------------------------------------------------------------

def extract_table_from_pdf(pdf_bytes: bytes) -> list[dict]:
    records = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                if not table or len(table) < 2:
                    continue

                header_row, header_idx = table[0], 0
                for i, row in enumerate(table):
                    row_text = " ".join((c or "").strip().lower() for c in row)
                    if "river basin" in row_text or "station" in row_text:
                        header_row, header_idx = row, i
                        break

                raw_headers = [re.sub(r"\s+", " ", (h or "")).strip() for h in header_row]
                num_cols    = max(len(r) for r in table)
                headers     = _normalise_headers(raw_headers, num_cols)

                for row in table[header_idx + 1:]:
                    if not row or all(not (c or "").strip() for c in row):
                        continue
                    if _is_subheader_row(row):
                        continue

                    record = {
                        headers[j]: (cell or "").strip()
                        for j, cell in enumerate(row)
                        if j < len(headers) and not headers[j].startswith("_drop_")
                    }

                    if sum(1 for v in record.values() if v) < 3:
                        continue

                    records.append(record)

    return records


# ---------------------------------------------------------------------------
# GCS upload
# ---------------------------------------------------------------------------

def upload_to_gcs(df: pd.DataFrame, gcs_path: str) -> None:
    """Upload a DataFrame as CSV to a gs:// path."""
    if not gcs_path.startswith("gs://"):
        raise ValueError(f"gcs_output_path must start with gs://, got: {gcs_path}")

    path_parts   = gcs_path[len("gs://"):].split("/", 1)
    bucket_name  = path_parts[0]
    blob_name    = path_parts[1] if len(path_parts) > 1 else ""

    csv_bytes = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob   = bucket.blob(blob_name)
    blob.upload_from_string(csv_bytes, content_type="text/csv")
    logger.info("Uploaded %d rows to %s", len(df), gcs_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape DMC River Water Level reports for a given date."
    )
    parser.add_argument("--gcs_output_path",  required=True,  help="GCS destination, e.g. gs://bucket/path/output.csv")
    parser.add_argument("--base_url",         default=DEFAULT_BASE_URL,    help="Base URL for DMC website")
    parser.add_argument("--listing_url",      default=DEFAULT_LISTING_URL, help="Listing page URL")
    parser.add_argument("--partition_value",  help="Injected by Airflow DagGenerator (data_interval_start). +1 day = today.")
    args = parser.parse_args()

    if args.partition_value:
        from datetime import datetime, timedelta
        target_date = (datetime.strptime(args.partition_value, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        logger.info("Derived target_date from partition_value + 1: %s", target_date)
    else:
        target_date = args.date

    if not target_date or not re.match(r"^\d{4}-\d{2}-\d{2}$", target_date):
        logger.error("Invalid or missing date. Provide --date or --partition_value in YYYY-MM-DD format.")
        sys.exit(1)

    logger.info("=== DMC Water Level Scraper | date=%s ===", target_date)

    # ------------------------------------------------------------------
    # Step 1: Find matching PDFs
    # ------------------------------------------------------------------
    logger.info("[1/3] Searching report listings...")
    listings = fetch_listing(target_date, args.base_url, args.listing_url)

    if not listings:
        logger.warning("No reports found for %s. Writing empty CSV.", target_date)
        upload_to_gcs(pd.DataFrame(), args.gcs_output_path)
        sys.exit(0)

    logger.info("Found %d report(s) for %s:", len(listings), target_date)
    for entry in listings:
        logger.info("  - %s (%s)", entry["title"], entry["time"])

    # ------------------------------------------------------------------
    # Step 2: Download and extract
    # ------------------------------------------------------------------
    logger.info("[2/3] Downloading and extracting PDF data...")
    all_records = []

    for entry in listings:
        try:
            pdf_bytes = download_pdf(entry["pdf_url"])
            records   = extract_table_from_pdf(pdf_bytes)
            for rec in records:
                rec["report_title"] = entry["title"]
                rec["report_date"]  = entry["date"]
                rec["report_time"]  = entry["time"]
            all_records.extend(records)
            logger.info("  Extracted %d rows from %s (%s)", len(records), entry["title"], entry["time"])
        except Exception as e:
            logger.error("  Error processing %s: %s", entry["pdf_url"], e)

    if not all_records:
        logger.warning("No data extracted from PDFs. Writing empty CSV.")
        upload_to_gcs(pd.DataFrame(), args.gcs_output_path)
        sys.exit(0)

    # ------------------------------------------------------------------
    # Step 3: Upload to GCS
    # ------------------------------------------------------------------
    logger.info("[3/3] Uploading %d rows to GCS...", len(all_records))
    df = pd.DataFrame(all_records)
    upload_to_gcs(df, args.gcs_output_path)

    logger.info("Preview (first 5 rows):\n%s", df.head().to_string())
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()