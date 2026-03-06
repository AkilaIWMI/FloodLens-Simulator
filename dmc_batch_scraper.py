"""
DMC River Water Level – Batch Date-Range Scraper
=================================================
Iterates day-by-day over a date range, downloads and parses each
DMC water-level PDF report, keeps only Kelani Ganga rows, and
appends everything into a single CSV.

Features
--------
* Checkpoint / resume  – already-processed dates are read from the
  existing output CSV so a restart continues from where it left off.
* Progress bar         – via tqdm (install: pip install tqdm).
* Filters              – only rows where Tributory/River == "Kelani Ganga"
                         are written; the redundant report_date column is
                         dropped and replaced by a clean `date` column.

Usage
-----
    python dmc_batch_scraper.py --start-date 2025-01-01 --end-date 2025-12-12
    python dmc_batch_scraper.py --start-date 2025-01-01 --end-date 2025-12-12 --output my_data.csv
"""

import argparse
import re
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Import the reusable functions from the existing single-day scraper.
# Both files must be in the same directory (or on sys.path).
from dmc_scraper import fetch_listing, download_pdf, extract_table_from_pdf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RIVER_FILTER = "Kelani Ganga"

# Columns to drop from the raw extraction output (redundant with our `date` column).
# NOTE: report_time is kept — it distinguishes multiple reports on the same day.
DROP_COLUMNS = {"report_title", "report_date"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def date_range(start: date, end: date):
    """Yield each date from start to end, inclusive."""
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def load_existing_dates(csv_path: Path) -> set:
    """
    Return the set of date strings already present in the output CSV.
    Used to skip days that were already processed (checkpoint/resume).
    """
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path, usecols=["date"], dtype=str)
        return set(df["date"].dropna().unique())
    except Exception:
        # If the file is malformed / empty, start fresh
        return set()


def scrape_one_day(date_str: str) -> list[dict]:
    """
    Fetch listings for date_str, download + parse every PDF, filter to
    Kelani Ganga rows, and return a list of record dicts.
    Returns an empty list if no reports are found or an error occurs.
    """
    try:
        listings = fetch_listing(date_str)
    except Exception as exc:
        tqdm.write(f"  [WARN] Could not fetch listing for {date_str}: {exc}")
        return []

    if not listings:
        tqdm.write(f"  [INFO] No reports found for {date_str} – skipping.")
        return []

    day_records = []
    for entry in listings:
        try:
            pdf_bytes = download_pdf(entry["pdf_url"])
            records = extract_table_from_pdf(pdf_bytes)
        except Exception as exc:
            tqdm.write(f"  [WARN] Could not process PDF {entry['pdf_url']}: {exc}")
            continue

        for rec in records:
            # Add metadata that the original scraper adds
            rec["report_title"] = entry["title"]
            rec["report_date"] = entry["date"]
            rec["report_time"] = entry["time"]

        day_records.extend(records)

    if not day_records:
        return []

    df = pd.DataFrame(day_records)

    # ---- Filter: keep only Kelani Ganga rows ----
    river_col = "Tributory/River"
    if river_col not in df.columns:
        tqdm.write(f"  [WARN] Column '{river_col}' not found for {date_str}.")
        return []

    df = df[df[river_col].str.strip() == RIVER_FILTER].copy()

    if df.empty:
        tqdm.write(f"  [INFO] No Kelani Ganga rows in reports for {date_str}.")
        return []

    # ---- Add clean `date` column, drop redundant columns ----
    df.insert(0, "date", date_str)
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    tqdm.write(f"  [OK]   {date_str}: {len(df)} Kelani Ganga rows extracted.")
    return df.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Scrape DMC water-level reports over a date range and save "
            "Kelani Ganga data to a single CSV."
        )
    )
    parser.add_argument(
        "--start-date",
        required=True,
        metavar="YYYY-MM-DD",
        help="First date of the range (inclusive).",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        metavar="YYYY-MM-DD",
        help="Last date of the range (inclusive).",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        help=(
            "Output CSV filename. "
            "Defaults to kelani_ganga_<start>_to_<end>.csv"
        ),
    )
    args = parser.parse_args()

    # ---- Validate dates ----
    date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    for label, val in [("--start-date", args.start_date), ("--end-date", args.end_date)]:
        if not date_pattern.match(val):
            print(f"Error: {label} '{val}' is not in YYYY-MM-DD format.")
            sys.exit(1)

    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date)

    if start_date > end_date:
        print("Error: --start-date must be on or before --end-date.")
        sys.exit(1)
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    output_file = Path(
         args.output
         or output_dir / f"kelani_ganga_{args.start_date}_to_{args.end_date}.csv"
    )

    # ---- Checkpoint: find already-processed dates ----
    processed_dates = load_existing_dates(output_file)
    if processed_dates:
        print(
            f"[Resume] Found existing output '{output_file}' with "
            f"{len(processed_dates)} date(s) already processed. "
            "Those dates will be skipped."
        )

    # ---- Build list of dates to process ----
    all_dates = [d.isoformat() for d in date_range(start_date, end_date)]
    pending_dates = [d for d in all_dates if d not in processed_dates]

    total = len(all_dates)
    done_already = total - len(pending_dates)

    print(
        f"\nDate range : {args.start_date} → {args.end_date}  ({total} days total)"
    )
    print(f"Already done : {done_already} day(s)")
    print(f"To process   : {len(pending_dates)} day(s)")
    print(f"Output file  : {output_file}\n")

    if not pending_dates:
        print("Nothing to do – all dates are already in the output file.")
        sys.exit(0)

    # ---- Scrape with progress bar ----
    write_header = not output_file.exists()

    with tqdm(
        total=len(pending_dates),
        desc="Scraping days",
        unit="day",
        dynamic_ncols=True,
    ) as pbar:
        for date_str in pending_dates:
            pbar.set_postfix_str(date_str)
            records = scrape_one_day(date_str)

            if records:
                df_day = pd.DataFrame(records)
                df_day.to_csv(
                    output_file,
                    mode="a",          # append
                    index=False,
                    header=write_header,
                    encoding="utf-8-sig",
                )
                write_header = False   # header written only once

            pbar.update(1)

    # ---- Summary ----
    print(f"\nDone! Output saved to: {output_file}")
    if output_file.exists():
        df_final = pd.read_csv(output_file, on_bad_lines="warn")
        print(f"Total rows in CSV : {len(df_final)}")
        print(f"Dates covered     : {df_final['date'].nunique()}")
        print(f"\nPreview (first 5 rows):\n")
        print(df_final.head().to_string())


if __name__ == "__main__":
    main()
