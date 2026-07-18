from __future__ import annotations

from pathlib import Path

from cleaning.cleaner import clean_dataframe
from cleaning.config import (
    EXTRACTED_DIR,
    PROCESSED_DIR,
    SUPPORTED_TABLES,
    RAW_DIR,
)
from cleaning.conversion import extract_zip_files
from cleaning.utils import (
    logger,
    read_txt_file,
    save_parquet,
)


def process_table(
    input_file: Path,
    output_file: Path,
    table_name: str,
) -> None:

    if output_file.exists():

        logger.info(
            "%s already exists. Skipping...",
            output_file.name,
        )

        return

    logger.info("-" * 70)
    logger.info("Processing: %s", input_file)

    df = read_txt_file(input_file)

    cleaned_df, _ = clean_dataframe(
        df=df,
        table_name=table_name,
    )

    save_parquet(
        cleaned_df,
        output_file,
    )


def process_quarter(
    quarter_dir: Path,
    table: str | None = None,
) -> None:

    logger.info("=" * 70)
    logger.info("Quarter: %s", quarter_dir.name)
    logger.info("=" * 70)

    output_dir = PROCESSED_DIR / quarter_dir.name
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tables = [table] if table else SUPPORTED_TABLES

    for table_name in tables:

        input_file = quarter_dir / f"{table_name}.txt"

        if not input_file.exists():

            logger.warning(
                "%s not found. Skipping...",
                input_file.name,
            )

            continue

        output_file = output_dir / f"{table_name}.parquet"

        process_table(
            input_file=input_file,
            output_file=output_file,
            table_name=table_name,
        )


def main(
    quarter: str | None = None,
    table: str | None = None,
) -> None:

    logger.info("")
    logger.info("=" * 70)
    logger.info("SEC XBRL CLEANING PIPELINE")
    logger.info("=" * 70)

    logger.info("Running raw data extraction step...")
    extract_zip_files(raw_dir=RAW_DIR, extract_dir=EXTRACTED_DIR, quarter=quarter)

    if not EXTRACTED_DIR.exists():

        logger.error("Extracted directory not found.")
        return

    if quarter:

        quarter_dir = EXTRACTED_DIR / quarter

        if not quarter_dir.exists():

            logger.error(
                "Quarter '%s' does not exist.",
                quarter,
            )

            return

        quarter_folders = [quarter_dir]

    else:

        quarter_folders = sorted(
            folder
            for folder in EXTRACTED_DIR.iterdir()
            if folder.is_dir()
        )

    logger.info(
        "Found %d quarter(s).",
        len(quarter_folders),
    )

    for folder in quarter_folders:

        process_quarter(
            folder,
            table=table,
        )

    logger.info("")
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="SEC XBRL Data Cleaning Pipeline"
    )

    parser.add_argument(
        "--quarter",
        type=str,
        help="Quarter to process (e.g. 2020q1)",
    )

    parser.add_argument(
        "--table",
        choices=SUPPORTED_TABLES,
        help="Specific table to process",
    )

    args = parser.parse_args()

    main(
        quarter=args.quarter,
        table=args.table,
    )