import zipfile
from pathlib import Path

def extract_zip_files(raw_dir: Path, extract_dir: Path, quarter: str | None = None) -> None:
    """
    Extracts ZIP files from raw_dir to extract_dir.
    If a specific quarter is provided, only that quarter's ZIP file is extracted.
    """
    extract_dir.mkdir(parents=True, exist_ok=True)

    if quarter:
        # Normalize the quarter name to check for zip file
        zip_file = raw_dir / f"{quarter}.zip"
        zip_files = [zip_file] if zip_file.exists() else []
    else:
        zip_files = sorted(raw_dir.glob("*.zip"))

    if not zip_files:
        if quarter:
            print(f" No ZIP file found for quarter '{quarter}' in {raw_dir}", flush=True)
        else:
            print(f" No ZIP files found in {raw_dir}", flush=True)
        return

    for zip_file in zip_files:
        destination = extract_dir / zip_file.stem

        if destination.exists() and any(destination.iterdir()):
            print(f"  {zip_file.name} already extracted so Skipping", flush=True)
            continue

        destination.mkdir(parents=True, exist_ok=True)

        print(f" Extracting {zip_file.name}...", flush=True)

        try:
            with zipfile.ZipFile(zip_file, "r") as archive:
                archive.extractall(destination)

            print(f" Successfully extracted to {destination}", flush=True)

        except zipfile.BadZipFile:
            print(f" {zip_file.name} is corrupted. Skipping...", flush=True)

        except Exception as e:
            print(f" Error extracting {zip_file.name}: {e}", flush=True)

    print("\n Extraction process completed.", flush=True)

if __name__ == "__main__":
    # Inside src/cleaning/, parents[2] resolves to the project root
    project_root = Path(__file__).resolve().parents[2]
    raw_dir = project_root / "data" / "raw"
    extract_dir = project_root / "data" / "extracted"
    extract_zip_files(raw_dir, extract_dir)
