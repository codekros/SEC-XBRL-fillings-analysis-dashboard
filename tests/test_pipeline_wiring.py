import zipfile
from pathlib import Path
import pytest

from cleaning.conversion import extract_zip_files

@pytest.fixture
def temp_raw_and_extract_dirs(tmp_path):
    """
    Creates temporary directories for raw zip files and extracted content.
    """
    raw_dir = tmp_path / "raw"
    extract_dir = tmp_path / "extracted"
    raw_dir.mkdir()
    extract_dir.mkdir()
    return raw_dir, extract_dir

def create_mock_zip(zip_path: Path, files_content: dict):
    """
    Helper to create a mock zip file with the given file names and contents.
    """
    with zipfile.ZipFile(zip_path, "w") as zf:
        for filename, content in files_content.items():
            zf.writestr(filename, content)

def test_extract_zip_files_all(temp_raw_and_extract_dirs):
    raw_dir, extract_dir = temp_raw_and_extract_dirs

    # Create mock zip files
    create_mock_zip(raw_dir / "2020q1.zip", {"sub.txt": "sub 1 content", "num.txt": "num 1 content"})
    create_mock_zip(raw_dir / "2020q2.zip", {"sub.txt": "sub 2 content"})

    # Run extraction
    extract_zip_files(raw_dir, extract_dir)

    # Assertions
    dest_q1 = extract_dir / "2020q1"
    dest_q2 = extract_dir / "2020q2"

    assert dest_q1.exists()
    assert (dest_q1 / "sub.txt").read_text() == "sub 1 content"
    assert (dest_q1 / "num.txt").read_text() == "num 1 content"

    assert dest_q2.exists()
    assert (dest_q2 / "sub.txt").read_text() == "sub 2 content"

def test_extract_zip_files_single_quarter(temp_raw_and_extract_dirs):
    raw_dir, extract_dir = temp_raw_and_extract_dirs

    # Create mock zip files
    create_mock_zip(raw_dir / "2020q1.zip", {"sub.txt": "sub 1 content"})
    create_mock_zip(raw_dir / "2020q2.zip", {"sub.txt": "sub 2 content"})

    # Run extraction targeting only 2020q2
    extract_zip_files(raw_dir, extract_dir, quarter="2020q2")

    # Assertions: 2020q2 should be extracted, 2020q1 should not
    assert not (extract_dir / "2020q1").exists()
    assert (extract_dir / "2020q2").exists()
    assert (extract_dir / "2020q2" / "sub.txt").read_text() == "sub 2 content"

def test_extract_zip_files_skips_existing(temp_raw_and_extract_dirs):
    raw_dir, extract_dir = temp_raw_and_extract_dirs

    # Create a mock zip
    create_mock_zip(raw_dir / "2020q1.zip", {"sub.txt": "new content"})

    # Pre-create the destination directory with some content
    dest_dir = extract_dir / "2020q1"
    dest_dir.mkdir(parents=True)
    existing_file = dest_dir / "sub.txt"
    existing_file.write_text("existing content")

    # Run extraction (should skip since destination exists and contains files)
    extract_zip_files(raw_dir, extract_dir)

    # Content should not be overwritten
    assert existing_file.read_text() == "existing content"

def test_extract_zip_files_missing_quarter(temp_raw_and_extract_dirs, capsys):
    raw_dir, extract_dir = temp_raw_and_extract_dirs

    # Call with non-existent quarter
    extract_zip_files(raw_dir, extract_dir, quarter="2020q1")

    # Ensure no directories were created and extraction reports failure/skip
    assert not (extract_dir / "2020q1").exists()
    captured = capsys.readouterr()
    assert "No ZIP file found for quarter" in captured.out
