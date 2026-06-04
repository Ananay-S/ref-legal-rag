import argparse
import json
import logging
import os
import sys
from pathlib import Path

from chunking import build_chunks, trim_document_header
from cleaning import clean_pages, normalize_document_text
from db_writer import compute_document_id, init_db, write_document
from metadata_extractor import extract_metadata
from pdf_extractor import extract_pages

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("preprocessing")


def build_document(metadata: dict, chunks: list[dict], text: str, doc_id: str) -> dict:
    return {
        "id": doc_id,
        "metadata": {
            "appellant": metadata["appellant"],
            "respondent": metadata["respondent"],
            "year": metadata["year"],
            "month": metadata["month"],
            "day": metadata["day"],
            "author": metadata["author"],
            "bench": metadata["bench"],
            "court": metadata["court"],
            "jurisdiction": metadata["jurisdiction"],
        },
        "text": text,
        "chunks": chunks,
    }


def write_artifacts(output_dir: Path, pdf_stem: str, doc_payload: dict, text_payload: str) -> tuple[Path, Path]:
    json_dir = output_dir / "json"
    txt_dir = output_dir / "text"
    json_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)

    json_path = json_dir / f"{pdf_stem}.json"
    txt_path = txt_dir / f"{pdf_stem}.txt"
    json_path.write_text(json.dumps(doc_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    txt_path.write_text(text_payload, encoding="utf-8")
    return json_path, txt_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Legal RAG preprocessing pipeline")
    parser.add_argument(
        "--pdf-dir",
        default=os.environ.get("PDF_DIR", "/data/pdfs"),
        help="Directory containing judgment PDFs",
    )
    parser.add_argument(
        "--db-path",
        default=os.environ.get("DB_PATH", "/data/db/legal.db"),
        help="Path to write/update the SQLite DB",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("OUTPUT_DIR", "/data/preprocessed"),
        help="Directory for deterministic JSON and debug text artifacts",
    )
    parser.add_argument("--dry-run", action="store_true", help="Process the first PDF only and print JSON to stdout")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    output_dir = Path(args.output_dir)
    logger.info("Initializing Legal RAG preprocessing pipeline")
    logger.info("PDF Directory: %s", pdf_dir)
    logger.info("Database Path: %s", args.db_path)
    logger.info("Output Directory: %s", output_dir)
    logger.info("Dry Run: %s", args.dry_run)

    if not pdf_dir.exists():
        logger.error("PDF directory does not exist: %s", pdf_dir)
        sys.exit(1)

    pdf_files = sorted([path for path in pdf_dir.iterdir() if path.suffix.lower() == ".pdf"])
    if not pdf_files:
        logger.error("No PDF files found in directory: %s", pdf_dir)
        sys.exit(1)

    if args.dry_run:
        pdf_files = [pdf_files[0]]
        logger.info("Dry-run active. Only processing the first file: %s", pdf_files[0].name)

    db_conn = None
    if not args.dry_run:
        db_parent = Path(args.db_path).parent
        db_parent.mkdir(parents=True, exist_ok=True)
        db_conn = init_db(args.db_path)

    success_count = 0
    failure_count = 0

    for pdf_path in pdf_files:
        logger.info("--- Starting pipeline for: %s ---", pdf_path.name)
        try:
            raw_pages = extract_pages(str(pdf_path))
            metadata = extract_metadata(raw_pages)
            cleaned_pages = clean_pages(raw_pages)
            raw_data = normalize_document_text(cleaned_pages)
            chunk_source = trim_document_header(raw_data)
            chunks = build_chunks(chunk_source)

            doc_id = compute_document_id(
                metadata["appellant"],
                metadata["respondent"],
                metadata["year"],
                metadata["month"],
                metadata["day"],
            )
            doc_payload = build_document(metadata, chunks, raw_data, doc_id)

            if args.dry_run:
                print(json.dumps(doc_payload, indent=2, ensure_ascii=False))
            else:
                write_document(db_conn, doc_id, doc_payload)
                write_artifacts(output_dir, pdf_path.stem, doc_payload, raw_data)

            success_count += 1
        except Exception as exc:
            logger.exception("Unrecoverable error while processing %s: %s", pdf_path, exc)
            failure_count += 1
            if args.dry_run:
                sys.exit(1)

    if db_conn is not None:
        db_conn.close()

    logger.info("--- Pipeline Execution Summary ---")
    logger.info("Total PDFs processed successfully: %s", success_count)
    logger.info("Total PDFs failed: %s", failure_count)

    if failure_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
