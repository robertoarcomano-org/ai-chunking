"""
chunk_text.py

Legge un file di testo, lo divide in chunk usando
RecursiveCharacterTextSplitter (langchain) e mostra a schermo
i chunk risultanti.

Installazione dipendenza richiesta:
    pip install langchain-text-splitters

Uso:
    python chunk_text.py input.txt
    python chunk_text.py input.txt --chunk-size 500 --chunk-overlap 50
"""

import argparse
import sys
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: str) -> list[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        documents = splitter.split_text(document)
        return documents


def leggi_file(percorso: str) -> str:
    path = Path(percorso)
    if not path.exists():
        raise FileNotFoundError(f"Il file non esiste: {percorso}")
    return path.read_text(encoding="utf-8")


def mostra_chunk(chunks: list[str]) -> None:
    print(f"\nTotale chunk generati: {len(chunks)}\n")
    for i, chunk in enumerate(chunks, start=1):
        print("=" * 60)
        print(f"CHUNK {i}/{len(chunks)}  (lunghezza: {len(chunk)} caratteri)")
        print("=" * 60)
        print(chunk)
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Divide un file di testo in chunk e li mostra a schermo."
    )
    parser.add_argument("input_file", help="Percorso del file di testo da dividere")
    parser.add_argument(
        "--chunk-size", type=int, default=1000, help="Dimensione massima di ogni chunk (default: 1000)"
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=200, help="Sovrapposizione tra chunk consecutivi (default: 200)"
    )
    args = parser.parse_args()

    try:
        testo = leggi_file(args.input_file)
    except FileNotFoundError as e:
        print(f"Errore: {e}", file=sys.stderr)
        sys.exit(1)

    chunker = TextChunker(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    chunks = chunker.chunk(testo)

    mostra_chunk(chunks)


if __name__ == "__main__":
    main()
