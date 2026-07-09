"""
chunk_text_paragraph_aware.py

Divide un file di testo in chunk allineati ai confini di PARAGRAFO
(o, quando necessario, di FRASE) — non spezza mai un paragrafo o una
frase a metà, a differenza di RecursiveCharacterTextSplitter puro.

Strategia:
1. Il testo viene diviso in paragrafi (separatore "\n\n").
2. I paragrafi vengono accorpati in un chunk finché la somma delle
   loro lunghezze non supera chunk_size.
3. Se un SINGOLO paragrafo è già più lungo di chunk_size, viene
   diviso in frasi (mai a metà frase) e le frasi vengono accorpate
   con la stessa logica.
4. Solo se una singola FRASE è più lunga di chunk_size (caso raro,
   es. testo senza punteggiatura), viene divisa per parola.

Installazione dipendenza richiesta:
    pip install nltk

Al primo utilizzo nltk scarica il modello per il sentence tokenizer:
    python -c "import nltk; nltk.download('punkt_tab')"

Uso:
    python chunk_text_paragraph_aware.py input.txt
    python chunk_text_paragraph_aware.py input.txt --chunk-size 1000 --chunk-overlap 100
"""

import argparse
import sys
from pathlib import Path

import nltk


class ParagraphAwareChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: str) -> list[str]:
        paragraphs = [p.strip() for p in document.split("\n\n") if p.strip()]

        # Espande i paragrafi troppo lunghi in una lista di "unità"
        # (paragrafo intero se ci sta, altrimenti le sue frasi)
        units = []
        for p in paragraphs:
            if len(p) <= self.chunk_size:
                units.append(p)
            else:
                units.extend(self._split_in_frasi(p))

        return self._accorpa_unita(units)

    def _split_in_frasi(self, paragrafo: str) -> list[str]:
        frasi = nltk.sent_tokenize(paragrafo, language="italian")
        risultato = []
        for f in frasi:
            if len(f) <= self.chunk_size:
                risultato.append(f)
            else:
                # Caso estremo: una frase da sola supera chunk_size.
                # Ultima risorsa: divide per parola.
                risultato.extend(self._split_per_parola(f))
        return risultato

    def _split_per_parola(self, frase: str) -> list[str]:
        parole = frase.split(" ")
        pezzi, corrente = [], ""
        for parola in parole:
            candidato = (corrente + " " + parola).strip()
            if len(candidato) <= self.chunk_size:
                corrente = candidato
            else:
                if corrente:
                    pezzi.append(corrente)
                corrente = parola
        if corrente:
            pezzi.append(corrente)
        return pezzi

    def _accorpa_unita(self, units: list[str]) -> list[str]:
        """Accorpa le unità (paragrafi o frasi) in chunk, rispettando
        chunk_size e applicando un overlap testuale tra chunk consecutivi."""
        chunks = []
        corrente = ""

        for unit in units:
            candidato = (corrente + "\n\n" + unit).strip() if corrente else unit

            if len(candidato) <= self.chunk_size:
                corrente = candidato
            else:
                if corrente:
                    chunks.append(corrente)
                # Nuovo chunk che riparte con overlap dal chunk precedente
                overlap_text = self._prendi_overlap(corrente)
                corrente = (overlap_text + "\n\n" + unit).strip() if overlap_text else unit

        if corrente:
            chunks.append(corrente)

        return chunks

    def _prendi_overlap(self, testo_precedente: str) -> str:
        """Estrae le ultime frasi complete del chunk precedente, fino a
        un massimo di chunk_overlap caratteri, per usarle come contesto
        di raccordo. Non taglia mai una frase a metà, né all'inizio né
        alla fine dell'overlap."""
        if not testo_precedente or self.chunk_overlap <= 0:
            return ""

        # Tokenizza in frasi l'ultimo paragrafo del testo precedente
        # (è sufficiente: l'overlap non deve mai coprire più di un chunk)
        ultimo_paragrafo = testo_precedente.split("\n\n")[-1]
        frasi = nltk.sent_tokenize(ultimo_paragrafo, language="italian")

        frasi_overlap = []
        lunghezza_totale = 0
        # Percorre le frasi dalla fine, accumulando finché resta
        # sotto chunk_overlap
        for frase in reversed(frasi):
            if lunghezza_totale + len(frase) > self.chunk_overlap and frasi_overlap:
                break
            frasi_overlap.insert(0, frase)
            lunghezza_totale += len(frase)

        return " ".join(frasi_overlap)


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
        description="Divide un file di testo in chunk allineati a paragrafi/frasi."
    )
    parser.add_argument("input_file", help="Percorso del file di testo da dividere")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Dimensione massima di ogni chunk (default: 1000)")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Sovrapposizione tra chunk consecutivi (default: 200)")
    args = parser.parse_args()

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        print("Scaricamento modello nltk 'punkt_tab' (una tantum)...")
        nltk.download("punkt_tab")

    try:
        testo = leggi_file(args.input_file)
    except FileNotFoundError as e:
        print(f"Errore: {e}", file=sys.stderr)
        sys.exit(1)

    chunker = ParagraphAwareChunker(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    chunks = chunker.chunk(testo)

    mostra_chunk(chunks)


if __name__ == "__main__":
    main()
