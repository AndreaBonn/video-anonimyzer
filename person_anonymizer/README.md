# Person Anonymizer Tool v6.0

Oscuramento automatico persone in video di sorveglianza con pipeline multi-strategia YOLO e revisione manuale OpenCV.

## Requisiti

- Python 3.9+
- ffmpeg installato nel PATH
- 8 GB RAM (16 GB consigliati per 1080p+)
- Display per modalita' manuale

## Installazione

```bash
cd person_anonymizer
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ffmpeg -version  # verifica
```

## Utilizzo

```bash
# Modalita' manuale (default consigliato)
python person_anonymizer.py video.mp4

# Solo automatico
python person_anonymizer.py video.mp4 --mode auto

# Output personalizzato
python person_anonymizer.py video.mp4 -o /output/risultato.mp4

# Senza file aggiuntivi
python person_anonymizer.py video.mp4 --mode auto --no-debug --no-report
```

## Opzioni CLI

| Argomento | Descrizione |
|-----------|-------------|
| `input` | Percorso video da elaborare |
| `--mode` / `-M` | `manual` o `auto` |
| `--output` / `-o` | Percorso file di output |
| `--method` / `-m` | `pixelation` o `blur` |
| `--no-debug` | Disabilita video debug |
| `--no-report` | Disabilita CSV report |

## Comandi revisione manuale

| Tasto | Azione |
|-------|--------|
| Freccia destra / Spazio | Avanti |
| Freccia sinistra | Indietro |
| Click sinistro | Aggiungi punto poligono |
| Invio | Chiudi poligono |
| Ctrl+Z | Annulla ultimo punto |
| D | Toggle modalita' elimina |
| Esc | Annulla poligono corrente |
| Q | Conferma e esci |

## File di output

| File | Descrizione |
|------|-------------|
| `*_anonymized.mp4` | Video con persone oscurate |
| `*_debug.mp4` | Video con poligoni visibili |
| `*_report.csv` | Statistiche rilevamento |
| `*_annotations.json` | Annotazioni per ripristino |

## Formati supportati

MP4, MOV, AVI, MKV, WebM
