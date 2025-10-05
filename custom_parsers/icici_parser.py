
import pdfplumber
import pandas as pd
from typing import Optional

SCHEMA = ['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance']

def parse(pdf_path: str) -> Optional[pd.DataFrame]:
    try:
        rows = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables or []:
                    for row in table or []:
                        if not row:
                            continue
                        first = str(row[0]).strip() if row[0] is not None else ""
                        if first.lower().startswith("date") or "karbon" in "".join(map(str,row)).lower():
                            continue
                        row = list(row)[:5]
                        while len(row) < 5:
                            row.append(None)
                        rows.append(row)
        df = pd.DataFrame(rows, columns=SCHEMA)
        if 'Description' in df.columns:
            df['Description'] = df['Description'].astype(str).str.replace('T o', 'To', regex=False)
        for c in df.columns:
            try:
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', ''), errors="coerce")
            except Exception:
                pass
        return df
    except FileNotFoundError:
        raise
    except Exception as e:
        raise RuntimeError(f"Local fallback parser failed: {e}")
