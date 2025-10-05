import pandas as pd
import re
from pypdf import PdfReader
import io

def parse(pdf_path: str) -> pd.DataFrame:
    # 1. PDF Text Extraction
    reader = PdfReader(pdf_path)
    text_content = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_content.append(page_text)
    
    # Split all extracted text into lines and strip whitespace
    # Filter out lines that are purely whitespace or empty
    lines = []
    for page_text in text_content:
        for line in page_text.split('\n'):
            stripped_line = line.strip()
            if stripped_line:
                lines.append(stripped_line)

    # 2. Define Regex Patterns
    # Captures the date in DD-MM-YYYY format
    DATE_PATTERN = r"(\d{2}-\d{2}-\d{4})" 
    # Captures two floating-point numbers with exactly two decimal places at the end of the string
    AMOUNT_BALANCE_PATTERN = r"(\d+\.\d{2})\s+(\d+\.\d{2})$" 
    
    # Pre-compile regex for efficiency
    TRANSACTION_START_REGEX = re.compile(DATE_PATTERN)
    AMOUNT_BALANCE_REGEX = re.compile(AMOUNT_BALANCE_PATTERN)

    # Patterns for lines to explicitly ignore (headers, footers, summaries etc.)
    # Use re.IGNORECASE for robustness
    IGNORE_PATTERNS = [
        re.compile(r"Date Description Debit Amt Credit Amt Balance", re.IGNORECASE),
        re.compile(r"ChatGPT Powered Karbon Bannk", re.IGNORECASE),
        re.compile(r"Date Description Amount Balance", re.IGNORECASE),
        re.compile(r"Total Balance:", re.IGNORECASE),
        re.compile(r"Page \d+ of \d+", re.IGNORECASE),
        re.compile(r"Account Summary", re.IGNORECASE),
        re.compile(r"Statement Period", re.IGNORECASE),
        re.compile(r"Opening Balance", re.IGNORECASE),
        re.compile(r"Closing Balance", re.IGNORECASE),
        re.compile(r"Transactions", re.IGNORECASE),
        re.compile(r"Account Number:", re.IGNORECASE),
        re.compile(r"Description Amount Balance", re.IGNORECASE), # Another possible header variation
        re.compile(r"Karbon Bank", re.IGNORECASE)
    ]

    # Keywords for Debit/Credit classification (case-insensitive)
    CREDIT_KEYWORDS = {"credit", "salary", "deposit", "interest", "refund", "received", "income", "gain", "credited", "cash in"}
    DEBIT_KEYWORDS = {"debit", "payment", "purchase", "withdrawal", "bill", "charge", "emi", "transfer", "spent", "paid", "fees", "fee", "sent", "debited", "cash out"}

    # 3. Iterate and Parse Transactions
    transactions_data = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # Check if the line is a potential start of a new transaction (starts with a date)
        date_match = TRANSACTION_START_REGEX.match(line)
        if not date_match:
            # If not a date, check if it's an ignorable line
            if any(p.search(line) for p in IGNORE_PATTERNS):
                i += 1
                continue
            # If it's neither a date nor an ignorable line, it's likely noise, skip
            i += 1
            continue

        # Found the start of a transaction
        current_date_str = date_match.group(1)
        transaction_lines_buffer = [line] # Buffer to hold all lines for the current transaction

        # Gather all subsequent lines that belong to this transaction
        j = i + 1
        while j < len(lines):
            next_line = lines[j]
            # Stop gathering if the next line is a new transaction or an ignorable pattern
            if TRANSACTION_START_REGEX.match(next_line) or any(p.search(next_line) for p in IGNORE_PATTERNS):
                break
            
            # If the line is not empty (already filtered by `lines` creation, but defensive)
            if next_line.strip():
                transaction_lines_buffer.append(next_line)
            j += 1
        
        # Combine all lines gathered for this transaction into a single string for easier parsing
        full_transaction_text = " ".join(transaction_lines_buffer)

        # Extract transaction value and balance from the end of the combined text
        amount_balance_match = AMOUNT_BALANCE_REGEX.search(full_transaction_text)

        if not amount_balance_match:
            # If amounts/balance are not found, this transaction is incomplete or malformed. Skip.
            i = j # Advance to the next potential transaction start
            continue

        transaction_value_str = amount_balance_match.group(1)
        balance_str = amount_balance_match.group(2)
        
        # Extract description: the text between the date and the transaction value
        # This requires careful indexing based on the combined string
        desc_start_idx = full_transaction_text.find(current_date_str) + len(current_date_str)
        desc_end_idx = amount_balance_match.start(1) # Start of the first amount captured by AMOUNT_BALANCE_REGEX
        
        current_description = full_transaction_text[desc_start_idx:desc_end_idx].strip()

        # Determine Debit/Credit based on keywords in the description
        debit_amt = 0.0
        credit_amt = 0.0
        transaction_value = float(transaction_value_str)

        lower_description = current_description.lower()

        # Check for credit and debit keywords
        is_credit = any(kw in lower_description for kw in CREDIT_KEYWORDS)
        is_debit = any(kw in lower_description for kw in DEBIT_KEYWORDS)

        if is_credit and not is_debit:
            credit_amt = transaction_value
        elif is_debit and not is_credit:
            debit_amt = transaction_value
        elif is_credit and is_debit: # Ambiguous case (e.g., "Transfer Credit" vs "Transfer Payment")
            # Prioritize specific debit terms if present along with credit terms, excluding terms that are also credit
            if any(dw in lower_description for dw in DEBIT_KEYWORDS - CREDIT_KEYWORDS): 
                debit_amt = transaction_value
            elif any(cw in lower_description for cw in CREDIT_KEYWORDS - DEBIT_KEYWORDS): 
                credit_amt = transaction_value
            else: # Still ambiguous, default to debit
                debit_amt = transaction_value
        else: # No specific keywords found, default to debit
            debit_amt = transaction_value

        transactions_data.append({
            'Date': current_date_str, # Keep as string (DD-MM-YYYY) as per previous feedback analysis
            'Description': current_description,
            'Debit Amt': debit_amt,
            'Credit Amt': credit_amt,
            'Balance': float(balance_str)
        })
        
        i = j # Advance the main loop index to the next unprocessed line

    # 4. Create pandas DataFrame
    df = pd.DataFrame(transactions_data)

    # 5. Type Conversion and Cleaning
    expected_columns = ['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance']
    # Ensure all expected columns are present, adding them if missing
    for col in expected_columns:
        if col not in df.columns:
            # Initialize with appropriate dtype to avoid errors later
            if col == 'Date':
                df[col] = pd.Series(dtype='object') # Store dates as strings
            elif col == 'Description':
                df[col] = pd.Series(dtype='object')
            else: # Numeric columns
                df[col] = 0.0

    # Ensure columns are in the exact specified order
    df = df[expected_columns]

    # Convert numeric columns, coercing errors to NaN, then fill NaN with 0
    numeric_cols = ['Debit Amt', 'Credit Amt', 'Balance']
    for col in numeric_cols:
        if col in df.columns:
            # pd.to_numeric handles non-numeric strings by converting to NaN if errors='coerce'
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)
        # No else needed, as missing columns are already initialized to 0.0 above.

    return df