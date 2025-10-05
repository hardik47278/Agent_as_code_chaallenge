import pypdf
import pandas as pd
import re

def parse(pdf_path: str) -> pd.DataFrame:
    """
    Parses a bank statement PDF and extracts transaction data into a pandas DataFrame.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        pd.DataFrame: A DataFrame with columns ['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance'].
                      Date column is datetime, amount columns are float.
    """

    # 1. PDF Text Extraction
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text()
    except Exception as e:
        # Handle potential file reading errors or PDF corruption
        print(f"Error reading PDF {pdf_path}: {e}")
        return pd.DataFrame(columns=['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance'])

    lines = text.split('\n')

    # 2. Define Regex Patterns
    DATE_PATTERN = r"^\d{2}-\d{2}-\d{4}"
    # Captures two floating point numbers at the end of the line, typically with two decimal places.
    AMOUNT_BALANCE_PATTERN = r"(\d{1,3}(?:,\d{3})*\.\d{2})\s+(\d{1,3}(?:,\d{3})*\.\d{2})$"
    # This pattern also handles comma as thousands separator for amounts.
    HEADER_PATTERN = r"Date Description Debit Amt Credit Amt Balance"
    # Specific non-transaction text identified in the plan
    NON_TRANSACTION_TEXT_INDICATORS = ["ChatGPT Powered Karbon Bannk"]

    # Keywords to determine Debit/Credit
    # These are illustrative and might need tuning based on actual bank statements.
    CREDIT_KEYWORDS = ['Credit', 'Salary', 'Deposit', 'Interest', 'Refund', 'Incoming', 'Income', 'Reversal', 'Receipt', 'Top-up']
    DEBIT_KEYWORDS = ['Debit', 'Payment', 'Purchase', 'Withdrawal', 'Bill', 'Charge', 'EMI', 'Transfer Out', 'Fee', 'Outgoing', 'POS']

    transactions_data = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip headers, footers, and known non-transaction lines
        if not line or \
           re.search(HEADER_PATTERN, line, re.IGNORECASE) or \
           any(indicator.lower() in line.lower() for indicator in NON_TRANSACTION_TEXT_INDICATORS):
            i += 1
            continue

        # 3. Iterate and Parse Transactions
        date_match = re.match(DATE_PATTERN, line)
        if date_match:
            current_date = date_match.group(0)

            # Attempt to extract transaction value and balance from the end of the line
            amount_balance_match = re.search(AMOUNT_BALANCE_PATTERN, line)
            
            transaction_value = 0.0
            balance = 0.0
            
            if amount_balance_match:
                try:
                    # Remove commas before converting to float
                    transaction_value = float(amount_balance_match.group(1).replace(',', ''))
                    balance = float(amount_balance_match.group(2).replace(',', ''))
                except ValueError:
                    # If conversion fails for some reason, default to 0.0
                    pass 
            else:
                # If amounts/balance not found on the line starting with a date,
                # we assume this line is not a valid transaction start as per the plan.
                i += 1
                continue # Skip to the next line

            # Initialize description from the current line
            # Remove date and the captured amount/balance patterns to get the raw description part
            description_part = line[len(current_date):]
            if amount_balance_match:
                # Only remove the matched amount/balance part if it was found
                description_part = description_part[:amount_balance_match.start() - len(current_date)]
            
            current_description_parts = [description_part.strip()]

            # Handle Multi-line Description
            j = i + 1
            while j < len(lines):
                next_line = lines[j].strip()
                # If the next line is empty, is a header, or starts with a date pattern,
                # it's considered the end of the current transaction's description.
                if not next_line or \
                   re.match(DATE_PATTERN, next_line) or \
                   re.search(HEADER_PATTERN, next_line, re.IGNORECASE) or \
                   any(indicator.lower() in next_line.lower() for indicator in NON_TRANSACTION_TEXT_INDICATORS):
                    break 
                
                current_description_parts.append(next_line)
                j += 1
            
            description = " ".join(current_description_parts).strip()

            # Determine Debit/Credit based on keywords in description
            debit_amt = 0.0
            credit_amt = 0.0

            # Prioritize credit keywords. If found, it's a credit.
            if any(keyword.lower() in description.lower() for keyword in CREDIT_KEYWORDS):
                credit_amt = transaction_value
            # Otherwise, it's considered a debit by default.
            else:
                debit_amt = transaction_value
            
            transactions_data.append({
                'Date': current_date,
                'Description': description,
                'Debit Amt': debit_amt,
                'Credit Amt': credit_amt,
                'Balance': balance
            })
            i = j # Advance the main iterator to the line after the current transaction's description
        else:
            i += 1 # Not a transaction start, move to the next line

    # 4. Create pandas DataFrame
    df = pd.DataFrame(transactions_data)

    # If no transactions were found, return an empty DataFrame with correct columns
    if df.empty:
        return pd.DataFrame(columns=['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance'])

    # 5. Type Conversion and Cleaning
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
    df['Debit Amt'] = pd.to_numeric(df['Debit Amt'], errors='coerce').fillna(0.0)
    df['Credit Amt'] = pd.to_numeric(df['Credit Amt'], errors='coerce').fillna(0.0)
    df['Balance'] = pd.to_numeric(df['Balance'], errors='coerce').fillna(0.0)

    # Ensure the DataFrame columns match the expected schema and are in the correct order
    df = df[['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance']]

    # 6. Return DataFrame
    return df