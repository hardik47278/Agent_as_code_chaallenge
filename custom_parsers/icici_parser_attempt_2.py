import pandas as pd
import re
from pypdf import PdfReader

def parse(pdf_path: str) -> pd.DataFrame:
    reader = PdfReader(pdf_path)
    all_text = []
    for page in reader.pages:
        all_text.append(page.extract_text())
    
    # Combine all text and split into lines
    lines = "\n".join(all_text).split('\n')
    lines = [line.strip() for line in lines if line.strip()] # Remove empty lines and strip whitespace

    # Define Regex Patterns
    # Date pattern at the beginning of the line
    DATE_PATTERN = re.compile(r"^\d{2}-\d{2}-\d{4}")
    # Amount and Balance pattern at the end of the line, allowing for comma separators and two decimal places
    AMOUNT_BALANCE_PATTERN = re.compile(r"(\d{1,3}(?:,\d{3})*\.\d{2})\s+(\d{1,3}(?:,\d{3})*\.\d{2})$")
    # Header pattern to identify and skip header rows
    HEADER_PATTERN = re.compile(r"Date\s+Description\s+Debit\s+Amt\s+Credit\s+Amt\s+Balance", re.IGNORECASE)

    transactions_data = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # Skip known non-transaction lines like headers or specific footers
        if HEADER_PATTERN.search(line) or "ChatGPT Powered Karbon Bannk" in line:
            i += 1
            continue

        date_match = DATE_PATTERN.match(line)
        if date_match:
            current_date = date_match.group(0)
            
            description_parts = []
            transaction_value = None
            balance = None
            
            # Start processing from the current line, potential multi-line description
            current_line_idx = i
            
            # Helper to clean and convert amount strings (e.g., "1,234.56" to 1234.56)
            def clean_amount_string(amount_str):
                return float(amount_str.replace(',', ''))

            # Loop to capture multi-line description and find amounts/balance
            while current_line_idx < len(lines):
                process_line = lines[current_line_idx].strip()
                if not process_line: # Skip empty lines that might appear in between transaction parts
                    current_line_idx += 1
                    continue

                # Check if this line signals the start of a new transaction or is a header
                # This ensures we don't consume lines belonging to the next transaction or irrelevant content
                if current_line_idx != i and (DATE_PATTERN.match(process_line) or HEADER_PATTERN.search(process_line) or "ChatGPT Powered Karbon Bannk" in process_line):
                    break # Break from inner loop, current transaction block ends
                
                # Determine the part of the line to search for amounts and append to description
                line_for_amount_check = process_line
                if current_line_idx == i: # On the first line, description starts after the date
                    line_for_amount_check = process_line[len(current_date):]
                    
                amount_balance_match = AMOUNT_BALANCE_PATTERN.search(line_for_amount_check)

                if amount_balance_match:
                    transaction_value = clean_amount_string(amount_balance_match.group(1))
                    balance = clean_amount_string(amount_balance_match.group(2))
                    
                    # Add description part up to the amount
                    desc_part = line_for_amount_check[:amount_balance_match.start(1)].strip()
                    if desc_part:
                        description_parts.append(desc_part)
                    
                    current_line_idx += 1 # Consume this line
                    break # Amounts found, transaction block ends here
                else:
                    # No amounts found on this line, it's entirely description or a non-transaction line
                    if current_line_idx == i:
                        desc_part = process_line[len(current_date):].strip()
                    else:
                        desc_part = process_line
                    
                    if desc_part: # Only add non-empty description parts
                        description_parts.append(desc_part)
                    
                    current_line_idx += 1
            
            # Join all collected description parts
            full_description = ' '.join(filter(None, description_parts)).strip()

            debit_amt = 0.0
            credit_amt = 0.0

            # Determine Debit/Credit based on description keywords if a transaction_value was found
            if transaction_value is not None:
                description_lower = full_description.lower()
                credit_keywords = ['credit', 'deposit', 'salary', 'interest', 'refund', 'incoming', 'payment received']
                debit_keywords = ['debit', 'payment', 'purchase', 'withdraw', 'bill', 'charge', 'transfer out', 'fee', 'emi', 'expense']

                is_credit = any(k in description_lower for k in credit_keywords)
                is_debit = any(k in description_lower for k in debit_keywords)

                if is_credit and not is_debit:
                    credit_amt = transaction_value
                elif is_debit and not is_credit: # Explicit debit keywords
                    debit_amt = transaction_value
                else: # Ambiguous or neither, default to debit as per plan
                    debit_amt = transaction_value

            transactions_data.append({
                'Date': current_date,
                'Description': full_description,
                'Debit Amt': debit_amt,
                'Credit Amt': credit_amt,
                'Balance': balance if balance is not None else 0.0
            })
            
            # Move the main iterator 'i' to the next line after the current transaction block
            # This is crucial to prevent re-processing lines already consumed by the current transaction
            i = current_line_idx 
        else:
            i += 1 # Move to the next line if no date match

    df = pd.DataFrame(transactions_data)

    # Define the exact required columns
    required_columns = ['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance']
    
    # Ensure all required columns exist, even if no data was parsed, and set default values/types
    for col in required_columns:
        if col not in df.columns:
            if col == 'Date':
                df[col] = pd.NaT # Not a Time
            elif col == 'Description':
                df[col] = ''
            else:
                df[col] = 0.0

    # Type Conversion and Cleaning
    if not df.empty:
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce') # Specify format
        for col in ['Debit Amt', 'Credit Amt', 'Balance']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(float)
        
        # Ensure column order
        df = df[required_columns]
    else: # If df is empty, create an empty DataFrame with correct columns and dtypes
        df = pd.DataFrame(columns=required_columns)
        df['Date'] = df['Date'].astype('datetime64[ns]')
        df['Description'] = df['Description'].astype(str)
        for col in ['Debit Amt', 'Credit Amt', 'Balance']:
            df[col] = df[col].astype(float)

    return df