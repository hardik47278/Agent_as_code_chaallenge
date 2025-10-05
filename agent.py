# agent.py
import os
import re
import sys
import argparse
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from typing import TypedDict
from pypdf import PdfReader
import importlib.util
import traceback
from langgraph.graph import StateGraph, END

# Load .env
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
genai.configure(api_key=GEMINI_API_KEY)

def clean_code_block(text: str) -> str:
    
    if not isinstance(text, str):
        return ""
    # remove triple backticks and language hints
    text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text, flags=re.MULTILINE)
    text = text.replace("```", "")
    return text.strip()

def extract_pdf_text(pdf_path: str) -> str:
    """Extracts text from a given PDF file (pypdf)."""
    if not os.path.exists(pdf_path):
        return "ERROR: PDF file not found."
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        
        return "\n".join(line.strip() for line in text.splitlines() if line.strip())
    except Exception as e:
        return f"ERROR reading PDF: {e}"

def _local_planner_fallback(target: str, csv_schema: list[str], pdf_sample: str) -> str:
    return (
        "1. Read PDF and locate transaction table header (Date, Description, Amount, Balance).\n"
        "2. Merge continuation lines where line doesn't start with a date.\n"
        "3. Extract columns, coerce numbers, return pandas DataFrame with schema "
        f"{csv_schema}.\n"
        "4. Use pypdf for extraction, pandas for manipulation, handle missing files gracefully."
    )

def _local_code_fallback(csv_schema: list[str]) -> str:
    
    return f'''\
# FALLBACK parser (agent generated)
import pandas as pd
from pypdf import PdfReader
from typing import Optional

SCHEMA = {csv_schema!r}

def parse(pdf_path: str) -> Optional[pd.DataFrame]:
    try:
        reader = PdfReader(pdf_path)
        rows = []
        for page in reader.pages:
            text = page.extract_text() or ""
            for line in text.splitlines():
                ln = line.strip()
                if not ln:
                    continue
                # naive heuristic: lines with a date at start (DD/MM/YYYY or YYYY-MM-DD or DD-MM-YYYY)
                if ln[:2].isdigit() and ("/" in ln or "-" in ln):
                    parts = ln.split()
                    # put entire remainder into Description if fewer tokens
                    if len(parts) >= 4:
                        date = parts[0]
                        amount = parts[-2]
                        balance = parts[-1]
                        description = " ".join(parts[1:-2])
                    else:
                        # fallback: place entire line into Description
                        date = parts[0]
                        description = " ".join(parts[1:])
                        amount = None
                        balance = None
                    rows.append([date, description, amount, balance])
        df = pd.DataFrame(rows, columns=SCHEMA[:4]) if rows else pd.DataFrame(columns=SCHEMA)
        # Coerce numeric columns
        for c in df.columns:
            if c.lower() in ('amount', 'balance', 'debit', 'credit'):
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', ''), errors='coerce')
        return df
    except Exception as e:
        raise RuntimeError(f"Fallback parser error: {{e}}")
'''

def safe_generate_text(prompt: str) -> str:
    """Call Gemini but return fallback marker on error."""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        resp = model.generate_content(prompt)
        
        return getattr(resp, "text", str(resp))
    except Exception as e:
        print(f"Warning: Gemini generation failed ({type(e).__name__}): {e}")
        return ""

# ----------------- Agent state typing -----------------

class AgentState(TypedDict):
    target: str
    pdf_path: str
    csv_path: str
    plan: str
    code: str
    feedback: str
    attempts_left: int
    attempt_no: int



def planner_node(state: AgentState) -> AgentState:
    print("---PLANNING---")
    csv_schema = pd.read_csv(state['csv_path']).columns.tolist()
    pdf_text_sample = extract_pdf_text(state['pdf_path'])[:4000]
    prompt = f"""
You are an expert Python developer. Create a concise step-by-step plan to write a Python function:

def parse(pdf_path: str) -> pd.DataFrame

Target bank: {state['target']}
Expected schema: {csv_schema}

PDF sample:
{pdf_text_sample}

Focus on: locating transactions, handling multi-line descriptions, extracting Date/Description/Amount/Balance.
Return only the plan (no code).
"""
    resp_text = safe_generate_text(prompt)
    if not resp_text:
        resp_text = _local_planner_fallback(state['target'], csv_schema, pdf_text_sample)
    state['plan'] = resp_text
    return state

def code_generator_node(state: AgentState) -> AgentState:
    attempt_no = state.get("attempt_no", 1)
    print(f"---GENERATING CODE (attempt {attempt_no})---")
    csv_schema = pd.read_csv(state['csv_path']).columns.tolist()

    prompt = f"""
You are an expert Python developer. Based on the plan below, write a complete Python script that defines:

def parse(pdf_path: str) -> pd.DataFrame

Constraints:
- The returned DataFrame must have columns exactly: {csv_schema}
- Use pypdf for PDF extraction and pandas for data manipulation.
- Do NOT include any markdown fences or explanatory text — return ONLY raw Python code.

Plan:
{state['plan']}

Previous feedback:
{state.get('feedback','No previous feedback')}
"""
    resp_text = safe_generate_text(prompt)
    cleaned = clean_code_block(resp_text or "")
    # If nothing or looks like explanation, fallback to local stable parser
    if not cleaned or len(cleaned) < 50:
        print("LLM produced empty/short output — using local fallback parser.")
        cleaned = _local_code_fallback(csv_schema)
    # save attempt code in state
    state['code'] = cleaned
    return state

def code_tester_node(state: AgentState) -> AgentState:
    print("---TESTING CODE---")
    parser_dir = "custom_parsers"
    os.makedirs(parser_dir, exist_ok=True)
    attempt_no = state.get("attempt_no", 1)
    parser_filename = f"{state['target']}_parser_attempt_{attempt_no}.py"
    parser_path = os.path.join(parser_dir, parser_filename)

    # write code
    with open(parser_path, "w", encoding="utf-8") as f:
        f.write(state['code'])

    # Try to import and run the parse() function
    module_name = f"parser_{state['target']}_attempt_{attempt_no}"
    try:
        # unload if previously loaded
        if module_name in sys.modules:
            del sys.modules[module_name]

        spec = importlib.util.spec_from_file_location(module_name, parser_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "parse"):
            raise AttributeError("Generated parser module has no `parse` function.")

        result_df = module.parse(state['pdf_path'])
        if not isinstance(result_df, pd.DataFrame):
            raise TypeError("parse() did not return a pandas DataFrame.")

        expected_df = pd.read_csv(state['csv_path'])

        # Compare frames exactly
        try:
            pd.testing.assert_frame_equal(result_df.reset_index(drop=True),
                                          expected_df.reset_index(drop=True),
                                          check_dtype=False,
                                          check_like=True)
            print("---TEST PASSED---")
            state['feedback'] = "success"
            # Save final parser to canonical name
            final_path = os.path.join(parser_dir, f"{state['target']}_parser.py")
            with open(final_path, "w", encoding="utf-8") as f:
                f.write(state['code'])
            print(f"Saved working parser to: {final_path}")
        except AssertionError as ae:
            err_msg = f"DataFrames not equal: {ae}"
            print("---TEST FAILED: Data mismatch---")
            state['feedback'] = err_msg
        except Exception as e:
            raise

    except Exception as e:
        tb = traceback.format_exc()
        print(f"---TEST FAILED: Exception during import/run---\n{type(e).__name__}: {e}\n{tb}")
        state['feedback'] = f"{type(e).__name__}: {e}\n{tb}"

    finally:
        # decrement attempts and increment attempt counter
        state['attempts_left'] = max(0, state['attempts_left'] - 1)
        state['attempt_no'] = state.get("attempt_no", 1) + 1
        return state

# ----------------- Graph control -----------------

def should_continue(state: AgentState):
    if state.get('feedback') == "success":
        return "end"
    if state.get('attempts_left', 0) <= 0:
        print("---MAX ATTEMPTS REACHED---")
        return "end"
    return "continue"

workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_node)
workflow.add_node("code_generator", code_generator_node)
workflow.add_node("code_tester", code_tester_node)
workflow.set_entry_point("planner")
workflow.add_edge("planner", "code_generator")
workflow.add_edge("code_generator", "code_tester")
workflow.add_conditional_edges(
    "code_tester",
    should_continue,
    {"continue": "code_generator", "end": END},
)
app = workflow.compile()

# ----------------- Main -----------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Agent for generating PDF parsers.")
    parser.add_argument("--target", required=True, help="The target bank, e.g., 'icici'")
    args = parser.parse_args()

    target_bank = args.target.strip().lower()
    # specific mapping for icici sample name if needed
    bank_prefix = "icic" if target_bank == "icici" else target_bank

    pdf_path = os.path.join("data", target_bank, f"{bank_prefix}_sample.pdf")
    csv_path = os.path.join("data", target_bank, f"{bank_prefix}_sample.csv")

    if not os.path.exists(pdf_path) or not os.path.exists(csv_path):
        print(f"Error: Data files not found for target '{target_bank}'.")
        print(f"  - Looked for PDF at: {pdf_path}")
        print(f"  - Looked for CSV at: {csv_path}")
        sys.exit(1)

    initial_state = {
        "target": target_bank,
        "pdf_path": pdf_path,
        "csv_path": csv_path,
        "plan": "",
        "code": "",
        "feedback": "",
        "attempts_left": 3,
        "attempt_no": 1,
    }

    final_state = app.invoke(initial_state)

    if final_state.get('feedback') == "success":
        print(f"\n✅ Successfully generated parser: custom_parsers/{target_bank}_parser.py")
    else:
        attempts_used = 3 - final_state.get('attempts_left', 0)
        print(f"\n❌ Failed to generate a working parser for {target_bank} after {attempts_used} attempts.")
        print("Last feedback:\n", final_state.get('feedback', '(no feedback)'))
        print("Check files under custom_parsers/ for saved attempts.")




