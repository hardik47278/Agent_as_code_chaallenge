from PyPDF2 import PdfReader
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import pandas as pd
from typing_extensions import TypedDict
from typing import Literal
import os

# ------------------- Setup -------------------
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="", api_key=api_key)  # Set your model name here

# ------------------- State Definition -------------------
class State(TypedDict):
    pdf_path: str
    csv_path: str
    extracted_text: str
    plan: str
    generated_code: str
    parser_output: pd.DataFrame
    reference_output: pd.DataFrame
    test_passed: bool
    retries: int

# ------------------- Node Functions -------------------
def read_pdf(state: State) -> dict:
    reader = PdfReader(state['pdf_path'])
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        text += page_text if page_text else ""
    return {"extracted_text": text}

def planner(state: State) -> dict:
    plan = f"""
1. Read PDF from {state['pdf_path']}.
2. Identify transaction lines containing Date, Description, Amount, Balance.
3. Extract each transaction field using regex or string parsing.
4. Structure data into pandas DataFrame with columns: [Date, Description, Amount, Balance].
5. Handle errors, missing values, or inconsistent formatting.
6. Compare output DataFrame to CSV schema in {state['csv_path']}.
"""
    return {"plan": plan}

def generate_code(state: State) -> dict:
    prompt = f"""
You are an agent responsible for writing a Python function 'parse_pdf' 
that extracts transactions from {state['pdf_path']} and structures them to match {state['csv_path']} 
based on the following plan:

{state['plan']}
"""
    response = llm.invoke(prompt)
    state['generated_code'] = response.content
    return {"generated_code": state['generated_code']}

def run_parser(state: State) -> dict:
    local_vars = {}
    try:
        exec(state['generated_code'], {}, local_vars)
        parse_fn = local_vars.get("parse_pdf")
        if not parse_fn:
            raise ValueError("Generated code must define parse_pdf function")
        df = parse_fn(state['pdf_path'])
        if not isinstance(df, pd.DataFrame):
            raise ValueError("parse_pdf must return a pandas DataFrame")
        return {"parser_output": df}
    except Exception as e:
        print("Parser execution failed:", e)
        return {"parser_output": pd.DataFrame()}

def load_csv(state: State) -> dict:
    df = pd.read_csv(state['csv_path'])
    return {"reference_output": df}

def run_test(state: State) -> dict:
    test_passed = state['parser_output'].equals(state['reference_output'])
    return {"test_passed": test_passed}

# ------------------- Conditional -------------------
def test_conditions(state: State) -> str:
    state['retries'] = state.get("retries", 0) + 1
    if state['test_passed'] or state['retries'] >= 3:
        return "END"
    else:
        # Reset parser output for retry
        state['parser_output'] = pd.DataFrame()
        return "generate_code"

# ------------------- Build Graph -------------------
graph = StateGraph(State)

# Add nodes
graph.add_node("START", START)
graph.add_node("read_pdf", read_pdf)
graph.add_node("planner", planner)
graph.add_node("generate_code", generate_code)
graph.add_node("run_parser", run_parser)
graph.add_node("load_csv", load_csv)
graph.add_node("run_test", run_test)
graph.add_node("END", END)

# Add edges
graph.add_edge("START", "read_pdf")
graph.add_edge("read_pdf", "planner")
graph.add_edge("planner", "generate_code")
graph.add_edge("generate_code", "run_parser")
graph.add_edge("run_parser", "load_csv")
graph.add_edge("load_csv", "run_test")
graph.add_conditional_edges("run_test", test_conditions)

# Compile graph
chat_graph = graph.compile()

# ------------------- CLI -------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="Bank folder name")
    args = parser.parse_args()

    base = f"data/{args.target}"
    init_state = {
        "pdf_path": f"{base}/{args.target}_sample.pdf",
        "csv_path": f"{base}/{args.target}_sample.csv",
        "extracted_text": "",
        "plan": "",
        "generated_code": "",
        "parser_output": pd.DataFrame(),
        "reference_output": pd.DataFrame(),
        "test_passed": False,
        "retries": 0
    }

    final_state = chat_graph.invoke(init_state)
    if final_state.get("test_passed"):
        print("✅ Test Passed")
    else:
        print("❌ Test Failed")
