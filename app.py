import os
from groq import Groq
from typing_extensions import TypedDict
import pandas as pd
from PyPDF2 import PdfReader
from langgraph.graph import StateGraph, START, END

# Load GROQ API key
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

# State definition
class State(TypedDict):
    pdf_path: str
    csv_path: str
    extracted_text: str
    plan: str
    generated_code: str
    parser_output: pd.DataFrame
    reference_output: pd.DataFrame
    test_passed: bool

# Nodes
def read_pdf_node(state: State) -> dict:
    reader = PdfReader(state['pdf_path'])
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        text += page_text if page_text else ""
    return {'extracted_text': text}


def load_csv_node(state:State)->dict:
    df = pd.read_csv(State['csv_path'])
    return {"refrence_output":df}

def planner_node(state: State) -> dict:
    plan = "1. Read PDF\n2. Extract transactions\n3. Structure as DataFrame\n4. Compare to CSV"
    return {'plan': plan}

def generate_code_node(state: State) -> dict:
    """Call Groq LLM to generate parsing code from extracted text + plan."""
    prompt = f"""
    You are an assistant that writes Python code.
    Given this extracted bank statement text:

    {state['extracted_text'][:2000]}  # truncated for safety

    And this plan:
    {state['plan']}

    Write Python code that parses the PDF into a pandas DataFrame
    with columns: [Date, Description, Amount, Balance].
    Return only code, no explanation.
    """
    
    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",  # or llama-3, gemma-7b, etc.
        messages=[
            {"role": "system", "content": "You are a helpful Python coding assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    
    code = response.choices[0].message.content
    return {'generated_code': code}

def run_test_node(state: State) -> dict:
    test_passed = state['parser_output'].equals(state['reference_output'])
    return {'test_passed': test_passed}

def test_conditions(state: State):
    state['retries'] = state.get("retries",0)+1
    if state['test_passed'] or state['retries'] >=3:
        return "END"
    else:
        "generate"

# Build graph
graph = StateGraph(State)
graph.add_node("START", START)
graph.add_node("plan", planner_node)
graph.add_node("generate", generate_code_node)
graph.add_node("run_test", run_test_node)
graph.add_node("END", END)

graph.add_edge("START", "plan")
graph.add_edge("plan", "generate")
graph.add_edge("generate", "run_test")
graph.add_conditional_edges("run_test", test_conditions)

chat_graph = graph.compile()

# Show diagram
from IPython.display import Image
display(Image(chat_graph.get_graph().draw_mermaid_png()))
