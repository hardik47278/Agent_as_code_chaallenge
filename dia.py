from langgraph.graph import StateGraph, END, START
from typing_extensions import TypedDict

# --- Define a simple state ---
class AgentState(TypedDict):
    pass  # You can extend this with real state variables

# --- Dummy node functions ---
def planner_node(state): return state
def code_generator_node(state): return state
def code_tester_node(state): return state
def should_continue(state): return "end"  # or "continue"

# --- Build the workflow using StateGraph ---
workflow = StateGraph(AgentState)

workflow.add_node("planner", planner_node)
workflow.add_node("code_generator", code_generator_node)
workflow.add_node("code_tester", code_tester_node)

workflow.add_edge(START, "planner")  # Entry point
workflow.add_edge("planner", "code_generator")
workflow.add_edge("code_generator", "code_tester")
workflow.add_conditional_edges(
    "code_tester",
    should_continue,
    {"continue": "code_generator", "end": END},
)

workflow.add_edge("code_tester", END)  # Optional direct end

# --- Compile the workflow ---
chat_graph = workflow.compile()

# --- Save diagram as PNG ---
with open("workflow.png", "wb") as f:
    f.write(chat_graph.get_graph().draw_mermaid_png())

print("Workflow diagram saved as workflow.png")
