from typing import TypedDict, Optional
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END


# ============================================================
# State
# ============================================================

class MyState(TypedDict):
    """Estado compartido entre todos los nodos del grafo."""
    user_input: str
    llm_output: str


# ============================================================
# Nodes
# ============================================================

def Agente1(state: MyState, config: Optional[RunnableConfig] = None):
    # Node logic here (e.g., LLM call, tool use, data processing)
    updated_state = {"llm_output": "Updated output based on input"}
    return updated_state


def Agente2(state: MyState, config: Optional[RunnableConfig] = None):
    # Node logic here (e.g., LLM call, tool use, data processing)
    updated_state = {"llm_output": "Updated output based on input"}
    return updated_state


def Agente3(state: MyState, config: Optional[RunnableConfig] = None):
    # Node logic here (e.g., LLM call, tool use, data processing)
    updated_state = {"llm_output": "Updated output based on input"}
    return updated_state


def Moderador(state: MyState, config: Optional[RunnableConfig] = None):
    # Node logic here (e.g., LLM call, tool use, data processing)
    updated_state = {"llm_output": "Updated output based on input"}
    return updated_state


def Ejecutor(state: MyState, config: Optional[RunnableConfig] = None):
    # Node logic here (e.g., LLM call, tool use, data processing)
    updated_state = {"llm_output": "Updated output based on input"}
    return updated_state


# ============================================================
# Edges
# ============================================================

def moderador_routing_function(state: MyState):
    if state["llm_output"] == "Agente1":
        return "Agente1"
    elif state["llm_output"] == "Agente2":
        return "Agente2"
    elif state["llm_output"] == "Agente3":
        return "Agente3"
    elif state["llm_output"] == "Ejecutor":
        return "Ejecutor"
    else:
        return "END"


# ============================================================
# Graph
# ============================================================

builder = StateGraph(MyState)

# Nodos
builder.add_node("Agente1", Agente1)
builder.add_node("Agente2", Agente2)
builder.add_node("Agente3", Agente3)
builder.add_node("Moderador", Moderador)
builder.add_node("Ejecutor", Ejecutor)

# Edges: START -> Agentes en paralelo -> Moderador
builder.add_edge(START, "Agente1")
builder.add_edge(START, "Agente2")
builder.add_edge(START, "Agente3")
builder.add_edge("Agente1", "Moderador")
builder.add_edge("Agente2", "Moderador")
builder.add_edge("Agente3", "Moderador")

# Conditional edges: Moderador decide el siguiente paso
builder.add_conditional_edges(
    "Moderador",
    moderador_routing_function,
    {"Agente1": "Agente1", "Agente2": "Agente2", "Agente3": "Agente3", "Ejecutor": "Ejecutor"}
)

builder.add_edge("Ejecutor", END)

workflow = builder.compile()
