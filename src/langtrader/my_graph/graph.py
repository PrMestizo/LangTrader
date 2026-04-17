import os
from typing import TypedDict, Optional
from langchain_core.runnables import RunnableConfig
# langchain_core.tools import tool - removido
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 1. El Estado Compartido (La Pizarra)
# ============================================================
class MyState(TypedDict):
    ticker: str            
    noticia: str           
    sentimiento_radar: str 
    analisis_tecnico: str
    analisis_fundamental: str
    analisis_sentimiento: str
    decision_final: str
    accion_ejecutada: str

# ============================================================
# 2. Las Herramientas (Los "ojos" y "manos" de los agentes)
# ============================================================
from langtrader.my_graph.tools import (
    buscar_sentimiento_social,
    analizar_grafica_1m,
    evaluar_dependencia_fundamental,
    ejecutar_orden_mercado
)

# Inicializamos el LLM (Asegúrate de tener OPENAI_API_KEY en tu .env)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ============================================================
# 3. Los Nodos (Los Agentes)
# ============================================================
def analista_sentimiento(state: MyState, config: Optional[RunnableConfig] = None):
    print(f"🌍 Sentimiento buscando reacciones sociales para {state['ticker']}...")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un analista de sentimiento de mercado. Usa tus herramientas para evaluar la reacción social minorista."),
        ("human", "Ticker: {ticker}\nNoticia: {noticia}")
    ])
    # Le atamos la herramienta de redes sociales
    agente = prompt | llm.bind_tools([buscar_sentimiento_social])
    respuesta = agente.invoke(state)
    return {"analisis_sentimiento": respuesta.content}

def analista_tecnico(state: MyState, config: Optional[RunnableConfig] = None):
    print(f"📈 Técnico analizando volumen y gráficas de {state['ticker']}...")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un analista técnico cuantitativo. Revisa la acción del precio a muy corto plazo (1 minuto)."),
        ("human", "Ticker: {ticker}\nNoticia: {noticia}")
    ])
    agente = prompt | llm.bind_tools([analizar_grafica_1m])
    respuesta = agente.invoke(state)
    return {"analisis_tecnico": respuesta.content}

def analista_fundamental(state: MyState, config: Optional[RunnableConfig] = None):
    print(f"🏢 Fundamental evaluando el impacto real en {state['ticker']}...")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un analista fundamental implacable. Tu objetivo es ver si la noticia destruye el valor de la empresa o es solo ruido."),
        ("human", "Ticker: {ticker}\nNoticia: {noticia}")
    ])
    agente = prompt | llm.bind_tools([evaluar_dependencia_fundamental])
    respuesta = agente.invoke(state)
    return {"analisis_fundamental": respuesta.content}

def moderador(state: MyState, config: Optional[RunnableConfig] = None):
    print(f"⚖️ Moderador sintetizando datos para {state['ticker']}...")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres el gestor de riesgos. Lee los análisis de tu equipo. Decide: BUY, SELL, HOLD, o REVISAR (si crees que falta información, hay contradicciones o fallos en las peticiones y deben volver a buscar). Tu respuesta DEBE empezar explícitamente con una de esas 4 palabras clave, seguido de una breve justificación."),
        ("human", "Noticia original: {noticia} (Sentimiento Radar: {sentimiento_radar})\n\n-- REPORTES --\nTécnico: {analisis_tecnico}\nFundamental: {analisis_fundamental}\nSentimiento Social: {analisis_sentimiento}")
    ])
    agente = prompt | llm
    respuesta = agente.invoke(state)
    return {"decision_final": respuesta.content}

def ejecutor(state: MyState, config: Optional[RunnableConfig] = None):
    print(f"🚀 Ejecutor procesando orden para {state['ticker']}...")
    decision = state['decision_final'].upper()
    
    if "BUY" in decision:
        print(f"🛒 Ejecutando orden de COMPRA para {state['ticker']}...")
        resultado = ejecutar_orden_mercado.invoke({"ticker": state["ticker"], "accion": "BUY", "cantidad": 1})
    elif "SELL" in decision:
        print(f"💸 Ejecutando orden de VENTA para {state['ticker']}...")
        resultado = ejecutar_orden_mercado.invoke({"ticker": state["ticker"], "accion": "SELL", "cantidad": 1})
    else:
        print(f"⏸️ Ninguna orden ejecutada. (Decisión: HOLD)")
        resultado = "Mantenido (HOLD). Ninguna orden ejecutada hacia la API."
        
    return {"accion_ejecutada": resultado}

# ============================================================
# 4. Conditional Edges
# ============================================================

def router_moderador(state: MyState):
    decision = state.get("decision_final", "").upper()
    if "REVISAR" in decision or "CORREGIR" in decision:
        print("🔄 El Moderador ha solicitado REVISIÓN. Volviendo a los Analistas...")
        return ["Analista_Tecnico", "Analista_Fundamental", "Analista_Sentimiento"]
    return ["Ejecutor"]

# ============================================================
# 5. Construcción del Grafo
# ============================================================
builder = StateGraph(MyState)

builder.add_node("Analista_Tecnico", analista_tecnico)
builder.add_node("Analista_Fundamental", analista_fundamental)
builder.add_node("Analista_Sentimiento", analista_sentimiento)
builder.add_node("Moderador", moderador)
builder.add_node("Ejecutor", ejecutor)

# El flujo: Inicio -> [Técnico, Fundamental, Sentimiento] en paralelo
builder.add_edge(START, "Analista_Tecnico")
builder.add_edge(START, "Analista_Fundamental")
builder.add_edge(START, "Analista_Sentimiento")

# Todos terminan y le pasan la información al Moderador
builder.add_edge("Analista_Tecnico", "Moderador")
builder.add_edge("Analista_Fundamental", "Moderador")
builder.add_edge("Analista_Sentimiento", "Moderador")

# Desde el Moderador, enrutamos condicionalmente a revisar o al Ejecutor
builder.add_conditional_edges("Moderador", router_moderador)

# El Ejecutor toma la decisión final de API y termina el proceso
builder.add_edge("Ejecutor", END)

workflow = builder.compile()