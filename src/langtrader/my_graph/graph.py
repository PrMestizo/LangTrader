import os
from typing import TypedDict, Optional
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
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

# ============================================================
# 2. Las Herramientas (Los "ojos" y "manos" de los agentes)
# ============================================================
@tool
def buscar_sentimiento_social(ticker: str) -> str:
    """Busca en Twitter (X) y Reddit la reacción del público minorista en los últimos 5 minutos."""
    # Aquí iría tu código real conectando a la API de Reddit o X.
    return f"Simulación: Mucho pánico en Reddit sobre {ticker}. Retail está vendiendo masivamente."

@tool
def analizar_grafica_1m(ticker: str) -> str:
    """Obtiene velas de 1 minuto y volumen en tiempo real para ver si las instituciones compran o venden."""
    # Aquí te conectarías a la API de Alpaca o Yahoo Finance
    return f"Simulación: Volumen institucional inusualmente alto en {ticker}. Están absorbiendo las ventas (comprando barato)."

@tool
def evaluar_dependencia_fundamental(ticker: str, contexto: str) -> str:
    """Revisa los balances para ver cuánto depende la empresa de la noticia actual."""
    # Aquí te conectarías a Financial Modeling Prep o similar
    return f"Simulación: El producto mencionado en la noticia representa solo el 5% de los ingresos de {ticker}."

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

def moderador_ejecutor(state: MyState, config: Optional[RunnableConfig] = None):
    print(f"⚖️ Ejecutor sintetizando datos para {state['ticker']}...")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres el gestor de riesgos y tomador de decisiones final. Lee los análisis de tu equipo. Si la caída es por pánico minorista pero el impacto fundamental es bajo y hay compras institucionales, compra. Decide: BUY, SELL o HOLD y justifica brevemente."),
        ("human", """
        Noticia original: {noticia} (Sentimiento Radar: {sentimiento_radar})
        
        -- REPORTES --
        Técnico: {analisis_tecnico}
        Fundamental: {analisis_fundamental}
        Sentimiento Social: {analisis_sentimiento}
        """)
    ])
    agente = prompt | llm
    respuesta = agente.invoke(state)
    return {"decision_final": respuesta.content}

# ============================================================
# 4. Construcción del Grafo
# ============================================================
builder = StateGraph(MyState)

builder.add_node("Analista_Tecnico", analista_tecnico)
builder.add_node("Analista_Fundamental", analista_fundamental)
builder.add_node("Analista_Sentimiento", analista_sentimiento)
builder.add_node("Ejecutor", moderador_ejecutor)

# El flujo: Inicio -> [Técnico, Fundamental, Sentimiento] en paralelo
builder.add_edge(START, "Analista_Tecnico")
builder.add_edge(START, "Analista_Fundamental")
builder.add_edge(START, "Analista_Sentimiento")

# Todos terminan y le pasan la información al Ejecutor
builder.add_edge("Analista_Tecnico", "Ejecutor")
builder.add_edge("Analista_Fundamental", "Ejecutor")
builder.add_edge("Analista_Sentimiento", "Ejecutor")

# El Ejecutor toma la decisión y termina el proceso
builder.add_edge("Ejecutor", END)

workflow = builder.compile()