import os
from typing import TypedDict, Optional
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from langtrader.logger import logger

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
    # --- Campos estructurados de la decisión ---
    decision_accion: str                # BUY, SELL, HOLD o REVISAR
    precio_stop_loss: float
    precio_take_profit: float
    justificacion: str
    accion_ejecutada: str
    intentos_revision: int

# ============================================================
# 1.5. Modelo Pydantic para Salida Estructurada
# ============================================================
class DecisionModerador(BaseModel):
    """Decisión estructurada del moderador/gestor de riesgos."""
    decision_accion: str = Field(
        description=(
            "La acción a tomar. DEBE ser estrictamente una de estas 4 opciones: "
            "'BUY' (comprar), 'SELL' (vender), 'HOLD' (mantener sin operar), "
            "o 'REVISAR' (si los reportes tienen fallos, contradicciones o falta "
            "información y los analistas deben volver a buscar)."
        )
    )
    precio_stop_loss: float = Field(
        description=(
            "Precio exacto de Stop-Loss. Nivel de precio donde se cierra la posición "
            "para limitar pérdidas. Para BUY: usar el mínimo reciente del Analista "
            "Técnico menos un pequeño margen (ej. mínimo - 0.5%). Para SELL: usar el "
            "máximo reciente más un pequeño margen (ej. máximo + 0.5%). "
            "Debe ser 0.0 si la acción es 'HOLD' o 'REVISAR'."
        )
    )
    precio_take_profit: float = Field(
        description=(
            "Precio exacto de Take-Profit. Nivel de precio donde se cierra la posición "
            "para asegurar beneficios. DEBE respetar un ratio Riesgo/Beneficio mínimo "
            "de 1:2 respecto al Stop-Loss. Por ejemplo, si el riesgo (distancia al SL) "
            "es $2, el beneficio (distancia al TP) debe ser al menos $4. "
            "Debe ser 0.0 si la acción es 'HOLD' o 'REVISAR'."
        )
    )
    justificacion: str = Field(
        description=(
            "Breve justificación de la decisión, explicando la lógica detrás de la "
            "acción elegida y los niveles de precio seleccionados."
        )
    )

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

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=8), reraise=True)
def safe_invoke(agente, state):
    return agente.invoke(state)

def analista_sentimiento(state: MyState, config: Optional[RunnableConfig] = None):
    logger.info(f"Sentimiento buscando reacciones sociales para {state['ticker']}...")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un analista de sentimiento de mercado. Usa tus herramientas para evaluar la reacción social minorista."),
        ("human", "Ticker: {ticker}\nNoticia: {noticia}")
    ])
    # Le atamos la herramienta de redes sociales
    agente = prompt | llm.bind_tools([buscar_sentimiento_social])
    try:
        respuesta = safe_invoke(agente, state)
        return {"analisis_sentimiento": respuesta.content}
    except Exception as e:
        logger.error(f"Error crítico en Analista Sentimiento tras reintentos: {e}")
        return {"analisis_sentimiento": "Error obteniendo sentimiento social (LLM Timeout/RateLimit)."}

def analista_tecnico(state: MyState, config: Optional[RunnableConfig] = None):
    logger.info(f"Técnico analizando volumen y gráficas de {state['ticker']}...")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un analista técnico cuantitativo. Revisa la acción del precio a muy corto plazo (1 minuto)."),
        ("human", "Ticker: {ticker}\nNoticia: {noticia}")
    ])
    agente = prompt | llm.bind_tools([analizar_grafica_1m])
    try:
        respuesta = safe_invoke(agente, state)
        return {"analisis_tecnico": respuesta.content}
    except Exception as e:
        logger.error(f"Error crítico en Analista Técnico tras reintentos: {e}")
        return {"analisis_tecnico": "Error analizando gráficas (LLM Timeout/RateLimit)."}

def analista_fundamental(state: MyState, config: Optional[RunnableConfig] = None):
    logger.info(f"Fundamental evaluando el impacto real en {state['ticker']}...")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un analista fundamental implacable. Tu objetivo es ver si la noticia destruye el valor de la empresa o es solo ruido."),
        ("human", "Ticker: {ticker}\nNoticia: {noticia}")
    ])
    agente = prompt | llm.bind_tools([evaluar_dependencia_fundamental])
    try:
        respuesta = safe_invoke(agente, state)
        return {"analisis_fundamental": respuesta.content}
    except Exception as e:
        logger.error(f"Error crítico en Analista Fundamental tras reintentos: {e}")
        return {"analisis_fundamental": "Error analizando fundamentales (LLM Timeout/RateLimit)."}

MAX_INTENTOS_REVISION = 2

def moderador(state: MyState, config: Optional[RunnableConfig] = None):
    intentos = state.get('intentos_revision', 0) + 1
    logger.info(f"Moderador sintetizando datos para {state['ticker']}... (intento {intentos})")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Eres un gestor de riesgos cuantitativo de élite. Tu trabajo es sintetizar los reportes de tu equipo de analistas y tomar una decisión de trading estructurada.

INSTRUCCIONES CRÍTICAS:
1. Extrae el PRECIO ACTUAL y el MÍNIMO RECIENTE del reporte del Analista Técnico.
2. Si decides BUY:
   - Coloca el Stop-Loss ligeramente por debajo del mínimo reciente (ej. mínimo - 0.5%).
   - Calcula la distancia de riesgo: precio_actual - stop_loss.
   - El Take-Profit DEBE estar a una distancia mínima de 2x esa distancia por encima del precio actual (ratio R/B mínimo 1:2).
3. Si decides SELL:
   - Coloca el Stop-Loss ligeramente por encima del máximo reciente (ej. máximo + 0.5%).
   - Calcula la distancia de riesgo: stop_loss - precio_actual.
   - El Take-Profit DEBE estar a una distancia mínima de 2x esa distancia por debajo del precio actual.
4. Si decides HOLD o REVISAR: todos los precios y riesgo deben ser 0.0.
5. El riesgo sugerido debe estar entre 1.0% y 2.0% del capital para BUY/SELL.
6. Usa REVISAR solo si los reportes tienen errores, fallos en las peticiones, contradicciones graves o falta información esencial."""),
        ("human", """Noticia original: {noticia} (Sentimiento Radar: {sentimiento_radar})

-- REPORTES DEL EQUIPO --
Técnico: {analisis_tecnico}
Fundamental: {analisis_fundamental}
Sentimiento Social: {analisis_sentimiento}""")
    ])
    agente_estructurado = prompt | llm.with_structured_output(DecisionModerador)
    try:
        orden = safe_invoke(agente_estructurado, state)
        
        logger.info(f"Decisión: {orden.decision_accion} | SL: {orden.precio_stop_loss} | TP: {orden.precio_take_profit}")
        logger.info(f"Justificación: {orden.justificacion}")

        return {
            "decision_accion": orden.decision_accion,
            "precio_stop_loss": orden.precio_stop_loss,
            "precio_take_profit": orden.precio_take_profit,
            "justificacion": orden.justificacion,
            "intentos_revision": intentos,
        }
    except Exception as e:
        logger.error(f"Error crítico en Moderador tras reintentos: {e}")
        return {
            "decision_accion": "HOLD",
            "precio_stop_loss": 0.0,
            "precio_take_profit": 0.0,
            "justificacion": "HOLD forzado por caída severa del LLM (Rate Limit/503).",
            "intentos_revision": intentos,
        }

def ejecutor(state: MyState, config: Optional[RunnableConfig] = None):
    logger.info(f"Ejecutor procesando orden para {state['ticker']}...")
    accion = state["decision_accion"].upper()
    
    if accion == "BUY":
        logger.info(f"Ejecutando orden de COMPRA para {state['ticker']}... SL: {state['precio_stop_loss']} | TP: {state['precio_take_profit']}")
        resultado = ejecutar_orden_mercado.invoke({
            "ticker": state["ticker"],
            "accion": "BUY",
            "stop_loss": state["precio_stop_loss"],
            "take_profit": state["precio_take_profit"]
        })
    elif accion == "SELL":
        logger.info(f"Ejecutando orden de VENTA para {state['ticker']}... SL: {state['precio_stop_loss']} | TP: {state['precio_take_profit']}")
        resultado = ejecutar_orden_mercado.invoke({
            "ticker": state["ticker"],
            "accion": "SELL",
            "stop_loss": state["precio_stop_loss"],
            "take_profit": state["precio_take_profit"]
        })
    else:
        logger.info(f"Ninguna orden ejecutada. (Decisión: {accion})")
        resultado = f"Mantenido ({accion}). Ninguna orden ejecutada hacia la API."
        
    return {"accion_ejecutada": resultado}

# ============================================================
# 4. Conditional Edges
# ============================================================
def router_moderador(state: MyState):
    accion = state.get("decision_accion", "").upper()
    intentos = state.get("intentos_revision", 0)

    if accion == "REVISAR":
        if intentos >= MAX_INTENTOS_REVISION:
            logger.warning(f"Circuit Breaker: {intentos} revisiones alcanzadas. Forzando HOLD para proteger capital.")
            state["decision_accion"] = "HOLD"
            state["precio_stop_loss"] = 0.0
            state["precio_take_profit"] = 0.0
            state["justificacion"] = f"HOLD forzado por circuit breaker tras {intentos} revisiones sin resolución."
            return ["Ejecutor"]
        logger.info(f"El Moderador ha solicitado REVISIÓN ({intentos}/{MAX_INTENTOS_REVISION}). Volviendo a los Analistas...")
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