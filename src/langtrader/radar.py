import asyncio
import os
import time
from dotenv import load_dotenv
from transformers import pipeline
from alpaca.data.live import NewsDataStream
from tenacity import retry, stop_after_attempt, wait_exponential

from langtrader.logger import logger

# IMPORTAMOS TU GRAFO DE AGENTES
from langtrader.my_graph.graph import workflow

# Cargar API Keys del archivo .env
load_dotenv()
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

PALABRAS_CLAVE = ["bancarrota", "adquisicion", "fraude", "dimision", "acuerdo"]

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=8), reraise=True)
async def safe_ainvoke(workflow, estado):
    return await workflow.ainvoke(estado)

logger.info("Cargando modelo FinBERT (esto tarda unos segundos)...")
nlp_finanzas = pipeline("sentiment-analysis", model="ProsusAI/finbert")
logger.info("FinBERT cargado.")


async def procesar_noticia(noticia):
    """Callback que se ejecuta cada vez que Alpaca recibe una noticia"""
    # Alpaca envía un objeto de noticia. Extraemos titular y el primer símbolo.
    titular = noticia.headline
    simbolos = noticia.symbols
    ticker = simbolos[0] if simbolos else "Desconocido"
    titular_lower = titular.lower()

    try:
        # Mostrar la noticia en crudo para confirmar que el WebSockets funciona
        logger.info(f"📰 [{ticker}] {titular}")

        # FILTRO A: Palabras clave
        if any(palabra in titular_lower for palabra in PALABRAS_CLAVE):
            logger.info(f"ALERTA: Palabra clave detectada en {ticker}!")
        
        # FILTRO B: Triaje con IA ligera
        # Corremos el pipeline en un thread separado para no bloquear el WebSockets de Alpaca
        resultado_nlp_list = await asyncio.to_thread(nlp_finanzas, titular)
        resultado_nlp = resultado_nlp_list[0]
        sentimiento = resultado_nlp["label"]
        confianza = resultado_nlp["score"]
        
        logger.info(f"FinBERT dictamina: {sentimiento} (Confianza: {confianza:.2f})")
        
        if sentimiento in ["negative", "positive"] and confianza > 0.85:
            logger.info(f"Noticia crítica. Despertando al Comité de LangGraph para {ticker}...")
            
            # --- CONEXIÓN CON TU ARCHIVO graph.py ---
            estado_inicial = {
                "ticker": ticker,
                "noticia": titular,
                "sentimiento_radar": sentimiento,
                "analisis_tecnico": "",
                "analisis_fundamental": "",
                "analisis_sentimiento": "",
                "decision_accion": "",
                "precio_stop_loss": 0.0,
                "precio_take_profit": 0.0,
                "justificacion": "",
                "accion_ejecutada": "",
                "intentos_revision": 0
            }
            
            # EJECUTAMOS TU GRAFO DE FORMA ASÍNCRONA (A PRUEBA DE FALLOS LLM)
            try:
                resultado = await safe_ainvoke(workflow, estado_inicial)
                
                logger.info(f"Moderador Veredicto: {resultado['decision_accion']} | SL: {resultado['precio_stop_loss']} | TP: {resultado['precio_take_profit']}")
                logger.info(f"Justificación: {resultado['justificacion']}")
                logger.info(f"Resultado de Ejecución: {resultado['accion_ejecutada']}")
            except Exception as e:
                logger.error(f"Fallback Global: LangGraph falló tras reintentos (ej. 429/503). Forzando HOLD para {ticker}.")
                logger.info("El bot sobrevivió al fallo de la API. Esperando la siguiente noticia.")
        else:
            logger.info("Falsa alarma. Los agentes siguen durmiendo.")
    except Exception as e:
        logger.error(f"Error procesando noticia para {ticker}: {e}", exc_info=True)


def escuchar_noticias():
    while True:
        try:
            logger.info("Conectando a Alpaca News Stream...")
            # Inicializamos el cliente de WebSockets
            news_stream = NewsDataStream(ALPACA_API_KEY, ALPACA_SECRET_KEY)
            
            # Suscribirse a todas las noticias ("*")
            news_stream.subscribe_news(procesar_noticia, "*")
            
            logger.info("Radar Rápido encendido. Escuchando mercado en tiempo real...")
            # Arrancamos el loop bloqueante de Alpaca (mantiene la conexión abierta)
            news_stream.run()
        except Exception as e:
            logger.error(f"WebSocket desconectado o error crítico: {e}. Intentando reconectar en 5 segundos...", exc_info=True)
            time.sleep(5)

if __name__ == "__main__":
    escuchar_noticias()
