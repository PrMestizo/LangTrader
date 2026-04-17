import asyncio
import os
from dotenv import load_dotenv
from transformers import pipeline
from alpaca.data.live import NewsDataStream

# IMPORTAMOS TU GRAFO DE AGENTES
from langtrader.my_graph.graph import workflow

# Cargar API Keys del archivo .env
load_dotenv()
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

PALABRAS_CLAVE = ["bancarrota", "adquisicion", "fraude", "dimision", "acuerdo"]

print("Cargando modelo FinBERT (esto tarda unos segundos)...")
nlp_finanzas = pipeline("sentiment-analysis", model="ProsusAI/finbert")
print("✅ FinBERT cargado.")


async def procesar_noticia(noticia):
    """Callback que se ejecuta cada vez que Alpaca recibe una noticia"""
    # Alpaca envía un objeto de noticia. Extraemos titular y el primer símbolo.
    titular = noticia.headline
    simbolos = noticia.symbols
    ticker = simbolos[0] if simbolos else "Desconocido"
    titular_lower = titular.lower()

    # Mostrar la noticia en crudo para confirmar que el WebSockets funciona
    print(f"📰 [{ticker}] {titular}")

    # FILTRO A: Palabras clave
    if any(palabra in titular_lower for palabra in PALABRAS_CLAVE):
        print(f"\n🚨 ALERTA: Palabra clave detectada en {ticker}!")
        
        # FILTRO B: Triaje con IA ligera
        resultado_nlp = nlp_finanzas(titular)[0]
        sentimiento = resultado_nlp["label"]
        confianza = resultado_nlp["score"]
        
        print(f"🧠 FinBERT dictamina: {sentimiento} (Confianza: {confianza:.2f})")
        
        if sentimiento in ["negative", "positive"] and confianza > 0.85:
            print(f"🔥 Noticia crítica. Despertando al Comité de LangGraph para {ticker}...\n")
            
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
                "riesgo_sugerido_porcentaje": 0.0,
                "justificacion": "",
                "accion_ejecutada": ""
            }
            
            # EJECUTAMOS TU GRAFO
            resultado = workflow.invoke(estado_inicial)
            
            print(f"\n🤖 Moderador Veredicto: {resultado['decision_accion']}")
            print(f"   SL: {resultado['precio_stop_loss']} | TP: {resultado['precio_take_profit']} | Riesgo: {resultado['riesgo_sugerido_porcentaje']}%")
            print(f"   Justificación: {resultado['justificacion']}")
            print(f"✅ Resultado de Ejecución: {resultado['accion_ejecutada']}")
        else:
            print("💤 Falsa alarma. Los agentes siguen durmiendo.")


def escuchar_noticias():
    print("📡 Conectando a Alpaca News Stream...")
    # Inicializamos el cliente de WebSockets
    news_stream = NewsDataStream(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    
    # Suscribirse a todas las noticias ("*")
    news_stream.subscribe_news(procesar_noticia, "*")
    
    print("🚀 Radar Rápido encendido. Escuchando mercado en tiempo real...")
    # Arrancamos el loop bloqueante de Alpaca (mantiene la conexión abierta)
    news_stream.run()

if __name__ == "__main__":
    escuchar_noticias()
