import asyncio
import json
from transformers import pipeline

# IMPORTAMOS TU GRAFO DE AGENTES
from my_graph.graph import workflow 

PALABRAS_CLAVE = ["bancarrota", "adquisicion", "fraude", "dimision", "acuerdo"]

print("Cargando modelo FinBERT (esto tarda unos segundos)...")
nlp_finanzas = pipeline("sentiment-analysis", model="ProsusAI/finbert")
print("✅ FinBERT cargado.")

async def escuchar_noticias():
    # En un entorno real, aquí iría la URL de tu proveedor (ej. Alpaca)
    uri = "wss://echo.websocket.events" # Usamos una genérica para que no falle al probar
    
    print("📡 Radar Rápido encendido. Escuchando mercado en tiempo real...")
    
    # SIMULAMOS LA LLEGADA DE UNA NOTICIA
    await asyncio.sleep(2)
    mensaje_json = '{"ticker": "DIS", "headline": "El CEO de Disney presenta su dimision tras acusaciones de fraude"}'
    datos_noticia = json.loads(mensaje_json)
    titular = datos_noticia["headline"].lower()
    ticker = datos_noticia["ticker"]

    # FILTRO A: Palabras clave
    if any(palabra in titular for palabra in PALABRAS_CLAVE):
        print(f"\n🚨 ALERTA: Palabra clave detectada en {ticker}!")
        
        # FILTRO B: Triaje con IA ligera
        resultado_nlp = nlp_finanzas(datos_noticia["headline"])[0]
        sentimiento = resultado_nlp["label"]
        confianza = resultado_nlp["score"]
        
        print(f"🧠 FinBERT dictamina: {sentimiento} (Confianza: {confianza:.2f})")
        
        if sentimiento in ["negative", "positive"] and confianza > 0.85:
            print(f"🔥 Noticia crítica. Despertando al Comité de LangGraph para {ticker}...\n")
            
            # --- AQUÍ CONECTAMOS CON TU ARCHIVO graph.py ---
            # Preparamos los datos iniciales que recibirá tu grafo
            estado_inicial = {
                "user_input": f"Analizar urgencia: {datos_noticia['headline']} para el ticker {ticker}",
                "llm_output": "" 
            }
            
            # EJECUTAMOS TU GRAFO
            resultado = workflow.invoke(estado_inicial)
            
            print(f"\n🤖 Veredicto Final del Grafo: {resultado['llm_output']}")
        else:
            print("💤 Falsa alarma. Los agentes siguen durmiendo.")

if __name__ == "__main__":
    asyncio.run(escuchar_noticias())