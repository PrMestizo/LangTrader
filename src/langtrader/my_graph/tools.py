import os
import yfinance as yf
from datetime import datetime, timedelta
from langchain_core.tools import tool
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from dotenv import load_dotenv

load_dotenv()
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

stock_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

@tool
def ejecutar_orden_mercado(
    ticker: str,
    accion: str,
    cantidad: int = 1,
    stop_loss: float = 0.0,
    take_profit: float = 0.0
) -> str:
    """Ejecuta una orden de compra (BUY) o venta (SELL) a precio de mercado en Alpaca.
    Si se proporcionan stop_loss y take_profit (> 0), envía una orden bracket (OTO)
    que incluye automáticamente las órdenes de salida vinculadas."""
    try:
        side = OrderSide.BUY if accion.upper() == "BUY" else OrderSide.SELL

        # Si tenemos SL y TP válidos, creamos una orden bracket (OTO)
        if stop_loss > 0 and take_profit > 0:
            market_order_data = MarketOrderRequest(
                symbol=ticker,
                qty=cantidad,
                side=side,
                time_in_force=TimeInForce.GTC,
                order_class=OrderClass.BRACKET,
                stop_loss={"stop_price": round(stop_loss, 2)},
                take_profit={"limit_price": round(take_profit, 2)}
            )
        else:
            # Orden simple sin bracket
            market_order_data = MarketOrderRequest(
                symbol=ticker,
                qty=cantidad,
                side=side,
                time_in_force=TimeInForce.GTC
            )
        
        orden = trading_client.submit_order(order_data=market_order_data)

        msg = f"Orden {accion} ejecutada para {cantidad} acc. de {ticker}. Status: {orden.status}"
        if stop_loss > 0 and take_profit > 0:
            msg += f" | Bracket: SL={stop_loss:.2f}, TP={take_profit:.2f}"
        return msg
    except Exception as e:
        return f"Fallo al ejecutar orden {accion} en {ticker}: {str(e)}"

@tool
def buscar_sentimiento_social(ticker: str) -> str:
    """Busca en noticias recientes la reacción del mercado y público sobre un ticker."""
    try:
        # Usamos Yahoo Finance como proxy de noticias y sentimiento reciente
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news:
            return f"No se encontraron noticias recientes para {ticker}."
        
        # Resumen de los titulares de las noticias
        titulares = []
        for n in news[:5]:
            if 'content' in n and 'title' in n['content']:
                titulares.append(n['content']['title'])
            elif 'title' in n:
                titulares.append(n['title'])
                
        titulares_str = " | ".join(titulares)
        
        return f"Titulares recientes relacionados con {ticker}: {titulares_str}."
    except Exception as e:
        return f"Error obteniendo resúmenes recientes de {ticker}: {str(e)}"

@tool
def analizar_grafica_1m(ticker: str) -> str:
    """Obtiene velas de 1 minuto y volumen reciente para ver acción institucional de precio."""
    try:
        # Obtener los ultimos datos desde la ultima fecha de mercado abierta.
        end_time = datetime.now() - timedelta(minutes=20) # Retrocedemos 20 minutos por el delay SIP gratuito
        start_time = end_time - timedelta(days=5) # Retrocedemos unos dias por si es fin de semana
        
        request_params = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Minute,
            start=start_time,
            end=end_time,
            limit=1000,
            feed=DataFeed.IEX # Especificar IEX para cuentas gratuitas
        )
        
        bars = stock_client.get_stock_bars(request_params)
        df = bars.df
        
        if df.empty:
            return f"No se encontraron datos recientes de velas de 1 minuto para {ticker}."
        
        # Extraer los últimos 15 minutos (si están agrupados por MultiIndex, obtener para el ticker)
        if hasattr(df.index, 'levels'):
             df = df.loc[ticker]
        df = df.tail(15) 
        
        avg_volume = df['volume'].mean()
        last_close = df['close'].iloc[-1]
        first_close = df['close'].iloc[0]
        
        tendencia = "ALCISTA (posible acción de compra corporativa)" if last_close > first_close else "BAJISTA (distribución/venta)"
        resumen = (f"En los últimos 15 minutos de mercado, el precio fue de {first_close:.2f} a {last_close:.2f}. "
                   f"Tendencia: {tendencia}. Volumen promedio: {avg_volume:.0f} acciones por min. "
                   f"Máximo del intradía reciente: {df['high'].max():.2f}, Mínimo: {df['low'].min():.2f}.")
        return resumen
    except Exception as e:
         return f"Error obteniendo gráficas de Alpaca para {ticker}: {str(e)}"

@tool
def evaluar_dependencia_fundamental(ticker: str, contexto: str) -> str:
    """Revisa los balances y el perfil de la empresa para ver cuánto depende de la noticia/fundamentales."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        sector = info.get('sector', 'Desconocido')
        industria = info.get('industry', 'Desconocido')
        market_cap = info.get('marketCap', 'Desconocido')
        resumen_negocio = info.get('longBusinessSummary', 'No disponible')[:450] + "..."
        
        return (f"Sector: {sector} | Industria: {industria} | Market Cap: {market_cap}\n"
                f"Resumen de Negocio: {resumen_negocio}\n"
                f"Contexto del evento: {contexto}. Considerar si afecta significativamente el core business.")
    except Exception as e:
        return f"Error obteniendo fundamentales de {ticker}: {str(e)}"
