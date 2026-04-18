import os
import math
import yfinance as yf
from datetime import datetime, timedelta
from langchain_core.tools import tool
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest, MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, QueryOrderStatus
from dotenv import load_dotenv

from langtrader.logger import logger

load_dotenv()
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

stock_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

def _obtener_precio_actual(ticker: str) -> float:
    """Obtiene el último precio del ticker vía Alpaca (IEX) con fallback a yfinance."""
    try:
        quote = stock_client.get_stock_latest_quote(StockLatestQuoteRequest(
            symbol_or_symbols=ticker,
            feed=DataFeed.IEX
        ))
        if ticker in quote and quote[ticker].ask_price > 0:
            return float(quote[ticker].ask_price)
    except Exception:
        pass
    # Fallback a yfinance
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1d")
    if not hist.empty:
        return float(hist['Close'].iloc[-1])
    raise ValueError(f"No se pudo obtener precio actual de {ticker}")

def _calcular_position_size(ticker: str, stop_loss: float, riesgo_porcentaje: float) -> tuple[int, float, float]:
    """Calcula la cantidad de acciones basándose en el equity, SL y % de riesgo.
    Retorna (cantidad, precio_actual, equity)."""
    account = trading_client.get_account()
    equity = float(account.equity)
    buying_power = float(account.buying_power)
    precio_actual = _obtener_precio_actual(ticker)
    
    capital_en_riesgo = equity * (riesgo_porcentaje / 100.0)
    riesgo_por_accion = abs(precio_actual - stop_loss)
    
    if riesgo_por_accion <= 0:
        raise ValueError(f"Riesgo por acción inválido: precio={precio_actual}, SL={stop_loss}")
    
    cantidad_por_riesgo = math.floor(capital_en_riesgo / riesgo_por_accion)
    cantidad_maxima_comprable = math.floor(buying_power / precio_actual)
    
    cantidad = min(cantidad_por_riesgo, cantidad_maxima_comprable)
    
    if cantidad <= 0:
        raise ValueError(f"Fondos insuficientes: no hay buying power para comprar 1 acción de {ticker}. Buying Power={buying_power}")
    
    return cantidad, precio_actual, equity

@tool
def ejecutar_orden_mercado(
    ticker: str,
    accion: str,
    stop_loss: float = 0.0,
    take_profit: float = 0.0,
    riesgo_porcentaje: float = 1.0
) -> str:
    """Ejecuta una orden de compra (BUY) o venta (SELL) a precio de mercado en Alpaca
    con Position Sizing dinámico. Calcula la cantidad de acciones arriesgando un
    porcentaje del equity de la cuenta basándose en la distancia al Stop-Loss.
    Si stop_loss y take_profit > 0, envía una Bracket Order (OTO)."""
    try:
        # --- Verificación de Reloj de Mercado ---
        reloj = trading_client.get_clock()
        if not reloj.is_open:
            return "Operación cancelada: El mercado está cerrado. Se evitan órdenes para prevenir gaps de apertura."

        # --- Prevención de Sobreexposición ---
        try:
            # Intenta obtener una posición abierta para el ticker
            trading_client.get_open_position(ticker)
            existe_posicion = True
        except Exception:
            # Si lanza excepción (generalmente APIError 40440000), no existe posición abierta
            existe_posicion = False
            
        ordenes_pendientes = trading_client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[ticker]))
        
        if existe_posicion or len(ordenes_pendientes) > 0:
            return f"Operación ignorada: Ya existe una posición abierta u orden pendiente para {ticker}. Previniendo sobreexposición."

        side = OrderSide.BUY if accion.upper() == "BUY" else OrderSide.SELL

        # --- Verificación de Activo Shortable (si es SELL) ---
        activo = trading_client.get_asset(ticker)
        if side == OrderSide.SELL and not activo.shortable:
            return f"Operación cancelada: El activo {ticker} no permite posiciones en corto (shortable) en Alpaca."

        # --- Obtención de Precio y Position Sizing ---
        if stop_loss > 0:
            cantidad, precio_actual, equity = _calcular_position_size(ticker, stop_loss, riesgo_porcentaje)
            logger.info(f"Position Sizing: Equity=${equity:,.2f} | Precio={precio_actual:.2f} | "
                        f"Riesgo/acc=${abs(precio_actual - stop_loss):.2f} | Cantidad={cantidad} acc.")
        else:
            cantidad = 1  # Fallback si no hay SL
            precio_actual = _obtener_precio_actual(ticker)
            logger.info(f"Sin Stop-Loss definido para {ticker}. Usando cantidad fija: {cantidad} | Precio={precio_actual:.2f}")

        # --- Cálculo del Limit Price (Protección contra Slippage del 0.5%) ---
        if side == OrderSide.BUY:
            limit_price = round(precio_actual * 1.005, 2)
        else:
            limit_price = round(precio_actual * 0.995, 2)

        # Si tenemos SL y TP válidos, creamos una orden bracket (OTO) con un Límite de entrada
        if stop_loss > 0 and take_profit > 0:
            order_data = LimitOrderRequest(
                symbol=ticker,
                limit_price=limit_price,
                qty=cantidad,
                side=side,
                time_in_force=TimeInForce.IOC,
                order_class=OrderClass.BRACKET,
                stop_loss=StopLossRequest(stop_price=round(stop_loss, 2)),
                take_profit=TakeProfitRequest(limit_price=round(take_profit, 2))
            )
        else:
            # Orden límite simple sin bracket
            order_data = LimitOrderRequest(
                symbol=ticker,
                limit_price=limit_price,
                qty=cantidad,
                side=side,
                time_in_force=TimeInForce.IOC
            )
        
        orden = trading_client.submit_order(order_data=order_data)

        msg = f"Orden {accion} ejecutada para {cantidad} acc. de {ticker}. Status: {orden.status}"
        if stop_loss > 0 and take_profit > 0:
            msg += f" | Bracket: SL={stop_loss:.2f}, TP={take_profit:.2f}"
        logger.info(msg)
        return msg
    except Exception as e:
        logger.error(f"Fallo al ejecutar orden {accion} en {ticker}: {e}")
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
        logger.error(f"Error obteniendo resúmenes recientes de {ticker}: {e}")
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
         logger.error(f"Error obteniendo gráficas de Alpaca para {ticker}: {e}")
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
        logger.error(f"Error obteniendo fundamentales de {ticker}: {e}")
        return f"Error obteniendo fundamentales de {ticker}: {str(e)}"
