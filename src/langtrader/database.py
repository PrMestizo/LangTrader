import sqlite3
from datetime import datetime
from langtrader.logger import logger

DB_PATH = "langtrader_history.db"

def init_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trade_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                ticker TEXT NOT NULL,
                noticia TEXT,
                decision_accion TEXT,
                precio_stop_loss REAL,
                precio_take_profit REAL,
                justificacion TEXT,
                resultado_ejecucion TEXT
            )
        ''')
        conn.commit()
        conn.close()
        logger.info("Base de datos SQLite inicializada correctamente.")
    except Exception as e:
        logger.error(f"Error inicializando la base de datos: {e}")

def registrar_trade(ticker: str, noticia: str, decision: str, sl: float, tp: float, justificacion: str, resultado: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute('''
            INSERT INTO trade_history (
                timestamp, ticker, noticia, decision_accion, 
                precio_stop_loss, precio_take_profit, justificacion, resultado_ejecucion
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, ticker, noticia, decision, sl, tp, justificacion, resultado))
        conn.commit()
        conn.close()
        logger.info(f"Trade registrado con éxito en DB para {ticker}.")
    except Exception as e:
        logger.error(f"Error registrando el trade en la DB: {e}")
