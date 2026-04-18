import logging
import sys

def setup_logger():
    # Evita crear múltiples handlers si ya está configurado
    if logging.getLogger().hasHandlers():
        return logging.getLogger("trader")

    logger = logging.getLogger("trader")
    logger.setLevel(logging.INFO)

    # Formato del log
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File Handler para guardar en trader.log
    file_handler = logging.FileHandler("trader.log", encoding="utf-8")
    file_handler.setFormatter(formatter)

    # Stream Handler para la consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Añadir handlers al logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logger()
