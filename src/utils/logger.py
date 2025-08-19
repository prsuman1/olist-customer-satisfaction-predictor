"""
Logging utilities for the project.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with both file and console handlers.
    
    Args:
        name: Logger name
        log_file: Optional log file path
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(exist_ok=True, parents=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_project_logger(name: str = "olist_ml") -> logging.Logger:
    """
    Get the main project logger.
    
    Args:
        name: Logger name
        
    Returns:
        Project logger instance
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(__file__).parent.parent.parent / "logs" / f"{name}_{timestamp}.log"
    
    return setup_logger(name, str(log_file))