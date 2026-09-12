import time
from functools import wraps
from typing import Callable, Any
from src.logging_conf import get_logger

logger = get_logger(__name__)

def timed(func: Callable) -> Callable:
    """Decorator to log the execution time of a function."""
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        logger.info(f"{func.__name__} executed in {duration_ms:.2f} ms")
        return result
    return wrapper
