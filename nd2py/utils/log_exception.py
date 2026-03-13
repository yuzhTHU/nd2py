import traceback


def log_exception(e: Exception) -> str:
    """Format exception for logging."""
    return f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
