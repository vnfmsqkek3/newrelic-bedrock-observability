import functools
import logging

logger = logging.getLogger("newrelic_bedrock_observability")


def handle_errors(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as ex:
            logger.error(f"Error in {func.__name__}: {ex}")
            return args[0] if len(args) > 0 else None
    return wrapper 