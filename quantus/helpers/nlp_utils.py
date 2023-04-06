from functools import lru_cache


@lru_cache
def is_nlpaug_available() -> bool:
    try:
        import nlpaug
        return True
    except ModuleNotFoundError:
        return False
