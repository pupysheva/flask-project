#!/usr/bin/python
# utf-8
from logger import log
from functools import wraps


def timer(text=''):
    """Декоратор, печатает время выполнения декорированной функции.
    Аргумент:
        Текст для печати
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            log('start', (timer.__name__, func.__name__))
            result = func(*args, **kwargs)
            log('finish{}'.format(': {}'.format(text) if text is not '' else ''), (timer.__name__, func.__name__))
            return result
        return wrapper
    return decorator
