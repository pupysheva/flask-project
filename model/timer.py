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
            log('start', (timer, func))
            result = func(*args, **kwargs)
            log('finish{}'.format(': {}'.format(text) if text != '' else ''), (timer, func))
            return result
        return wrapper
    return decorator
