from builtins import print as builtin_print
import datetime


def print(*args, sep=' ', end='\n', file=None):
    builtin_print(_get_format_datetime(), *args, sep=sep, end=end, file=file)


def _get_format_datetime():
    return datetime.datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')