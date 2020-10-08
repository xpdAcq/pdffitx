import typing as tp


def get_value(dct: dict, keys: tuple) -> tp.Any:
    """Get the value by walking along a key chain."""
    result = dct
    for key in keys:
        result = result[key]
    return result


def only_letter(s: str) -> str:
    """Filter out letters in the string."""
    return ''.join([c for c in s if c.isalpha()])


def only_digit(s: str) -> str:
    """Filter out the digits in the string."""
    return ''.join([c for c in s if c.isdigit()])
