"""Utility functions for simulation operations."""


def set_nested_attr(obj: object, path: str, value: any):
    """Set a nested attribute on an object using dot notation.

    Navigate through nested objects using a dotted path string and set
    the final attribute to the specified value.

    Args:
        obj: The root object to modify.
        path: Dot-separated path to the attribute (e.g., "attr1.attr2.attr3").
        value: The value to set.

    Example:
        >>> class Inner:
        ...     x = 1
        >>> class Outer:
        ...     inner = Inner()
        >>> obj = Outer()
        >>> set_nested_attr(obj, "inner.x", 42)
        >>> obj.inner.x
        42
    """
    keys = path.split(".")
    for key in keys[:-1]:
        obj = getattr(obj, key)
    setattr(obj, keys[-1], value)
