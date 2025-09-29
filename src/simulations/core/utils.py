def set_nested_attr(obj: object, path: str, value: any):
    """Set a value on a nested attribute of an object using dot notation."""

    keys = path.split(".")
    for key in keys[:-1]:
        obj = getattr(obj, key)
    setattr(obj, keys[-1], value)
