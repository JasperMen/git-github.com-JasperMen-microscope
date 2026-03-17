import typing
try:
    from typing_extensions import Self as _Self
    if not hasattr(typing, 'Self'):
        typing.Self = _Self
except Exception:
    try:
        typing.Self = typing.Any
    except Exception:
        pass
