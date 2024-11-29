import typing

class NotImplementedClass(object):
    def __getattr__(self, obj):
        return NotImplemented
    def __setattr__(self, name: str, value: typing.Any) -> None:
        raise NotImplementedError()
