

class CodePosition():
    def __init__(self, frame, tracer=None, node=None, parameter=None):
        self.frame = frame

    @property
    def code(self):
        return self.frame.f_code

    @property
    def description(self):
        code = self.code
        desc = f"{code.co_filename}:"
        desc += f"{code.co_name}"
        desc += f".{code.co_firstlineno})"
        return desc

    def __repr__(self):
        return self.description


def get_code_pos_from_frame(frame):
    return CodePosition(frame).description
