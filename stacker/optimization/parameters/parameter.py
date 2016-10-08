class Parameter:
    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return self.name.__hash__()