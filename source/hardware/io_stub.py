class SyncMultiWriter():
    def __init__(self, h, p, ids, attrs):
        pass

    def write(self, attrs):
        pass

class SyncMultiReader():
    def __init__(self, h, p, ids, attrs):
        self.length = len(ids)

    def read(self):
        return [0]*self.length

class MultiReader():
    def __init__(self, h, p, ids, attrs):
        self.length = len(attrs)
    def read(self):
        return [0]*self.length

print("ALL I/O OPERATIONS HAVE BEEN STUBBED")

