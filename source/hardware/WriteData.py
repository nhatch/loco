# THIS IS A STUB

class SyncMultiWriter():
    def __init__(self, h, p, ids, attrs):
        pass

    def write(self, attrs):
        pass

import hardware.ReadData as ReadData
import hardware.dabs as dabs

class SyncMultiReaderStub():
    def __init__(self, h, p, ids, attrs):
        self.length = len(ids)

    def read(self):
        return [0]*self.length

class MultiReaderStub():
    def __init__(self, h, p, ids, attrs):
        self.length = len(attrs)

    def read(self):
        return [0]*self.length

# Stub the other I/O classes as well
setattr(ReadData, 'SyncMultiReader', SyncMultiReaderStub)
setattr(dabs, 'MultiReader', MultiReaderStub)
print("ALL I/O OPERATIONS HAVE BEEN STUBBED")
