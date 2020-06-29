import psutil
from psutil._common import bytes2human

class Memory:
    def __init__(self):
        self.virtual = psutil.virtual_memory()
        self.swap = psutil.swap_memory()

    @property
    def free_memory(self):
        return getattr(self.virtual, 'free')

    @property
    def total_memory(self):
        return getattr(self.virtual, 'total')

    def calculate_usage(self, ndoubles):
        return 8*ndoubles

    def remaining_memory(self, ndoubles):
        return self.free_memory - self.calculate_usage(ndoubles)
