from abc import ABC, abstractmethod

class Placement(ABC):
    def __init__(self):
        self.name = "Placement"

    @abstractmethod
    def generate_allocation(self, topology, applications, users):
        """
        Generate allocation using the placement algorithm.
        """
        