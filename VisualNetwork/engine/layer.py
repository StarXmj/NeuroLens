# engine/layer.py
import random
from config.settings import DEFAULT_NODES

class Neuron:
    _id_counter = 1
    def __init__(self):
        self.id = f"N{Neuron._id_counter}"
        Neuron._id_counter += 1
        self.value = round(random.uniform(0.1, 1.0), 2)
        self.x = 0
        self.y = 0

class Link:
    _id_counter = 1
    def __init__(self, n1, n2):
        self.id = f"L{Link._id_counter}"
        Link._id_counter += 1
        self.value = round(random.uniform(-1.0, 1.0), 2)
        self.n1 = n1
        self.n2 = n2

class Layer:
    def __init__(self, name, nodes=DEFAULT_NODES, activation="ReLU"):
        self.name = name
        self.nodes = nodes
        self.activation = "Input" if name == "Input" else activation
        self.neurons = [Neuron() for _ in range(nodes)]