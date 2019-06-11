import math

def func_sigmoid(x):
  return 1 / (1 + math.exp(-x))

def derivada_sigmoid(resultado):
    derivada = resultado * (1 - resultado)
    return derivada