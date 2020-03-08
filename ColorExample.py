'''
    ColorExample.py

    An example implementation of the neural network.
'''

from Brain import Brain
from config import SAMPLE_ITERATIONS

# Input: r, g, b background color, Output: desired text color
data = [
    {'input': dict(r=0.62, g=0.72, b=0.88), 'output': dict(white=1, black=0)},
    {'input': dict(r=0.1, g=0.84, b=0.72), 'output': dict(white=1, black=0)},
    {'input': dict(r=0.33, g=0.24, b=0.29), 'output': dict(white=0, black=1)},
    {'input': dict(r=0.74, g=0.78, b=0.86), 'output': dict(white=1, black=0)},
    {'input': dict(r=0.31, g=0.35, b=0.41), 'output': dict(white=0, black=1)},
    {'input': dict(r=0.59, g=1.0, b=0.35), 'output': dict(white=0, black=1)}
]

brain = Brain()
brain.train(data, iterations=SAMPLE_ITERATIONS)
result = brain.run([0.0, 1.0, 0.65])
print("\n\nOUTPUT: {0}".format(result))
