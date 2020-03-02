from Brain import Brain
from config import SAMPLE_ITERATIONS

# Input: r, g, b, Output: white, black
data = [
    {'input': [0.62, 0.72, 0.88], 'output': [1, 0]},
    {'input': [0.1, 0.84, 0.72], 'output': [1, 0]},
    {'input': [0.33, 0.24, 0.29], 'output': [0, 1]},
    {'input': [0.74, 0.78, 0.86], 'output': [1, 0]},
    {'input': [0.31, 0.35, 0.41], 'output': [0, 1]},
    {'input': [0.59, 1.0, 0.35], 'output': [0, 1]}
]

brain = Brain()
brain.train(data, iterations=SAMPLE_ITERATIONS)
result = brain.run([0.0, 1.0, 0.65])
print("\n\nOUTPUT: {0}".format(result))
