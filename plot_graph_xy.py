import matplotlib.pyplot as plt
from utils import remap
y = remap([x*x for x in range(0,15)], [1.0, 8.0])
x = range(0,len(y))

plt.figure()
plt.plot(x, y, marker='o') # 'o' creates circular markers at each data point
plt.grid(True)

# Annotating each point with its y value
for i, value in enumerate(y):
    plt.annotate(f"{value:.3f}", (x[i], y[i]))

plt.show()