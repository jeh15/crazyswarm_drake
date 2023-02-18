import numpy as np

prev_p = np.array([1, 1], dtype=float)
obst = np.array([2, 2], dtype=float)
p = np.array([1.5, 1], dtype=float)

v = obst - prev_p
w = obst - p

mag = np.linalg.norm(v)
proj = (np.dot(w, v) / np.dot(v, v)) * mag

what = np.dot(w, v)

print(mag)
print(proj)
print(what)