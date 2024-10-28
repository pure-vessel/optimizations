import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random
import scipy.linalg

random.seed(85377666522016058)

def projection(f, grad, a, b):
    x_star, _, _, _ = scipy.linalg.lstsq(a, b)
    if np.linalg.norm(a @ x_star - b) > 1e-9:
        raise scipy.linalg.LinAlgError("No linear system solutions found")
    basis = scipy.linalg.null_space(a)
    print("x_star:", x_star)
    print("basis:", basis)
    return (basis.shape[1], lambda x: basis @ x + x_star, lambda x: f(basis @ x + x_star), lambda x: np.transpose(basis) @ grad(basis @ x + x_star))


eps = 2e-5
learning_rate = 1e-5

def descent_steps(f, grad, x):
    points = [x]
    while True:
        x2 = x - learning_rate * grad(x)
        points.append(x2)
        if np.linalg.norm(x - x2) < eps:
            return points
        x = x2

def linear_limited_descent(f, grad, a, b, limit):
    dim, convert, f_proj, grad_proj = projection(f, grad, a, b)
    point = np.array([random.uniform(*limit) for _ in range(dim)])
    print("point:", point)
    return np.array(list(map(convert, descent_steps(f_proj, grad_proj, point))))

def rosenbrock(xy):
    x = xy[0]
    y = xy[1]
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

def rosenbrock_grad(xy):
    x = xy[0]
    y = xy[1]
    return np.array([400 * x ** 3 - 400 * x * y + 2 * x - 2, 200 * y - 200 * x ** 2])


a = np.array([[-1, 1]])
b = np.array([3])

trace = linear_limited_descent(rosenbrock, rosenbrock_grad, a, b, limit=(-3, 3))

x = np.linspace(-2, 0, 100)
y = np.linspace(1, 3, 100)
x, y = np.meshgrid(x, y)
z = rosenbrock((x, y))

xl = np.linspace(-2, 0, 100)
yl = xl + 3
zl = rosenbrock((xl, yl))

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlim([-2, 0])
ax.set_ylim([1, 3])
ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=.8)
ax.plot(xl, yl, zl, color="black", linewidth=0.5)
trace_x = trace[:, 0]
trace_y = trace[:, 1]
trace_z = np.array([rosenbrock((trace_x[i], trace_y[i])) for i in range(len(trace_x))])
ax.plot(trace_x, trace_y, trace_z, c='red', marker='o', markersize=1)
ax.scatter(trace_x[-1], trace_y[-1], trace_z[-1], c='green', marker='o', linewidths=5)
print(trace[0], '-->', trace[-1], "steps:", len(trace))
plt.show()



