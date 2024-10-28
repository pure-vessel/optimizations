import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random

random.seed(85377666522016058)

eps = 1e-9

def newton_steps(f, grad, hess_inv, x):
    points = [x]
    while True:
        delta = hess_inv(x) @ grad(x)
        if np.linalg.norm(delta) < eps:
            return points
        x = x - delta
        points.append(x)


# minimize f while g(x) = 0
def limited_newton(f, f_grad, f_hess, g, g_grad, g_hess, limits, points=10):
    # L(x, l) = f(x) - l * g(x)
    # grad L = [grad f - l * grad g, -g]
    # H_L = [
    #     [H_f - l * H_g, -grad g],
    #     [-grad g,       0]
    # ]
    def l(xl):
        x = xl[:-1]
        l = xl[-1]
        return f(x) - l * g(x)
    def l_grad(xl):
        x = xl[:-1]
        l = xl[-1]
        return np.append(f_grad(x) - l * g_grad(x), [-g(x)])
    def l_hess(xl):
        x = xl[:-1]
        l = xl[-1]
        neg_g_grad = -g_grad(x)
        return [
            *np.append(np.array(f_hess(x) - l * g_hess(x)), np.transpose([neg_g_grad]), axis=1),
            np.array([*neg_g_grad, 0])
        ]
    l_hess_inv = lambda xl: np.linalg.inv(l_hess(xl))
    traces = []
    for i in range(points):
        point = np.append(np.array(list(map(lambda x: random.uniform(*x), limits))), random.uniform(-1, 1))
        traces.append(np.array(newton_steps(l, l_grad, l_hess_inv, point)))
    return traces


def rosenbrock(xy):
    x = xy[0]
    y = xy[1]
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

def rosenbrock_grad(xy):
    x = xy[0]
    y = xy[1]
    return np.array([400 * x ** 3 - 400 * x * y + 2 * x - 2, 200 * y - 200 * x ** 2])

def rosenbrock_hess(xy):
    x = xy[0]
    y = xy[1]
    return np.array([
        [1200 * x ** 2 - 400 * y + 2, -400 * x],
        [-400 * x,                    200]
    ])

def limit(xy):
    x = xy[0]
    y = xy[1]
    return x ** 2 + y

def limit_grad(xy):
    x = xy[0]
    y = xy[1]
    return np.array([2 * x, 1])

def limit_hess(xy):
    x = xy[0]
    y = xy[1]
    return np.array([
        [2, 0],
        [0, 0]
    ])

# traces = limited_newton(rosenbrock, rosenbrock_grad, rosenbrock_hess, limit, limit_grad, limit_hess, newton_steps(1, stop_by_epsilon(1e-9)), limits=((-1, 1), (-.5, 1.5)), points=10)
traces = limited_newton(rosenbrock, rosenbrock_grad, rosenbrock_hess, limit, limit_grad, limit_hess, limits=((-1, 1), (-.5, 1.5)), points=10)

x = np.linspace(-1, 1, 100)
y = np.linspace(-.5, 1.5, 100)
x, y = np.meshgrid(x, y)
z = rosenbrock((x, y))

xl = np.linspace(-1, 1, 100)
yl = -xl ** 2
zl = rosenbrock((xl, yl))

for i, trace in enumerate(traces):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-.5, 1.5])
    ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=.8)
    ax.plot(xl, yl, zl, color="black")
    trace_x = trace[:, 0]
    trace_y = trace[:, 1]
    # ignoring l parameter
    trace_z = np.array([rosenbrock((trace_x[i], trace_y[i])) for i in range(len(trace_x))])
    ax.plot(trace_x, trace_y, trace_z, c='red', marker='o', markersize=1)
    print(trace[0], '-->', trace[-1], "steps:", len(trace))
plt.show()
