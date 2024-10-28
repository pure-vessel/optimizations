import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random
from collections import deque

random.seed(85377666522016058)

def wolfe_conds_line_search(c1, c2, alpha_max_start=1, alpha_max_multiplier=1.3):
    def search(phi, dphi_dalpha):
        phi_zero = phi(0)
        dphi_dalpha_zero = dphi_dalpha(0)

        def zoom(alpha_l, alpha_r):
            while True:
                alpha = (alpha_l + alpha_r) / 2
                phi_alpha = phi(alpha)
                dphi_dalpha_alpha = dphi_dalpha(alpha)
                if phi_alpha > phi_zero + c1 * alpha * dphi_dalpha_zero or phi_alpha >= phi(alpha_l):
                    alpha_r = alpha
                    continue
                elif dphi_dalpha_alpha >= c2 * dphi_dalpha_zero:
                    return alpha
                elif (alpha_r - alpha_l) * dphi_dalpha_alpha >= 0:
                    alpha_r = alpha_l
                alpha_l = alpha
        alpha_max = alpha_max_start
        while phi(alpha_max) <= phi_zero + c1 * alpha_max * dphi_dalpha_zero:
            alpha_max *= alpha_max_multiplier
        alpha_prev = 0
        alpha = alpha_max / 2
        while True:
            phi_alpha = phi(alpha)
            dphi_dalpha_alpha = dphi_dalpha(alpha)
            if phi_alpha > phi_zero + c1 * alpha * dphi_dalpha_zero or phi_alpha >= phi(alpha_prev):
                return zoom(alpha_prev, alpha)
            elif dphi_dalpha_alpha >= c2 * dphi_dalpha_zero:
                return alpha
            elif dphi_dalpha_alpha >= 0:
                return zoom(alpha, alpha_max)
            alpha_prev = alpha
            alpha = (alpha + alpha_max) / 2
    return search


L_BFGS_line_search = wolfe_conds_line_search(1e-4, 0.9)

def evaluate_phi(f, x, p):
    return lambda alpha: f(x + alpha * p)

def evaluate_dphi_dalpha(grad, x, p):
    return lambda alpha: np.dot(grad(x + alpha * p), p)

eps = 1e-9

def L_BFGS(f, grad, m, x):
    direction_start = -grad(x)
    points = [x, x + direction_start * L_BFGS_line_search(evaluate_phi(f, x, direction_start), evaluate_dphi_dalpha(grad, x, direction_start))]
    s = deque([points[1] - x])
    y = deque([grad(points[1]) + direction_start])
    x = points[1]

    while True:
        q = grad(x)
        alpha = [None for _ in s]
        rho = [1 / np.dot(si, yi) for (si, yi) in zip(s, y)]
        for i in range(len(s) - 1, -1, -1):
            alpha[i] = rho[i] * np.dot(s[i], q)
            q -= alpha[i] * y[i]
        direction = np.dot(s[-1], y[-1]) / np.dot(y[-1], y[-1]) * q
        for i in range(len(s)):
            direction += s[i] * (alpha[i] - rho[i] * np.dot(y[i], direction))
        direction = -direction
        alpha = L_BFGS_line_search(evaluate_phi(f, x, direction), evaluate_dphi_dalpha(grad, x, direction))

        if np.linalg.norm(alpha * direction) < eps:
            return points
        x1 = x + alpha * direction
        s.append(x1 - x)
        y.append(grad(x1) - grad(x))
        if len(s) > m:
            s.popleft()
            y.popleft()
        x = x1
        points.append(x)


# Supposed, square is forgotten in task, 'cause original function has no local minimums

def rosenbrock(x, y):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

def rosenbrock_grad(x, y):
    return [400 * x ** 3 - 400 * x * y + 2 * x - 2, 200 * y - 200 * x ** 2]

def test_function(args):
    return sum([rosenbrock(args[2 * i], args[2 * i + 1]) for i in range(len(args) // 2)])

def test_function_grad(args):
    return np.array([e for i in range(len(args) // 2) for e in rosenbrock_grad(args[2 * i], args[2 * i + 1])])

N = 20
M = 15

start_point = np.array([e for _ in range(N // 2) for e in [random.uniform(-1, 1.5), random.uniform(-.5, 1.5)]])
trace = np.array(L_BFGS(test_function, test_function_grad, M, start_point))

traces2d = [trace[:, 2 * i : 2 * i + 2] for i in range(N // 2)]

x = np.linspace(-1, 1.5, 100)
y = np.linspace(-.5, 1.5, 100)
x, y = np.meshgrid(x, y)
z = rosenbrock(x, y)

print(trace[0], '\n-->\n', trace[-1], "\nsteps: ", len(trace), sep='')

for i, trace in enumerate(traces2d):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlim([-1, 1.5])
    ax.set_ylim([-.5, 1.5])
    ax.set_zlim([0, 500])
    ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=.8)
    trace_x = trace[:, 0]
    trace_y = trace[:, 1]
    trace_z = np.array([rosenbrock(trace_x[i], trace_y[i]) for i in range(len(trace_x))])
    ax.plot(trace_x, trace_y, trace_z, c='red', marker='o', markersize=1)
plt.show()

