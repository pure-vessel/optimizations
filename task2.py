import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random

random.seed(85377666522016058)

eps = 2e-5

# method can be Nesterov's method or EMA or anything else. more_data_start is additional data needed for method (like `v` in Nesterov's method).
def descent_function(method, more_data_start):
    def descent_steps(f, grad, x):
        more_data = more_data_start
        points = [x]
        while True:
            x2, more_data = method(x, f, grad, more_data)
            points.append(x2)
            if np.linalg.norm(x - x2) < eps:
                return points
            x = x2
    return descent_steps

def simple_gradient_descent(learning_rate):
    return lambda x, f, g, none: (x - learning_rate * g(x), None)

def exponential_moving_average(alpha, gamma):
    def method(x, f, g, v):
        v2 = gamma * v + (1 - gamma) * g(x)
        return (x - alpha * v2, v2)
    return method

def nesterov_method(alpha, gamma):
    def method(x, f, g, v):
        v2 = gamma * v + (1 - gamma) * g(x - alpha * gamma * v)
        return (x - alpha * v2, v2)
    return method


def limited_descent(f, f_grad, g, g_grad, steps_function, limits):
    # L(x, l) = f(x) - l * g(x)
    # grad L = [grad f - l * grad g, -g]
    def l(xl):
        x = xl[:-1]
        l = xl[-1]
        return f(x) - l * g(x)
    def l_grad(xl):
        x = xl[:-1]
        l = xl[-1]
        #print(xl)
        #print(np.array([*(f_grad(x) - l * g_grad(x)), -g(x)]))
        return np.array([*(f_grad(x) - l * g_grad(x)), -g(x)])
    point = np.append(np.array(list(map(lambda x: random.uniform(*x), limits))), random.uniform(-1, 1))
    return np.array(steps_function(l, l_grad, point))[:, :-1]



def rosenbrock(xy):
    x = xy[0]
    y = xy[1]
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

def rosenbrock_grad(xy):
    x = xy[0]
    y = xy[1]
    return np.array([400 * x ** 3 - 400 * x * y + 2 * x - 2, 200 * y - 200 * x ** 2])

def limit(xy):
    x = xy[0]
    y = xy[1]
    return x ** 2 + y

def limit_grad(xy):
    x = xy[0]
    y = xy[1]
    return np.array([2 * x, 1])

# method = (simple_gradient_descent(1e-5), None)
# method = (exponential_moving_average(1e-5, 0.9), 0)
method = (nesterov_method(1e-5, 0.95), 0)
trace = limited_descent(rosenbrock, rosenbrock_grad, limit, limit_grad, steps_function = descent_function(*method), limits=((-1, 1), (-.5, 1.5)))

x = np.linspace(-1, 1, 100)
y = np.linspace(-.5, 1.5, 100)
x, y = np.meshgrid(x, y)
z = rosenbrock((x, y))

xl = np.linspace(-1, 1, 100)
yl = -xl ** 2
zl = rosenbrock((xl, yl))

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlim([-1, 1])
ax.set_ylim([-.5, 1.5])
ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=.8)
ax.plot(xl, yl, zl, color="black")
trace_x = trace[:, 0]
trace_y = trace[:, 1]
trace_z = np.array([rosenbrock((trace_x[i], trace_y[i])) for i in range(len(trace_x))])
ax.plot(trace_x, trace_y, trace_z, c='red', marker='o', markersize=1)
ax.scatter(trace_x[-1], trace_y[-1], trace_z[-1], c='green', marker='o', linewidths=5)
print(trace[0], '-->', trace[-1], "steps:", len(trace))
plt.show()




