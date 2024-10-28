import numpy as np
import random
import itertools
import math

random.seed(85377666522016058)

def crossover(fst, snd):
    bpoint = random.randint(1, len(fst) - 1)
    start = fst[:bpoint]
    start_set = set(start)
    return tuple(list(start) + [j for j in snd[bpoint:] if j not in start_set] + [j for j in snd[:bpoint] if j not in start_set])

def mutate(perm):
    n = len(perm)
    i = 0
    j = 1
    while i == (j + 1) % n or j == (i + 1) % n:
        i, j = random.sample(range(n), 2)
    if i > j:
        i, j = j, i
    return tuple(list(perm[:i + 1]) + list(reversed(perm[i + 1:j + 1])) + list(perm[j + 1:]))

def evaluate_goodness(graph, perm):
    return sum([graph[perm[i]][perm[(i + 1) % len(perm)]] for i in range(len(perm))])

def genetic(graph, iterations, n, m, mu, nu):
    duplicates_cnt = m - n
    population = list(map(lambda p: (p, evaluate_goodness(graph, p)), random.sample(list(itertools.permutations(range(len(graph)))), n)))
    for it in range(iterations):
        population = population + random.choices(population, k=duplicates_cnt)
        for i in range(m):
            if random.random() < mu:
                p = mutate(population[i][0])
                population[i] = (p, evaluate_goodness(graph, p))
        for i in range(m):
            for j in range(m):
                if i != j and random.random() < nu:
                    child = crossover(population[i][0], population[j][0])
                    population.append((child, evaluate_goodness(graph, child)))
        population.sort(key=lambda p: p[1])
        population = population[:n]
    return population[0]

def random_graph(size, metric):
    points = [(random.random(), random.random()) for _ in range(size)]
    return [[metric(points[i], points[j]) for i in range(size)] for j in range(size)]


graph = random_graph(10, lambda p1, p2: math.hypot(p1[0] - p2[0], p1[1] - p2[1]))
for r in graph:
    print(*r, sep=' ')
if len(graph) <= 10:
    opt = (None, 1e500)
    for p in itertools.permutations(range(len(graph))):
        g = evaluate_goodness(graph, p)
        if g < opt[1]:
            opt = (p, g)
    print(f"\nOptimum is\n{opt[1]}\t{opt[0]}\n")
opt_found = genetic(graph, 20, 1000, 1500, 0.1, 1e-5)
print(f"Genetic algorithm found\n{opt_found[1]}\t{opt_found[0]}\n")



