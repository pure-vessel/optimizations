import numpy as np
import random

"""
Z = dot(c, x) with Ax <= b and x >= 0    |-->    Z = dot(c', x') with A'x' = b and x' >= 0

x' = [x_1, x_2, ..., x_n,   y_1, ..., y_m]
y are slack variables
"""
def add_slack(A, b, c):
    return (
        np.concatenate((A, np.eye(len(A))), axis=1),
        np.concatenate((c, np.zeros(len(A)))),
    )


"""
Makes the following matrix:
[1 c^T 0]
[0  A  b]
"""
def make_simplex_method_matrix(A, b, c):
    return np.concatenate(([[1]] + [[0]] * len(A), np.concatenate(([c], A)), np.concatenate(([0], b)).reshape(len(A) + 1, 1)), axis=1)

# maximizes c^T @ x.
def simplex_method(matrix, basic_columns):
    n = matrix.shape[1] - 2
    m = matrix.shape[0] - 1
    print("Matrix:", matrix, sep='\n')
    print("Basic columns:", basic_columns)
    while True:
        pivot_column = (None, 0)
        for i in range(1, n + 1):
            if i not in basic_columns and matrix[0][i] < pivot_column[1]:
                pivot_column = (i, matrix[0][i])
        if pivot_column[0] == None:
            return matrix[0][-1]
        pivot_column = pivot_column[0]
        print("Pivot column:", pivot_column, end='\t')
        pivot_row = (None, 1e500)
        for j in range(1, m + 1):
            if matrix[j][pivot_column] > 0 and matrix[j][-1] / matrix[j][pivot_column] < pivot_row[1]:
                pivot_row = (j, matrix[j][-1] / matrix[j][pivot_column])
        if pivot_row == (None, 1e500):
            return 1e500 # no upper limits found
        pivot_row = pivot_row[0]
        print("Pivot row:", pivot_row)
        matrix[pivot_row] /= matrix[pivot_row][pivot_column]
        for j in range(m + 1):
            if j != pivot_row:
                matrix[j] -= matrix[pivot_row] * matrix[j][pivot_column]
        print("Matrix:", matrix, sep='\n')
        basic_columns[pivot_row - 1] = pivot_column
        print("Basic columns:", basic_columns)

"""
maximizes c^T @ x
Ax <= b
x >= 0
coefficients in b may be negative, meaning two-phase simplex method can be needed
"""
def solve_lp(A, b, c):
    A1, c1 = add_slack(A, b, c)
    matrix = make_simplex_method_matrix(A1, b, c1)
    basic_columns = list(range(len(c) + 1, len(c1) + 1))
    two_phase_needed = False
    for bi in b:
        if bi < 0:
            two_phase_needed = True
            break
    if not two_phase_needed:
        print("No two-phase needed")
        return simplex_method(matrix, basic_columns)
    row0 = [1] + [0] * (matrix.shape[1] - 1)
    for i, bi in enumerate(b):
        if bi < 0:
            artificial = np.zeros(len(matrix), dtype='float64')
            artificial[i + 1] = 1
            matrix[i + 1] *= -1
            basic_columns[i] = matrix.shape[1]
            matrix = np.insert(matrix, matrix.shape[1] - 1, artificial, axis=1)
            row0.append(1)
        else:
            basic_columns[i] += 1
    row0.append(0)
    row0 = np.array(row0, dtype='float64')
    matrix = np.insert(matrix, 0, np.zeros(len(matrix)), axis=1)
    for i, bi in enumerate(b):
        if bi < 0:
            row0 -= matrix[i + 1]
    matrix = np.insert(matrix, 0, row0, axis=0)
    basic_columns.insert(0, None)
    print("Running first phase")
    if abs(simplex_method(matrix, basic_columns)) >= 1e-9:
        return -1e500 # no solution exists
    matrix = matrix[1:, 1:]
    matrix = np.delete(matrix, np.s_[len(c1) + 1:-1], axis=1)
    basic_columns = list(map(lambda c: c - 1, basic_columns[1:]))
    print("Running second phase")
    return simplex_method(matrix, basic_columns)




print("Running example 1")
A = np.array([[1, 1], [-1, 3], [0, 1]], dtype='float64')
b = np.array([3, 1, 3], dtype='float64')
c = np.array([-1, -1], dtype='float64')
print(solve_lp(A, b, c))
print('\n--------------------------------------\n')
print("Running example 2")
A = np.array([[1, -1], [1, -3], [0, 1]], dtype='float64')
b = np.array([-1, -1, 3], dtype='float64')
c = np.array([-1, -1], dtype='float64')
print(solve_lp(A, b, c))
