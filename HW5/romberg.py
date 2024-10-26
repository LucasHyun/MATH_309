import numpy as np
from typing import Callable

def f1(x): return np.sin(x)

def f2(x): return np.sqrt(x)

def f3(x): return np.cos(2*x) * np.exp(-x)


def recursive_trapezoid(f: Callable, a: float, b: float, n: int) -> float:
    """
    Compute the integral of a function using the recursive trapezoid rule.

    Parameters:
    f (Callable): The function to integrate.
    a (float): The start of the interval.
    b (float): The end of the interval.
    n (int): The number of subintervals.

    Returns:
    float: The approximate integral of the function over [a, b].
    """
    # Base case: if n is 1, use the simple trapezoid rule
    if n == 1:
        return (b - a) * (f(a) + f(b)) / 2

    # Calculate the step size
    h = (b - a) / n
    # Generate n+1 equally spaced points between a and b
    x = np.linspace(a, b, n + 1)
    # Apply the trapezoid rule recursively
    return h * (0.5 * (f(x[0]) + f(x[-1])) + np.sum(f(x[1:-1])))

def romberg(f: Callable, a: float, b: float, n: int) -> np.ndarray:
    """
    Compute the integral of a function using Romberg integration.

    Parameters:
    f (Callable): The function to integrate.
    a (float): The start of the interval.
    b (float): The end of the interval.
    n (int): The number of levels of Romberg integration.

    Returns:
    np.ndarray: The Romberg integration table.
    """
    R = np.zeros((n, n))  # Initialize the Romberg table with zeros.

    # First column using recursive trapezoid rule
    for i in range(n):
        R[i, 0] = recursive_trapezoid(f, a, b, 2 ** i)

    # Compute remaining columns using Richardson extrapolation
    for j in range(1, n):
        for i in range(n - j):
            R[i, j] = (4 ** j * R[i + 1, j - 1] - R[i, j - 1]) / (4 ** j - 1)

    return R

# Test for first integral: sin(x) from 0 to pi
n = 6
print("Romberg Integration for sin(x) from 0 to pi:")
R1 = romberg(f1, 0, np.pi, n)
print("Romberg table:")
for i in range(n):
    print([f"{x:.10f}" for x in R1[i, :n - i]])
print(f"Final approximation: {R1[0, n - 1]:.10f}")
print(f"Actual value: 2.0000000000")
print()

# Test for second integral: sqrt(x) from 0 to 1
print("Romberg Integration for sqrt(x) from 0 to 1:")
R2 = romberg(f2, 0, 1, n)
print("Romberg table:")
for i in range(n):
    print([f"{x:.10f}" for x in R2[i, :n - i]])
print(f"Final approximation: {R2[0, n - 1]:.10f}")
print(f"Actual value: {2 / 3:.10f}")

print("Romberg Integration for cos(2x)*exp(-x) from 0 to pi/2:")
R3 = romberg(f3, 0, np.pi / 2, n)
print("Romberg table:")
for i in range(n):
    print([f"{x:.10f}" for x in R3[i, :n - i]])
print(f"Final approximation: {R3[0, n - 1]:.10f}")
print(f"Actual value: 0.2415759")