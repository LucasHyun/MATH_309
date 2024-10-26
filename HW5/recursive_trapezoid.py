#Recursive Trapezoid Rule
#The recursive trapezoid rule is a method to compute the integral of a function.
#The function is f(x) = sin(x) from 0 to pi.
#The exact integral is 2.0.
import numpy as np
from typing import Callable
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

