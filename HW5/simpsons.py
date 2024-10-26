#Code simpson's rule to compute the integral of a function
# The function is f(x) = sin(x) from 0 to pi
# The exact integral is 2.0
# The error is calculated as the difference between the exact integral and the computed integral
# The error is printed to the screen
# The code is run with N = 50382
import numpy as np
from typing import Callable
def f(x):
    return np.sin(x)

def simpson_method(f: Callable, a: float, b: float, n: int) -> float:
    """
    Compute integral using Simpson's rule

    Parameters:
        f: Function to integrate
        a: Lower bound
        b: Upper bound
        n: Number of intervals (must be even)
    """
    if n % 2 != 0:
        n += 1  # Ensure n is even

    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)

    # Simpson's 1/3 rule
    return h / 3 * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2]) + y[-1])

#Calculating the error
Simpson = simpson_method(f,0.0,np.pi,204)
print("Simpson: ", Simpson)
Error = Simpson - 2.0
print("Error: ", Error)

#check if error is less than 10^-9
if abs(Error) < 1.0e-9:
    print("The error is less than 10^-9")
