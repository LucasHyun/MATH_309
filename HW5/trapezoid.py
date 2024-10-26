#Code Trapezoid rule seperately to compute the integral of a function
# The function is f(x) = sin(x) from 0 to pi
# The exact integral is 2.0
# The error is calculated as the difference between the exact integral and the computed integral
# The error is printed to the screen
# The code is run with N = 50382

import numpy as np
from typing import Callable
def f(x):
    return np.sin(x)

def trapezoid_method(f: Callable, a: float, b: float, n: int) -> float:
    """
    Compute integral using trapezoidal rule

    Parameters:
        f: Function to integrate
        a: Lower bound
        b: Upper bound
        n: Number of intervals
    """
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])


#Calculating the error
Trapezoid = trapezoid_method(f,0.0,np.pi,50382)
print("Trapezoid: ", Trapezoid)
Error = Trapezoid - 2.0
print("Error: ", Error)

#check if error is less than 10^-9
if abs(Error) < 1.0e-9:
    print("The error is less than 10^-9")

#The testing results show that the error is less than 10^-9
# Trapezoid:  1.9999999993519302
# Error:  -6.480698200306279e-10
# The error is less than 10^-9
