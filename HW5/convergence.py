import numpy as np
import matplotlib.pyplot as plt
from typing import Callable


def f(x):
    """Test function: f(x) = sin(x)"""
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


# Set up integration parameters
a, b = 0, np.pi  # Integration interval [0, pi]
exact = 2.0  # Exact integral of sin(x) from 0 to pi

# Generate n values as powers of 2
n_values = [2 ** i for i in range(2, 13)]
h_values = [(b - a) / n for n in n_values]

# Calculate errors for both methods
trap_errors = []
simp_errors = []

for n in n_values:
    trap_result = trapezoid_method(f, a, b, n)
    simp_result = simpson_method(f, a, b, n)

    trap_errors.append(abs(trap_result - exact))
    simp_errors.append(abs(simp_result - exact))

# Create log-log plot
plt.figure(figsize=(10, 6))
plt.loglog(h_values, trap_errors, 'o-', label='Trapezoidal Rule')
plt.loglog(h_values, simp_errors, 's-', label="Simpson's Rule")

# Add reference lines
ref_h = np.array([h_values[0], h_values[-1]])
ref_2 = trap_errors[0] * (ref_h / ref_h[0]) ** 2  # O(h^2) reference
ref_4 = simp_errors[0] * (ref_h / ref_h[0]) ** 4  # O(h^4) reference

plt.loglog(ref_h, ref_2, 'k--', alpha=0.5, label='O(h^2) Reference')
plt.loglog(ref_h, ref_4, 'k:', alpha=0.5, label='O(h^4) Reference')

# Customize plot
plt.grid(True)
plt.xlabel('Step size (h)')
plt.ylabel('Absolute Error')
plt.title('Convergence Analysis of Numerical Integration Methods')
plt.legend()

# Calculate and display slopes
trap_slope = np.polyfit(np.log(h_values), np.log(trap_errors), 1)[0]
simp_slope = np.polyfit(np.log(h_values), np.log(simp_errors), 1)[0]

plt.text(0.02, 0.98,
         f'Trapezoidal slope: {abs(trap_slope):.2f}\nSimpson slope: {abs(simp_slope):.2f}',
         transform=plt.gca().transAxes,
         verticalalignment='top')

plt.show()
plt.savefig('convergence.png')
# Print convergence analysis results
print("\nConvergence Rate Analysis:")
print(f"Trapezoidal Rule experimental convergence rate: {abs(trap_slope):.2f} (theoretical: 2)")
print(f"Simpson's Rule experimental convergence rate: {abs(simp_slope):.2f} (theoretical: 4)")

# Print detailed error table
print("\nDetailed Error Analysis:")
print("    N    |    h    | Trap Error | Simp Error")
print("-" * 45)
for n, h, te, se in zip(n_values, h_values, trap_errors, simp_errors):
    print(f"{n:7d} | {h:.2e} | {te:.2e} | {se:.2e}")

# result
# Convergence Rate Analysis:
# Trapezoidal Rule experimental convergence rate: 2.00 (theoretical: 2)
# Simpson's Rule experimental convergence rate: 4.00 (theoretical: 4)
#
# Detailed Error Analysis:
#     N    |    h    | Trap Error | Simp Error
# ---------------------------------------------
#       4 | 7.85e-01 | 1.04e-01 | 4.56e-03
#       8 | 3.93e-01 | 2.58e-02 | 2.69e-04
#      16 | 1.96e-01 | 6.43e-03 | 1.66e-05
#      32 | 9.82e-02 | 1.61e-03 | 1.03e-06
#      64 | 4.91e-02 | 4.02e-04 | 6.45e-08
#     128 | 2.45e-02 | 1.00e-04 | 4.03e-09
#     256 | 1.23e-02 | 2.51e-05 | 2.52e-10
#     512 | 6.14e-03 | 6.27e-06 | 1.57e-11
#    1024 | 3.07e-03 | 1.57e-06 | 9.84e-13
#    2048 | 1.53e-03 | 3.92e-07 | 6.13e-14
#    4096 | 7.67e-04 | 9.80e-08 | 4.00e-15
