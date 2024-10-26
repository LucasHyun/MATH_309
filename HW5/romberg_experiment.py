import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

# Define the function f1 as sin(x)
def f1(x): return np.sin(x)

# Define the function f2 as sqrt(x)
def f2(x): return np.sqrt(x)

# Define the function f3 as cos(2x) * exp(-x)
def f3(x): return np.cos(2 * x) * np.exp(-x)

# Define the recursive trapezoid rule function
def recursive_trapezoid(f: Callable, a: float, b: float, n: int) -> float:
    """Compute integral using recursive trapezoid rule"""
    # Calculate the step size
    h = (b - a) / n
    # Generate n+1 equally spaced points between a and b
    x = np.linspace(a, b, n + 1)
    # Apply the trapezoid rule recursively
    return h * (0.5 * (f(x[0]) + f(x[-1])) + np.sum(f(x[1:-1])))

# Define the Romberg integration function
def romberg(f: Callable, a: float, b: float, n: int) -> np.ndarray:
    """Compute Romberg integration table"""
    # Initialize the Romberg table with zeros
    R = np.zeros((n, n))

    # First column using recursive trapezoid rule
    for i in range(n):
        R[i, 0] = recursive_trapezoid(f, a, b, 2 ** i)

    # Compute remaining columns using Richardson extrapolation
    for j in range(1, n):
        for i in range(n - j):
            R[i, j] = (4 ** j * R[i + 1, j - 1] - R[i, j - 1]) / (4 ** j - 1)

    return R

# Define the function to analyze convergence of Romberg integration
def analyze_convergence(f: Callable, a: float, b: float, exact_val: float, max_level: int = 10):
    """Analyze convergence of Romberg integration"""
    # Define the range of levels to analyze
    levels = range(2, max_level + 1)
    # Initialize the list to store errors
    errors = []

    # Compute the Romberg integration and errors for each level
    for n in levels:
        R = romberg(f, a, b, n)
        error = abs(R[0, -1] - exact_val)
        errors.append(error)

    return np.array(list(levels)), np.array(errors)

# Setup test cases for different functions and intervals
test_cases = [
    {
        'f': f1,
        'name': 'sin(x)',
        'interval': (0, np.pi),
        'exact': 2.0,
        'color': 'blue'
    },
    {
        'f': f2,
        'name': 'sqrt(x)',
        'interval': (0, 1),
        'exact': 2 / 3,
        'color': 'red'
    },
    {
        'f': f3,
        'name': 'cos(2x)exp(-x)',
        'interval': (0, np.pi / 2),
        'exact': 0.2415759,
        'color': 'green'
    }
]

# Perform convergence analysis and plot results
plt.figure(figsize=(12, 8))

for case in test_cases:
    # Analyze convergence for each test case
    levels, errors = analyze_convergence(
        case['f'],
        case['interval'][0],
        case['interval'][1],
        case['exact']
    )

    # Plot error vs level
    plt.semilogy(levels, errors, 'o-', label=case['name'], color=case['color'])

    # Calculate and print the experimental convergence rate
    convergence_rate = np.polyfit(np.log(levels), np.log(errors), 1)[0]
    print(f"\nConvergence analysis for {case['name']}:")
    print(f"Experimental convergence rate: {abs(convergence_rate):.2f}")

    # Perform detailed analysis for one level
    n = 6
    R = romberg(case['f'], case['interval'][0], case['interval'][1], n)
    print("\nRomberg table:")
    for i in range(n):
        row = [f"{x:.10f}" for x in R[i, :n - i]]
        print(f"n={2 ** i}: {row}")

    # Analyze and print diagonal element errors
    diag_errors = [abs(R[i, i] - case['exact']) for i in range(n)]
    print("\nDiagonal element errors:")
    for i, error in enumerate(diag_errors):
        print(f"Level {i}: {error:.2e}")

# Configure plot settings
plt.grid(True)
plt.xlabel('Integration Level (n)')
plt.ylabel('Absolute Error (log scale)')
plt.title('Romberg Integration Convergence Analysis')
plt.legend()

# Add theoretical reference lines for O(h^4) and O(h^8)
ref_levels = np.array([levels[0], levels[-1]])
ref_4 = errors[0] * (ref_levels[0] / ref_levels) ** 4
ref_8 = errors[0] * (ref_levels[0] / ref_levels) ** 8

plt.semilogy(ref_levels, ref_4, 'k--', alpha=0.5, label='O(h^4) Reference')
plt.semilogy(ref_levels, ref_8, 'k:', alpha=0.5, label='O(h^8) Reference')
plt.legend()

# Show and save the plot
plt.show()
plt.savefig('romberg_convergence.png')

# Additional analysis: Print error reduction ratios between successive levels
print("\nError reduction ratios between successive levels:")
for case in test_cases:
    print(f"\n{case['name']}:")
    n = 6
    R = romberg(case['f'], case['interval'][0], case['interval'][1], n)
    for i in range(1, n):
        ratio = abs(R[0, i - 1] - case['exact']) / abs(R[0, i] - case['exact'])
        print(f"Level {i - 1} to {i}: {ratio:.2f}")

# Output:
# Error reduction ratios between successive levels:
#
# sin(x):
# Level 0 to 1: 21.19
# Level 1 to 2: 66.04
# Level 2 to 3: 257.53
# Level 3 to 4: 1025.36
# Level 4 to 5: 4096.92
#
# sqrt(x):
# Level 0 to 1: 5.83
# Level 1 to 2: 3.21
# Level 2 to 3: 2.91
# Level 3 to 4: 2.85
# Level 4 to 5: 2.83
#
# cos(2x)exp(-x):
# Level 0 to 1: 11.13
# Level 1 to 2: 121.32
# Level 2 to 3: 189.17
# Level 3 to 4: 129.88
# Level 4 to 5: 0.75