import numpy as np
import matplotlib.pyplot as plt

def lspline(t, y, xint):
    """
    Perform linear spline interpolation

    :param t: Original time points (input x coordinates)
    :param y: Original y values
    :param xint: x coordinates to interpolate
    :return: Interpolated y values
    """
    n = len(t)  # Number of original data points
    yint = np.zeros_like(xint)  # Initialize array to store interpolated y values

    for i in range(1, n):
        # Create mask for xint values in the current interval
        mask = (xint >= t[i - 1]) & (xint <= t[i])

        # Apply linear interpolation formula
        # y = y1 + (y2-y1)/(x2-x1) * (x-x1)
        yint[mask] = y[i - 1] + (y[i] - y[i - 1]) / (t[i] - t[i - 1]) * (xint[mask] - t[i - 1])

    return yint  # Return interpolated y values

# Natural Cubic spline by solving the tridiagonal system of equations
def cspline(t, y, xint):
    """
    Perform cubic spline interpolation

    :param t: Input x coordinates
    :param y: Input y coordinates
    :param xint: x coordinates to interpolate
    :return: Interpolated y values
    """
    n = len(t)
    h = np.diff(t)  # Intervals between consecutive x coordinates

    # Set up coefficient matrix A and vector b
    A = np.zeros((n, n))
    b = np.zeros(n)

    # Natural spline condition: S''(t_0) = S''(t_n) = 0
    A[0, 0] = 1
    A[-1, -1] = 1

    # Set continuity and smoothness conditions
    for i in range(1, n - 1):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        # Second derivative continuity condition
        b[i] = 6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    # Calculate second derivatives (C values)
    C = np.linalg.solve(A, b)

    # Calculate spline coefficients and interpolate
    cs = []
    for x in xint:
        for i in range(n - 1):
            if t[i] <= x <= t[i + 1]:
                dx = x - t[i]
                # Cubic spline equation:
                # S(x) = y[i] + t3*dx + t2*dx^2 + t1*dx^3
                t1 = (C[i + 1] - C[i]) / (6 * h[i])
                t2 = C[i] / 2
                t3 = (y[i + 1] - y[i]) / h[i] - (h[i] * (C[i + 1] + 2 * C[i])) / 6
                cs.append(y[i] + dx * (t3 + dx * (t2 + dx * t1)))
                break
        else:
            cs.append(None)  # x is out of input range

    return cs  # Return interpolated y values

# Test your code with the following data:
ti = [1.2, 1.5, 1.6, 2.0, 2.2]
yi = [0.4275, 1.139, 0.8736, -0.9751, -0.1536]

# Compute x = 1.8 for both linear and cubic spline
x = 1.8
y_ls = lspline(ti, yi, np.array([x]))
y_cs = cspline(ti, yi, np.array([x]))
print("My custom implementation")
print('Linear spline at x = 1.8:', y_ls)
print('Cubic spline at x = 1.8:', y_cs)

# Compare the results with the original implementation of the spline function (linear, cubic) in Python.
from scipy.interpolate import CubicSpline, interp1d
print("Scipy implementation")
print('Linear spline at x = 1.8:', interp1d(ti, yi, kind='linear')(1.8))
print('Cubic spline at x = 1.8:', CubicSpline(ti, yi)(1.8))

# Define mountain points
mountain1_x = [0, 0.5, 1]
mountain1_y = [0, 2, 0]
mountain2_x = [0.7, 1, 1.4, 1.5, 1.6, 2, 2.3]
mountain2_y = [0, 0.5, 1.3, 1.5, 1.3, 0.5, 0]

# Green curve points
land_x = [0, 0.5, 1, 1.5, 2.3]
land_y = [0, 0.1, 0, 0.1, 0]

# Sun's top half control points for the spline
upper_sun_x = [1.65, 1.8, 1.95]
upper_sun_y = [2.35, 2.7, 2.35]

# Sun's bottom half control points for the spline
lower_sun_x = [1.65, 1.8, 1.95]
lower_sun_y = [2.35, 2, 2.35]  # Adjusted to create a mirrored lower arc

# Generate waves using custom cspline
wave_control_x = np.linspace(0, 2.3, 10)
wave_control_y = np.array([0, 0.03, -0.02, 0.05, -0.05, 0.04, -0.03, 0.06, -0.01, 0])
waves_x = np.linspace(0, 2.3, 500)
waves_y = cspline(wave_control_x, wave_control_y, waves_x)

# Create plot
plt.figure(figsize=(10, 6))

# First mountain
x_range1 = np.linspace(0, 1, 500)
mountain1_spline = cspline(mountain1_x, mountain1_y, x_range1)
plt.fill_between(x_range1, mountain1_spline, color='gray', alpha=0.6, zorder=1)

# Second mountain
x_range2 = np.linspace(0.7, 2.3, 1000)
mountain2_spline = cspline(mountain2_x, mountain2_y, x_range2)
plt.fill_between(x_range2, mountain2_spline, color='darkgray', alpha=0.7, zorder=2)

# Green curve
land_spline = cspline(land_x, land_y, np.linspace(0, 2.3, 500))
plt.plot(np.linspace(0, 2.3, 500), land_spline, color='lightgreen', linewidth=5, zorder=3)

# Draw waves
for shift in np.linspace(-0.3, -0.2, 3):
    plt.plot(waves_x, shift + waves_y, color='deepskyblue', zorder=0)

# Draw sun using cubic splines for the top half and bottom half
upper_sun_spline = cspline(upper_sun_x, upper_sun_y, np.linspace(1.65, 1.95, 100))
lower_sun_spline = cspline(lower_sun_x, lower_sun_y, np.linspace(1.95, 1.65, 100))

plt.plot(np.linspace(1.65, 1.95, 100), upper_sun_spline, color='tomato', linewidth=3, alpha=0.7, zorder=4)
plt.plot(np.linspace(1.65, 1.95, 100), lower_sun_spline, color='tomato', linewidth=3, alpha=0.7, zorder=4)

# Set plot limits and hide axes
plt.xlim(0, 2.3)
plt.ylim(-0.5, 3)
plt.axis('off')

# Display plot
plt.tight_layout()
plt.show()

# Save plot
plt.savefig('./mountain_sun_spline.png')