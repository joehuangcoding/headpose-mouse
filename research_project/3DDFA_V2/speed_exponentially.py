import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import math

# Initial speed scaling factor
initial_speed_scaling_factor = 0.1

# Sample data for the absolute difference in yaw angles
yaw_differences = np.arange(0, 50, 0.1)

# Calculate speed_yaw for each absolute difference
speed_yaw_values = [math.exp(diff * initial_speed_scaling_factor) for diff in yaw_differences]

# Create the initial plot
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)  # Adjust the bottom to make room for the slider

# Plotting
line, = ax.plot(yaw_differences, speed_yaw_values, label='Speed_yaw')
plt.title('Speed_yaw vs. Absolute Yaw Difference')
plt.xlabel('Absolute Yaw Difference')
plt.ylabel('Speed_yaw')
plt.legend()

# Add a slider for adjusting the speed_scaling_factor
ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
speed_scaling_factor_slider = Slider(ax_slider, 'Speed Scaling Factor', 0.001, 1.0, valinit=initial_speed_scaling_factor)

# Function to update the plot when the slider is moved
def update(val):
    speed_scaling_factor = speed_scaling_factor_slider.val
    speed_yaw_values = [math.exp(diff * speed_scaling_factor) for diff in yaw_differences]
    line.set_ydata(speed_yaw_values)
    fig.canvas.draw_idle()

# Attach the update function to the slider
speed_scaling_factor_slider.on_changed(update)

plt.show()
