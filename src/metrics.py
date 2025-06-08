import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from datetime import datetime
import numpy as np

def plot_metrics():
    filepath = 'data/metrics_6_4_25.csv'
    
    # Format today's date
    today_str = datetime.today().strftime('%Y%m%d')
    save_filename = f'metrics_plot_{today_str}.png'
    save_path = os.path.join('data', save_filename)

    # Load the metrics
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"No metrics file found at {filepath}")
        return
    if df.empty:
        print("metrics.csv is empty.")
        return

    # Extract columns
    weights = df['Weight']
    e_grid_home = df['E_grid_home']
    e_back_feed = df['E_back_feed']

    # Normalize energy values
    e_grid_home_norm = e_grid_home - e_grid_home.min()
    e_back_feed_norm = e_back_feed - e_back_feed.min()

    # Compute distance from origin in normalized space
    distances = np.sqrt(e_grid_home_norm**2 + e_back_feed_norm**2)
    min_index = distances.idxmin()  # Index of optimum point

    # Plotting
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot all points
    ax.scatter(weights, e_grid_home_norm, e_back_feed_norm, c='blue', s=60, marker='o', label='All Points')

    # Highlight optimum point in red
    ax.scatter(weights[min_index], e_grid_home_norm[min_index], e_back_feed_norm[min_index],
               c='red', s=100, marker='^', label=f'Optimum (Weight={weights[min_index]})')

    for w, x, y in zip(weights, e_grid_home_norm, e_back_feed_norm):
        ax.text(w, x, y, f'{w}', fontsize=8, ha='center', va='bottom')

    ax.set_xlabel('Weight')
    ax.set_ylabel('Energy from Grid to Home (kWh, normalized)')
    ax.set_zlabel('Energy Fed Back to Grid (kWh, normalized)')
    ax.set_title('Normalized Energy Metrics vs Weight')

    ax.legend()
    plt.tight_layout()

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the figure
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")
    print(f"Most optimum weight is: {weights[min_index]}")

    plt.show()

if __name__ == "__main__":
    plot_metrics()