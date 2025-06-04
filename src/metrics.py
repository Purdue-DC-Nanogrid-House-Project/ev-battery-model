import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_metrics():
    filepath = 'data/metrics.csv'

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

    # Plotting
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Bigger dots using 's', optional color gradient by weight
    scatter = ax.scatter(weights, e_grid_home, e_back_feed, c='blue', s=60, marker='o')

    # Add weight labels to each point
    for w, x, y in zip(weights, e_grid_home, e_back_feed):
        ax.text(w, x, y, f'{w}', fontsize=9, ha='center', va='bottom')

    # Labels
    ax.set_xlabel('Weight')
    ax.set_ylabel('Energy from Grid to Home (kWh)')
    ax.set_zlabel('Energy Fed Back to Grid (kWh)')
    ax.set_title('Energy Metrics vs Weight')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_metrics()