import numpy as np
from scipy.stats import gaussian_kde



def generate_heat_map(base_stations, grid_size, bandwidth, min_x, max_x, min_y, max_y):
    """
    base_stations: array of shape (N, 2) with x, y coordinates of base stations
    grid_size: heat map resolution
    bandwidth: KDE smoothing parameter. The higher the value, the smoother the heat map (less peaks).
    min_x, max_x, min_y, max_y: limits of the map
    """
    x = base_stations[:, 0]
    y = base_stations[:, 1]
    
    # Create the KDE
    kde = gaussian_kde(np.vstack([x, y]), bw_method=bandwidth)
    
    # Create the meshgrid
    xi, yi = np.meshgrid(
        np.linspace(min_x, max_x, grid_size),
        np.linspace(min_y, max_y, grid_size)
    )
    coords = np.vstack([xi.ravel(), yi.ravel()])
    zi = kde(coords).reshape(xi.shape)
    
    # Normalize to use as a probability distribution
    zi_norm = zi / zi.sum()
    
    return xi, yi, zi_norm


def generate_heat_map_bordered(base_stations, grid_size=100, bandwidth=0.5, border=3):
    """
    base_stations: array of shape (N, 2) with x, y coordinates of base stations
    grid_size: heat map resolution
    bandwidth: KDE smoothing parameter
    border: fraction of the range (for example, 0.1 adds a 10% margin to each side)
    """
    x = base_stations[:, 0]
    y = base_stations[:, 1]
    
    # Calculate the limits with a margin proportional to the range
    x_min = x.min() - border
    x_max = x.max() + border
    y_min = y.min() - border
    y_max = y.max() + border
    
    return generate_heat_map(base_stations, grid_size, bandwidth, x_min, x_max, y_min, y_max)


def sample_users(xi, yi, zi_norm, num_users=100, seed=123456789):
    """
    xi, yi: meshgrid coordinates of the heat map
    zi_norm: normalized density
    """
    flat_probs = zi_norm.ravel()
    flat_probs /= flat_probs.sum()  # ensure sum is 1
    
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(flat_probs), size=num_users, p=flat_probs)
    
    x_coords = xi.ravel()[indices]
    y_coords = yi.ravel()[indices]
    
    return np.vstack((x_coords, y_coords)).T
