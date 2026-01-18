import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def grf_1D(lb, up, num, l):
    x = np.linspace(lb, up, num)  # Discretize the interval [0, 1]

    # Covariance matrix using the Gaussian kernel
    def gaussian_kernel(x, l):
        """ Generate a Gaussian kernel matrix with correlation length l. """
        x = x[:, np.newaxis]  # Convert to column vector
        # Squared exponential kernel
        return np.exp(-0.5 * (x - x.T) ** 2 / l ** 2)

    covariance_matrix = gaussian_kernel(x, l)

    # Sample from the multivariate normal distribution
    mean = np.zeros(num)  # Zero mean
    random_field = multivariate_normal.rvs(mean=mean, cov=covariance_matrix)

    max_val = np.max(np.abs(random_field))  # Find the maximum absolute value
    random_field = 0.9 * random_field / max_val
    return x, random_field

def grf_2D(node, l):
    # node: a list containing all the nodes (x, y) coordinates
    # l: correlation length
    # return: a list of random field values at each node
    x = [n[0] for n in node]
    y = [n[1] for n in node]
    x = np.array(x)
    y = np.array(y)
    num = len(x)
    # Covariance matrix using the Gaussian kernel
    def gaussian_kernel(x, y, l):
        """ Generate a Gaussian kernel matrix with correlation length l. """
        x = x[:, np.newaxis]
        y = y[:, np.newaxis]
        # Squared exponential kernel
        return np.exp(-0.5 * (x - x.T) ** 2 / l ** 2 - 0.5 * (y - y.T) ** 2 / l ** 2)

    covariance_matrix = gaussian_kernel(x, y, l)
    # Sample from the multivariate normal distribution
    mean = np.zeros(num)  # Zero mean
    random_field = multivariate_normal.rvs(mean=mean, cov=covariance_matrix)
    return random_field

def batch_grf_1d(x, l, batch_size=1, sigma=1.0, mean=None, jitter=1e-8, random_state=None):
    """
    Draw samples from a 1D Gaussian Random Field with SE kernel:
        K_ij = sigma^2 * exp(-0.5 * (x_i - x_j)^2 / l^2)

    Parameters
    ----------
    x : (n,) array_like
        1D locations.
    l : float
        Correlation length (must be > 0).
    batch_size : int, optional
        Number of independent GRF samples to draw. Default is 1.
    sigma : float, optional
        Marginal standard deviation. Default is 1.0.
    mean : None, float, or (n,) array_like, optional
        Mean value(s) at each x. If None, uses 0.
    jitter : float, optional
        Small diagonal term added to K for numerical stability.
    random_state : None, int, or np.random.Generator, optional
        Seed or Generator for reproducibility.

    Returns
    -------
    samples : (batch_size, n) ndarray
        Each row is one GRF sample evaluated at x.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n == 0:
        raise ValueError("`x` must contain at least one location.")
    if l <= 0:
        raise ValueError("`l` (correlation length) must be positive.")

    # Build covariance matrix via pairwise squared distances
    dx2 = (x[:, None] - x[None, :])**2
    K = (sigma ** 2) * np.exp(-0.5 * dx2 / (l ** 2))

    # Add jitter on the diagonal for numerical stability
    K[np.diag_indices(n)] += jitter

    # Random number generator
    rng = (np.random.default_rng(random_state)
           if not isinstance(random_state, np.random.Generator)
           else random_state)

    # Try Cholesky; if it fails (rare), fall back to eigen-decomposition
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(K)
        w = np.clip(w, 0.0, None)
        L = V @ np.diag(np.sqrt(w))

    # Draw standard normals and correlate via L
    Z = rng.standard_normal(size=(batch_size, n))
    samples = Z @ L.T

    # Add mean if provided
    if mean is not None:
        mu = (np.full(n, float(mean)) if np.isscalar(mean)
              else np.asarray(mean, dtype=float).ravel())
        if mu.shape != (n,):
            raise ValueError("`mean` must be scalar or have shape (n,).")
        samples = samples + mu

    return samples


if __name__ == '__main__':
    # n_points = 100  # Number of points in the domain
    # correlation_length = 0.5
    # correlation_length = 0.5
    # x, random_field = grf_1D(0, 1, num=100, l=correlation_length)
    # plt.figure(figsize=(10, 5))
    # plt.plot(x, random_field, label='Gaussian Random Field')
    # plt.title(f'1D Gaussian Random Field with Correlation Length l = {correlation_length}')
    # plt.xlabel('x')
    # plt.ylabel('Field Value')
    # plt.legend()
    # plt.show()

    # Visualize 2D Gaussian Random Field
    # Define the domain
    x_len = 1.0
    y_len = 1.0
    num_x = num_y = 16
    node = np.array([[i * x_len / num_x, j * y_len / num_y] for i in range(num_x + 1) for j in range(num_y + 1)])
    random_field = grf_2D(node, l=0.3)
    plt.figure()
    # Plot the contour plot
    plt.tricontourf([n[0] for n in node], [n[1] for n in node], random_field, levels=100)
    plt.colorbar()
    plt.title("2D Gaussian Random Field")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
