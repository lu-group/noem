import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def grf_1Dv2(x, l):
    # Covariance matrix using the Gaussian kernel
    def gaussian_kernel(x, l):
        """ Generate a Gaussian kernel matrix with correlation length l. """
        x = x[:, np.newaxis]  # Convert to column vector
        # Squared exponential kernel
        return np.exp(-0.5 * (x - x.T) ** 2 / l ** 2)

    covariance_matrix = gaussian_kernel(x, l)

    # Sample from the multivariate normal distribution
    mean = np.zeros(len(x))  # Zero mean
    random_field = multivariate_normal.rvs(mean=mean, cov=covariance_matrix)
    # max_val = np.max(np.abs(random_field))  # Find the maximum absolute value
    # random_field = 0.9 * random_field / max_val
    return x, random_field

def grf_1D_lognormal(lb, ub, num, l, mean, std):
    # Generate the points at which to sample
    x = np.linspace(lb, ub, num)

    # Calculate the parameters for the underlying normal distribution
    mean_normal = np.log(mean ** 2 / np.sqrt(std ** 2 + mean ** 2))
    std_normal = np.sqrt(np.log(1 + (std ** 2 / mean ** 2)))
    l_normal = l * np.log(1 + (std_normal ** 2 / mean_normal ** 2)) / (std_normal ** 2 / mean_normal ** 2)
    print("mean_normal: ", mean_normal)
    print("std_normal: ", std_normal)
    print("l_normal: ", l_normal)

    # Create the covariance matrix based on the exponential decay correlation function
    covariance_matrix = np.zeros((num, num))
    for i in range(num):
        for j in range(num):
            distance = np.abs(x[i] - x[j])
            covariance_matrix[i, j] = (std_normal ** 2) * np.exp(-0.5 * (distance / l_normal) ** 2)

    # Generate the correlated normal random variables
    normal_random_field = np.random.multivariate_normal(mean_normal * np.ones(num), covariance_matrix)

    # Convert the normal distribution to log-normal
    log_normal_random_field = np.exp(normal_random_field)

    return x, log_normal_random_field


def grf_2D_lognormal(node_loc, l, mean, std):
    # Extract the number of nodes
    num_nodes = node_loc.shape[0]

    # Calculate the parameters for the underlying normal distribution
    mean_normal = np.log(mean ** 2 / np.sqrt(std ** 2 + mean ** 2))
    std_normal = np.sqrt(np.log(1 + (std ** 2 / mean ** 2)))
    l_normal = l * np.log(1 + (std_normal ** 2 / mean_normal ** 2)) / (std_normal ** 2 / mean_normal ** 2)

    # Create the covariance matrix based on the exponential decay correlation function
    distance_matrix = np.linalg.norm(node_loc[:, np.newaxis, :] - node_loc[np.newaxis, :, :], axis=2)
    covariance_matrix = (std_normal ** 2) * np.exp(-0.5 * (distance_matrix / l_normal) ** 2)

    # Generate the correlated normal random variables
    normal_random_field = np.random.multivariate_normal(mean_normal * np.ones(num_nodes), covariance_matrix)

    # Convert the normal distribution to log-normal
    log_normal_random_field = np.exp(normal_random_field)

    return log_normal_random_field

def grf_2D_normal(node_loc, l, mean, std):
    # Extract the number of nodes
    if type(node_loc) == list:
        node_loc = np.array(node_loc)
    num_nodes = node_loc.shape[0]

    # Create the covariance matrix based on the exponential decay correlation function
    distance_matrix = np.linalg.norm(node_loc[:, np.newaxis, :] - node_loc[np.newaxis, :, :], axis=2)
    covariance_matrix = std ** 2 * np.exp(-0.5 * (distance_matrix / l) ** 2)

    # Generate the correlated normal random variables
    normal_random_field = np.random.multivariate_normal(mean * np.ones(num_nodes), covariance_matrix)

    return normal_random_field

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
    # max_val = np.max(np.abs(random_field))  # Find the maximum absolute value
    # random_field = 0.9 * random_field / max_val
    return x, random_field

def grf_2D(node, l, std=1):
    # node: a list containing all the nodes (x, y) coordinates
    # l: correlation length
    # return: a list of random field values at each node
    x = [n[0] for n in node]
    y = [n[1] for n in node]
    x = np.array(x)
    y = np.array(y)
    num = len(x)
    # Covariance matrix using the Gaussian kernel
    def gaussian_kernel(x, y, l, std=1):
        """ Generate a Gaussian kernel matrix with correlation length l. """
        x = x[:, np.newaxis]
        y = y[:, np.newaxis]
        # Squared exponential kernel
        return (std ** 2) * np.exp(-0.5 * (x - x.T) ** 2 / l ** 2 - 0.5 * (y - y.T) ** 2 / l ** 2)

    covariance_matrix = gaussian_kernel(x, y, l, std)
    # Sample from the multivariate normal distribution
    mean = np.zeros(num)  # Zero mean
    random_field = multivariate_normal.rvs(mean=mean, cov=covariance_matrix)
    return random_field




if __name__ == '__main__':
    x = np.linspace(0, 1, 100)
    l = 0.3
    x, random_field = grf_1Dv2(x, l)
    random_field = np.exp(random_field)
    random_field = [min(k, 5) for k in random_field]
    random_field = [max(k, 0.4) for k in random_field]
    plt.plot(x, random_field)
    plt.show()


