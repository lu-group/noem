import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def grf(lb, up, num, l):
    x = np.linspace(lb, up, num)  

    def gaussian_kernel(x, l):
        """ Generate a Gaussian kernel matrix with correlation length l. """
        x = x[:, np.newaxis]  
        return np.exp(-0.5 * (x - x.T) ** 2 / l ** 2)

    covariance_matrix = gaussian_kernel(x, l)

    mean = np.zeros(num)  # Zero mean
    random_field = multivariate_normal.rvs(mean=mean, cov=covariance_matrix)
    return x, random_field

if __name__ == '__main__':
    n_points = 100  
    correlation_length = 0.5
    correlation_length = 0.5
    x, random_field = grf(0, 1, num=100, l=correlation_length)
    plt.figure(figsize=(10, 5))
    plt.plot(x, random_field, label='Gaussian Random Field')
    plt.title(f'1D Gaussian Random Field with Correlation Length l = {correlation_length}')
    plt.xlabel('x')
    plt.ylabel('Field Value')
    plt.legend()
    plt.show()
