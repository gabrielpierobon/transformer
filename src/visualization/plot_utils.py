import matplotlib.pyplot as plt
import numpy as np

def plot_random_subsequences(subsequences, n, max_length):
    """
    Plots N random subsequences from the array of subsequences.

    Args:
        subsequences: Numpy array with all subsequences
        n: Number of subsequences to plot
        max_length: The maximum length of the subsequences (for x-axis scaling)
    """
    plt.figure(figsize=(15, 8))

    # Choose n random indices without replacement
    random_indices = np.random.choice(subsequences.shape[0], n, replace=False)

    for index in random_indices:
        # Select the subsequence
        subsequence = subsequences[index].flatten()
        # Find the first non-zero value (where the actual data starts)
        start_index = np.where(subsequence != 0)[0][0]
        # Plot the subsequence
        plt.plot(range(start_index, max_length), subsequence[start_index:], label=f'Subsequence {index}')

    plt.xlabel('Time Steps')
    plt.ylabel('Sales (Scaled)')
    plt.title(f'Random {n} Subsequences')
    plt.legend()
    plt.grid(True)
    plt.show() 