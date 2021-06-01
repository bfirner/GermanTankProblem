import math
from scipy import special
import sys

def computeExpectedMaxSN(samples, max_number = 1000):
    """Compute the expected maximum serial number.

    Returns bayesian median, bayesian average, frequentist predictions.
    """
    k = len(samples)
    m = max(samples)
    results = []
    # Probability of max to max+10 being the total number
    for j in range(m, m+30):
        result = 0
        # Loop through priors for the total number of tanks, n
        max_n = 1000
        for n in range (j, max_n):
            # Explanation:
            # result = result + (j is max | n total, k samples) / probability(n tanks given k observed)
            result = result + special.comb(m-1,k-1) / special.comb(n, k) / (max_n - k)
        #print(f"{j}: {result}")
        results.append(result)

    # Find the median
    total = sum(results)
    roving_sum = 0
    idx = 0
    while roving_sum < total / 2.0:
        roving_sum += results[idx]
        idx += 1
    # The median occurs between idx-1+m and idx+m
    bayes_median = idx + m

    # Now find the mean
    roving_sum = 0
    idx = 0
    while results[idx] > total / len(results):
        roving_sum += results[idx]
        idx += 1
    # The mean occurs between idx-1+m and idx+m
    bayes_mean = idx + m

    # Now find the frequntist value
    # Basically estimate the average gap between samples, (m-k)/k and add that on to the maximum value.
    freq = m + m/k - 1

    return bayes_median, bayes_mean, freq

if __name__ == "__main__":
    samples = [int(i) for i in sys.argv[1:]]
    print(f"samples are {samples}")
    b_median, b_mean, freq = computeExpectedMaxSN(samples)

    print(f"Median occurs between {b_median - 1} and {b_median}")
    print(f"Mean occurs between {b_mean - 1} and {b_mean}")
    print(f"Frequentist estimate is {freq}")
