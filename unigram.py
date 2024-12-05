import numpy as np

frequency_count = np.load("frequency_count.npy")
unigram_distribution = np.log(frequency_count / np.sum(frequency_count))
uniform_distribution = np.asarray([np.log(1 / len(frequency_count))] * 50277)

np.save("unigram_dist.npy", unigram_distribution)
np.save("uniform_dist.npy", uniform_distribution)
