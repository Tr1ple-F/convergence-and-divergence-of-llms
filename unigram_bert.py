import ipdb
import numpy as np

frequency_count = np.load("frequency_count_bert.npy")
ipdb.set_trace()
unigram_distribution = np.log((frequency_count+1) / np.sum(frequency_count))
uniform_distribution = np.asarray([np.log(1 / len(frequency_count))] * 30522)

np.save("unigram_dist_bert.npy", unigram_distribution)
np.save("uniform_dist_bert.npy", uniform_distribution)
