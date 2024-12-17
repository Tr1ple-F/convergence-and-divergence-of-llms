import sys

import numpy as np
import pandas as pd

from utils import seeds_config, strip, tagged_tokens, pos_tags

config = seeds_config()
revisions = config['revisions']
models = config['model_names']
seeds = config['seeds']

noun_tags = ['NN', 'NNS', 'NNP', 'NNPS']
verb_tags = ['VB', 'VBG', 'VBN', 'VBD', 'VBZ']

data = []
for model1 in models:
    for revision1 in revisions:
        for seed1 in seeds:
            kl_data = np.load(
                f'../working_dir/{sys.argv[1]}/results/seeds/{model1.replace("/", "-")}-{revision1}-seed{seed1}-kl.npy')
            surprisal_data = np.load(
                f'../working_dir/{sys.argv[1]}/results/seeds/{model1.replace("/", "-")}-{revision1}-seed{seed1}-surprisal.npy')

            pos_kl = []
            pos_surprisal = []
            for i, pos_tag in enumerate(pos_tags()):
                pos_indices = [idx - 1 for idx, tag in enumerate(tagged_tokens()) if tag[1] == pos_tag["tag"]]
                kl_averages = np.mean(kl_data[:, pos_indices], axis=1)
                pos_kl.append(kl_averages)
                surprisal_averages = np.mean(surprisal_data[pos_indices])
                pos_surprisal.append(surprisal_averages)

            # Noun average
            noun_indices = [idx - 1 for idx, tag in enumerate(tagged_tokens()) if tag[1] in noun_tags]
            noun_kl = np.mean(kl_data[:, noun_indices], axis=1)
            noun_surprisal = np.mean(surprisal_data[noun_indices])

            # Verb average
            verb_indices = [idx - 1 for idx, tag in enumerate(tagged_tokens()) if tag[1] in verb_tags]
            verb_kl = np.mean(kl_data[:, verb_indices], axis=1)
            verb_surprisal = np.mean(surprisal_data[verb_indices])

            overall_average = np.mean(kl_data, axis=1)

            i = 0
            for model2 in models:
                for revision2 in revisions:
                    for seed2 in seeds:
                        if model1 == model2 and revision1 == revision2:
                            surprisal_data2 = np.load(f'../working_dir/{sys.argv[1]}/results/seeds/{model2.replace("/", "-")}-{revision2}-seed{seed2}-surprisal.npy')
                            row = {'Model 1': strip(model1), 'Revision 1': strip(revision1), "Seed 1": seed1,
                                   "Model 2": strip(model2), "Revision 2": strip(revision2), "Seed 2": seed2,
                                   'KL Average': overall_average[i], 'Surprisal Average': np.mean(surprisal_data), 'Surprisal Average 2': np.mean(surprisal_data2)}

                            for j, pos_tag in enumerate(pos_tags()):
                                row[f'KL Average - {pos_tag["tag"]}'] = pos_kl[j][i]
                                row[f'Surprisal Average - {pos_tag["tag"]}'] = pos_surprisal[j]

                            row['KL Average - Nouns'] = noun_kl[i]
                            row['Surprisal Average - Nouns'] = noun_surprisal
                            row['KL Average - Verbs'] = verb_kl[i]
                            row['Surprisal Average - Verbs'] = verb_surprisal

                            data.append(row)
                        i += 1

            print("Done with", model1, revision1, seed1)

df = pd.DataFrame(data)
df.to_csv(f'../working_dir/{sys.argv[1]}/output/seeds_pos_current_dataframe.csv')
