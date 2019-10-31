import numpy as np
import pandas as pd
import csv
import sys

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

__author__ = 'aGn'


def apply_pca(file_: 'word_vec', ratio_similarity: "ratio_similarity to real data")->'applied_pca':
    df = pd.read_csv(file_, header=None, delimiter=" ", quoting=csv.QUOTE_NONE, encoding='utf-8')
    limited_df = df.iloc[:, 1:]  # Remove first column
    scaled_df = scale(limited_df)
    pca = PCA()
    pca.fit(scaled_df)
    w = pca.components_.T
    y = pca.fit_transform(scaled_df)

    reached_ratio = 0
    for best_indices, ratio_ in enumerate(pca.explained_variance_ratio_):
        reached_ratio += ratio_
        if reached_ratio >= ratio_similarity:
            print(f"Reached ratio percentage is: {reached_ratio:.0%}")
            break

    print(f"Number of PC is first {best_indices + 1} column(s)")
    file_name = 'pca_applied_on_' + file_
    pca_applied = np.zeros((len(df), best_indices + 1))
    pca_applied = pd.DataFrame(pca_applied)
    pca_applied.iloc[:, 1:] = y[:, :best_indices]
    pca_applied.iloc[:, 0] = df.iloc[:, 0]
    np.savetxt(file_name, pca_applied.values, fmt='%s')

    return file_name


def usage():
    return "exec type: python %s word_vec desired_ratio" % sys.argv[0]


if __name__ == '__main__':
    print(apply_pca.__annotations__)
    if len(sys.argv) == 3:
        word_vec = sys.argv[1]
        desired_ratio = sys.argv[2]
    else:
        print(usage())
        sys.exit(1)

    print('Please wait ...')
    applied_pca = apply_pca(word_vec, ratio_similarity=desired_ratio)
    print(f"Your destination file is: {applied_pca}")
    # print('Press any key to continue ...')
    # sys.stdin.read(1)
