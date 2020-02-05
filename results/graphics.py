import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import pandas as pd


def plot_rs_wp(model_prefix, title, metric='hit', version='standard'):
    '''
    Plots graphs of weighted pooling comparison
    '''
    df = pd.read_csv("model_scores_rs_{}.csv".format(version))

    plt.xlabel('K')
    plt.ylabel(metric + '@K')
    plt.title(title)

    for wp in [1, 10, .1, .01]:
        model = model_prefix + str(wp) + ".pt"
        cur_results = df.loc[df['model'] == model]

        x_vals = range(1, 16)
        y_vals = cur_results[metric].tolist()
        plt.plot(x_vals, y_vals, label='λ=' + str(wp), linewidth=1, marker='o', markersize=3)

    plt.axes().yaxis.set_minor_locator(AutoMinorLocator())
    plt.axes().set_xticks([1,3,5,7,9,11,13,15], minor=True)
    plt.legend()
    plt.grid()

    #plt.show()
    plt.savefig(title + '.png')
    plt.close()


def plot_versions(title, metric='hit'):
    plt.xlabel('K')
    plt.ylabel(metric + '@K')
    plt.title(title)
    x_vals = range(1, 16)

    df = pd.read_csv("model_scores_rs_sample.csv")

    model = "rs_all_sample_5_wp_1.pt"
    cur_results = df.loc[df['model'] == model]
    y_vals = cur_results[metric].tolist()
    plt.plot(x_vals, y_vals, label='kprn-r sample', linewidth=1, marker='o', markersize=3)

    model = "rs_all_no_rel_sample_5_wp_1.pt"
    cur_results = df.loc[df['model'] == model]
    y_vals = cur_results[metric].tolist()
    plt.plot(x_vals, y_vals, label='kprn sample', linewidth=1, marker='o', markersize=3)

    df = pd.read_csv("results/model_scores_rs_standard.csv")

    model = "rs_all_wp_1.pt"
    cur_results = df.loc[df['model'] == model]
    y_vals = cur_results[metric].tolist()
    plt.plot(x_vals, y_vals, label='kprn-r', linewidth=1, marker='o', markersize=3)

    model = "rs_all_no_rel_wp_1.pt"
    cur_results = df.loc[df['model'] == model]
    y_vals = cur_results[metric].tolist()
    plt.plot(x_vals, y_vals, label='kprn', linewidth=1, marker='o', markersize=3)


    plt.axes().yaxis.set_minor_locator(AutoMinorLocator())
    plt.axes().set_xticks([1,3,5,7,9,11,13,15], minor=True)
    plt.legend()
    plt.grid()

    #plt.show()
    plt.savefig('title + '.png')
    plt.close()


def main():
    # plot_rs_wp("rs_all_sample_5_wp_", "rs kprn-r hit@K with Path Sampling", "hit", "sample")
    # plot_rs_wp("rs_all_sample_5_wp_", "rs kprn-r ndcg@K with Path Sampling", "ndcg", "sample")
    #
    # plot_rs_wp("rs_all_no_rel_sample_5_wp_", "rs kprn hit@K with Path Sampling", "hit", "sample")
    # plot_rs_wp("rs_all_no_rel_sample_5_wp_", "rs kprn ndcg@K with Path Sampling", "ndcg", "sample")
    #
    # plot_rs_wp("rs_all_wp_", "rs kprn-r hit@K", "hit", "standard")
    # plot_rs_wp("rs_all_wp_", "rs kprn-r ndcg@K", "ndcg", "standard")
    #
    # plot_rs_wp("rs_all_no_rel_wp_", "rs kprn hit@K", "hit", "standard")
    # plot_rs_wp("rs_all_no_rel_wp_", "rs kprn ndcg@K", "ndcg", "standard")
    #
    # plot_versions("rs relation and sampling comparison hit@K", "hit")
    # plot_versions("rs relation and sampling comparison ndcg@K", "ndcg")

    #TODO: dense graphs, baselines, and comparisons


if __name__ == "__main__":
    main()
