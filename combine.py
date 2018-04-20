import os
import argparse
import joblib
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob

def main():
    parser = argparse.ArgumentParser(
        description="")
    parser.add_argument("--dir", dest="dir",
                        type=str, help='iteration', default=".")
    parser.add_argument("--num_instruments", dest="num_instruments",
                        type=int, help='iteration', default=2)
    parser.add_argument("--n_samples", dest="n_samples",
                        type=int, help='iteration', default=4000)
    parser.add_argument("--num_steps", dest="num_steps",
                        type=int, help='iteration', default=100)
    parser.add_argument("--strength", dest="strength",
                        type=float, help='iteration', default=1.0)
    parser.add_argument("--jitter", dest="jitter",
                        type=int, help='iteration', default=1)
    parser.add_argument("--n_critics", dest="n_critics",
                        type=int, help='iteration', default=50)
    parser.add_argument("--func", dest="func",
                        type=str, default="abs")
    parser.add_argument("--dgp_two", dest="dgp_two",
                        type=int, help='iteration', default=0)
    parser.add_argument("--radius", dest='radius', type=int, default=50)
    opts = parser.parse_args(sys.argv[1:])
    opts.dir = os.path.join(
        opts.dir, 'func_{}_n_insts_{}_n_steps_{}_n_samples_{}_strength_{}_jitter_{}_n_crits_{}_radius_{}_dgp_two_{}'.format(
                        opts.func, opts.num_instruments, opts.num_steps, opts.n_samples,
                        opts.strength, opts.jitter, opts.n_critics, opts.radius, opts.dgp_two))

    results_dist = []
    results_grid = []
    for fname in glob.glob(os.path.join(opts.dir, 'distributional*')):
        results_dist += joblib.load(fname)
    for fname in glob.glob(os.path.join(opts.dir, 'grid*')):
        results_grid += joblib.load(fname)

    results_dist = np.array(results_dist)
    num_nans_dist = np.sum(np.any(np.isnan(results_dist), axis=1))
    results_dist = results_dist[[~np.any(np.isnan(results_dist), axis=1)]]
    results_grid = np.array(results_grid)
    num_nans_grid = np.sum(np.any(np.isnan(results_grid), axis=1))
    results_grid = results_grid[[~np.any(np.isnan(results_grid), axis=1)]]

    plt.figure(figsize=(15, 10))
    plt.violinplot(results_dist,
                    showmeans=False,
                    showmedians=True)
    plt.ylim(None, 1.2)

    plt.xticks([y + 1 for y in range(results_dist.shape[1])],
            ['mse_best', 'mse_final', 'mse_avg', 'mse_deep_iv', 'mse_2sls_poly', 'mse_2sls', 'mse_direct_poly', 'mse_direct_nn'])
    plt.title("Test on Treatment Distribution. R-square: 1- MSE/Var")
    plt.savefig(os.path.join(opts.dir, 'mse_dist.png'))

    plt.figure(figsize=(15, 10))
    plt.violinplot(results_grid,
                    showmeans=False,
                    showmedians=True)
    plt.ylim(None, 1.2)
    plt.xticks([y + 1 for y in range(results_grid.shape[1])],
            ['mse_best', 'mse_final', 'mse_avg', 'mse_deep_iv', 'mse_2sls_poly', 'mse_2sls', 'mse_direct_poly', 'mse_direct_nn'])
    plt.title("Test on Treatment Grid. R-square: 1- MSE/Var")
    plt.savefig(os.path.join(opts.dir, 'mse_grid.png'))

    plt.figure(figsize=(24, 10))
    plt.subplot(1, 2, 1)
    plt.violinplot(results_dist,
                    showmeans=False,
                    showmedians=True)
    plt.ylim(None, 1.2)

    plt.xticks([y + 1 for y in range(results_dist.shape[1])],
            ['best', 'final', 'avg', 'DeepIV', '2SLS_poly', '2SLS', 'direct_poly', 'direct_NN'])
    plt.title("Test on Treatment Distribution. R-square: 1- MSE/Var")
    plt.subplot(1, 2, 2)
    plt.violinplot(results_grid,
                    showmeans=False,
                    showmedians=True)
    plt.ylim(None, 1.2)
    plt.xticks([y + 1 for y in range(results_dist.shape[1])],
            ['best', 'final', 'avg', 'DeepIV', '2SLS_poly', '2SLS', 'direct_poly', 'direct_NN'])
    plt.title("Test on Treatment Grid. R-square: 1- MSE/Var")
    plt.savefig(os.path.join(opts.dir, 'joint.png'))
    print("Grid:{}".format(np.median(results_grid, axis=0)))
    print("Dist:{}".format(np.median(results_dist, axis=0)))
    with open(os.path.join(opts.dir, "summary_results"), 'w') as f:
        f.write("Grid NaNs: {} \n".format(num_nans_grid))
        f.write("Grid median: {} \n".format(np.median(results_grid, axis=0)))
        f.write("Grid 5: {} \n".format(np.percentile(results_grid, 5, axis=0)))
        f.write("Grid 95: {} \n".format(np.percentile(results_grid, 95, axis=0)))
        f.write("Dist NaNs: {} \n".format(num_nans_dist))
        f.write("Dist: {} \n".format(np.median(results_dist, axis=0)))
        f.write("Dist 5: {} \n".format(np.percentile(results_dist, 5, axis=0)))
        f.write("Dist 95: {} \n".format(np.percentile(results_dist, 95, axis=0)))


if __name__ == "__main__":
    main()