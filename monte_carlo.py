import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from datetime import datetime
import joblib
import argparse
from deep_gmm import DeepGMM
import itertools
import inspect
from utils import generate_random_pw_linear


def deep_iv_fit(z, t, y, epochs=100, hidden=[128, 64, 32]):
    from deepiv.models import Treatment, Response
    import deepiv.architectures as architectures
    import deepiv.densities as densities
    from keras.layers import Input, Dense
    from keras.models import Model
    n = z.shape[0]
    dropout_rate = min(1000. / (1000. + n), 0.5)
    batch_size = 100
    images = False
    act = "relu"
    n_components = 10
    instruments = Input(shape=(z.shape[1],), name="instruments")
    treatment_input = instruments
    est_treat = architectures.feed_forward_net(treatment_input, lambda x: densities.mixture_of_gaussian_output(x, n_components),
                                               hidden_layers=hidden,
                                               dropout_rate=dropout_rate, l2=0.0001,
                                               activations=act)

    treatment_model = Treatment(inputs=[instruments], outputs=est_treat)
    treatment_model.compile('adam',
                            loss="mixture_of_gaussians",
                            n_components=n_components)

    treatment_model.fit([z], t, epochs=epochs, batch_size=batch_size)

    # Build and fit response model
    treatment = Input(shape=(t.shape[1],), name="treatment")
    response_input = treatment
    est_response = architectures.feed_forward_net(response_input, Dense(1),
                                                  activations=act,
                                                  hidden_layers=hidden,
                                                  l2=0.001,
                                                  dropout_rate=dropout_rate)
    response_model = Response(treatment=treatment_model,
                              inputs=[treatment],
                              outputs=est_response)
    response_model.compile('adam', loss='mse')
    response_model.fit([z], y, epochs=epochs, verbose=1,
                       batch_size=batch_size, samples_per_batch=2)

    return response_model


def get_data(n_samples, n_instruments, iv_strength, tau_fn, dgp_two):
    # Construct dataset
    confounder = np.random.normal(0, 1, size=(n_samples, 1))
    z = np.random.normal(0, 1, size=(n_samples, n_instruments))
    if dgp_two:
        p = 2 * z[:, 0].reshape(-1, 1) * (z[:, 0] > 0).reshape(-1, 1) * iv_strength \
            + 2 * z[:, 1].reshape(-1, 1) * (z[:, 1] < 0).reshape(-1, 1) * iv_strength \
            + 2 * confounder * (1 - iv_strength) + \
            np.random.normal(0, .1, size=(n_samples, 1))
    else:
        p = 2 * z[:, 0].reshape(-1, 1) * iv_strength \
            + 2 * confounder * (1 - iv_strength) + \
            np.random.normal(0, .1, size=(n_samples, 1))
    y = tau_fn(p) + 2 * confounder + \
        np.random.normal(0, .1, size=(n_samples, 1))
    return z, p, y


def test(test_id, dir, strength_scale=.5, n_samples=4000, num_instruments=2, num_treatments=1, num_outcomes=1,
         num_steps=100, jitter=True, n_critics=50, func='abs', radius=50, dgp_two=False):
    print("Parameters: {}".format(locals()))
    with open(os.path.join(dir, "params_{}.txt".format(test_id)), 'w') as f:
        f.write("Parameters: {}".format(locals()))

    np.random.seed(test_id)

    if func=='abs':
        def tau_fn(x): return np.abs(x)
    elif func=='2dpoly':
        def tau_fn(x): return -1.5 * x + .9 * (x**2)
    elif func=='sigmoid':
        def tau_fn(x): return 2/(1+np.exp(-2*x))
    elif func=='sin':
        def tau_fn(x): return np.sin(x)
    elif func=='step':
        def tau_fn(x): return 1. * (x<0) + 2.5 * (x>=0)
    elif func=='3dpoly':
        def tau_fn(x): return -1.5 * x + .9 * (x**2) + x**3
    elif func=='linear':
        def tau_fn(x): return x
    elif func=='rand_pw':
        pw_linear = generate_random_pw_linear()
        def tau_fn(x):             
            return np.reshape(np.array([pw_linear(x_i) for x_i in x.flatten()]), x.shape)

    iv_strength = strength_scale
    degree_benchmarks = 3

    # Network parameters
    hidden_layers = [1000, 1000, 1000]

    # Generate data
    data_z, data_p, data_y = get_data(
        n_samples, num_instruments, iv_strength, tau_fn, dgp_two)
    print(data_p.shape)
    print(data_z.shape)
    print(data_y.shape)
    if num_instruments >= 2:
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.scatter(data_z[:, 0], data_p, label='p vs z1')
        plt.legend()
        plt.subplot(1, 3, 2)
        plt.scatter(data_z[:, 1], data_p, label='p vs z2')
        plt.legend()
        plt.subplot(1, 3, 3)
        plt.scatter(data_p, data_y, label='y vs p')
        plt.legend()
        plt.savefig(os.path.join(dir, 'data_{}.png'.format(test_id)))

    # We reset the whole graph
    dgmm = DeepGMM(n_critics=n_critics, num_steps=num_steps, store_step=5, learning_rate_modeler=0.007,
                   learning_rate_critics=0.007, critics_jitter=jitter, dissimilarity_eta=0.0,
                   cluster_type='kmeans', critic_type='Gaussian', critics_precision=None,
                   min_cluster_size=radius,  # num_trees=5,
                   eta_hedge=0.11, bootstrap_hedge=False,
                   l1_reg_weight_modeler=0.0, l2_reg_weight_modeler=0.0,
                   dnn_layers=hidden_layers, dnn_poly_degree=1,
                   log_summary=False, summary_dir='./graphs_monte', display_step=20, random_seed=test_id)
    inst_inds = np.arange(num_instruments)
    np.random.shuffle(inst_inds)
    dgmm.fit(data_z[:, inst_inds], data_p, data_y)

    test_min = np.percentile(data_p, 10)
    test_max = np.percentile(data_p, 90)
    test_grid = np.array(list(itertools.product(
        np.linspace(test_min, test_max, 100), repeat=num_treatments)))
    print(test_grid.shape)

    _, test_data_p, _ = get_data(
        5 * n_samples, num_instruments, iv_strength, tau_fn, dgp_two)
    print(test_data_p.shape)
    clip_edges = ((test_data_p > test_min) & (
        test_data_p < test_max)).flatten()
    test_data_p = test_data_p[clip_edges, :]

    best_fn_grid = dgmm.predict(test_grid.reshape(-1, 1), model='best')
    final_fn_grid = dgmm.predict(test_grid.reshape(-1, 1), model='final')
    avg_fn_grid = dgmm.predict(test_grid.reshape(-1, 1), model='avg')
    best_fn_dist = dgmm.predict(test_data_p, model='best')
    final_fn_dist = dgmm.predict(test_data_p, model='final')
    avg_fn_dist = dgmm.predict(test_data_p, model='avg')

    ########################
    # Plot alone
    ########################
    plt.figure(figsize=(10, 10))
    plt.plot(test_grid, avg_fn_grid, label='AvgANN y=g(p)')
    plt.plot(test_grid, best_fn_grid, label='BestANN y=g(p)')
    plt.plot(test_grid, final_fn_grid, label='FinalANN y=g(p)')
    plt.plot(test_grid, tau_fn(test_grid), label='true y=g(p)')
    plt.xlabel('Treatment')
    plt.ylabel('Outcome')
    plt.legend()
    plt.savefig(os.path.join(dir, 'deep_gmm_{}.png'.format(test_id)))

    ##################################
    # Benchmarks
    ##################################
    from sklearn.linear_model import LinearRegression, MultiTaskElasticNet, ElasticNet
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
    from sklearn.neural_network import MLPRegressor

    direct_poly = Pipeline([('poly', PolynomialFeatures(
        degree=degree_benchmarks)), ('linear', LinearRegression())])
    direct_poly.fit(data_p, data_y.flatten())
    direct_poly_fn_grid = direct_poly.predict(test_grid.reshape(-1, 1))
    direct_poly_fn_dist = direct_poly.predict(test_data_p)

    direct_nn = MLPRegressor(hidden_layer_sizes=hidden_layers)
    direct_nn.fit(data_p, data_y.flatten())
    direct_nn_fn_grid = direct_nn.predict(test_grid.reshape(-1, 1))
    direct_nn_fn_dist = direct_nn.predict(test_data_p)

    plf = PolynomialFeatures(degree=degree_benchmarks)
    sls_poly_first = MultiTaskElasticNet()
    sls_poly_first.fit(plf.fit_transform(data_z), plf.fit_transform(data_p))
    sls_poly_second = ElasticNet()
    sls_poly_second.fit(sls_poly_first.predict(
        plf.fit_transform(data_z)), data_y)
    sls_poly_fn_grid = sls_poly_second.predict(
        plf.fit_transform(test_grid.reshape(-1, 1)))
    sls_poly_fn_dist = sls_poly_second.predict(plf.fit_transform(test_data_p))

    sls_first = LinearRegression()
    sls_first.fit(data_z, data_p)
    sls_second = LinearRegression()
    sls_second.fit(sls_first.predict(data_z), data_y)
    sls_fn_grid = sls_second.predict(test_grid.reshape(-1, 1))
    sls_fn_dist = sls_second.predict(test_data_p)

    ######
    # Deep IV
    #####
    # We reset the whole graph
    with tf.name_scope("DeepIV"):
        deep_iv = deep_iv_fit(data_z, data_p, data_y,
                              epochs=100, hidden=hidden_layers)
        deep_iv_fn_grid = deep_iv.predict(test_grid.reshape(-1, 1))
        deep_iv_fn_dist = deep_iv.predict(test_data_p)

    plt.figure(figsize=(40, 10))
    plt.subplot(1, 7, 1)
    plt.plot(test_grid, avg_fn_grid, label='AvgANN y=g(p)')
    plt.plot(test_grid, best_fn_grid, label='BestANN y=g(p)')
    plt.plot(test_grid, final_fn_grid, label='FinalANN y=g(p)')
    plt.plot(test_grid, tau_fn(test_grid), label='true y=g(p)')
    plt.xlabel('Treatment')
    plt.ylabel('Outcome')
    plt.legend()
    plt.subplot(1, 7, 2)
    plt.plot(test_grid, deep_iv_fn_grid, label='DeepIV')
    plt.plot(test_grid, tau_fn(test_grid), label='true y=g(p)')
    plt.xlabel('Treatment')
    plt.ylabel('Outcome')
    plt.legend()
    plt.subplot(1, 7, 3)
    plt.plot(test_grid, sls_poly_fn_grid, label='2SLS_poly')
    plt.plot(test_grid, tau_fn(test_grid), label='true y=g(p)')
    plt.xlabel('Treatment')
    plt.ylabel('Outcome')
    plt.legend()
    plt.subplot(1, 7, 4)
    plt.plot(test_grid, sls_fn_grid, label='2SLS')
    plt.plot(test_grid, tau_fn(test_grid), label='true y=g(p)')
    plt.xlabel('Treatment')
    plt.ylabel('Outcome')
    plt.legend()
    plt.subplot(1, 7, 5)
    plt.plot(test_grid, direct_poly_fn_grid, label='Direct poly')
    plt.plot(test_grid, tau_fn(test_grid), label='true y=g(p)')
    plt.xlabel('Treatment')
    plt.ylabel('Outcome')
    plt.legend()
    plt.subplot(1, 7, 6)
    plt.plot(test_grid, direct_nn_fn_grid, label='Direct ANN')
    plt.plot(test_grid, tau_fn(test_grid), label='true y=g(p)')
    plt.xlabel('Treatment')
    plt.ylabel('Outcome')
    plt.legend()
    plt.subplot(1, 7, 7)
    plt.scatter(data_p, data_y, color='blue', label='Data')
    plt.plot(test_grid, tau_fn(test_grid), color='red', label='true y=g(p)')
    plt.xlabel('Treatment')
    plt.ylabel('Outcome')
    plt.legend()
    plt.savefig(os.path.join(dir, 'benchmarks_{}.png'.format(test_id)))

    def mse_test(y_true, y_pred):
        return 1 - np.mean((y_pred.flatten() - y_true.flatten())**2) / np.var(y_true.flatten())

    mse_best = mse_test(tau_fn(test_data_p), best_fn_dist)
    mse_final = mse_test(tau_fn(test_data_p), final_fn_dist)
    mse_avg = mse_test(tau_fn(test_data_p), avg_fn_dist)
    mse_2sls_poly = mse_test(tau_fn(test_data_p), sls_poly_fn_dist)
    mse_direct_poly = mse_test(tau_fn(test_data_p), direct_poly_fn_dist)
    mse_direct_nn = mse_test(tau_fn(test_data_p), direct_nn_fn_dist)
    mse_2sls = mse_test(tau_fn(test_data_p), sls_fn_dist)
    mse_deep_iv = mse_test(tau_fn(test_data_p), deep_iv_fn_dist)

    on_p_dist = [mse_best, mse_final, mse_avg, mse_deep_iv,
                 mse_2sls_poly, mse_2sls, mse_direct_poly, mse_direct_nn]

    mse_best = mse_test(tau_fn(test_grid), best_fn_grid)
    mse_final = mse_test(tau_fn(test_grid), final_fn_grid)
    mse_avg = mse_test(tau_fn(test_grid), avg_fn_grid)
    mse_2sls_poly = mse_test(tau_fn(test_grid), sls_poly_fn_grid)
    mse_direct_poly = mse_test(tau_fn(test_grid), direct_poly_fn_grid)
    mse_direct_nn = mse_test(tau_fn(test_grid), direct_nn_fn_grid)
    mse_2sls = mse_test(tau_fn(test_grid), sls_fn_grid)
    mse_deep_iv = mse_test(tau_fn(test_grid), deep_iv_fn_grid)

    on_p_grid = [mse_best, mse_final, mse_avg, mse_deep_iv,
                 mse_2sls_poly, mse_2sls, mse_direct_poly, mse_direct_nn]

    return on_p_dist, on_p_grid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="")
    parser.add_argument("--iteration", dest="iteration",
                        type=int, help='iteration', default=0)
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
    parser.add_argument("--dgp_two", dest="dgp_two",
                        type=int, help='iteration', default=0)
    parser.add_argument("--n_critics", dest="n_critics",
                        type=int, help='iteration', default=50)
    parser.add_argument("--func", dest="func",
                        type=str, default="abs")
    parser.add_argument("--radius", dest='radius', type=int, default=50)
    opts = parser.parse_args(sys.argv[1:])
    opts.dir = os.path.join(
        opts.dir, 'func_{}_n_insts_{}_n_steps_{}_n_samples_{}_strength_{}_jitter_{}_n_crits_{}_radius_{}_dgp_two_{}'.format(
                        opts.func, opts.num_instruments, opts.num_steps, opts.n_samples,
                        opts.strength, opts.jitter, opts.n_critics, opts.radius, opts.dgp_two))
    if not os.path.exists(opts.dir):
        os.makedirs(opts.dir)
    np.random.seed(opts.iteration)
    print(opts.iteration)

    dist, grid = test(opts.iteration, opts.dir, strength_scale=opts.strength, n_samples=opts.n_samples,
                      num_instruments=opts.num_instruments, num_steps=opts.num_steps, jitter=(
                          opts.jitter == 1),
                      n_critics=opts.n_critics, func=opts.func, radius=opts.radius, dgp_two=(opts.dgp_two==1) )

    print(dist)
    print(grid)
    joblib.dump([dist], os.path.join(opts.dir, "distributional_{}".format(opts.iteration)))
    joblib.dump([grid], os.path.join(opts.dir, "grid_{}".format(opts.iteration)))
