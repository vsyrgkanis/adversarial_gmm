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
from utils import plot_3d

def deep_iv_fit(x, z, t, y, epochs=100, hidden=[128, 64, 32]):
    from deepiv.models import Treatment, Response
    import deepiv.architectures as architectures
    import deepiv.densities as densities
    from keras.layers import Input, Dense
    from keras.models import Model
    from keras.layers.merge import Concatenate
    n = z.shape[0]
    dropout_rate = min(1000./(1000. + n), 0.5)
    batch_size = 100
    images = False
    act = "relu"
    n_components = 10
    instruments = Input(shape=(z.shape[1],), name="instruments")
    features = Input(shape=(x.shape[1],), name="features")
    treatment_input = Concatenate(axis=1)([instruments, features])
    est_treat = architectures.feed_forward_net(treatment_input, lambda x: densities.mixture_of_gaussian_output(x, n_components),
                                            hidden_layers=hidden,
                                            dropout_rate=dropout_rate, l2=0.0001,
                                            activations=act)

    treatment_model = Treatment(inputs=[instruments, features], outputs=est_treat)
    treatment_model.compile('adam',
                            loss="mixture_of_gaussians",
                            n_components=n_components)

    treatment_model.fit([z, x], t, epochs=epochs, batch_size=batch_size)

    # Build and fit response model
    treatment = Input(shape=(t.shape[1],), name="treatment")
    response_input = Concatenate(axis=1)([features, treatment])
    est_response = architectures.feed_forward_net(response_input, Dense(1),
                                                activations=act,
                                                hidden_layers=hidden,
                                                l2=0.001,
                                                dropout_rate=dropout_rate)
    response_model = Response(treatment=treatment_model,
                          inputs=[features, treatment],
                          outputs=est_response)
    response_model.compile('adam', loss='mse')
    response_model.fit([z, x], y, epochs=epochs, verbose=1,
                    batch_size=batch_size, samples_per_batch=2)
    
    return response_model

def get_data(n_samples, n_instruments, iv_strength, tau_fn, n_features):
    # Construct dataset
    confounder = np.random.normal(0, 1, size=(n_samples, 1))
    x = np.random.normal(0, 1, size=(n_samples, n_features))
    z = np.random.normal(0, 1, size=(n_samples, n_instruments))
    #p = z[:, 0].reshape(-1, 1) * (z[:, 0]>0).reshape(-1, 1) * iv_strength[0, 0] \
    #    + z[:, 1].reshape(-1, 1) * (z[:, 1]<0).reshape(-1, 1) * iv_strength[1, 0] \
    #    + confounder + \
    #    np.random.normal(0, .1, size=(n_samples, 1))
    p = z[:, 0].reshape(-1, 1) * iv_strength[0, 0] \
        + confounder + \
        np.random.normal(0, .1, size=(n_samples, 1))
    y = tau_fn(x, p) + 2 * confounder + \
        np.random.normal(0, .1, size=(n_samples, 1))
    return x, z, p, y

def test(test_id, dir, strength_scale, n_samples, num_features, num_instruments, num_treatments, num_outcomes):

    def tau_fn(x, p): return (-1.5 * x + .9 * (x**2)) * p #np.abs(p) * x # #np.abs(x) #-1.5 * x + .9 * (x**2)# 2/(1+np.exp(-2*x)) #-1.5 * x + .9 * (x**2) #np.abs(x) #-1.5 * x + .9 * (x**2)  #np.abs(x) #-1.5 * x + .9 * (x**2) #np.sin(x) #1. * (x<0) + 2.5 * (x>=0) #np.abs(x)  # 1. * (x<0) + 3. * (x>=0) #-1.5 * x + .9 * (x**2)  #-1.5 * x + .9 * (x**2) #np.abs(x) #-1.5 * x + .9 * (x**2) + x**3 #-1.5 * x + .9 * (x**2) + x**3 # np.sin(x) #-1.5 * x + .9 * (x**2) + x**3 #np.sin(x) #-1.5 * x + .9 * (x**2) + x**3 #np.sin(x) #np.abs(x) #np.sin(x) #2/(1+np.exp(-2*x)) #2/(1+np.exp(-2*x)) #1.5 * x - .9 * (x**2) #2/(1+np.exp(-2*x))#-1.5 * x + .9 * (x**2)
    iv_strength = strength_scale * np.random.uniform(1., 1.1, size=(num_instruments, 1))
    degree_benchmarks = 3

    # Network parameters
    hidden_layers = [1000, 1000, 1000]

    # Generate data
    data_x, data_z, data_treatment, data_y = get_data(
        n_samples, num_instruments, iv_strength, tau_fn, num_features)
    data_z = np.concatenate((data_z, data_x), axis=1)
    data_p = np.concatenate((data_treatment, data_x), axis=1)
    num_instruments = num_features + num_instruments
    num_treatments = num_features + num_treatments
    print(data_p.shape) 
    print(data_z.shape)
    print(data_y.shape)
    if num_instruments>=2:
        plt.figure()
        plt.subplot(1, 4, 1)
        plt.scatter(data_z[:, 0], data_p[:, 0], label='p vs z1')
        plt.legend()
        plt.subplot(1, 4, 2)
        plt.scatter(data_z[:, 1], data_p[:, 0], label='p vs z2')
        plt.legend()
        plt.subplot(1, 4, 3)
        plt.scatter(data_p[:, 0], data_y)
        plt.legend()
        plt.subplot(1, 4, 4)
        plt.scatter(data_p[:, 1], data_y)
        plt.legend()
        plt.savefig(os.path.join(dir, 'data_{}.png'.format(test_id)))
    
    # We reset the whole graph  
    dgmm = DeepGMM(n_critics=70, num_steps=200, store_step=5, learning_rate_modeler=0.01,
                    learning_rate_critics=0.01, critics_jitter=True, dissimilarity_eta=0.0,
                    cluster_type='kmeans', critic_type='Gaussian', critics_precision=None, min_cluster_size=200, #num_trees=5,
                    eta_hedge=0.16, bootstrap_hedge=False,
                    l1_reg_weight_modeler=0.0, l2_reg_weight_modeler=0.0,
                    dnn_layers=hidden_layers, dnn_poly_degree=1, 
                    log_summary=False, summary_dir='./graphs_monte') 
    dgmm.fit(data_z, data_p, data_y)

    test_min = np.percentile(data_p, 10)
    test_max = np.percentile(data_p, 90)
    test_grid = np.array(list(itertools.product(np.linspace(test_min, test_max, 100), repeat=num_treatments)))
    print(test_grid.shape)

    test_data_x, _, test_data_treatment, _ = get_data(
        5*n_samples, num_instruments, iv_strength, tau_fn, num_features)
    test_data_p = np.concatenate((test_data_treatment, test_data_x), axis=1)
    print(test_data_p.shape)
    clip_edges = (np.all((test_data_p > test_min), axis=1) & np.all((test_data_p < test_max), axis=1)).flatten()
    test_data_p = test_data_p[clip_edges, :]
    test_data_treatment = test_data_treatment[clip_edges, :]
    test_data_x = test_data_x[clip_edges, :]
    print(test_data_p.shape)

    best_fn_grid = dgmm.predict(test_grid, model='best')
    final_fn_grid = dgmm.predict(test_grid, model='final')
    avg_fn_grid = dgmm.predict(test_grid, model='avg')
    best_fn_dist = dgmm.predict(test_data_p, model='best')
    final_fn_dist = dgmm.predict(test_data_p, model='final')
    avg_fn_dist = dgmm.predict(test_data_p, model='avg')


    ##################################
    # Benchmarks
    ##################################
    from sklearn.linear_model import LinearRegression, MultiTaskElasticNet, ElasticNet
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
    from sklearn.neural_network import MLPRegressor
    
    direct_poly = Pipeline([('poly', PolynomialFeatures(degree=degree_benchmarks)), ('linear', LinearRegression())])
    direct_poly.fit(data_p, data_y.flatten())
    direct_poly_fn_grid = direct_poly.predict(test_grid)
    direct_poly_fn_dist = direct_poly.predict(test_data_p)

    direct_nn = MLPRegressor(hidden_layer_sizes=hidden_layers)
    direct_nn.fit(data_p, data_y.flatten())
    direct_nn_fn_grid = direct_nn.predict(test_grid)
    direct_nn_fn_dist = direct_nn.predict(test_data_p)

    plf = PolynomialFeatures(degree=degree_benchmarks)
    sls_poly_first = MultiTaskElasticNet()
    sls_poly_first.fit(plf.fit_transform(data_z), plf.fit_transform(data_p))
    sls_poly_second = ElasticNet()
    sls_poly_second.fit(sls_poly_first.predict(plf.fit_transform(data_z)), data_y)
    sls_poly_fn_grid = sls_poly_second.predict(plf.fit_transform(test_grid))
    sls_poly_fn_dist = sls_poly_second.predict(plf.fit_transform(test_data_p))

    sls_first = LinearRegression()
    sls_first.fit(data_z, data_p)
    sls_second = LinearRegression()
    sls_second.fit(sls_first.predict(data_z), data_y)
    sls_fn_grid = sls_second.predict(test_grid)
    sls_fn_dist = sls_second.predict(test_data_p)

    ###### 
    # Deep IV
    #####
    # We reset the whole graph
    with tf.name_scope("DeepIV"):
        deep_iv = deep_iv_fit(data_x, data_z, data_treatment, data_y, epochs=10, hidden=hidden_layers)
        deep_iv_fn_grid = deep_iv.predict([test_grid[:, 1], test_grid[:, 0]])
        deep_iv_fn_dist = deep_iv.predict([test_data_x, test_data_treatment])


    plt.figure()
    plot_3d(test_grid, tau_fn(test_grid[:, [1]], test_grid[:, [0]]).flatten())
    plt.savefig(os.path.join(dir, 'true_{}.png'.format(test_id)))

    print(avg_fn_grid.shape)
    plt.figure()
    plot_3d(test_grid, avg_fn_grid.flatten())
    plt.savefig(os.path.join(dir, 'avg_fn_{}.png'.format(test_id)))
    
    plt.figure()
    plot_3d(test_grid, best_fn_grid.flatten())
    plt.savefig(os.path.join(dir, 'best_fn_{}.png'.format(test_id)))
    
    plt.figure()
    plot_3d(test_grid, final_fn_grid.flatten())
    plt.savefig(os.path.join(dir, 'final_fn_{}.png'.format(test_id)))

    plt.figure()
    plot_3d(test_grid, deep_iv_fn_grid.flatten())
    plt.savefig(os.path.join(dir, 'deep_iv_{}.png'.format(test_id)))

    plt.figure()
    plot_3d(test_grid, sls_poly_fn_grid.flatten())
    plt.savefig(os.path.join(dir, 'sls_poly_{}.png'.format(test_id)))

    plt.figure()
    plot_3d(test_grid, sls_fn_grid.flatten())
    plt.savefig(os.path.join(dir, 'sls_{}.png'.format(test_id)))

    plt.figure()
    plot_3d(test_grid, direct_poly_fn_grid.flatten())
    plt.savefig(os.path.join(dir, 'direct_poly_{}.png'.format(test_id)))

    plt.figure()
    plot_3d(test_grid, direct_nn_fn_grid.flatten())
    plt.savefig(os.path.join(dir, 'direct_nn_{}.png'.format(test_id)))


    def mse_test(y_true, y_pred):
        return 1 - np.mean((y_pred.flatten() - y_true.flatten())**2) / np.var(y_true.flatten())
    
    mse_best = mse_test(tau_fn(test_data_x, test_data_treatment), best_fn_dist)
    mse_final = mse_test(tau_fn(test_data_x, test_data_treatment), final_fn_dist)
    mse_avg = mse_test(tau_fn(test_data_x, test_data_treatment), avg_fn_dist)
    mse_2sls_poly = mse_test(tau_fn(test_data_x, test_data_treatment), sls_poly_fn_dist)
    mse_direct_poly = mse_test(tau_fn(test_data_x, test_data_treatment), direct_poly_fn_dist)
    mse_direct_nn = mse_test(tau_fn(test_data_x, test_data_treatment), direct_nn_fn_dist)
    mse_2sls = mse_test(tau_fn(test_data_x, test_data_treatment), sls_fn_dist)
    mse_deep_iv = mse_test(tau_fn(test_data_x, test_data_treatment), deep_iv_fn_dist)


    on_p_dist = [mse_best, mse_final, mse_avg, mse_deep_iv, mse_2sls_poly, mse_2sls, mse_direct_poly, mse_direct_nn]
    
    mse_best = mse_test(tau_fn(test_grid[:, [1]], test_grid[:, [0]]), best_fn_grid)
    mse_final = mse_test(tau_fn(test_grid[:, [1]], test_grid[:, [0]]), final_fn_grid)
    mse_avg = mse_test(tau_fn(test_grid[:, [1]], test_grid[:, [0]]), avg_fn_grid)
    mse_2sls_poly = mse_test(tau_fn(test_grid[:, [1]], test_grid[:, [0]]), sls_poly_fn_grid)
    mse_direct_poly = mse_test(tau_fn(test_grid[:, [1]], test_grid[:, [0]]), direct_poly_fn_grid)
    mse_direct_nn = mse_test(tau_fn(test_grid[:, [1]], test_grid[:, [0]]), direct_nn_fn_grid)
    mse_2sls = mse_test(tau_fn(test_grid[:, [1]], test_grid[:, [0]]), sls_fn_grid)
    mse_deep_iv = mse_test(tau_fn(test_grid[:, [1]], test_grid[:, [0]]), deep_iv_fn_grid)

    on_p_grid = [mse_best, mse_final, mse_avg, mse_deep_iv, mse_2sls_poly, mse_2sls, mse_direct_poly, mse_direct_nn]

    return on_p_dist, on_p_grid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="")
    parser.add_argument("--iteration", dest="iteration",
                        type=int, help='iteration', default=0)
    parser.add_argument("--dir", dest="dir",
                        type=str, help='dir', default=".")
    parser.add_argument("--num_instruments", dest="num_instruments",
                        type=int, help='num_instruments', default=1)
    parser.add_argument("--num_features", dest="num_features",
                        type=int, help='num_features', default=1)
    parser.add_argument("--num_treatments", dest="num_treatments",
                        type=int, help='num_treatments', default=1)
    parser.add_argument("--num_outcomes", dest="num_outcomes",
                        type=int, help='num_outcomes', default=1)
    parser.add_argument("--n_samples", dest="n_samples",
                        type=int, help='n_samples', default=4000)      
    parser.add_argument("--strength", dest="strength",
                        type=float, help='iteration', default=1.0)                      
    opts = parser.parse_args(sys.argv[1:])
    if not os.path.exists(opts.dir):
        os.makedirs(opts.dir)
    np.random.seed(opts.iteration)
    print(opts.iteration)

    dist, grid = test(opts.iteration, opts.dir, opts.strength, opts.n_samples, opts.num_features, opts.num_instruments, opts.num_treatments, opts.num_outcomes)
    if os.path.exists(os.path.join(opts.dir, 'distributional')):
        results_dist = joblib.load(os.path.join(opts.dir, 'distributional'))
        results_dist.append(dist)
    else:
        results_dist = [dist]

    if os.path.exists(os.path.join(opts.dir, 'grid')):
        results_grid = joblib.load(os.path.join(opts.dir, 'grid'))
        results_grid.append(grid)
    else:
        results_grid = [grid]
    print(dist)
    print(grid)
    joblib.dump(results_dist, os.path.join(opts.dir, 'distributional'))
    joblib.dump(results_grid, os.path.join(opts.dir, 'grid'))

