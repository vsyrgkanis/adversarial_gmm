
# Adversarial Generalized Method of Moments

Prototype Code for paper: [Adversarial Generalized Method of Moments](https://arxiv.org/abs/1803.07164) by Greg Lewis and Vasilis Syrgkanis

## Library Files

The main file is deep\_gmm.py that contains the main class. The file gmm\_game\_graph.py contains all tensorflow expressions related to the adversarial gmm game. The utils.py file contains some general purpose utilities. The monte\_carlo.py file contains an example of monte carlo simulations of the class and comparisons with other instrumental variable methods. 


## Example Use:
```python
dgmm = DeepGMM(n_critics=n_critics, num_steps=num_steps, store_step=5, learning_rate_modeler=0.007,
               learning_rate_critics=0.007, critics_jitter=jitter, dissimilarity_eta=0.0,
               cluster_type='kmeans', critic_type='Gaussian', critics_precision=None,
               min_cluster_size=radius,  num_trees=5,
               eta_hedge=0.11, bootstrap_hedge=False,
               l1_reg_weight_modeler=0.0, l2_reg_weight_modeler=0.0,
               dnn_layers=hidden_layers, dnn_poly_degree=1,
               log_summary=False, summary_dir='./graphs_monte', display_step=20, random_seed=test_id)
dgmm.fit(data_z, data_p, data_y)
test_min = np.percentile(data_p, 10)
test_max = np.percentile(data_p, 90)
test_grid = np.linspace(test_min, test_max, 100)
best_fn_grid = dgmm.predict(test_grid.reshape(-1, 1), model='best')
final_fn_grid = dgmm.predict(test_grid.reshape(-1, 1), model='final')
avg_fn_grid = dgmm.predict(test_grid.reshape(-1, 1), model='avg')
```

## Parameter explanation:
```python
class DeepGMM:

    def __init__(self, n_critics=50, batch_size_modeler=200, num_steps=30, store_step=10,
                 display_step=20, check_loss_step=10, train_ratio=(1, 1), hedge_step=1,
                 eta_hedge=0.16, loss_clip_hedge=2, bootstrap_hedge=True,
                 learning_rate_modeler=0.01, learning_rate_critics=0.01, critics_jitter=False, critics_precision=None,
                 cluster_type='forest', critic_type='Gaussian', min_cluster_size=50, num_trees=5,
                 l1_reg_weight_modeler=0., l2_reg_weight_modeler=0.,
                 dnn_layers=[1000, 1000, 1000], dnn_poly_degree=1,
                 dissimilarity_eta=0.0, log_summary=True, summary_dir='./graphs',
                 random_seed=None):
        ''' Initialize parameters
        Parameters
        n_critics: number of critic functions
        batch_size_modeler: batch size for modeler gradient step
        num_steps: training steps
        store_step: at how many steps to store the function for calculating avg function
        display_step: at how many steps to print some info
        check_loss_step: at how many steps to check the loss for calculating the best function
        train_ratio: ratio of (modeler, critics) updates
        hedge_step: at how many steps to update the meta-critic with hedge
        eta_hedge: step size of hedge
        loss_clip_hedge: clipping of the moments so that hedge doesn't blow up
        bootstrap_hedge: whether to draw bootstrap subsamples for Hedge update
        learning_rate_modeler: step size for the modeler gradient descent
        learning_rate_critics: step size for the critics gradient descents
        critics_jitter: whether to perform gradient descent on the parameters of the critics
        critics_precision: the radius of the critics in number of samples
        cluster_type: ('forest' | 'kmeans' | 'random_points') which method to use to select the center of the different critics
        critic_type: ('Gaussian' | 'Uniform') whether to put a gaussian or a uniform on the sample points of the cluster
        min_cluster_size: how many points to include in each cluster of points 
        num_trees: only for the forest cluster type, how many trees to build
        l1_reg_weight_modeler: l1 regularization of modeler parameters
        l2_reg_weight_modeler: l2 regularization of modeler parameters
        dnn_layers: (list of int) sizes of fully connected layers
        dnn_poly_degree: how many polynomial features to create as input to the dnn
        dissimilarity_eta: coefficient in front of dissimilarity loss for flexible critics
        log_summary: whether to log the summary using tensorboard
        summary_dir: where to store the summary
        '''

  def fit(self, data_z, data_p, data_y):
    ''' Fits the treatment response model.
    Parameters
    data_z: (n x d np array) of instruments
    data_p: (n x p np array) of treatments
    data_y: (n x 1 np array) of outcomes
    '''
 
  def predict(self, data_p):
    ''' Predicts outcome for each treatment vector.
    Parameters
    data_p: (n x p np array) of treatments
    
    Returns
    y_pred: (n x 1 np array) of counterfacual outcome predictions for each treatment
    '''
```

## Monte Carlo Simulations

```bash
python monte_carlo.py --iteration $number --dir $dir --num_instruments $dimension --n_samples $samples --num_steps $num_steps --func $func --radius $radius --n_critics $n_critics --strength $strength --jitter $jitter --dgp_two $dgp_two
```

Look also in all\_experiments.sh for a shell script that executes many monte carlo experiments and then also combines the results in summary plots.
