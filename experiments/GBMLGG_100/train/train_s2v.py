import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

import src.envs.core as ising_env
from experiments.utils import load_graph_set, mk_dir
from src.agents.dqn.dqn import DQN
from src.agents.dqn.utils import TestMetric
from src.envs.utils import (SetGraphGenerator,
                            RandomErdosRenyiGraphGenerator,
                            EdgeType, RewardSignal, ExtraAction,
                            OptimisationTarget, SpinBasis,
                            Observable)
from src.networks.mpnn_bipartite_q import MPNN
from copy import deepcopy

try:
    import seaborn as sns
    plt.style.use('seaborn')
except ImportError:
    pass

import time

def run(save_loc="GBMLGG_100/s2v"):

    print("\n----- Running {} -----\n".format(os.path.basename(__file__)))

    ####################################################
    # SET UP ENVIRONMENTAL AND VARIABLES
    ####################################################

    gamma = 1
    step_fact = 1

    env_args = {'observables':[Observable.SPIN_STATE],
                'reward_signal':RewardSignal.DENSE,
                'extra_action':ExtraAction.NONE,
                'optimisation_target':OptimisationTarget.PVALUE,
                'spin_basis':SpinBasis.BINARY,
                'norm_rewards':False,
                'memory_length':None,
                'horizon_length':None,
                'stag_punishment':None,
                'basin_reward':None,
                'reversible_spins':False}

    ####################################################
    # SET UP TRAINING AND TEST GRAPHS
    ####################################################

    k = 4
    n_spins_train = k

    # train_graph_generator = RandomErdosRenyiGraphGenerator(n_spins=n_spins_train,p_connection=0.15,edge_type=EdgeType.DISCRETE)

    ####
    # Pre-generated test graphs
    ####
    # graph_save_loc = "/home2/wsdm/gyy/eco-dqn_v1/_graphs/testing/ER_200spin_p15_50graphs.pkl"
    # graphs_test = load_graph_set(graph_save_loc)
    # n_tests = len(graphs_test)
    n_tests = 2

    # test_graph_generator = SetGraphGenerator(graphs_test, ordered=True)

    ####################################################
    # SET UP TRAINING AND TEST ENVIRONMENTS
    ####################################################
    train_list = [('COADREAD', 100, 15), ('GBMLGG', 100, 6), ('STAD', 100, 15)]
    test_list = [('LUAD', 100, 20), ('COADREAD', 200, 30), ('GBMLGG', 200, 12)]

    mut_file_path = '/home2/wsdm/gyy/comet_v2/example_datasets/temp/{}_our_pnum={}.m2'

    train_envs = [ising_env.make("SpinSystem",
                                 mut_file_path.format(cancer_name, str(pnum)),
                                 int(n_spins_train*step_fact),
                                 minFreq=minfreq,
                                 **env_args) for cancer_name, pnum, minfreq in train_list]

    test_envs = deepcopy(train_envs) + [ising_env.make("SpinSystem",
                                      mut_file_path.format(cancer_name, str(pnum)),
                                      int(n_spins_train*step_fact),
                                      minFreq=minfreq,
                                      **env_args) for cancer_name, pnum, minfreq in test_list]

    '''
    n_spins_test = train_graph_generator.get().shape[0]
    test_envs = [ising_env.make("SpinSystem",
                                mut_file_path,
                                int(n_spins_test*step_fact),
                                **env_args)]
    '''

    ####################################################
    # SET UP FOLDERS FOR SAVING DATA
    ####################################################

    data_folder = os.path.join(save_loc, 'data')
    network_folder = os.path.join(save_loc, 'network')

    mk_dir(data_folder)
    mk_dir(network_folder)
    # print(data_folder)
    network_save_path = os.path.join(network_folder, 'network.pth')
    test_save_path = os.path.join(network_folder, 'test_scores.pkl')
    loss_save_path = os.path.join(network_folder, 'losses.pkl')

    ####################################################
    # SET UP AGENT
    ####################################################

    nb_steps = 10000000

    network_fn = lambda: MPNN(n_obs_in_g=train_envs[0].observation_space.shape[1] + 1,
                              n_layers=2,
                              n_features=32,
                              n_hid_readout=[],
                              tied_weights=False)

    agent = DQN(train_envs,

                network_fn,

                init_network_params=None,
                init_weight_std=0.5,

                double_dqn=False,
                clip_Q_targets=True,

                replay_start_size=200,
                replay_buffer_size=3200,  # 20000
                gamma=gamma,  # 1
                update_target_frequency=10,  # 500

                update_learning_rate=True,
                initial_learning_rate=1e-2,
                peak_learning_rate=1e-2,
                peak_learning_rate_step=2000,
                final_learning_rate=1e-3,
                final_learning_rate_step=4000,

                update_frequency=4,  # 1
                minibatch_size=64,  # 128
                max_grad_norm=None,
                weight_decay=0,

                update_exploration=True,
                initial_exploration_rate=1,
                final_exploration_rate=0.1,  # 0.05
                final_exploration_step=10000,  # 40000

                adam_epsilon=1e-8,
                logging=False,
                loss="mse",

                save_network_frequency=4000,
                network_save_path=network_save_path,

                evaluate=True,
                test_envs=test_envs,
                test_episodes=n_tests,
                test_frequency=500,  # 10000
                test_save_path=test_save_path,
                test_metric=TestMetric.CUMULATIVE_REWARD,

                seed=None
                )

    print("\n Created DQN agent with network:\n\n", agent.network)

    #############
    # TRAIN AGENT
    #############
    start = time.time()
    agent.learn(timesteps=nb_steps, verbose=True)
    print(time.time() - start)

    agent.save()


    ############
    # PLOT - learning curve
    ############
    data = pickle.load(open(test_save_path,'rb'))
    data = np.array(data)

    fig_fname = os.path.join(network_folder,"training_curve")

    plt.plot(data[:,0],data[:,1])
    plt.xlabel("Training run")
    plt.ylabel("Mean reward")
    if agent.test_metric==TestMetric.ENERGY_ERROR:
      plt.ylabel("Energy Error")
    elif agent.test_metric==TestMetric.BEST_ENERGY:
      plt.ylabel("Best Energy")
    elif agent.test_metric==TestMetric.CUMULATIVE_REWARD:
      plt.ylabel("Cumulative Reward")
    elif agent.test_metric==TestMetric.MAX_CUT:
      plt.ylabel("Max Cut")
    elif agent.test_metric==TestMetric.FINAL_CUT:
      plt.ylabel("Final Cut")

    plt.savefig(fig_fname + ".png", bbox_inches='tight')
    plt.savefig(fig_fname + ".pdf", bbox_inches='tight')

    plt.clf()

    ############
    # PLOT - losses
    ############
    data = pickle.load(open(loss_save_path,'rb'))
    data = np.array(data)

    fig_fname = os.path.join(network_folder,"loss")

    N=50
    data_x = np.convolve(data[:,0], np.ones((N,))/N, mode='valid')
    data_y = np.convolve(data[:,1], np.ones((N,))/N, mode='valid')

    plt.plot(data_x,data_y)
    plt.xlabel("Timestep")
    plt.ylabel("Loss")

    plt.yscale("log")
    plt.grid(True)

    plt.savefig(fig_fname + ".png", bbox_inches='tight')
    plt.savefig(fig_fname + ".pdf", bbox_inches='tight')

if __name__ == "__main__":
    run()