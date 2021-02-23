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

    test_list = ['HNSC', 'ACC', 'LGG', 'KIPAN', 'UVM', 'CESC', 'BRCA', 'UCEC', 'OV',
                 'DLBC', 'STAD', 'UCS', 'PRAD', 'CHOL', 'PAAD', 'TGCT', 'LUAD', 'STES',
                 'GBMLGG', 'LIHC', 'BLCA', 'KIRC', 'KIRP', 'COAD', 'GBM', 'THCA',
                 'READ', 'PCPG', 'COADREAD', 'LUSC', 'KICH', 'SARC']
    # test_list = [('GBMLGG', 400, 5), ('GBMLGG', 500, 5), ('GBMLGG', 600, 5), ('GBMLGG', 700, 5)]
    # test_list = ['STAD', 'GBMLGG', 'COADREAD']

    # mut_file_path = '/home2/wsdm/gyy/comet_v1/example_datasets/temp/{}_our_pnum={}.m2'
    mut_file_path = '/home2/wsdm/gyy/comet_v1/example_datasets/our/{}_our.m2'


    test_envs = [ising_env.make("SpinSystem",
                                mut_file_path.format(cancer_name),
                                int(n_spins_train * step_fact),
                                minFreq=5,
                                **env_args) for cancer_name in test_list]
    '''
    test_envs = [ising_env.make("SpinSystem",
                                      mut_file_path.format(cancer_name, str(pnum)),
                                      int(n_spins_train*step_fact),
                                      minFreq=minfreq,
                                      **env_args) for cancer_name, pnum, minfreq in test_list]
    '''

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

    network_fn = lambda: MPNN(n_obs_in_g=test_envs[0].observation_space.shape[1] + 1,
                              n_layers=2,
                              n_features=32,
                              n_hid_readout=[],
                              tied_weights=False)

    agent = DQN(test_envs,

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
    # EVAL AGENT
    #############

    agent.load('/home2/wsdm/gyy/eco-dqn_v2/experiments/GBMLGG_100/train/GBMLGG_100/s2v/network/network32000.pth')
    agent.evaluate_agent()



if __name__ == "__main__":
    run()