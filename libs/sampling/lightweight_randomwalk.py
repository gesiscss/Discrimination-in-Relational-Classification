"""
Bayesian inference in Friendster graph
Author: Jithin K. Sreedharan (jithin.sreedharan@inria.fr)

usage: python hyp_test_friendster.py

"""
from __future__ import division

import matplotlib
matplotlib.use('Agg')

import networkx as nx
import gzip
import random
import math
import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt
import brewer2mpl
from libs.sampling.network_sample import NetworkSample
from libs.utils.loggerinit import *
import sys



class LightWeightRandomWalk(NetworkSample):

    def __init__(self, G, pseeds):
        super(LightWeightRandomWalk, self).__init__(max(nx.connected_component_subgraphs(G), key=len),pseeds)
        self.plotfn = None
        self.no_runs = None
        self.no_tous = None
        self.size_super_node = None
        self.data_to_plot = None

    def set_parameters(self, plotfn, no_runs=100, no_tours=100, size_super_node=200):
        self.plotfn = plotfn
        self.no_runs = no_runs
        self.no_tous = no_tours
        self.size_super_node = size_super_node

    def set_plotfn(self, plotfn):
        self.plotfn = plotfn

    def compute_sample(self):

        estimation_fn = 1
        no_runs = 100
        no_tours = 100
        no_tours_hist = no_runs * no_tours  # total no. of tours needed for plotting the histogram

        batch = int(math.floor(no_tours ** 0.5))  # No of tours which to be grouped (function values added) for Bayesian approximation
        batch_hist = no_tours  # Batch size in histogram calculation. Essentially batch size here is actual no of tours.
        size_super_node = 200

        G_no_edges = self.G.number_of_edges()
        G_no_nodes = self.G.number_of_nodes()

        print("Begin function estimation from tours")
        super_node = random.sample(self.G.nodes(), size_super_node)  # super-node formation from uniform samples

        # Finding super_node's neighbours outside super node
        nbr_out_sup = []
        sum_temp = 0
        F_org_sup_1 = 0
        i = 0
        for node_s in super_node:
            print("super node ", i)
            i += 1
            list_s_n = [node_s_n for node_s_n in nx.neighbors(self.G, node_s) if node_s_n not in super_node]
            nbr_out_sup = nbr_out_sup + list_s_n
            F_org_sup_1 += sum([self.G.degree(node_s) * self.G.degree(node_s_n) for node_s_n in list_s_n])

        vol_sup_node = len(nbr_out_sup)
        ##---------------------------------------------------
        ## According to the selected function, do the estimation with the defined number of tours
        (S) = self._random_walk_tour_estimate_fn_1(super_node, no_tours_hist, batch_hist, nbr_out_sup)
        (S_1) = self._random_walk_tour_estimate_fn_1(super_node, no_tours, batch, nbr_out_sup)

        print("Finished function estimation from tours")

        H = self.G.subgraph(super_node)

        print("INPUT--- no of tours:", no_tours, "no of runs for plotting histogram:", no_runs)
        print("INPUT--- Super node technique:", estimation_fn, "volume of super node:", vol_sup_node)

        ### To calculate the actual value of function
        F_org = sum([self.G.degree(i)*self.G.degree(j) for (i,j) in self.G.edges()])
        F_org_sup_2 = sum([self.G.degree(i) * self.G.degree(j) for (i, j) in H.edges()])
        F_org_sup = F_org_sup_1 + F_org_sup_2

        print("OUTPUT-- F_original", F_org)
        F_org_except_sup = F_org - F_org_sup

        self.data_to_plot = [S, batch_hist, S_1, batch, vol_sup_node, no_tours, F_org_except_sup]
        self.plot_apprxn(self.data_to_plot[0],self.data_to_plot[1],self.data_to_plot[2],self.data_to_plot[3],self.data_to_plot[4],self.data_to_plot[5],self.data_to_plot[6])

    def _random_walk_tour_estimate_fn_1(self, super_node, no_tours, batch, nbr_out_sup):
        """
        1. Returns estimate from tours
        2. function  fn_1: f(D_t,D_{t+1}) = D_t.D_{t+1}
    
        Parameters
        ----------
        super_node	  - node set  for "super node"
        no_tours	  - number of tours
        batch         - batch size for averaging (see last equation in Theorem 2, it is represented as \sqrt(m)
        """
        R = []
        crawled_edges = set()
        crawled_nodes = set()

        for tour_i in range(1, no_tours + 1):
            print("tour no:", tour_i)
            sample = random.choice(nbr_out_sup);
            crawled_nodes.add(sample)
            T = 1
            R_k = 0
            redo_tour_flag = 0
            while True:
                neighbors = nx.neighbors(self.G, sample)
                sample_new = random.choice(neighbors)
                ##To calculate the crawling percentage ##
                crawled_nodes.add(sample_new)
                if ((sample, sample_new) not in crawled_edges) and ((sample_new, sample) not in crawled_edges):
                    crawled_edges.add((sample, sample_new))
                ##--------------------------------------------
                if sample_new not in super_node:
                    R_k = R_k + self.G.degree(sample) * self.G.degree(sample_new)
                    T = T + 1
                    sample = sample_new
                else:
                    break

            R.append(R_k)
            no_crwld_edge = len(crawled_edges)
        no_crawled_nodes = len(crawled_nodes)
        self.sample = list(crawled_nodes)
        S = [sum(R[i:batch + i]) for i in range(0, len(R), batch)]
        return (S)

    def plot_apprxn(self, S, batch, S_1, batch_1, vol_sup_node, no_tours, F_org_except_sup):

        if self.plotfn is None:
            logging.warning('plotfn is None.')
            return

        bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
        colors = bmap.mpl_colors
        params = {
            'axes.labelsize': 10,
            'font.size': 8,
            'legend.fontsize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'text.usetex': False,
            'figure.figsize': [6, 5]
        }
        plt.rcParams.update(params)
        ax = plt.axes(frameon=0)
        plt.grid()
        ax.set_xlabel('$\hat{\mu}(\mathcal{D}_m(S_n))$')

        S = np.array(S) * (vol_sup_node / (2 * batch))
        plt.hist(S, normed=True, alpha=0.5, color=colors[1], label='Empirical distribution')

        S_1 = np.array(S_1) * (vol_sup_node / (2 * batch_1))
        m_0 = 0;
        mu_0 = 0;
        nu_0 = 0;
        sigma_0 = 1
        n_stt = len(S_1)  # 31 #no_tours # "n" for student-t
        nu_tild = nu_0 + n_stt
        S1_avg = np.mean(S_1)
        S1_sum = sum(S_1)
        mu_tild = (m_0 * mu_0 + n_stt * S1_avg) / (m_0 + n_stt)

        temp_second_term = sum((S_1 - S1_avg) ** 2)
        temp_third_term = (m_0 * n_stt * (S1_avg - mu_0) ** 2) / (m_0 + n_stt)
        sigma_tild = np.sqrt((nu_0 * (sigma_0) ** 2 + temp_second_term + temp_third_term) / ((nu_0 + n_stt) * (m_0 + n_stt)))

        samples_temp = np.linspace(t.ppf(10 ** (-15), nu_tild, loc=mu_tild, scale=sigma_tild), t.ppf((1 - 10 ** (-15)), nu_tild, loc=mu_tild, scale=sigma_tild), 50 ** 4)
        #    samples_temp = np.linspace(4*10**9,8*10**9, 50**4)
        plt.plot(samples_temp, t.pdf(samples_temp, nu_tild, loc=mu_tild, scale=sigma_tild), color=colors[0], linewidth=2, linestyle='-', label='Approximate posterior')
        plt.axvline(x=F_org_except_sup, ymin=0, color="blue", alpha=0.75, label='True value', linewidth=2)

        legend = plt.legend(loc='best')
        frame = legend.get_frame()
        frame.set_facecolor('0.9')
        frame.set_edgecolor('0.75')

        plt.title('Network (Light-Weight Random Walk)')
        plt.savefig(self.plotfn)
        print("Saved the figure for Friendster graph in {}".format(self.plotfn))
        print("Normalized percentage error in true value: {}".format( 100 * (F_org_except_sup - S1_avg) / F_org_except_sup ) )
        plt.close()


    def get_seednodes(self):
        return self.sample[:self.nseeds]



# def random_walk_tour_estimate_fn_1(G, super_node, no_tours, batch, nbr_out_sup):
#     """
#     1. Returns estimate from tours
#     2. function  fn_1: f(D_t,D_{t+1}) = D_t.D_{t+1}
#
#     Parameters
#     ----------
#     G:            - networkx graph
#     super_node	  - node set  for "super node"
#     no_tours	  - number of tours
#     batch         - batch size for averaging (see last equation in Theorem 2, it is represented as \sqrt(m)
#     """
#     R = []
#     crawled_edges = set()
#     crawled_nodes = set()
#
#     for tour_i in range(1, no_tours + 1):
#         print
#         "tour no:", tour_i
#         sample = random.choice(nbr_out_sup);
#         crawled_nodes.add(sample)
#         T = 1
#         R_k = 0
#         redo_tour_flag = 0
#         while True:
#             neighbors = nx.neighbors(G, sample)
#             sample_new = random.choice(neighbors)
#             ##To calculate the crawling percentage ##
#             crawled_nodes.add(sample_new)
#             if ((sample, sample_new) not in crawled_edges) and ((sample_new, sample) not in crawled_edges):
#                 crawled_edges.add((sample, sample_new))
#             ##--------------------------------------------
#             if sample_new not in super_node:
#                 R_k = R_k + G.degree(sample) * G.degree(sample_new)
#                 T = T + 1
#                 sample = sample_new
#             else:
#                 break
#
#         R.append(R_k)
#         no_crwld_edge = len(crawled_edges)
#     no_crawled_nodes = len(crawled_nodes)
#     S = [sum(R[i:batch + i]) for i in range(0, len(R), batch)]
#     return (S)
#
#
# ##-------------------------------------------------------------------------------------------------------
#
# def random_walk_tour_estimate_fn_2(G, super_node, no_tours, batch, nbr_out_sup):
#     """
#     1. Returns estimate from tours
#     2. function  fn_2: f(D_t,D_{t+1}) = 1 if D_t.D_{t+1} >50
#
#     Parameters
#     ----------
#     G:            - networkx graph
#     super_node	  - node set  for "super node"
#     no_tours	  - number of tours
#     batch         - batch size for averaging (see last equation in Theorem 2, it is represented as \sqrt(m)
#     """
#
#     R = []
#     crawled_edges = set()
#     for tour_i in range(1, no_tours + 1):
#         print
#         "tour no:", tour_i
#         sample = random.choice(nbr_out_sup);
#         T = 1
#         R_k = 0
#         redo_tour_flag = 0
#         while True:
#             neighbors = nx.neighbors(G, sample)
#             sample_new = random.choice(neighbors)
#             ## To calculate the crawling percentage ##
#             if ((sample, sample_new) not in crawled_edges) and ((sample_new, sample) not in crawled_edges):
#                 crawled_edges.add((sample, sample_new))
#             ##--------------------------------------------
#             if sample_new not in super_node:
#                 R_k = R_k + int((G.degree(sample) + G.degree(sample_new)) > 50)
#                 T = T + 1
#                 sample = sample_new
#             else:
#                 break
#
#         R.append(R_k)
#         no_crwld_edge = len(crawled_edges)
#     S = [sum(R[i:batch + i]) for i in range(0, len(R), batch)]
#     return (S)
#
#
# ##---------------------------------------------------------------------------------------------------------
#
#
# def plot_apprxn(S, batch, S_1, batch_1, vol_sup_node, no_tours, F_org_except_sup):
#     bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
#     colors = bmap.mpl_colors
#     params = {
#         'axes.labelsize': 10,
#         'text.fontsize': 8,
#         'legend.fontsize': 10,
#         'xtick.labelsize': 10,
#         'ytick.labelsize': 10,
#         'text.usetex': False,
#         'figure.figsize': [6, 5]
#     }
#     plt.rcParams.update(params)
#     ax = plt.axes(frameon=0)
#     plt.grid()
#     ax.set_xlabel('$\hat{\mu}(\mathcal{D}_m(S_n))$')
#
#     S = np.array(S) * (vol_sup_node / (2 * batch))
#     plt.hist(S, normed=True, alpha=0.5, color=colors[1], label='Empirical distribution')
#
#     S_1 = np.array(S_1) * (vol_sup_node / (2 * batch_1))
#     m_0 = 0;
#     mu_0 = 0;
#     nu_0 = 0;
#     sigma_0 = 1
#     n_stt = len(S_1)  # 31 #no_tours # "n" for student-t
#     nu_tild = nu_0 + n_stt
#     S1_avg = np.mean(S_1)
#     S1_sum = sum(S_1)
#     mu_tild = (m_0 * mu_0 + n_stt * S1_avg) / (m_0 + n_stt)
#
#     temp_second_term = sum((S_1 - S1_avg) ** 2)
#     temp_third_term = (m_0 * n_stt * (S1_avg - mu_0) ** 2) / (m_0 + n_stt)
#     sigma_tild = np.sqrt((nu_0 * (sigma_0) ** 2 + temp_second_term + temp_third_term) / ((nu_0 + n_stt) * (m_0 + n_stt)))
#
#     samples_temp = np.linspace(t.ppf(10 ** (-15), nu_tild, loc=mu_tild, scale=sigma_tild), t.ppf((1 - 10 ** (-15)), nu_tild, loc=mu_tild, scale=sigma_tild), 50 ** 4)
#     #    samples_temp = np.linspace(4*10**9,8*10**9, 50**4)
#     plt.plot(samples_temp, t.pdf(samples_temp, nu_tild, loc=mu_tild, scale=sigma_tild), color=colors[0], linewidth=2, linestyle='-', label='Approximate posterior')
#     plt.axvline(x=F_org_except_sup, ymin=0, color="blue", alpha=0.75, label='True value', linewidth=2)
#
#     legend = plt.legend(loc='best')
#     frame = legend.get_frame()
#     frame.set_facecolor('0.9')
#     frame.set_edgecolor('0.75')
#
#     plt.title('Friendster Network')
#     plt.savefig('plot_HypRW.pdf')
#     print
#     "Saved the figure for Friendster graph in 'plot_HypRW.pdf'"
#
#     print
#     "Normalized percentage error in true value: ", 100 * (F_org_except_sup - S1_avg) / F_org_except_sup
#
#
# if __name__ == "__main__":
#     print("Loading graph (Trimmed)...")
#     G = nx.read_edgelist("friendster_community1_trimmed.edgelist", nodetype=int)
#     print("Finished loading graph ")
#     print("**************************************************")
#     estimation_fn = raw_input("Which function is needed?\
# 		        \n \n 1. Estimation function: f(D_t,D_{t+1}) = D_t.D_{t+1} \
# 		        \n \n 2. Estimation function: f(D_t,D_{t+1}) = 1 if D_t + D_{t+1} > 50 \
# 		        \nEnter: ");
#     estimation_fn = int(estimation_fn)
#
#     print
#     "********************************"
#     print
#     "Input the parameters: "
#
#     no_runs = raw_input("No of runs for plotting histogram: ")
#     no_tours = raw_input("No of tours: ")
#
#     no_runs = int(no_runs)
#     no_tours = int(no_tours)  # No. of tours needed for calculating the Bayesian approximation
#     no_tours_hist = no_runs * no_tours  # total no. of tours needed for plotting the histogram
#
#     batch = int(math.floor(no_tours ** 0.5))  # No of tours which to be grouped (function values added) for Bayesian approximation
#     batch_hist = no_tours  # Batch size in histogram calculation. Essentially batch size here is actual no of tours.
#
#     size_super_node = raw_input("Size of super node: ")
#     size_super_node = int(size_super_node)
#
#     G_no_edges = G.number_of_edges()
#     G_no_nodes = G.number_of_nodes()
#
#     print
#     "Begin function estimation from tours"
#
#     super_node = random.sample(G.nodes(), size_super_node)  # super-node formation from uniform samples
#
#     # Finding super_node's neighbours outside super node
#     nbr_out_sup = []
#     sum_temp = 0
#     F_org_sup_1 = 0
#     i = 0
#     for node_s in super_node:
#         print
#         "super node ", i
#         i += 1
#         list_s_n = [node_s_n for node_s_n in nx.neighbors(G, node_s) if node_s_n not in super_node]
#         nbr_out_sup = nbr_out_sup + list_s_n
#         if estimation_fn in [1]:
#             F_org_sup_1 += sum([G.degree(node_s) * G.degree(node_s_n) for node_s_n in list_s_n])
#         if estimation_fn in [2]:
#             F_org_sup_1 += sum([1 for node_s_n in list_s_n if G.degree(node_s) + G.degree(node_s_n) > 50])
#
#     vol_sup_node = len(nbr_out_sup)
#     ##---------------------------------------------------
#
#     ## According to the selected function, do the estimation with the defined number of tours
#     if estimation_fn in [1]:
#         (S) = random_walk_tour_estimate_fn_1(G, super_node, no_tours_hist, batch_hist, nbr_out_sup)
#         (S_1) = random_walk_tour_estimate_fn_1(G, super_node, no_tours, batch, nbr_out_sup)
#
#     if estimation_fn in [2]:
#         (S) = random_walk_tour_estimate_fn_2(G, super_node, no_tours_hist, batch_hist, nbr_out_sup)
#         (S_1) = random_walk_tour_estimate_fn_2(G, super_node, no_tours, batch, nbr_out_sup)
#
#     print
#     "Finished function estimation from tours"
#     print
#     "***************************************"
#
#     H = G.subgraph(super_node)
#
#     print
#     "INPUT--- no of tours:", no_tours, "no of runs for plotting histogram:", no_runs
#     print
#     "INPUT--- Super node technique:", estimation_fn, "volume of super node:", vol_sup_node
#
#     ### To calculate the actual value of function
#     if estimation_fn in [1]:
#         F_org = 5588081356  # sum([G.degree(i)*G.degree(j) for (i,j) in G.edges()])
#         F_org_sup_2 = sum([G.degree(i) * G.degree(j) for (i, j) in H.edges()])
#         F_org_sup = F_org_sup_1 + F_org_sup_2
#
#     if estimation_fn in [2]:
#         F_org = 1164872  # sum([1 for (i,j) in G.edges() if G.degree(i)+G.degree(j)>50])
#         F_org_sup_2 = sum([1 for (i, j) in H.edges() if G.degree(i) + G.degree(j) > 50])
#         F_org_sup = F_org_sup_1 + F_org_sup_2
#
#     print
#     "OUTPUT-- F_original", F_org
#
#     F_org_except_sup = F_org - F_org_sup
#
#     plot_apprxn(S, batch_hist, S_1, batch, vol_sup_node, no_tours, F_org_except_sup)

