import multiprocessing as mp
import time
import sys
import argparse
import os
import numpy as np
import random
import math

core = 8


def cal_total_size(R):
    total = 0
    for rr in R:
        total += sys.getsizeof(rr)
    print('total = {}'.format(total))


def read_dataset(network):
    nodes = []
    edges = []
    inv_edges = []

    with open(network, 'r') as f_net:
        line = f_net.readline()
        graph_info = line.split(" ")
        node_cnt = int(graph_info[0])
        edge_cnt = int(graph_info[1])

        for i in range(0, node_cnt + 1):
            nodes.append(i)
            edges.append({})
            inv_edges.append({})

        for i in range(0, edge_cnt):
            line = f_net.readline()
            start, end, weight = line.split(" ")

            adj_list = edges[int(start)]
            adj_list[int(end)] = float(weight)

            inv_adj_list = inv_edges[int(end)]
            inv_adj_list[int(start)] = float(weight)

    return nodes, edges, inv_edges


def sampling(epsilon, l):
    R = []
    LB = 1
    n = len(nodes)
    new_epsilon = epsilon * math.sqrt(2)

    log_cnk = log_Cnk(n, size)
    log_2 = math.log(2)
    log_n = math.log(n)

    for i in range(1, int(math.log2(n - 1)) + 1):
        x = n / (math.pow(2, i))
        theta_i = ((2 + 2 / 3 * new_epsilon)
                   * (log_cnk + l * log_n + math.log(math.log2(n)))
                   * n) / pow(new_epsilon, 2) / x
        while len(R) <= theta_i:
            R.append(gen_RR())

        s_i, FR = node_selection(R)
        if n * FR >= (1 + new_epsilon) * x:
            LB = n * FR / (1 + new_epsilon)
            break

    alpha = math.sqrt(l * log_n + log_2)
    beta = math.sqrt((1 - 1 / math.e) * (log_cnk + l * log_n + log_2))
    lambda_ = 2 * n * math.pow(((1 - 1 / math.e) * alpha + beta), 2) * math.pow(epsilon, -2)
    theta = lambda_ / LB

    while len(R) <= theta:
        R.append(gen_RR())

    # cal_total_size(R)

    return R


def sampling_bound_with_time():
    R = []
    while time.time() < time_out:
        R.append(gen_RR())

    # cal_total_size(R)

    return R


def log_Cnk(n, k):
    cnk = 0
    for i in range(n - k + 1, n + 1):
        cnk += math.log(i)
    for i in range(1, k + 1):
        cnk -= math.log(i)
    return cnk


def gen_RR():
    node = random.randint(1, len(nodes) - 1)

    if model == 'IC':
        return gen_RR_IC(node)
    elif model == 'LT':
        return gen_RR_LT(node)


def gen_RR_IC(node):
    act_set = [node]
    activated_nodes = [node]

    while act_set:
        new_act_set = []
        for seed in act_set:
            adj_list = inv_edges[seed]
            for neighbor in adj_list:
                if neighbor in activated_nodes:
                    continue
                rand = random.random()
                if rand < adj_list[neighbor]:
                    activated_nodes.append(neighbor)
                    new_act_set.append(neighbor)
            act_set = new_act_set
    return list(set(activated_nodes))


def gen_RR_LT(node):
    act = node
    activated_nodes = [node]
    keep_adding = True

    while keep_adding:
        keep_adding = False

        adj_list = inv_edges[act]

        if len(adj_list) == 0:
            break
        act = random.sample(adj_list.keys(), 1)[0]
        if act not in activated_nodes:
            activated_nodes.append(act)
            keep_adding = True
    return list(set(activated_nodes))


def node_selection(R):
    sk = set()

    occurrence = {}
    occurrence_rr_idx = {}

    # In which rr set the nodes appear
    for rr_i in range(0, len(R)):
        rr = R[rr_i]
        for node in rr:
            if node not in occurrence_rr_idx:
                occurrence_rr_idx[node] = set()
            occurrence_rr_idx[node].add(rr_i)

    # How many times does each node appear in rr sets in total
    for node_i in occurrence_rr_idx:
        occurrence[node_i] = len(occurrence_rr_idx[node_i])

    # Add the node with the most occurrences into sk
    # Then delete its appearances
    total_diff = 0
    for i in range(1, size + 1):
        max_occ = -1
        v = None
        for node_i in occurrence_rr_idx:
            occ = occurrence[node_i]
            if occ > max_occ:
                v = node_i
                max_occ = occ
        sk.add(v)

        total_diff += len(occurrence_rr_idx[v])

        for rr_i in occurrence_rr_idx[v]:
            rr = R[rr_i]
            for node in rr:
                if node == v:
                    continue
                occurrence[node] -= 1
                occurrence_rr_idx[node].remove(rr_i)
        del occurrence[v]
        del occurrence_rr_idx[v]
    return list(set(sk)), total_diff / len(R)


def IMM(epsilon, l):
    n = len(nodes)
    l = l * (1 + math.log(2) / math.log(n))
    R = sampling(epsilon, l)
    # R = sampling_bound_with_time()
    # print("Size of R:{}".format(sys.getsizeof(R)))
    # print("R length = {}".format(len(R)))
    Sk, no_use = node_selection(R)
    return Sk


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-i', '--network', type=str, default='../dataset/in_100000_250000_1.txt')
    parser.add_argument('-i', '--network', type=str, default='../dataset/NetHEPT.txt')
    parser.add_argument('-k', '--size', type=str, default='500')
    parser.add_argument('-m', '--model', type=str, default='IC')
    parser.add_argument('-t', '--time_limit', type=int, default=120)
    args = parser.parse_args()

    time_out = time.time() + int(args.time_limit / 2)
    # time_start = time.time()

    nodes, edges, inv_edges = read_dataset(os.path.abspath(args.network))

    size = int(args.size)
    model = args.model

    seeds = IMM(epsilon=0.1, l=1)

    for seed in seeds:
        print(seed)

    # time_end = time.time()
    # print("time: {}".format(time_end - time_start))
