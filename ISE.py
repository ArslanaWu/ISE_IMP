# -i ./dataset/network.txt -s ./dataset/network_seeds.txt -m IC -t 60
# -i ./dataset/network.txt -s ./dataset/network_seeds.txt -m LT -t 60
# -i ./dataset/NetHEPT.txt -s ./dataset/network_seeds.txt -m IC -t 60
# -i ./dataset/NetHEPT.txt -s ./dataset/network_seeds.txt -m LT -t 60

import multiprocessing as mp
import time
import sys
import argparse
import os
import numpy as np

nodes = []
edges = []
activity = []  # True: activated
seed_set = []
repeat_time = 1000


def read_dataset():
    global nodes, edges, activity, seed_set

    with open(seed, 'r') as f_seed:
        for line in f_seed.readlines():
            seed_set.append(int(line))
    print("Seed_set is {}".format(seed_set))

    with open(network, 'r') as f_net:
        line = f_net.readline()
        graph_info = line.split(" ")
        node_cnt = int(graph_info[0])
        edge_cnt = int(graph_info[1])

        for i in range(0, node_cnt + 1):
            nodes.append(i)
            edges.append({})
            activity.append(i in seed_set)

        print("Node = {}".format(nodes))

        for i in range(0, edge_cnt):
            line = f_net.readline()
            edge_info = line.split(" ")

            adj_list = edges[int(edge_info[0])]
            adj_list[int(edge_info[1])] = float(edge_info[2])

        print("edges = {}".format(edges))


def one_LT_sample():
    act_set = seed_set.copy()
    # todo:random sample thresholds
    thresh = []
    cnt = len(act_set)
    while not len(act_set) == 0:
        new_act_set = []
        for seed in act_set:
            for neighbor in edges[seed]:
                # todo:cal weight of activates neighbors
                w_total = 0
                if w_total >= thresh[neighbor]:
                    activity[neighbor] = True
                    new_act_set.append(neighbor)
        cnt += len(new_act_set)
        act_set = new_act_set
    return cnt


def one_IC_sample():
    act_set = seed_set.copy()
    cnt = len(act_set)

    while not len(act_set) == 0:
        new_act_set = []
        for seed in act_set:
            for neighbor in edges[seed]:
                # todo:active
                # if activated, activity[neighbor] = True
                if activity[neighbor]:
                    new_act_set.append(neighbor)
        cnt += len(new_act_set)
        act_set = new_act_set
    return cnt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--network', type=str, default='./dataset/network.txt')
    parser.add_argument('-s', '--seed', type=str, default='./dataset/network_seeds.txt')
    parser.add_argument('-m', '--model', type=str, default='IC')
    parser.add_argument('-t', '--time_limit', type=int, default=60)

    args = parser.parse_args()
    network = os.path.abspath(args.network)
    seed = os.path.abspath(args.seed)
    model = args.model
    time_limit = int(args.time_limit)
    print("Input is {}\n{}\n{}\n{}".format(network, seed, model, time_limit))

    read_dataset()

    all_sample = 0
    for i in range(0, repeat_time):
        if model == 'IC':
            one_sample = one_IC_sample()
        else:
            one_sample = one_LT_sample()
        all_sample += one_sample

    print("Result is {}".format(all_sample / repeat_time))

    sys.stdout.flush()
