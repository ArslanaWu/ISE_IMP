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
seed_set = []
repeat_time = 1000


def read_dataset():
    global nodes, edges, seed_set

    with open(network, 'r') as f_net:
        line = f_net.readline()
        graph_info = line.split(" ")
        node_cnt = int(graph_info[0])
        edge_cnt = int(graph_info[1])

        for i in range(0, node_cnt + 1):
            nodes.append(i)
            edges.append({})

        for i in range(0, edge_cnt):
            line = f_net.readline()
            edge_info = line.split(" ")

            adj_list = edges[int(edge_info[0])]
            adj_list[int(edge_info[1])] = float(edge_info[2])

    with open(seed, 'r') as f_seed:
        for line in f_seed.readlines():
            seed_set.append(int(line))


def one_LT_sample():
    activity = [False] * (len(nodes) + 1)
    for seed in seed_set:
        activity[seed] = True

    act_set = seed_set.copy()
    thresh = np.random.uniform(size=len(nodes))
    cnt = len(act_set)
    while not len(act_set) == 0:
        new_act_set = []
        for seed in act_set:
            adj_list = edges[seed]
            for neighbor in adj_list:
                if activity[neighbor]:
                    continue
                # todo:cal weight of activates neighbors
                w_total = 0
                if w_total >= thresh[neighbor]:
                    activity[neighbor] = True
                    new_act_set.append(neighbor)
        cnt += len(new_act_set)
        act_set = new_act_set
    return cnt


def one_IC_sample():
    activity = [False] * (len(nodes) + 1)
    for seed in seed_set:
        activity[seed] = True

    act_set = seed_set.copy()
    cnt = len(act_set)

    while not len(act_set) == 0:
        new_act_set = []
        for seed in act_set:
            adj_list = edges[seed]
            for neighbor in adj_list:
                if activity[neighbor]:
                    continue
                rand = np.random.random()
                if rand < adj_list[neighbor]:
                    activity[neighbor] = True
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

    read_dataset()

    all_sample = 0
    time_out = time.time() + time_limit - 1
    sample_cnt = 0
    if model == 'IC':
        while True:
            if time.time() > time_out:
                break
            if sample_cnt > repeat_time:
                break
            sample_cnt += 1
            one_sample = one_IC_sample()
            # print(one_sample)
            all_sample += one_sample
    else:
        while True:
            if time.time() > time_out:
                break
            if sample_cnt > repeat_time:
                break
            sample_cnt += 1
            one_sample = one_LT_sample()
            all_sample += one_sample

    # print("Result is {}".format(all_sample / repeat_time))
    print(all_sample / sample_cnt)

    sys.stdout.flush()
