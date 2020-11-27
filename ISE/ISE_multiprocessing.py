# -i ./dataset/network.txt -s ./dataset/network_seeds.txt -m IC -t 5
# -i ./dataset/network.txt -s ./dataset/network_seeds.txt -m LT -t 5
# -i ./dataset/NetHEPT.txt -s ./dataset/network_seeds.txt -m IC -t 5
# -i ./dataset/NetHEPT.txt -s ./dataset/network_seeds.txt -m LT -t 5

import multiprocessing as mp
import time
import sys
import argparse
import os
import numpy as np
import random

core = 8


def read_dataset(network, seed):
    nodes = []
    edges = []
    inv_edges = []
    seed_set = []

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
            edge_info = line.split(" ")

            adj_list = edges[int(edge_info[0])]
            adj_list[int(edge_info[1])] = float(edge_info[2])

            inv_adj_list = inv_edges[int(edge_info[1])]
            inv_adj_list[int(edge_info[0])] = float(edge_info[2])

    with open(seed, 'r') as f_seed:
        for line in f_seed.readlines():
            seed_set.append(int(line))

    return nodes, edges, inv_edges, seed_set


def LT(nodes, edges, inv_edges, seed_set, time_out):
    sample_sum = 0
    sample_num = 0

    while True:
        if time.time() > time_out:
            break

        np.random.seed(random.randint(0, 100000))

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

                    w_total = 0
                    sub_adj_list = inv_edges[neighbor]
                    for sub_neighbor in sub_adj_list:
                        if activity[sub_neighbor]:
                            w_total += sub_adj_list[sub_neighbor]

                    if w_total >= thresh[neighbor]:
                        activity[neighbor] = True
                        new_act_set.append(neighbor)
            cnt += len(new_act_set)
            act_set = new_act_set

        sample_sum += cnt
        sample_num += 1
    return sample_sum, sample_num


def IC(nodes, edges, seed_set, time_out):
    sample_sum = 0
    sample_num = 0

    while True:
        if time.time() > time_out:
            break

        np.random.seed(random.randint(0, 100000))

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

        sample_sum += cnt
        sample_num += 1
    return sample_sum, sample_num


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-i', '--network', type=str, default='../dataset/in_100000_250000_1.txt')
    parser.add_argument('-i', '--network', type=str, default='../dataset/NetHEPT.txt')
    parser.add_argument('-s', '--seed', type=str, default="C:\\Users\\dbg\\Desktop\\seed_new.txt")
    parser.add_argument('-m', '--model', type=str, default='IC')
    parser.add_argument('-t', '--time_limit', type=int, default=10)
    args = parser.parse_args()

    time_out = time.time() + int(args.time_limit) - 5

    nodes, edges, inv_edges, seed_set = read_dataset(
        os.path.abspath(args.network),
        os.path.abspath(args.seed)
    )

    pool = mp.Pool(processes=core)
    sample_list = []
    model = args.model
    if model == 'IC':
        for i in range(core):
            sample_list.append(pool.apply_async(
                func=IC,
                args=(nodes, edges, seed_set, time_out))
            )
    else:
        for i in range(core):
            sample_list.append(pool.apply_async(
                func=LT,
                args=(nodes, edges, inv_edges, seed_set, time_out))
            )
    pool.close()
    pool.join()

    sample_sum = 0
    sample_num = 0
    for sample in sample_list:
        sample_sum += sample.get()[0]
        sample_num += sample.get()[1]

    # print('Run time: {}'.format(sample_num))
    print(sample_sum / sample_num)

    sys.stdout.flush()
