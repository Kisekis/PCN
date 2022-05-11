import networkx as nx
import matplotlib.pyplot as plt
import os
import json
import random
import numpy as np
import itertools
import csv


# read file
def readFile():
    allNode = []
    path = './nodes/'
    for file_name in [file for file in os.listdir(path)]:
        with open(path + file_name) as json_file:
            data = json.load(json_file)
            allNode.append(data)

    return allNode


# create 2 graphs,directed graph "G" and undirected graph "undirected_G",assign random balance for both sides
# G is used to randomWalk , undirected_G is used to find subgraph with BFS
def createGraph(allNode):
    G = nx.DiGraph()
    undirected_G = nx.Graph()
    for node in allNode:
        for channel in node["channels"]:
            if G.has_edge(node["id"], channel["neightbor"]) or G.has_edge(channel["neightbor"], node["id"]):
                continue
            capacity = float(channel["capacity"][:-4])
            balance1 = random.uniform(0, capacity)
            balance2 = capacity - balance1
            if balance1 > balance2:
                G.add_edge(node["id"], channel["neightbor"], capacity=capacity, balance1=balance1, balance2=balance2)
                undirected_G.add_edge(node["id"], channel["neightbor"], capacity=capacity, balance1=balance1,
                                      balance2=balance2)
            else:
                G.add_edge(channel["neightbor"], node["id"], capacity=capacity, balance1=balance2, balance2=balance1)
                undirected_G.add_edge(channel["neightbor"], node["id"], capacity=capacity, balance1=balance2,
                                      balance2=balance1)
    return G, undirected_G


# bfs from source on undirected_G
def bfs(source, N):
    visited = []
    q = []
    visited.append(source)
    q.append(source)

    while q and len(visited) < N:
        s = q.pop(0)
        count = 0
        for node in undirected_G.neighbors(s):
            count += 1
            if node not in visited and count < N / 10:
                visited.append(node)
                q.append(node)
    return visited


# get induced subgraph of bfs tree nodes
def generateSubgraph(source, N):
    subgraph_nodes = bfs(source, N)
    H = G.subgraph(subgraph_nodes)  # induced subgraph
    return H


# random walk on directed graph G
def randomWalk(H, path, node, dest, L):
    path.append(node)
    if node == dest:
        return True
    if len(path) > L:
        return False
    neighbors = [node for node in H.neighbors(node)]
    if len(neighbors) == 0:
        return False
    next_walk = random.sample(neighbors, 1)[0]
    return randomWalk(H, path, next_walk, dest, L)


# random select an 枯竭的(?) edge on subgraph H
def selectEdge(H, alpha):
    while True:
        edge = random.sample(list(H.edges), 1)[0]
        beta = H[edge[0]][edge[1]]["balance2"] / H[edge[0]][edge[1]]["capacity"]
        if beta < alpha:
            return edge


def test(parameter, edge, H, data):
    L = parameter[0]
    K = parameter[1]
    fail_count = 0
    success_count = 0
    max_cycle_len = 0
    for i in range(0, K):
        s = edge[1]
        t = edge[0]
        path = []
        if randomWalk(H, path, s, t, L):
            success_count += 1
            max_cycle_len = max(max_cycle_len, len(path))
        else:
            fail_count += 1
    rate = success_count / K
    data += [alpha, L, K, rate, max_cycle_len]
    print("success : " + str(success_count))
    print("fail : " + str(fail_count))
    print("rate : " + str(rate))
    print("max_cycle len : " + str(max_cycle_len))


#     edge_colors = ['red' if e == edge else 'black' for e in H.edges]
#     nx.draw_networkx(H, arrows=True,edge_color = edge_colors,with_labels=False,node_size = 10, pos = nx.random_layout(H))
#     print(edge)

allNode = readFile()
G, undirected_G = createGraph(allNode)

# parameters
parameter_list = []
parameter = [
    [10, 20, 30],  # L
    [100, 200]  # K
]
for element in itertools.product(*parameter):
    parameter_list.append(element)

N = [100, 200, 300, 400, 500, 1000]
alpha = 0.5

#
data_list = []
source = random.sample(list(G.nodes), 1)[0]
for n in N:
    H = generateSubgraph(source, n)
    edge = selectEdge(H, alpha)
    for p in parameter_list:
        data = [n]
        test(p, edge, H, data)
        data_list.append(data)
        print(data)
        print("------------------------------------")

# save
with open("out.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(data_list)

# analyze output
arr = np.array(data_list)


# filter all arrays with N = N, L = L, K = K , -1 means ignoring that parameter
def filterArr(arr, N, L, K):
    filter_arr = []
    for element in arr:
        c1 = N == -1 or element[0] == N
        c2 = L == -1 or element[2] == K
        c3 = K == -1 or element[3] == K
        if c1 and c2 and c3:
            filter_arr.append(True)
        else:
            filter_arr.append(False)
    return filter_arr


# plot
figure, axis = plt.subplots(2, 3, figsize=(15, 15))

arr0 = arr[filterArr(arr, 100, -1, 200)]
arr1 = arr[filterArr(arr, 200, -1, 200)]
arr2 = arr[filterArr(arr, 300, -1, 200)]
arr3 = arr[filterArr(arr, 400, -1, 200)]
arr4 = arr[filterArr(arr, 500, -1, 200)]
arr5 = arr[filterArr(arr, 1000, -1, 200)]

axis[0, 0].plot(arr0[:, 2], arr0[:, 4])
axis[0, 0].set_title("N = " + str(arr0[0][0]))

axis[0, 1].plot(arr1[:, 2], arr1[:, 4])
axis[0, 1].set_title("N = " + str(arr1[0][0]))

axis[1, 0].plot(arr2[:, 2], arr2[:, 4])
axis[1, 0].set_title("N = " + str(arr2[0][0]))

axis[1, 1].plot(arr3[:, 2], arr3[:, 4])
axis[1, 1].set_title("N = " + str(arr3[0][0]))

axis[0, 2].plot(arr4[:, 2], arr4[:, 4])
axis[0, 2].set_title("N = " + str(arr4[0][0]))

axis[1, 2].plot(arr5[:, 2], arr5[:, 4])
axis[1, 2].set_title("N = " + str(arr5[0][0]))
plt.show()
