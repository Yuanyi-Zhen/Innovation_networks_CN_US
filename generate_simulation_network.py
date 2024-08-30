import networkx as nx
from matplotlib import pyplot as plt
import random
import math


def bridge_preferential_attachment(g, all_node):
    cycle_order = [i for i in range(group_num)]
    deg = [0] * group_num
    vis = [[False] * group_num for i in range(group_num)]
    node_degree = [[0] * single_group_node_num for g in range(group_num)]
    for t in range(group_num):
        for j in range(group_num):
            if t == j:
                vis[t][j] = True
    random.shuffle(cycle_order)
    for i in cycle_order:
        odr = []
        for j in cycle_order:
            if not vis[i][j]: odr.append(j)
        m = len(odr)
        for j in range(m):
            for k in range(j + 1, m):
                if deg[odr[j]] < deg[odr[k]]:
                    tmp = odr[j]
                    odr[j] = odr[k]
                    odr[k] = tmp
        cnt = 0
        for j in range(m):
            vis[i][odr[j]] = True
            vis[odr[j]][i] = True
            deg[i] += 1
            deg[odr[j]] += 1
            send_node_index = preferential_attachment_rule_new(node_degree[i])
            send_node = all_node[i][send_node_index]
            recieve_node_index = preferential_attachment_rule_new(node_degree[odr[j]])
            recieve_node = all_node[odr[j]][recieve_node_index]
            node_degree[i][send_node_index] += 1
            node_degree[odr[j]][send_node_index] += 1
            g.add_edge(send_node, recieve_node)
            cnt += 1
            if cnt == bridge_num: break


def preferential_attachment_rule_new(lst):
    sorted_list = sorted(enumerate(lst), key=lambda x: x[1], reverse=True)
    sorted_index = [index for index, _ in sorted_list]
    result = [intensity, intensity + (1 - intensity) * 0.8, 1]
    random_number = random.random()
    index = next((i for i, x in enumerate(result) if random_number <= x), -1)
    selected_nodes_index = sorted_index[index]
    return selected_nodes_index


def star_bridge_connet(g, core_node):
    star_order = [i for i in range(group_num)]
    deg = [0] * group_num
    vis = [[False] * group_num for i in range(group_num)]
    for t in range(group_num):
        for j in range(group_num):
            if t == j:
                vis[t][j] = True
    random.shuffle(star_order)
    for i in star_order:
        odr = []
        for j in star_order:
            if not vis[i][j]: odr.append(j)
        m = len(odr)
        for j in range(m):
            for k in range(j + 1, m):
                if deg[odr[j]] < deg[odr[k]]:
                    tmp = odr[j]
                    odr[j] = odr[k]
                    odr[k] = tmp
        cnt = 0
        for j in range(m):
            vis[i][odr[j]] = True
            vis[odr[j]][i] = True
            deg[i] += 1
            deg[odr[j]] += 1
            g.add_edge(core_node[i][0], core_node[odr[j]][0])
            cnt += 1
            if cnt == bridge_num: break


def create_random_edges(n, seed=None):
    if seed is not None:
        random.seed(seed)
    nodes = list(range(n))
    edges = []
    connections = {i: 0 for i in nodes}
    unvisited = set(nodes)
    current_node = random.choice(nodes)
    unvisited.remove(current_node)
    while unvisited:
        next_node = random.choice(list(unvisited))
        edges.append((current_node, next_node))
        connections[current_node] += 1
        connections[next_node] += 1
        unvisited.remove(next_node)
        current_node = next_node
    for node in nodes:
        def get_possible_nodes():
            return [n for n in nodes if n != node and connections[n] < 2]
        possible_nodes = get_possible_nodes()
        random.shuffle(possible_nodes)
        for neighbor in possible_nodes:
            if connections[node] < 2 and connections[neighbor] < 2:
                edges.append((node, neighbor))
                connections[node] += 1
                connections[neighbor] += 1
            if connections[node] == 2:
                break
    if all(connections[node] >= 1 for node in nodes):
        return edges
    else:
        print('wrong')


def cycle_network(group_num, n):
    g = nx.Graph()
    all_node = [[] for emp in range(group_num)]
    for j in range(0, group_num):
        for i in range(j * n, (j + 1) * n):
            g.add_node(i)
            all_node[j].append(i)
        for i in range(j * n, (j + 1) * n - 1):
            g.add_edge(i, i + 1)
        g.add_edge((j + 1) * n - 1, j * n)
    bridge_preferential_attachment(g, all_node)
    nx.to_pandas_edgelist(g).to_csv(cycle_save_path, index=False)


def star_network(group_num, n):
    g = nx.Graph()
    core_node = [[] for emp in range(group_num)]
    edge_node = [[] for emp in range(group_num)]
    all_node = [[] for emp in range(group_num)]
    for j in range(0, group_num):
        g.add_node(j * n)
        core_node[j].append(j * n)
        all_node[j].append(j * n)
        for i in range(j * n + 1, (j + 1) * n):
            g.add_node(i)
            g.add_edge(j * n, i)
            edge_node[j].append(i)
            all_node[j].append(i)
    star_bridge_connet(g, core_node)
    nx.to_pandas_edgelist(g).to_csv(star_save_path,
                                    index=False)


def degree_preserving_random_network(group_num, n):
    g = nx.Graph()
    all_nodes = []
    for j in range(group_num):
        edges = create_random_edges(n, seed=42)
        edges = [(pa[0] + j * n, pa[1] + j * n) for pa in edges]
        all_nodes.append(list(set([p for pa in edges for p in pa])))
        g.add_edges_from(edges)
    bridge_preferential_attachment(g, all_nodes)
    nx.to_pandas_edgelist(g).to_csv(random_save_path, index=False)


if __name__ == '__main__':
    group_num = 3
    single_group_node_num = 10
    intensity = 0.5
    bridge_num = 1
    cycle_network(group_num, single_group_node_num)
    intens = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
    # the number of nodes 800
    # bridges = [1, 2, 8, 12, 16]
    # bridges = [8]
    # bridges = [1]
    # #the number of nodes 800---each10nodes--80subgroups
    # bridges = [1, 2, 4, 16, 24, 32]
    # the number of nodes 8000--each10nodes--800subgroups
    bridges = [8, 24, 48, 160, 240, 320]
    # the number of nodes 2400,8000
    # bridges = [4, 12, 24, 80, 120, 160]
    # bridges = [24, 80]
    for intensity in intens:
        print('\nthis is {} preferential'.format(intensity))
        for bridge_num in bridges:
            for times in range(1):
                print('\n-------This is times {}-------'.format(times))
                # group_num = 40
                group_num = 800
                # group_num = 800
                # group_num = 400
                # single_group_node_nums = [20]
                single_group_node_nums = [10]
                for single_group_node_num in single_group_node_nums:
                    save_path = 'data/rewire_random_20240113/'
                    cycle_save_path = save_path + 'cycle_net/{}cycle_each{}nodes_{}preferetial_{}bridges_times{}.csv' \
                        .format(group_num, single_group_node_num, intensity, bridge_num, times)
                    star_save_path = save_path + 'star_net/{}star_each{}nodes_{}bridges_times{}.csv' \
                        .format(group_num, single_group_node_num, bridge_num, times)
                    random_save_path = save_path + 'random_net/{}star_each{}nodes_{}bridges_times{}.csv' \
                        .format(group_num, single_group_node_num, bridge_num, times)
                    print('\n------new network--------')
                    print('group_num is', group_num, '\nsingle_node_num is', single_group_node_num, '\nbridge_num is',
                          bridge_num)
                    print('----cycle network-----')
                    cycle_network(group_num, single_group_node_num)
                    print('----star network-----')
                    star_network(group_num, single_group_node_num)
                    print('----random network-----')
                    degree_preserving_random_network(group_num, single_group_node_num)
