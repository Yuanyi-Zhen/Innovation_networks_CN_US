import os
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter


def gini_coef(wealth):
    wealths = [ind[1] / len(wealth) for ind in Counter(wealth).items()]
    cum_wealths = np.cumsum(sorted(np.append(wealths, 0)))
    sum_wealths = cum_wealths[-1]
    xarray = np.array(range(0, len(cum_wealths))) / np.float(len(cum_wealths) - 1)
    yarray = cum_wealths / sum_wealths
    B = np.trapz(yarray, x=xarray)
    A = 0.5 - B
    return A / (A + B)


def node_network_feature(df, file_name):
    print('this is ---', file_name)
    g = nx.from_pandas_edgelist(df, source='source', target='target')
    g.remove_edges_from(nx.selfloop_edges(g))
    largest = max(nx.connected_components(g), key=len)
    g = g.subgraph(largest)
    node_num = g.number_of_nodes()
    edge_num = g.number_of_edges()
    print('node_num is {}, edge_num is {}'.format(node_num, edge_num))
    print('----------node level statistics--------')
    print('degree')
    uwd = dict(nx.degree(g))
    print('degree centrality')
    udc = nx.degree_centrality(g)
    print('betweeness centrality')
    ubc = nx.betweenness_centrality(g)
    print('betweeness')
    ub = nx.betweenness_centrality(g, normalized=False)
    print('closeness centrality')
    ucce = nx.closeness_centrality(g)
    print('eigenvector centrality')
    uec = nx.eigenvector_centrality(g, max_iter=1000)
    print('pagerank')
    upr = nx.pagerank(g)
    print('clustering')
    ucc = nx.clustering(g)
    print('rich_club')
    urc = nx.rich_club_coefficient(g, seed=42, normalized=False)
    print('node distribution result saving----')
    distri_data = [(key, uwd.get(key, None), udc.get(key, None)
                    , ubc.get(key, None)
                    , ub.get(key, None), ucce.get(key, None)
                    , uec.get(key, None)
                    , upr.get(key, None), ucc.get(key, None)
                    , urc.get(key, None))
                   for key in uwd.keys()]
    distri_df = pd.DataFrame(distri_data
                             , columns=['node', 'degree'
            , 'degree centrality', 'betweeness centrality'
            , 'betweeness', 'closeness centrality'
            , 'eigenvector centrality', 'pagerank'
            , 'clustering', 'rich_club'])

    distri_df.to_csv(result_path + 'star_net/node_feature_distribution/nf_{}.csv'.format(file_name)
                     , index=False)

    print('gini-----')
    gini_degree = gini_coef(list(uwd.values()))
    gini_betweeness = gini_coef(list(ubc.values()))
    gini_closeness = gini_coef(list(ucce.values()))
    gini_eigen = gini_coef(list(uec.values()))
    gini_pagerank = gini_coef(list(upr.values()))
    gini_rich_club = gini_coef(list(urc.values()))

    global ud
    ud = nx.density(g)
    ad = np.mean(list(uwd.values()))
    das = nx.degree_assortativity_coefficient(g)
    abc = np.mean(list(ubc.values()))
    acc = np.mean(list(ucce.values()))
    aec = np.mean(list(uec.values()))
    apr = np.mean(list(upr.values()))
    uda = nx.diameter(g)
    upl = nx.average_shortest_path_length(g)
    uge = nx.global_efficiency(g)
    ule = nx.local_efficiency(g)
    cc = nx.average_clustering(g)
    net_res = (file_name, node_num, edge_num
               , gini_degree, gini_betweeness, gini_closeness
               , gini_eigen, gini_pagerank, gini_rich_club
               , ud, ad, das, abc, acc
               , aec, apr, uda, upl
               , uge, ule, cc
               )
    cols = ['network', 'node_num', 'edge_num'
        , 'gini_degree', 'gini_betweeness', 'gini_closeness'
        , 'gini_eigen', 'gini_pagerank', 'gini_rich_club'
        , 'density', 'average degree'
        , 'degree assortativity coefficient'
        , 'average betweenness centrality'
        , 'average closeness centrality'
        , 'average eigenvector centrality'
        , 'average pagerank'
        , 'diameter'
        , 'average shortest path length'
        , 'global efficiency', 'local efficiency'
        , 'average clustering coefficient']

    print('network features saving-----')
    network_result = pd.DataFrame([net_res], columns=cols)
    network_result.to_csv(result_path + 'star_net/whole_network_statistics/net_features_{}.csv'
                          .format(file_name)
                          , index=False)
    return uwd, ubc, ucce, uec, upr, urc


def distribution(lst, feature_name):
    if feature_name == 'degree':
        ddict = {ind[0]: ind[1] for ind in Counter(list(lst.values())).items()}
    else:
        ddict = {ind[0]: ind[1] / len(lst) for ind in Counter(list(lst.values())).items()}
    ddicts = dict(sorted(ddict.items(), key=lambda x: x[0]))
    return ddicts


if __name__ == '__main__':
    data_path = 'data/rewire_random_20240113/'
    result_path = 'result/random_rewire_result20240113/'
    cycle_files = os.listdir(data_path + 'cycle_net/')
    star_files = os.listdir(data_path + 'star_net/')
    star_files = sorted(star_files, key=lambda x: [float(x.split('_')[0][:-4]), float(x.split('_')[1][4:-5]),
                                                   float(x.split('_')[2][:-7])])[9:]
    print(star_files)
    star_cycle_group = []
    for ind, i in enumerate(star_files):
        print('this is the No {}/{} group'
              .format(ind, len(star_files)), i)
        df_star = pd.read_csv(data_path + 'star_net/' + i)
        names = ['degree', 'betweenness_centrality'
            , 'closeness_centrality'
            , 'eigenvector_centrality', 'pagerank'
            , 'rich_club']
        node_network_feature(df_star, i[:-11])