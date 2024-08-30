import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import scipy.stats as stats


def fourty_three_domain():
    domains = pd.read_csv('../../5patent/data/43domains_800-8000nodes.csv')['domain_name'].unique()
    domains = [d[5:] for d in domains]
    print(len(domains))
    consistent = pd.read_csv(data_path)
    consistent = consistent[consistent['concept_name_level'].isin(domains)]
    print(consistent.shape)
    consistent.to_csv('../../5patent/result/43patent_network_domain_info.csv', index=False)


def com_comparison():
    print('\n', types)
    consistent = pd.read_csv(data_path)
    print('original_shape', consistent.shape)
    if types != 'movie':
        level_domain = pd.read_csv(level_domain_path)['concept_name_level'].to_list()
        consistent = consistent[consistent['concept_name_level'].isin(level_domain)]
        print(consistent.shape)

    consistent['cn_largest_percen_all_net'] = consistent['cn_largest_component_node_num'] / consistent['cn_nodes_num']
    consistent['us_largest_percen_all_net'] = consistent['us_largest_component_node_num'] / consistent['us_nodes_num']
    consistent['cn_component_large_30'] = consistent['cn_nodes_num_each_component'].apply(
        lambda x: len([c for c in x[2:-2].split(',')[1:] if int(c) >= 30]))
    consistent['us_component_large_30'] = consistent['us_nodes_num_each_component'].apply(
        lambda x: len([c for c in x[2:-2].split(',')[1:] if int(c) >= 30]))

    consistent['cn_component_large_30/all_nodes_num'] = consistent.apply(
        lambda x: len([c for c in x['cn_nodes_num_each_component'][2:-2].split(',')[1:] if int(c) >= 30]) /
                  x['cn_nodes_num'], axis=1)
    consistent['us_component_large_30/all_nodes_num'] = consistent.apply(
        lambda x: len([c for c in x['us_nodes_num_each_component'][2:-2].split(',')[1:] if int(c) >= 30]) /
                  x['us_nodes_num'], axis=1)

    consistent['cn_compon_large30_nodes_num'] = consistent.apply(
        lambda row: sum([int(c) for c in row['cn_nodes_num_each_component'][2:-2].split(',')[1:] if int(c) >= 30]),
        axis=1)
    consistent['us_compon_large30_nodes_num'] = consistent.apply(
        lambda row: sum([int(c) for c in row['us_nodes_num_each_component'][2:-2].split(',')[1:] if int(c) >= 30]),
        axis=1)
    consistent['cn_compon_large30_percen_all_net'] = consistent.apply(
        lambda row: sum([int(c) for c in row['cn_nodes_num_each_component'][2:-2].split(',')[1:] if int(c) >= 30]) /
                    row['cn_nodes_num'], axis=1)
    consistent['us_compon_large30_percen_all_net'] = consistent.apply(
        lambda row: sum([int(c) for c in row['us_nodes_num_each_component'][2:-2].split(',')[1:] if int(c) >= 30]) /
                    row['us_nodes_num'], axis=1)

    consistent['cn_largest_edge_ratio'] = consistent['cn_largest_component_node_num'] / consistent[
        'cn_compon_large30_nodes_num']
    consistent['us_largest_edge_ratio'] = consistent['us_largest_component_node_num'] / consistent[
        'us_compon_large30_nodes_num']

    us_largest_edge_ratio_filtered = consistent['us_largest_edge_ratio'][
        np.isfinite(consistent['us_largest_edge_ratio'])]
    cn_largest_edge_ratio_filtered = consistent['cn_largest_edge_ratio'][
        np.isfinite(consistent['cn_largest_edge_ratio'])]

    # Calculate mean without infinite values
    us_largest_edge_ratio = np.mean(us_largest_edge_ratio_filtered)
    cn_largest_edge_ratio = np.mean(cn_largest_edge_ratio_filtered)

    # Check if both arrays have non-zero length to prevent division by zero
    if len(us_largest_edge_ratio_filtered) > 0 and len(cn_largest_edge_ratio_filtered) > 0:
        us_cn_largest_edge_ratio = us_largest_edge_ratio / cn_largest_edge_ratio
    else:
        us_cn_largest_edge_ratio = None  # Handle division by zero gracefully, you can adjust this based on your requirements
    us_cn_large30_component_gap = consistent['us_component_large_30'].sum() / consistent['cn_component_large_30'].sum()
    us_cn_large30_component_nodes_gap = consistent['us_compon_large30_percen_all_net'].sum() / consistent[
        'cn_compon_large30_percen_all_net'].sum()
    # print(consistent.head(2))
    # print('us_cn_large30_component_gap', us_cn_large30_component_gap)
    # print('us_cn_large30_component_nodes_gap', us_cn_large30_component_nodes_gap)
    # print('us_largest_edge_ratio', us_largest_edge_ratio)
    # print('cn_largest_edge_ratio', cn_largest_edge_ratio)
    # print('us_cn_largest_edge_ratio', us_cn_largest_edge_ratio)

    cn_largest_edge_ratio = consistent['cn_largest_component_node_num'].sum() / consistent[
        'cn_compon_large30_nodes_num'].sum()
    us_largest_edge_ratio = consistent['us_largest_component_node_num'].sum() / consistent[
        'us_compon_large30_nodes_num'].sum()
    # print('cn_largest_edge_ratio', cn_largest_edge_ratio)
    # print('us_largest_edge_ratio', us_largest_edge_ratio)

    consistent.to_csv(save_path + '{}_{}domains_{}.csv'
                      .format(types, consistent.shape[0], types), index=False)

    col1 = ['concept_name_level', 'us_component_num', 'cn_component_num']
    col2 = ['concept_name_level', 'us_largest_percen_all_net', 'cn_largest_percen_all_net']
    col3 = ['concept_name_level', 'us_component_large_30', 'cn_component_large_30']
    col4 = ['concept_name_level', 'us_compon_large30_percen_all_net', 'cn_compon_large30_percen_all_net']
    col5 = ['concept_name_level', 'us_component_large_30/all_nodes_num', 'cn_component_large_30/all_nodes_num']
    col6 = ['concept_name_level', 'us_component_large_30', 'cn_component_large_30']

    df1 = consistent[col1].rename(columns={'us_component_num': 'us', 'cn_component_num': 'cn'})
    df2 = consistent[col2].rename(columns={'us_largest_percen_all_net': 'us'
        , 'cn_largest_percen_all_net': 'cn'})
    df3 = consistent[col3].rename(columns={'us_component_large_30': 'us', 'cn_component_large_30': 'cn'})
    df4 = consistent[col4].rename(
        columns={'us_compon_large30_percen_all_net': 'us', 'cn_compon_large30_percen_all_net': 'cn'})
    df5 = consistent[col5].rename(
        columns={'us_component_large_30/all_nodes_num': 'us', 'cn_component_large_30/all_nodes_num': 'cn'})
    df6 = consistent[col6].rename(
        columns={'us_component_large_30': 'us', 'cn_component_large_30': 'cn'})

    # plt.figure(figsize=(3, 3))
    # sns.catplot(
    #     data=df1, y=columns[0], x="group",
    #     kind="violin", split=True
    #     , palette=palette_us_cn, alpha=0.6)
    # plt.legend(bbox_to_anchor=(1.25, 1))
    # plt.show()

    # print(df1.shape)
    # # print(df1.head(3))
    # plt.figure(figsize=(4, 5))
    # sns.boxplot(data=df1)
    # plt.title('Network Component Count')
    # plt.show()

    # print(df2.shape)
    # # print(df2.head(3))
    # plt.figure(figsize=(4, 5))
    # sns.boxplot(data=df2)
    # plt.title('Largest Component Node Ratio in Network')
    # plt.show()

    # palette_us_cn = {'us': '#599CB4', 'US': '#599CB4'
    #     , 'CN': '#C25759', 'cn': '#C25759'}
    #
    # plt.figure(figsize=(3, 3))
    # plt.rcParams['font.family'] = 'Gill Sans'
    # plt.rcParams['font.size'] = 12
    #
    # plt.figure(figsize=(3, 3))
    # sns.catplot(
    #     data=df2
    #     # , y=columns[0], x="group"
    #     ,kind="violin", split=True
    #     , palette=palette_us_cn
    #     , alpha=0.5)
    # # plt.legend(bbox_to_anchor=(1.25, 1))
    # plt.ylabel('Percentage')
    # plt.xlabel('Group')
    # # plt.title('Largest Component Node Ratio in Network')
    # plt.tight_layout()
    # # plt.savefig('{}.svg'.format(1), dpi=500)
    # plt.show()

    # print(df3.shape)
    # # print(df3.head(3))
    # plt.figure(figsize=(4, 5))
    # sns.boxplot(data=df3)
    # plt.title('Number of Connected Components \n with over 30 Nodes')
    # plt.show()

    # print(df4.shape)
    # # print(df4.head(3))
    # plt.figure(figsize=(4, 5))
    # sns.boxplot(data=df4)
    # plt.title('Comparing Node Proportions \n in  Connected Components \nwith over 30 Nodes')
    # plt.show()

    # plt.figure(figsize=(3, 3))
    # sns.catplot(
    #     data=df4
    #     # , y=columns[0], x="group"
    #     ,kind="violin", split=True
    #     , palette=palette_us_cn
    #     , alpha=0.5)
    # # plt.legend(bbox_to_anchor=(1.25, 1))
    # # plt.ylabel('Percentage')
    # plt.xlabel('Group')
    # # plt.title('Comparing Node Proportions \n in  Connected Components \nwith over 30 Nodes')
    # plt.tight_layout()
    # plt.savefig('{}.svg'.format(2), dpi=500)
    # plt.show()
    # return df2, df4
    # return df2, df5
    return df2, df6


def figure_combine_svg(df2_sci, df4_sci, df2_tech, df4_tech, df2_mov, df4_mov):
    plt.rcParams['font.family'] = 'Gill Sans'
    plt.rcParams['font.size'] = 15
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    sns.despine()
    labels = ['a', 'b', 'c']
    for i, label in enumerate(labels):
        fig.text(0.02, 0.92 - i * 0.33, label
                 , fontsize=38, fontweight='bold'
                 , ha='center', fontfamily='Gill Sans')
    sns.violinplot(
        data=df2_sci,
        ax=axes[0, 0],
        split=True,
        palette=palette_us_cn,
        # alpha=0.5
    )
    axes[0, 0].set_ylabel('Percentage')
    axes[0, 0].set_title('Largest Component \n Node Ratio')
    sns.violinplot(
        data=df4_sci,
        ax=axes[0, 1],
        split=True,
        palette=palette_us_cn,
        # alpha=0.5
    )
    # axes[0, 1].set_title('Node Proportions in \n Large Connected Components (>=30)')
    # axes[0, 1].set_title('Component Proportions in \n Large Connected Components (>=30)')
    axes[0, 1].set_title('Number of Large \n Connected Components(>=30)')

    sns.violinplot(
        data=df2_tech,
        ax=axes[1, 0],
        split=True,
        palette=palette_us_cn,
        # alpha=0.5
    )
    axes[1, 0].set_ylabel('Percentage')
    sns.violinplot(
        data=df4_tech,
        ax=axes[1, 1],
        split=True,
        palette=palette_us_cn,
        # alpha=0.5
    )

    sns.violinplot(
        data=df2_mov,
        ax=axes[2, 0],
        split=True,
        palette=palette_us_cn,
        # alpha=0.5
    )
    axes[2, 0].set_ylabel('Percentage')
    sns.violinplot(
        data=df4_mov,
        ax=axes[2, 1],
        split=True,
        palette=palette_us_cn,
        # alpha=0.5
    )

    plt.tight_layout()
    plt.savefig('ED2.svg', dpi=500)
    plt.show()


def t_test(df):
    # test Homogeneity of variance
    # print('\n')
    # l = levene(df['us']+df['cn'])
    # # print('The result of the homogeneity of variances test for'+feature+ 'is', l)
    # if l.pvalue > 0.05:
    #     # print('equality of variances')
    #     variance_homo = True
    # else:
    #     # print('heteroscedasticity')
    #     variance_homo = False
    cn_mean_variance = df['cn'].to_list()
    us_mean_variance = df['us'].to_list()
    # t_statistics, p_value = stats.ttest_ind(cn_mean_variance, us_mean_variance, equal_var=variance_homo)
    t_statistics, p_value = stats.ttest_ind(cn_mean_variance, us_mean_variance)
    if p_value <= 0.05 and t_statistics > 0:
        print('cn>us significant', 't is', t_statistics, 'p_value is', p_value)
    if p_value <= 0.05 and t_statistics < 0:
        print('cn<us significant', 't is', t_statistics, 'p_value is', p_value)
    if p_value > 0.05 and t_statistics > 0:
        print('cn>us insignificant', 't is', t_statistics, 'p_value is', p_value)
    if p_value > 0.05 and t_statistics < 0:
        print('cn<us insignificant', 't is', t_statistics, 'p_value is', p_value)


if __name__ == '__main__':
    palette_us_cn = {'us': '#599CB4', 'US': '#599CB4'
        , 'CN': '#C25759', 'cn': '#C25759'}
    # largest_range = 300
    # largest_range = 800
    types = 'science'
    # data_path = '../../4coauthor/result/616domains_scientist.csv'
    # all domain
    data_path = '../../4coauthor/result/6w_network_domain_info.csv'
    level_domain_path = '../../4coauthor/turnover_position/network_rolling_window/1w9_sci_domains_20240711.csv'
    save_path = '../../4coauthor/result/'
    # df2_sci, df4_sci = com_comparison()
    # df2_sci, df5_sci = com_comparison()
    df2_sci, df6_sci = com_comparison()
    t_test(df2_sci)
    # t_test(df4_sci)
    # t_test(df5_sci)
    t_test(df6_sci)

    types = 'patent'
    # all domain
    data_path = '../../5patent/result/5w_patent_network_domain_info.csv'
    # data_path = '../../5patent/result/43patent_network_domain_info.csv'
    save_path = '../../5patent/result/'
    level_domain_path = '../../5patent/180_tech_domain_20240713.csv'
    # fourty_three_domain()
    # df2_tech, df4_tech = com_comparison()
    # df2_tech, df5_tech = com_comparison()
    df2_tech, df6_tech = com_comparison()
    t_test(df2_tech)
    # t_test(df4_tech)
    # t_test(df5_tech)
    t_test(df6_tech)

    types = 'movie'
    # data_path = '../../9movie/result/us_cn_4director_writer_star_network_comp_statistics_process' \
    #             '/4_movie_largest_component_20240520.csv'
    data_path = '../../9movie/result/us_cn_all_director_writer_star_network_comp_statistics_process' \
                '/24_movie_largest_component_20240611.csv'
    save_path = '../../9movie/result/'
    # df2_mov, df4_mov = com_comparison()
    # df2_mov, df5_mov = com_comparison()
    df2_mov, df6_mov = com_comparison()
    t_test(df2_mov)
    # t_test(df4_mov)
    # t_test(df5_mov)
    t_test(df6_mov)
    # figure_combine_svg(df2_sci, df4_sci, df2_tech, df4_tech, df2_mov, df4_mov)
    # figure_combine_svg(df2_sci, df5_sci, df2_tech, df5_tech, df2_mov, df5_mov)
    figure_combine_svg(df2_sci, df6_sci, df2_tech, df6_tech, df2_mov, df6_mov)
