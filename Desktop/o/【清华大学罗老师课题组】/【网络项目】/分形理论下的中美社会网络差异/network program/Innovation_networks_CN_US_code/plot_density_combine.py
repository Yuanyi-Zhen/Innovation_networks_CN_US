import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def dataset(types, kind):
    if types == 'simulation_whole_network_feature':
        df = pd.read_csv('../data/ideal_type_800_8000_three_types_network_clean_20240115.csv') \
            .rename(columns={'path length': 'Path length'
            , 'mean closeness centrality': 'Mean closeness centrality'
            , 'gini_degree': 'Gini degree'
            , 'global efficiency': 'Global efficiency'})
    if types == 'empirical_whole_network_feature':
        if kind == 'author':
            df = pd.read_csv('../data2/author_variable_20240125.csv') \
                .rename(columns={'average shortest path length': 'Path length'
                , 'average closeness centrality': 'Mean closeness centrality'
                , 'gini_degree': 'Gini degree'
                , 'global efficiency': 'Global efficiency'
                                 })
        if kind == 'patent':
            df = pd.read_csv('../data2/patent_variable_20240125.csv') \
                .rename(columns={'average shortest path length': 'Path length'
                , 'average closeness centrality': 'Mean closeness centrality'
                , 'global efficiency': 'Global efficiency'
                , 'gini_degree': 'Gini degree'
                                 })
        if kind == 'movie':
            df = pd.read_csv('../data2/movie_4types/cn_us_director_writer_star_800-8000_restrict_1.csv') \
                .rename(columns={'average shortest path length': 'Path length'
                , 'average closeness centrality': 'Mean closeness centrality'
                , 'global efficiency': 'Global efficiency'
                , 'gini_degree': 'Gini degree'
                                 })

    if types == 'cultural_mobility_domain' or types == 'social_mobility':
        if kind == 'author':
            if types == 'cultural_mobility_domain':
                df = pd.read_csv('/project/jevans/yuanyi/fractal_network_task/!paper_figures/!visualization/'
                                 'science_cultural_mobility_domain_20240727.csv')
            elif types == 'social_mobility':
                df = pd.read_csv('/project/jevans/yuanyi/fractal_network_task/!paper_figures/!visualization'
                                 '/science_social_mobility_20240727.csv')
        if kind == 'patent':
            if types == 'cultural_mobility_domain':
                df = pd.read_csv('/project/jevans/yuanyi/fractal_network_task/!paper_figures/!visualization'
                                 '/technology_cultural_mobility_domain_20240727.csv')
            elif types == 'social_mobility':
                df = pd.read_csv('/project/jevans/yuanyi/fractal_network_task/!paper_figures/!visualization'
                                 '/technology_social_mobility_20240727.csv')
        if kind == 'movie':
            if types == 'cultural_mobility_domain':
                df = pd.read_csv('/project/jevans/yuanyi/fractal_network_task/!paper_figures/!visualization'
                                 '/art_cultural_mobility_domain_20240727.csv')
            elif types == 'social_mobility':
                df = pd.read_csv('/project/jevans/yuanyi/fractal_network_task/!paper_figures/!visualization'
                                 '/art_social_mobility_20240727.csv')

    if types == 'cultural_mobility_person_across_time':
        if kind == 'author':
            df = pd.read_csv('/project/jevans/yuanyi/fractal_network_task/!paper_figures/!visualization/'
                             'science_cultural_mobility_person_20240727.csv')
        if kind == 'patent':
            df = pd.read_csv('/project/jevans/yuanyi/fractal_network_task/!paper_figures/!visualization/'
                             'technology_cultural_mobility_person_20240727.csv')
        if kind == 'movie':
            df = pd.read_csv('/project/jevans/yuanyi/fractal_network_task/!paper_figures/!visualization/'
                             'art_cultural_mobility_person_20240727.csv')

    return df


def kdeplot(df, columns):
    plt.rcParams['font.family'] = 'Gill Sans'
    plt.rcParams['font.size'] = 12
    if types == 'simulation_whole_network_feature':
        palette = palette_simu
    else:
        palette = palette_us_cn
    for column in columns:
        if types == 'simulation_whole_network_feature':
            plt.figure(figsize=(3.5, 3.5))
        else:
            plt.figure(figsize=(3.5, 3.5))

        sns.kdeplot(
            data=df, x=column, hue="group",
            fill=True
            , common_norm=False
            , palette=palette
            # , alpha=0.6
            , linewidth=1.4
            , edgecolor='black'
            , legend=True
            # , legend=False
        )
        sns.despine()
        plt.xlabel('Degree Centrality Rank \n Average Correlation')
        plt.subplots_adjust(bottom=0.5)
        plt.ylabel('Density', fontname='Gill Sans', fontsize=12)
        # plt.ylabel('')
        plt.tight_layout()
        # plt.figtext(0.6, 0.01, column, ha='center', fontname='Gill Sans', fontsize=12)
        plt.savefig('{}.svg'.format(column), dpi=500)
        plt.show()


def vilon_single(df, columns):
    plt.figure(figsize=(3, 3))
    plt.rcParams['font.family'] = 'Gill Sans'
    plt.rcParams['font.size'] = 12
    sns.catplot(
        data=df, y=columns[0], x="group",
        kind="violin", split=True
        , palette=palette_us_cn
        # , alpha=0.9
    )
    plt.ylabel('')
    plt.show()


def figure_4_cultural_mobility_domain_combine():
    types = 'cultural_mobility_domain'
    df_sci = dataset(types, kind='author')
    df_tech = dataset(types, kind='patent')
    df_mov = dataset(types, kind='movie')
    columns = ['Breadth']

    plt.rcParams['font.family'] = 'Gill Sans'
    plt.rcParams['font.size'] = 15

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    sns.despine()
    sns.violinplot(data=df_sci, y=columns[0], x="group",
                   split=True, palette=palette_us_cn
                   # , alpha=0.5
                   , ax=ax[0])
    ax[0].set_xlabel("")
    sns.violinplot(data=df_tech, y=columns[0], x="group",
                   split=True, palette=palette_us_cn
                   # , alpha=0.5
                   , ax=ax[1])
    ax[1].set_xlabel("")
    sns.violinplot(data=df_mov, y=columns[0], x="group",
                   split=True, palette=palette_us_cn
                   # , alpha=0.5
                   , ax=ax[2])
    ax[2].set_xlabel("")
    labels = ['a', 'b', 'c']
    for i, label in enumerate(labels):
        ax[i].text(0.5, -0.1, label, transform=ax[i].transAxes,
                   fontsize=35, fontweight='bold', va='top'
                   , ha='center', fontfamily='Gill Sans')
    plt.tight_layout()
    plt.savefig('culture_mobility_domain.svg', dpi=500)
    plt.show()


def cultural_mobility_domain_combine_robustness():
    types = 'cultural_mobility_domain'
    df_sci = dataset(types, kind='author')
    df_tech = dataset(types, kind='patent')
    df_mov = dataset(types, kind='movie')
    columns = ['mean_variance_to_centroid']

    plt.rcParams['font.family'] = 'Gill Sans'
    plt.rcParams['font.size'] = 15

    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    sns.despine()
    sns.violinplot(data=df_sci, y=columns[0], x="group",
                   split=True, palette=palette_us_cn
                   # , alpha=0.5
                   , ax=ax[0])
    ax[0].set_xlabel("")
    sns.violinplot(data=df_tech, y=columns[0], x="group",
                   split=True, palette=palette_us_cn
                   # , alpha=0.5
                   , ax=ax[1])
    ax[1].set_xlabel("")
    ax[1].set_ylabel("")
    sns.violinplot(data=df_mov, y=columns[0], x="group",
                   split=True, palette=palette_us_cn
                   # , alpha=0.5
                   , ax=ax[2])
    ax[2].set_xlabel("")
    ax[2].set_ylabel("")
    labels = ['a', 'b', 'c']
    for i, label in enumerate(labels):
        ax[i].text(0.5, -0.1, label, transform=ax[i].transAxes,
                   fontsize=35, fontweight='bold', va='top'
                   , ha='center', fontfamily='Gill Sans')
    plt.tight_layout()
    plt.savefig('culture_mobility_domain.svg', dpi=500)
    plt.show()


def person_social_mobility_robustness():
    t1 = 'social_mobility'
    x_label = 'Betweenness Centrality Rank Average Correlation'
    social_sci = dataset(t1, kind='author')
    tech_sci = dataset(t1, kind='patent')
    mov_sci = dataset(t1, kind='movie')
    column = ['degree_average_cor']
    plt.rcParams['font.family'] = 'Gill Sans'
    plt.rcParams['font.size'] = 12
    fig, axe = plt.subplots(1, 3, figsize=(6, 4))
    sns.despine()
    sns.kdeplot(
        data=social_sci, x=column[0], hue="group",
        fill=True
        , common_norm=False
        , ax=axe[0]
        , palette=palette_us_cn
        # , alpha=0.6
        , linewidth=1.4
        , edgecolor='black'
        , legend=False
    )
    axe[0].set_xlabel('')
    sns.kdeplot(
        data=tech_sci, x=column[0], hue="group",
        fill=True
        , common_norm=False
        , ax=axe[1]
        , palette=palette_us_cn
        # , alpha=0.6
        , linewidth=1.4
        , edgecolor='black'
        , legend=False
    )
    axe[1].set_xlabel(x_label)
    axe[1].set_ylabel('')
    sns.kdeplot(
        data=mov_sci, x=column[0], hue="group",
        fill=True
        , common_norm=False
        , ax=axe[2]
        , palette=palette_us_cn
        # , alpha=0.6
        , linewidth=1.4
        , edgecolor='black'
        , legend=False
    )
    axe[2].set_xlabel('')
    axe[2].set_ylabel('')
    labels = ['a', 'b', 'c']
    for i, label in enumerate(labels):
        axe[i].text(0.5, -0.2, label, transform=axe[i].transAxes,
                    fontsize=32, fontweight='bold', va='top'
                    , ha='center', fontfamily='Gill Sans')
    plt.tight_layout()
    plt.savefig('social_mobility_betweenness.svg', dpi=500)
    plt.show()


def person_culture_mobility_robustness():
    plt.rcParams['font.family'] = 'Gill Sans'
    plt.rcParams['font.size'] = 12
    fig, axe = plt.subplots(1, 3, figsize=(10, 4))
    sns.despine()
    t2 = 'cultural_mobility_person_across_time'
    culture_sci = dataset(t2, kind='author')
    culture_tech = dataset(t2, kind='patent')
    culture_mov = dataset(t2, kind='movie')
    column_t2 = ['mean_variance_to_centroid']
    sns.kdeplot(
        data=culture_sci, x=column_t2[0], hue="group",
        fill=True
        , common_norm=False
        , ax=axe[0]
        , palette=palette_us_cn
        # , alpha=0.6
        , linewidth=1.4
        , edgecolor='black'
        , legend=False
    )
    axe[0].set_xlabel('')
    sns.kdeplot(
        data=culture_tech, x=column_t2[0], hue="group",
        fill=True
        , common_norm=False
        , ax=axe[1]
        , palette=palette_us_cn
        # , alpha=0.6
        , linewidth=1.4
        , edgecolor='black'
        , legend=False
    )
    axe[1].set_xlabel('Mean_variance_to_centroid')
    axe[1].set_ylabel('')
    sns.kdeplot(
        data=culture_mov, x=column_t2[0], hue="group",
        fill=True
        , common_norm=False
        , ax=axe[2]
        , palette=palette_us_cn
        # , alpha=0.6
        , linewidth=1.4
        , edgecolor='black'
        , legend=False
    )
    axe[2].set_xlabel('')
    axe[2].set_ylabel('')
    labels = ['a', 'b', 'c']
    for i, label in enumerate(labels):
        axe[i].text(0.5, -0.2, label, transform=axe[i].transAxes,
                    fontsize=30, fontweight='bold', va='top'
                    , ha='center', fontfamily='Gill Sans')

    plt.tight_layout()
    plt.savefig('person_culture_mobility.svg', dpi=500)
    plt.show()


def empirical_network_features_combine():
    types = 'empirical_whole_network_feature'
    sci = dataset(types, kind='author').rename(columns={'Path length': 'Path Length'
        , 'Global efficiency': 'Global Efficiency'
        , 'Mean closeness centrality': 'Closeness Efficiency', 'Gini degree': 'Degree Inequality'})
    tech = dataset(types, kind='patent').rename(columns={'Path length': 'Path Length'
        , 'Global efficiency': 'Global Efficiency'
        , 'Mean closeness centrality': 'Closeness Efficiency', 'Gini degree': 'Degree Inequality'})
    mov = dataset(types, kind='movie').rename(columns={'Path length': 'Path Length'
        , 'Global efficiency': 'Global Efficiency'
        , 'Mean closeness centrality': 'Closeness Efficiency', 'Gini degree': 'Degree Inequality'})
    columns = ['Path Length', 'Global Efficiency', 'Closeness Efficiency', 'Degree Inequality']

    plt.rcParams['font.family'] = 'Gill Sans'
    plt.rcParams['font.size'] = 12
    fig, axes = plt.subplots(3, 4, figsize=(8, 6))
    sns.despine()
    for i, data in enumerate([sci, tech, mov]):
        for j, column in enumerate(columns):
            sns.kdeplot(
                data=data, x=column, hue="group",
                fill=True, common_norm=False,
                ax=axes[i, j]
                , palette=palette_us_cn
                #, alpha=0.6
                , linewidth=1.4
                , edgecolor='black', legend=False
            )
            if j != 0:
                axes[i, j].set_ylabel("")
            if i != 2:
                axes[i, j].set_xlabel("")
    for i, label in enumerate(['a', 'b', 'c']):
        fig.text(0.02, 0.94 - i * 0.33, label
                 , fontsize=25, fontweight='bold'
                 , ha='center', fontfamily='Gill Sans')

    plt.tight_layout()
    plt.savefig('empirical_network_features.svg', dpi=500)
    plt.show()


def simulation_network_features_combine():
    types = 'simulation_whole_network_feature'
    sim_df = dataset(types,kind=None).rename(columns={'Path length': 'Path Length'
        , 'Global efficiency': 'Global Efficiency'
        , 'Mean closeness centrality': 'Closeness Efficiency', 'Gini degree': 'Degree Inequality'})
    columns = ['Path Length', 'Global Efficiency', 'Closeness Efficiency', 'Degree Inequality']

    plt.rcParams['font.family'] = 'Gill Sans'
    plt.rcParams['font.size'] = 12
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    sns.despine()
    for j, column in enumerate(columns):
        row, col = divmod(j, 2)
        sns.kdeplot(
            data=sim_df, x=column, hue="group",
            fill=True, common_norm=False,
            ax=axes[row,col]
            , palette=palette_simu
            # ,alpha=0.6
            , linewidth=1.4
            , edgecolor='black'
            , legend=False
        )
        axes[row,col].set_xlabel(column)

    positions = [(0.02, 0.94), (0.51, 0.94), (0.02, 0.45), (0.52, 0.45)]
    labels = ['d', 'e', 'f','g']
    for i, (label, (x, y)) in enumerate(zip(labels, positions)):
        fig.text(x, y, label, fontsize=25, fontweight='bold'
                 , ha='center', fontfamily='Gill Sans')

    plt.tight_layout()
    plt.savefig('simulation_network_features.svg', dpi=500)
    plt.show()


def figure_3_person_social_culture_mobility_combine():
    t1 = 'social_mobility'
    social_sci = dataset(t1, kind='author')
    social_tech = dataset(t1, kind='patent')
    social_mov = dataset(t1, kind='movie')
    column = ['degree_average_cor']
    plt.rcParams['font.family'] = 'Gill Sans'
    plt.rcParams['font.size'] = 12
    fig, axe = plt.subplots(2, 3, figsize=(8, 6))
    sns.despine()
    sns.kdeplot(
        data=social_sci, x=column[0], hue="group",
        fill=True
        , common_norm=False
        , ax=axe[0, 0]
        , palette=palette_us_cn
        # , alpha=0.6
        , linewidth=1.4
        , edgecolor='black'
        , legend=False
    )
    axe[0, 0].set_xlabel(('Degree Centrality Rank \n Average Correlation'))
    sns.kdeplot(
        data=social_tech, x=column[0], hue="group",
        fill=True
        , common_norm=False
        , ax=axe[0, 1]
        , palette=palette_us_cn
        # , alpha=0.6
        , linewidth=1.4
        , edgecolor='black'
        , legend=False
    )
    axe[0, 1].set_xlabel(('Degree Centrality Rank \n Average Correlation'))
    axe[0, 1].set_ylabel('')
    sns.kdeplot(
        data=social_mov, x=column[0], hue="group",
        fill=True
        , common_norm=False
        , ax=axe[0, 2]
        , palette=palette_us_cn
        # , alpha=0.6
        , linewidth=1.4
        , edgecolor='black'
        , legend=False
    )
    axe[0, 2].set_xlabel(('Degree Centrality Rank \n Average Correlation'))
    axe[0, 2].set_ylabel('')

    t2 = 'cultural_mobility_person_across_time'
    culture_sci = dataset(t2, kind='author')
    culture_sci = culture_sci[culture_sci['Breadth'] < 4]
    culture_tech = dataset(t2, kind='patent')
    culture_tech = culture_tech[culture_tech['Breadth'] < 2]
    culture_mov = dataset(t2, kind='movie')
    culture_mov = culture_mov[culture_mov['Breadth'] < 6]
    column_t2 = ['Breadth']
    sns.kdeplot(
        data=culture_sci, x=column_t2[0], hue="group",
        fill=True
        , common_norm=False
        , ax=axe[1, 0]
        , palette=palette_us_cn
        # , alpha=0.6
        , linewidth=1.4
        , edgecolor='black'
        , legend=False
    )
    axe[1, 0].set_xlabel('Breadth')
    sns.kdeplot(
        data=culture_tech, x=column_t2[0], hue="group",
        fill=True
        , common_norm=False
        , ax=axe[1, 1]
        , palette=palette_us_cn
        # , alpha=0.6
        , linewidth=1.4
        , edgecolor='black'
        , legend=False
    )
    axe[1, 1].set_xlabel('Breadth')
    axe[1, 1].set_ylabel('')
    sns.kdeplot(
        data=culture_mov, x=column_t2[0], hue="group",
        fill=True
        , common_norm=False
        , ax=axe[1, 2]
        , palette=palette_us_cn
        # , alpha=0.6
        , linewidth=1.4
        , edgecolor='black'
        , legend=False
    )
    axe[1, 2].set_xlabel('Breadth')
    axe[1, 2].set_ylabel('')

    plt.tight_layout()
    plt.savefig('person_social_culture_mobility.svg', dpi=500)
    plt.show()


if __name__ == '__main__':
    # types = 'simulation_whole_network_feature'
    # types = 'empirical_whole_network_feature'
    # types = 'cultural_mobility_domain'
    types = 'cultural_mobility_person_across_time'
    # types = 'social_mobility'

    # kind = 'author'
    # kind = 'patent'
    kind = 'movie'
    palette_simu = {'cycle': '#599CB4'
        , 'random': '#B696B6'
        , 'star': '#C25759'}
    palette_us_cn = {'us': '#599CB4', 'US': '#599CB4', 'CN': '#C25759', 'cn': '#C25759'}
    if types == 'simulation_whole_network_feature' or types == 'empirical_whole_network_feature':
        columns = ['Path length', 'Global efficiency', 'Mean closeness centrality', 'Gini degree']
    elif types == 'cultural_mobility_domain' or types == 'cultural_mobility_person_across_time':
        # columns = ['mean_variance_to_centroid', 'breadth']
        columns = ['mean_variance_to_centroid']
        columns = ['Breadth']
    elif types == 'social_mobility':
        # columns = ['degree_average_cor', 'betweeness centrality_average_cor']
        columns = ['degree_average_cor']
        # columns = ['betweeness centrality_average_cor']
    df = dataset(types,kind)
    # print(df.columns)
    # print(df.shape)
    # rigeplot(df, columns)
    # kdeplot(df, columns)
    # kdeplot_legend_overlap(df, columns)
    # vilon(df, columns)
    # vilon_single(df, columns)

    # Fig 1
    # simulation_network_features_combine()

    # Fig 2
    # empirical_network_features_combine()

    # Fig 3
    # figure_3_person_social_culture_mobility_combine()
    # person_social_mobility_robustness()
    # person_culture_mobility_robustness()

    # Fig 4
    figure_4_cultural_mobility_domain_combine()
    # cultural_mobility_domain_combine()
    # cultural_mobility_domain_combine_robustness()
