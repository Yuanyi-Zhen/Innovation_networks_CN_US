import pandas as pd
from collections import Counter
import warnings
import itertools
import os

warnings.filterwarnings('ignore')


def split_list(input_list, year_window):
    chunks = []
    start = input_list[0]
    end = start + year_window
    # while end <= input_list[-1]:
    while end <= 2020:
        chunks.append((start, end))
        start += 1
        end += 1
    return chunks


def author_id_to_network_data(df, year, network_type='non-cumulated'):
    lst = []
    for index, row in df.iterrows():
        pairs = itertools.combinations(eval(row['authors_id']), 2)
        for pair in pairs:
            lst.append((pair[0], pair[1], row['publication_year']))
    result = pd.DataFrame(lst, columns=['author1', 'author2', 'year'])
    if network_type == 'cumulated':
        result = result[(result['year'] <= year[1]) & (result['year'] >= year[0])]
        results = result.groupby(['author1', 'author2']).size() \
            .reset_index().rename(columns={0: 'weight'}) \
            .sort_values(by='weight', ascending=False)
        results['year'] = year[1]
        # print(results.shape)
    else:
        result = result[result['year'] == year]
        result = result.groupby(['author1', 'author2', 'year']).size() \
            .reset_index().rename(columns={0: 'weight'}) \
            .sort_values(by='weight', ascending=False)
        columns = ['author1', 'author2', 'weight', 'year']
        result = result[columns]
    return results


def build_domain_coauthor_network(domain, network_type='non-cumulated', only_last_year=False):
    dataset = data[data['concepts_name_level'].str.contains(domain)]
    chinese_dataset = dataset[dataset['authors_country_flag'] == 'CN'][['authors_id', 'publication_year']]
    us_dataset = dataset[dataset['authors_country_flag'] == 'US'][['authors_id', 'publication_year']]
    # print(dataset.shape)
    print('chinese_dataset', chinese_dataset.shape)
    print('us_dataset', us_dataset.shape)
    years = sorted(list(set(list(dataset['publication_year']))))
    print(years)
    cn_years = sorted(list(set(list(chinese_dataset['publication_year']))))
    us_years = sorted(list(set(list(us_dataset['publication_year']))))
    cn_year_window = split_list(cn_years, rolling_year)
    us_year_window = split_list(us_years, rolling_year)
    print('cn_window num', len(cn_year_window))
    print('us_window num', len(us_year_window))
    for cn_year in cn_year_window:
        print('\nthis is {}'.format(str(cn_year[0]) + '-' + str(cn_year[1])))
        chinese_author_net = author_id_to_network_data(chinese_dataset, cn_year, network_type)
        cn_author_num = len(set(list(chinese_author_net['author1'])
                                + list(chinese_author_net['author2'])))
        chinese_author_net.to_csv(
            save_path + 'cn/{}_{}_cn_network_data.csv'.format(str(cn_year[0]) + '_' + str(cn_year[1]),
                                                              domain.replace(' ', '_'))
            , index=False)
    for us_year in us_year_window:
        print('\nthis is {}'.format(str(us_year[0]) + '-' + str(us_year[1])))
        # print('the number of papers in us:')
        us_author_net = author_id_to_network_data(us_dataset, us_year, network_type)
        us_author_num = len(set(list(us_author_net['author1'])
                                + list(us_author_net['author2'])))
        # print('the number of authors in us', us_author_num)
        # res.append('{}_{}_us_network_data.csv'.format(us_year, domain.replace(' ', '_')))
        us_author_net.to_csv(save_path + 'us/{}_{}_us_network_data.csv'.format(str(us_year[0]) + '_' + str(us_year[1]),
                                                                               domain.replace(' ', '_'))
                             , index=False)


if __name__ == "__main__":
    rolling_year = 5
    path = '../result/works_result_all.csv'
    # domains = pd.read_csv('../../result/616domains_scientist.csv')['concept_name_level'].to_list()
    domains = pd.read_csv('../../result/6w_network_domain_info.csv')
    domains = domains[domains['concept_level'] == '2']['concept_name_level'].unique()
    domains = sorted(domains)
    print(len(domains))
    # domains = ['Dorsum-2']
    cn_window_num = []
    us_window_num = []
    # 1 create folder
    nn = 0
    for d in domains:
        print(nn)
        for c in ['cn', 'us']:
            # os.makedirs('data/{}/{}'.format(d,c))
            # os.makedirs('data_6w/{}/{}'.format(d, c))
            os.makedirs('data_1w9/{}/{}'.format(d, c))
            # os.makedirs('result/{}/{}'.format(d, c))
            # os.makedirs('result_6w/{}/{}'.format(d, c))
            os.makedirs('result_1w9/{}/{}'.format(d, c))
        nn += 1
    # 2 build network: 5 years rolling window
    data = pd.read_csv('../../result/works_result_all.csv')
    # n = 250
    n = 0
    for domain in domains:
        # save_path = 'data/{}/'.format(domain)
        # save_path = 'data_6w/{}/'.format(domain)
        save_path = 'data_1w9/{}/'.format(domain)
        print('-----{}/{}------'.format(n, len(domains)))
        print('\n-----domain:{}------'.format(domain))
        build_domain_coauthor_network(domain, network_type='cumulated'
                                      , only_last_year=True)
        n += 1
