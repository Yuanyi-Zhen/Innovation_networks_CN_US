import pandas as pd
import os
import numpy as np
import warnings
from functools import reduce

warnings.filterwarnings("ignore")


def select_muture_period(domain, country):
    path = 'result_1w9/{}/{}/'.format(domain, country)
    file_lst = os.listdir(path)
    file_lst = sorted([fl for fl in file_lst if int(fl.split('_')[0]) >= 2010 and int(fl.split('_')[0]) <= 2015])
    res = []
    for f in file_lst:
        df = pd.read_csv(path + f)
        # print(df.shape)
        res.append(df)
    res_df = reduce(lambda left, right: pd.merge(left, right, on='node', how='outer'), res)
    res_df.to_csv('result_1w9/{}/{}window_{}_{}_result_2010-2020.csv'
                  .format(domain, rolling_year, domain, country))


def rank_rank_correlation():
    cor_result = {'domain': domain, 'country': country}
    cn_file = pd.read_csv('result_1w9/{}/{}window_{}_{}_result_2010-2020.csv'
                          .format(domain, rolling_year, domain, country))
    for fea_rank in list(cn_file.columns)[1:]:
        cn_file[fea_rank + '_rank'] = cn_file[fea_rank] \
            .rank(method='min', na_option='keep', ascending=False)
    for f in feature_cols:
        cot = [rc for rc in cn_file.columns if '_rank' in rc and f in rc]
        cot_pair = [[cot[i], cot[i + 1]] for i in range(len(cot) - 1)]
        corre_lst = []
        corre_shape = []
        each_dimention = []
        for cp in cot_pair:
            # print(cp)
            fi_tmp = cn_file[cp[0]].dropna()
            each_dimention.append(fi_tmp.shape[0])
            file_tmp = cn_file[cp].dropna(subset=cp)
            corre_shape.append(file_tmp.shape[0])
            # file_tmp.to_csv('test_11_{}.csv'.format(cp),index=False)
            # print(file_tmp[cp[0]],file_tmp[cp[1]])
            correlation = file_tmp[cp[0]].corr(file_tmp[cp[1]], method='spearman')
            # print(correlation)
            if not pd.isnull(correlation):
                corre_lst.append(correlation)
            elif pd.isnull(correlation) and file_tmp.shape[0] != 0 \
                    and file_tmp[cp[0]].equals(file_tmp[cp[1]]):
                corre_lst.append(1)
            elif pd.isnull(correlation) and file_tmp.shape[0] == 0:
                corre_lst.append(0)
        cor_result["corre_num"] = corre_shape
        cor_result["each_time_stap_dimention"] = each_dimention
        cor_result["corre_num_ratio"] = [x / y for x, y in zip(corre_shape, each_dimention)]

        cor_result["{}_corre_lst".format(f)] = corre_lst
        cor_result["{}_average_cor".format(f)] = np.mean(corre_lst)
        cor_result["{}_std_cor".format(f)] = np.std(corre_lst)
        return cor_result


if __name__ == '__main__':
    rolling_year = 5
    domains = pd.read_csv('../../result/6w_network_domain_info.csv')
    domains = list(domains[domains['concept_level'] == '2']['concept_name_level'].unique())
    domains.remove('2')
    domains = sorted(domains)
    print(len(domains))


    for nu, domain in enumerate(domains):
        for country in ['cn', 'us']:
            print('-----No.{}--{}_{}-----'.format(nu, domain, country))
            try:
                select_muture_period(domain, country)
            except:
                print('wrong', domain)

    feature_cols = ['degree']
    res_lst = []
    for nu, domain in enumerate(domains):
        print(nu)
        try:
            for country in ['cn', 'us']:
                # for country in ['us']:
                res_dict = rank_rank_correlation()
                res_lst.append(res_dict)
                # print(res_lst)
        except:
            print('wrong', domain)
    df = pd.DataFrame(res_lst)
    print(df.shape)
    print(df.head(2))
    df.to_csv('2010-2020_degree_correlation_res_20240720.csv', index=False)

    feature_cols = ['betweeness centrality']
    res_lst = []
    for nu, domain in enumerate(domains):
        print(nu)
        try:
            for country in ['cn', 'us']:
                res_dict = rank_rank_correlation()
                res_lst.append(res_dict)
                # print(res_lst)
        except:
            print('wrong', domain)
    df = pd.DataFrame(res_lst)
    print(df.shape)
    print(df.head(2))
    df.to_csv('2010-2020_betweeness_correlation_res_20240720.csv', index=False)

    feature_cols = ['closeness centrality']
    res_lst = []
    for nu, domain in enumerate(domains):
        # print(domain)
        print(nu)
        try:
            for country in ['cn', 'us']:
                res_dict = rank_rank_correlation()
                res_lst.append(res_dict)
                # print(res_lst)
        except:
            print('wrong', domain)
    df = pd.DataFrame(res_lst)
    print(df.shape)
    print(df.head(2))
    df.to_csv('2010-2020_closeness_correlation_res_20240720.csv', index=False)
