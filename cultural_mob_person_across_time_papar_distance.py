import numpy as np
import pandas as pd
import itertools
import os
from collections import defaultdict
from tqdm import *
import sys


def alt_cosine(vector_pair):
    return (np.inner(vector_pair[0], vector_pair[1]) / (
            np.linalg.norm(vector_pair[0]) * np.linalg.norm(vector_pair[1]))).item()


def person_across_time_paper_distance(work_idss):
    # paper_embed = np.load('result/domain_embedding/{}_works_embedding.npy'.format(domain)
    #                       , allow_pickle=True).item()
    work_id_embed = paper_embed
    centoid_lst = []  # list of centroid vector in time series
    for work_ids in work_idss:
        # 1 calculate the centroid vector in each time window
        total_vector = [work_id_embed[id] for id in work_ids if id in work_id_embed]
        if len(total_vector) > 1:
            centroid_vector = sum(total_vector) / len(total_vector)
            centoid_lst.append(centroid_vector)
        elif len(total_vector) == 1:
            centroid_vector = total_vector[0]
            centoid_lst.append(centroid_vector)
    # print(len(centoid_lst))
    # according the centroid_lst
    # 1 calculate the centroid vector of time-series centroid vector list
    if len(centoid_lst) >= 2:
        lst_centroid_vector = sum(centoid_lst) / len(centoid_lst)
        variance_to_centroid_lst = [np.linalg.norm(tv - lst_centroid_vector) for tv in centoid_lst]
        std_variance_to_cen = np.std(variance_to_centroid_lst)
        mean_variance_to_centroid = np.average(variance_to_centroid_lst)
        max_variance_to_centroid = max(variance_to_centroid_lst)
        min_variance_to_centroid = min(variance_to_centroid_lst)
        # 2 breadth: similarity between one author's papers--- # higher values equal broader breadth
        sims = [alt_cosine(p) for p in itertools.combinations(centoid_lst, 2)]
        # print(sims)
        breadth = (((sum(sims) / float(len(sims)) + 1) * -50) + 100)
        return pd.Series(
            [std_variance_to_cen, mean_variance_to_centroid, max_variance_to_centroid, min_variance_to_centroid,
             breadth])
    else:
        return pd.Series([None, None, None, None, None])


def cn_us_authors(domain, country):
    print('this is {}_{}'.format(country, domain))
    country_author_work = nano_data[nano_data['authors_country_flag'] == country][
        ['work_id', 'authors_id', 'publication_year']]
    country_author_work['authors_id'] = country_author_work['authors_id'].apply(eval)
    country_df = country_author_work.explode('authors_id').reset_index(drop=True)
    country_df['time_range'] = country_df['publication_year'].apply(
        lambda x: [tw for tw in time_window if x <= tw[1] and x >= tw[0]] if any(
            x <= tw[1] and x >= tw[0] for tw in time_window) else '')
    country_df = country_df.explode('time_range').reset_index(drop=True)
    middle_df = country_df.groupby(['authors_id', 'time_range'])['work_id'].agg(list).reset_index()
    final_df = middle_df.groupby(['authors_id']).agg(list).reset_index() \
        .rename(columns={'time_range': 'time_window', 'work_id': 'work_ids'})
    final_df['time_window_num'] = final_df['time_window'].apply(lambda x: len(x))
    final_df['work_num_time_window'] = final_df['work_ids'].apply(lambda x: [len(xx) for xx in x])
    final_df['total_work_num'] = final_df['work_num_time_window'].apply(lambda x: sum(x))
    final_df['total_work_num_no_overlap'] = final_df['work_ids'].apply(
        lambda x: len(set([tt for xx in x for tt in xx])))
    final_df['group'] = country
    final_df['domain'] = domain
    print('the number of all authors', final_df.shape)
    final_df = final_df[final_df['total_work_num_no_overlap'] >= 2]
    print('the number of authors publised more than 2 papers', final_df.shape)
    final_df = final_df[final_df['time_window_num'] >= 2]
    print('the number of authors in more than 2 time windows', final_df.shape)
    if final_df.shape[0]>0:
        final_df[['std_variance_to_cen', 'mean_variance_to_centroid', 'max_variance_to_centroid'
            , 'min_variance_to_centroid', 'breadth']] \
            = final_df['work_ids'].apply(person_across_time_paper_distance)
        print(final_df.shape)
        # print(final_df.head(3))
        final_df['group'] = country
        # final_df.to_csv(
        #     'result/person_across_time_cn_us_paper_change_result_process/{}_{}_author_paper_change.csv'
        #         .format(domain, country), index=False)
        final_df.to_csv(
            'result/1200w_person_across_time_cn_us_paper_change_result_process/{}_{}_author_paper_change.csv'
                .format(domain, country), index=False)
        return final_df
    else:
        return None


if __name__ == '__main__':
    time_window = [(2010, 2015), (2011, 2016), (2012, 2017), (2013, 2018), (2014, 2019), (2015, 2020)]
    domains = pd.read_csv('1w9_sci_domains_20240711.csv')['concept_name_level'].unique()
    finished = [item.replace('_cn_us_paper_change.csv', '').replace('_us_paper_change.csv', '')
                    .replace('_cn_paper_change.csv', '') for item in
                os.listdir('result/1200w_person_level_across_time_cn_us_paper_change_result/')]
    domains = [d for d in domains if d not in finished]
    domains = sorted(domains)
    print(len(domains))
    data = pd.read_csv('data/works_all_last_20240303.csv')
    paper_embed = np.load('result/1200w_paper_embedding.npy', allow_pickle=True).item()
    for inc, domain in tqdm(enumerate(domains), total=len(domains)):
        print('\n----No.{}/{}---'.format(inc, len(domains)))
        nano_data = data[data['concepts_name_level'].str.contains(domain)]
        nano_data = nano_data[(nano_data['publication_year'] >= 2010)
                              & (nano_data['publication_year'] <= 2020)]
        print(nano_data.shape)
        # try:
        us_df = cn_us_authors(domain, 'US')
        cn_df = cn_us_authors(domain, 'CN')
        if us_df is not None and cn_df is not None:
            pd.concat([us_df, cn_df], axis=0).to_csv(
                'result/1200w_person_level_across_time_cn_us_paper_change_result'
                '/{}_cn_us_paper_change.csv'.format(domain),
                index=False)
        elif us_df is None and cn_df is not None:
            cn_df.to_csv('result/1200w_person_level_across_time_cn_us_paper_change_result'
                         '/{}_cn_paper_change.csv'.format(domain),
                index=False)
        elif cn_df is None and us_df is not None:
            us_df.to_csv('result/1200w_person_level_across_time_cn_us_paper_change_result'
                         '/{}_us_paper_change.csv'.format(domain),
                index=False)
