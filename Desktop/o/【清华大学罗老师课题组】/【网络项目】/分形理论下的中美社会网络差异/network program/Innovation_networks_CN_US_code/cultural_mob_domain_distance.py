import numpy as np
import pandas as pd
import os
import itertools


def alt_cosine(vector_pair):
    return (np.inner(vector_pair[0], vector_pair[1]) / (
            np.linalg.norm(vector_pair[0]) * np.linalg.norm(vector_pair[1]))).item()


def paper_distance(work_ids):
    work_ids = list(work_ids)
    paper_embed = np.load('result/domain_embedding/{}_works_embedding.npy'.format(domain), allow_pickle=True).item()
    work_id_embed = paper_embed
    total_vector = [work_id_embed[id] for id in work_ids if id in work_id_embed]
    # print(len(total_vector))
    # 1 distance: between centroid vector and one paper
    centroid_vector = sum(total_vector) / len(total_vector)
    # print(len(centroid_vector))
    variance_to_centroid_lst = [np.linalg.norm(tv - centroid_vector) for tv in total_vector]
    std_variance_to_cen = np.std(variance_to_centroid_lst)
    mean_variance_to_centroid = np.average(variance_to_centroid_lst)
    max_variance_to_centroid = max(variance_to_centroid_lst)
    min_variance_to_centroid = min(variance_to_centroid_lst)
    # print(std_variance_to_cen, mean_variance_to_centroid)
    # 2 breadth: similarity between one author's papers--- # higher values equal broader breadth
    sims = [alt_cosine(p) for p in itertools.combinations(total_vector, 2)]
    breadth = (((sum(sims) / float(len(sims)) + 1) * -50) + 100)
    return pd.Series(
        [std_variance_to_cen, mean_variance_to_centroid, max_variance_to_centroid, min_variance_to_centroid, sims,
         breadth])


def entire_domain(domain, country):
    # cn_domain_authors = \
    #     data[(data['concepts_name_level'].str.contains(domain)) & (data['authors_country_flag'] == 'CN')][
    #         ['work_id', 'authors_id']]
    print('\n this is {}_{}'.format(country, domain))
    country_df = nano_data[nano_data['authors_country_flag'] == country][['work_id', 'authors_country_flag']]
    author_works = country_df.groupby('authors_country_flag')['work_id'].apply(list).reset_index()
    author_works['work_num'] = author_works['work_id'].apply(lambda x: len(x))
    print('the number of works', author_works['work_num'])
    author_works[['std_variance_to_cen', 'mean_variance_to_centroid', 'max_variance_to_centroid'
        , 'min_variance_to_centroid', 'sims_lst', 'breadth']] \
        = author_works['work_id'].apply(paper_distance)
    print(author_works.shape)
    print(author_works.head(3))
    author_works['group'] = country
    author_works.to_csv(
        'result/domain_cn_us_paper_change_result_process/{}_{}_author_paper_change.csv'
            .format(domain, country), index=False)
    return author_works


if __name__ == '__main__':
    domains = sorted([f[:-20] for f in os.listdir('result/domain_embedding/')])
    # print(len(domains), domains)
    for inc, domain in enumerate(domains):
        print('----No.{}/{}---'.format(inc, len(domains)))
        nano_data = pd.read_csv('data/domain_data/{}.csv'.format(domain))
        print(nano_data.shape)
        us_df = entire_domain(domain, 'US')
        cn_df = entire_domain(domain, 'CN')
        pd.concat([us_df, cn_df], axis=0).to_csv(
            'result/domain_level_cn_us_paper_change_result/{}_cn_us_paper_change.csv'.format(domain), index=False)
