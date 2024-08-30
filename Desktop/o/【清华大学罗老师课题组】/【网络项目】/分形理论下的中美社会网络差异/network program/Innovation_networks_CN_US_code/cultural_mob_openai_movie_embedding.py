import pandas as pd
import tiktoken
from openai import OpenAI


def num_tokens_from_string(string: str, encoding_name='cl100k_base') -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def get_embedding(text, model="text-embedding-3-small"):
    global r
    r += 1
    print(r)
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def embeded_file(path, country, feature):
    df = pd.read_csv(path)
    if country == 'cn':
        df[feature] = df['movie_name'] + '。' + df['description']
    df[feature] = df[feature].apply(lambda x: x[:8192])
    df['text_embedding'] = df[feature] \
        .apply(lambda x: get_embedding(x, model='text-embedding-3-small'))
    df['num_tokens'] = df[feature].apply(num_tokens_from_string)
    print('----saving----')
    df.to_csv('data/{}/embedded_{}_movie_info.csv'.format(country, country), index=False)


if __name__ == '__main__':
    client = OpenAI(api_key='')
    r = 0
    cn_path = '../data/chinese_movie/cn_movie_new_20240505'
    embeded_file(cn_path, 'cn', 'movie_name_description')
    us_path = '../data/us_movie/us_movie_info_20240329.csv'
    embeded_file(us_path, 'us', 'title_description')
