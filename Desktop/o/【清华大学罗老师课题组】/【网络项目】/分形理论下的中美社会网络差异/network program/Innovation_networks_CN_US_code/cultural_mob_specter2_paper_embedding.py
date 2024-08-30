from transformers import AutoTokenizer
from adapters import AutoAdapterModel


def papers_embedding(papers):
    # load model and tokenizer
    print('loading tokenizer')
    tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
    # load base model
    print('loading model')
    model = AutoAdapterModel.from_pretrained('allenai/specter2_base')
    model.load_adapter('allenai/specter2', source="hf", load_as="proximity", set_active=True)
    # load the adapter(s) as per the required task, provide an identifier for the adapter in load_as argument and activate it
    model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)
    # concatenate title and abstract
    text_batch = [str(d['title']) + tokenizer.sep_token + (str(d.get('abstract')) or '') for d in papers]

    # preprocess the input
    inputs = tokenizer(text_batch, padding=True, truncation=True,
                       return_tensors="pt", return_token_type_ids=False, max_length=512)
    output = model(**inputs)
    # take the first token in the batch as the embedding
    embeddings = output.last_hidden_state[:, 0, :]
    return embeddings
