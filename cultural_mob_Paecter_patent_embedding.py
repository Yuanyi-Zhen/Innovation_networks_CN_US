from sentence_transformers import SentenceTransformer


def patent_embedding(patents):
    model = SentenceTransformer('mpi-inno-comp/paecter')
    embeddings = model.encode(patents)
    return embeddings
