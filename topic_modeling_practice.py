"""Practice code for topic modeling approaches referenced in README.

This script demonstrates:
1. Latent Semantic Analysis (LSA)
2. Latent Dirichlet Allocation (LDA)
3. Combined Topic Models using SBERT embeddings
4. BERTopic (English & Korean)

Each section can be run independently. The examples use a very small
in-memory dataset so that students can easily experiment without additional
datasets.
"""

from __future__ import annotations

from typing import List


# Sample documents for English and Korean examples
ENGLISH_TEXTS: List[str] = [
    "I like machine learning and natural language processing.",
    "Topic modeling is a technique for summarizing documents.",
    "Latent semantic analysis uses SVD.",
    "Latent Dirichlet allocation assumes a generative process.",
    "Embedding models help capture semantics.",
    "BERTopic leverages embeddings for clustering.",
]

KOREAN_TEXTS: List[str] = [
    "머신 러닝과 자연어 처리가 좋아요.",
    "토픽 모델링은 문서를 요약하는데 사용됩니다.",
    "LSA는 SVD를 사용합니다.",
    "LDA는 생성 모델을 가정합니다.",
    "임베딩 모델은 의미를 포착합니다.",
    "BERTopic은 임베딩을 활용한 클러스터링을 합니다.",
]


def lsa_example(documents: List[str], n_topics: int = 2) -> None:
    """Run a simple LSA example and print top words per topic."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(documents)

    svd = TruncatedSVD(n_components=n_topics, random_state=42)
    lsa = svd.fit_transform(tfidf)

    terms = vectorizer.get_feature_names_out()
    for i, comp in enumerate(svd.components_):
        terms_in_topic = [terms[idx] for idx in comp.argsort()[-5:][::-1]]
        print(f"LSA Topic {i + 1}:", terms_in_topic)
    print()


def lda_example(documents: List[str], n_topics: int = 2) -> None:
    """Run a simple LDA example and print top words per topic."""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation

    vectorizer = CountVectorizer(stop_words="english")
    dtm = vectorizer.fit_transform(documents)

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(dtm)

    terms = vectorizer.get_feature_names_out()
    for idx, topic in enumerate(lda.components_):
        top_terms = [terms[i] for i in topic.argsort()[-5:][::-1]]
        print(f"LDA Topic {idx + 1}:", top_terms)
    print()


def combined_topic_model_example(documents: List[str], n_topics: int = 2) -> None:
    """Combined topic model using SBERT embeddings and HDBSCAN clustering."""
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import NMF
    import hdbscan
    import numpy as np

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(documents)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
    clusters = clusterer.fit_predict(embeddings)

    clustered_docs = {c: [] for c in set(clusters) if c != -1}
    for doc, label in zip(documents, clusters):
        if label != -1:
            clustered_docs[label].append(doc)

    vectorizer = CountVectorizer(stop_words="english")
    for label, docs in clustered_docs.items():
        dtm = vectorizer.fit_transform(docs)
        nmf = NMF(n_components=1, random_state=42)
        nmf.fit(dtm)
        terms = vectorizer.get_feature_names_out()
        topic_terms = [terms[i] for i in nmf.components_[0].argsort()[-5:][::-1]]
        print(f"Combined Topic {label}:", topic_terms)
    print()


def bertopic_example(documents: List[str], language: str = "english") -> None:
    """Run BERTopic and print discovered topics."""
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(stop_words="english" if language == "english" else None)
    topic_model = BERTopic(language=language, vectorizer_model=vectorizer)
    topics, _ = topic_model.fit_transform(documents)

    topic_info = topic_model.get_topic_info()
    print(topic_info[["Topic", "Name"]])
    print()


if __name__ == "__main__":
    print("--- LSA Example (English) ---")
    lsa_example(ENGLISH_TEXTS)

    print("--- LDA Example (English) ---")
    lda_example(ENGLISH_TEXTS)

    print("--- Combined Topic Model Example (English) ---")
    combined_topic_model_example(ENGLISH_TEXTS)

    print("--- BERTopic Example (English) ---")
    bertopic_example(ENGLISH_TEXTS)

    print("--- BERTopic Example (Korean) ---")
    bertopic_example(KOREAN_TEXTS, language="multilingual")
