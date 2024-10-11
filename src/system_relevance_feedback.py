import pandas as pd
import pyterrier as pt
import yaml
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import numpy as np

if not pt.java.started():
    pt.java.add_package("com.github.terrierteam", "terrier-prf", "-SNAPSHOT")

BASE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)
RESULTS_PATH = BASE_PATH + "/results"

with open(BASE_PATH + "/LongEval/metadata.yml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)


def extract_top_terms(texts, top_n=10, query=None):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        sums = tfidf_matrix.sum(axis=0)
        data = []
        for col, term in enumerate(feature_names):
            data.append((term, sums[0, col]))
        top_terms = sorted(data, key=lambda x: x[1], reverse=True)[:top_n]
    except ValueError:
        print(">>> No relevant token found for feedback")
        print(texts)
        top_terms = []
    return top_terms


def extend_topics_with_relevance_feedback(
    sub_collection, train_docids, history, known_topics, queryid_map, docid_map, conn
):

    history_doc_ids_filtered = (
        docid_map[docid_map[sub_collection].isin(train_docids)][history]
        .dropna(subset=history, how="all")
        .values.flatten()
        .tolist()
    )
    history_doc_ids_filtered = list(set(history_doc_ids_filtered))
    if np.nan in history_doc_ids_filtered:
        history_doc_ids_filtered.remove(np.nan)

    extended_topics = []
    no_history = []
    for _, topic in known_topics.iterrows():
        queries_to_extend = queryid_map.loc[topic["qid"]].dropna().to_list()

        query = """SELECT url, text_en, qrel.docid
        FROM qrel
        JOIN document ON qrel.docid = document.docid
        WHERE queryid IN (%s)
        AND relevance > 0""" % ",".join(
            "?" * len(queries_to_extend)
        )

        rel_docs = pd.read_sql_query(query, conn, params=queries_to_extend)
        rel_docs = rel_docs[rel_docs["docid"].isin(history_doc_ids_filtered)]

        texts = (
            rel_docs.drop_duplicates(subset="url")["text_en"]
            .str.replace("\n", " ")
            .tolist()
        )

        if len(texts) == 0:
            extended_topics.append(topic)
            no_history.append(topic)
        else:
            extension_terms = [item[0] for item in extract_top_terms(texts)]
            extension_terms = " ".join(extension_terms)
            topic["query"] = topic["query"] + " " + extension_terms
            extended_topics.append(topic)

    print(">>> No history:", len(no_history), "from", len(known_topics))
    return pd.DataFrame(extended_topics)


def system_relevance_feedback(
    train_docids, sub_collection, topics, index, history, fold_no
):

    print(">>> Use history:", history)
    run_name = f"/BM25+RF_{sub_collection}_F{fold_no}-{''.join(history)}"
    conn = sqlite3.connect(BASE_PATH + "/database.db")

    docid_map = pd.read_csv(
        BASE_PATH + "/document-groups-relevant.csv.gz", compression="gzip"
    )
    queryid_map = pd.read_csv(BASE_PATH + "/query_id_map.gz", compression="gzip")

    queryid_map = queryid_map.dropna(subset=[sub_collection]).set_index(sub_collection)[
        history
    ]

    new_topics = topics[
        topics["qid"].isin(queryid_map[queryid_map.isna().all(axis=1)].index)
    ]
    known_topics = topics[
        topics["qid"].isin(queryid_map.dropna(subset=history, how="all").index)
    ]
    extended_topics = extend_topics_with_relevance_feedback(
        sub_collection,
        train_docids,
        history,
        topics,
        queryid_map,
        docid_map,
        conn,
    )
    # new_topics, extended_topics = get_relevance_feedback_topics(topics, history, conn)

    # new_topics = topics[topics["qid"].isin(new_topics)]
    # extended_topics = pd.DataFrame(extended_topics)

    BM25 = pt.terrier.Retriever(index, wmodel="BM25", verbose=True)

    # print(">>> Run RM3 with pseudo relevance feedback")
    # rm3_pipe = BM25 >> pt.rewrite.RM3(index) >> BM25
    # run_with_pseudo_feedback = rm3_pipe.transform(new_topics)

    print(">>> Run BM25 with relevance feedback")
    run_with_feedback = BM25.transform(extended_topics)

    # merged_run = pd.concat([run_with_feedback, run_with_pseudo_feedback])
    merged_run = pd.concat([run_with_feedback])

    pt.io.write_results(merged_run, RESULTS_PATH + run_name)

    return pt.io.read_results(RESULTS_PATH + run_name)
