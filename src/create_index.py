import pyterrier as pt
import yaml
import os
import json
import pandas as pd


if not pt.started():
    pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])


BASE_PATH = "../data"
RESULTS_PATH = BASE_PATH + "/results"

with open(BASE_PATH + "/LongEval/metadata.yml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)


def doc_itter(sub_collection, docids: set):
    """Itterate over docs in a subcollection and skip every doc in docids set"""
    documents_path = os.path.join(
        BASE_PATH, config["subcollections"][sub_collection]["documents"]["json"]["en"]
    )
    documents = [
        os.path.join(documents_path, path) for path in os.listdir(documents_path)
    ]
    for doc_split_path in documents:
        with open(doc_split_path, "r") as f:
            docs = json.load(f)
            for doc in docs:
                docno = doc["id"]
                if docno not in docids:
                    yield {"docno": docno, "text": doc["contents"]}


def create_index(sub_collection, fold, docids):
    index_path = BASE_PATH + f"/index/{sub_collection}_{fold}"
    if os.path.exists(index_path):
        return pt.IndexFactory.of(index_path)

    iter_indexer = pt.IterDictIndexer(
        index_path, meta={"docno": 20, "text": 8096}, verbose=True, threads=8
    )
    indexref = iter_indexer.index(doc_itter(sub_collection, docids))
    return indexref


def load_data(sub_collection):
    # Test collection
    topics = pd.read_csv(
        BASE_PATH
        + "/"
        + config["subcollections"][sub_collection]["topics"]["test"]["tsv"]["en"],
        sep="\t",
        names=["qid", "query"],
    )
    qrels = pd.read_csv(
        BASE_PATH + "/" + config["subcollections"][sub_collection]["qrels"]["test"],
        sep=" ",
        names=["qid", "Q0", "docno", "relevance"],
    )

    # ID maps
    docid_map = pd.read_csv(
        "../data/document-groups-relevant.csv.gz", compression="gzip"
    )
    docid_map_patch = (
        docid_map[[sub_collection, "t" + str(int(sub_collection[-1]) - 1)]]
        .dropna()
        .set_index(sub_collection)
        .to_dict()["t" + str(int(sub_collection[-1]) - 1)]
    )

    queryid_map = pd.read_csv("../data/query_id_map.csv")
    queryid_map = (
        queryid_map[[sub_collection, "t" + str(int(sub_collection[-1]) - 1)]]
        .dropna()
        .set_index(sub_collection)
        .to_dict()["t" + str(int(sub_collection[-1]) - 1)]
    )

    return topics, qrels, docid_map_patch, queryid_map
