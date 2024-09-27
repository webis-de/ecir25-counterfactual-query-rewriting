import pyterrier as pt
import yaml
import os
import json
import pandas as pd


if not pt.java.started():
    pt.java.add_package("com.github.terrierteam", "terrier-prf", "-SNAPSHOT")


BASE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)
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

    return topics, qrels


def load_folds(path=BASE_PATH + "/splits.json"):

    with open(path, "r") as f:
        folds = json.load(f)

    for k in folds.keys():
        for fold in folds[k].keys():
            for key in folds[k][fold].keys():
                folds[k][fold][key] = set(folds[k][fold][key])

    return folds

