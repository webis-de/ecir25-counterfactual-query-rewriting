class PerGroupTransformer(TransformerBase, ABC):
    """
    Copied from ir_axioms: https://github.com/webis-de/ir_axioms/blob/main/ir_axioms/backend/pyterrier/transformers.py
    """

    group_columns: Set[str]
    optional_group_columns: Set[str] = {}
    verbose: bool = False
    description: Optional[str] = None
    unit: str = "it"

    @abstractmethod
    def transform_group(self, topics_or_res: DataFrame) -> DataFrame:
        pass

    def _all_group_columns(self, topics_or_res: DataFrame) -> Set[str]:
        return self.group_columns | {
            column for column in self.optional_group_columns
            if column in topics_or_res.columns
        }

    @final
    def transform(self, topics_or_res: DataFrame) -> DataFrame:
        require_columns(topics_or_res, self.group_columns)

        query_rankings: DataFrameGroupBy = topics_or_res.groupby(
            by=list(self._all_group_columns(topics_or_res)),
            as_index=False,
            sort=False,
        )
        if self.verbose:
            # Show progress during reranking queries.
            tqdm.pandas(
                desc=self.description,
                unit=self.unit,
            )
            query_rankings = query_rankings.progress_apply(
                self.transform_group
            )
        else:
            query_rankings = query_rankings.apply(self.transform_group)
        return query_rankings.reset_index(drop=True)


@dataclass(frozen=True)
class KeyQueryReRanker(AxiomTransformer):
    name = "KeyQueryReRanker"
    description = "Reranking query with keyquery"

    index: Optional[Union[Index, IndexRef, Path, str]] = None
    dataset: Optional[Union[Dataset, IRDSDataset]] = None
    contents_accessor: Optional[ContentsAccessor] = "text"
    verbose: bool = False
    counterfactual_index_docno_length: int = 50
    counterfactual_index_text_length: int = 4096


    def transform_query_ranking(
            self,
            query: Query,
            documents: Sequence[RankedDocument],
            topics_or_res: DataFrame,
    ) -> DataFrame:
        # Rerank documents.
        reranked_documents = self._axiom.rerank_kwiksort(
            self._context, query, documents, self.pivot_selection
        )

        # Convert reranked documents back to data frame.
        reranked = DataFrame({
            "docno": [doc.id for doc in reranked_documents],
            "rank": [doc.rank for doc in reranked_documents],
            "score": [doc.score for doc in reranked_documents],
        })

        # Remove original scores and ranks.
        original_ranking = topics_or_res.copy()
        del original_ranking["rank"]
        del original_ranking["score"]

        # Merge with new scores.
        reranked = reranked.merge(original_ranking, on="docno")
        return reranked

    def get_oracle_retrieval_results(topics, oracle_index, overlapping_queries):
        ret = []

        for _, topic in topics.iterrows():
            r = 0
            for hit in sorted(oracle_index[__normalize_queries(overlapping_queries[topic['qid']])], key=lambda x: x['relevance'], reverse=True):
                r += 1
                ret += [{'qid': topic['qid'], 'query': topic['query'], 'docno': 'ADD_' + hit['doc_id'], 'rank': r, 'score': 100-r, 'run_id': 'oracle'}]

        ret = pd.DataFrame(ret)
        return pt.transformer.get_transformer(ret)

    def build_reformulation_index(self, oracle_index, bm25_raw, topics, pt_dataset):
        additional_docs = {}

        for i in oracle_index:
             for j in oracle_index[i]:
                additional_docs[j['doc_id']] = j['doc']

        additional_docs = [{'docno': 'ADD_' + k, 'text': v} for k, v in additional_docs.items()]
        print(f'Have {len(additional_docs)} documents from the oracle.')

        doc_ids = []

        for _, i in bm25_raw(topics).iterrows():
            if i['qid'] in overlapping_queries:
                doc_ids.append(i['docno'])

        doc_ids = set(doc_ids)

        docs_for_reformulation = []

        for i in tqdm(pt_dataset.get_corpus_iter()):
            if i['docno'] not in doc_ids:
                continue
            docs_for_reformulation += [i]

        print(f'Have {len(additional_docs)} documents for reformulation.')

        iter_indexer = pt.IterDictIndexer("/tmp/reformulation-index", meta={'docno': self.counterfactual_index_docno_length, 'text': self.counterfactual_index_text_length}, overwrite=True)
        return iter_indexer.index(tqdm(docs_for_reformulation + additional_docs, 'Index'))

