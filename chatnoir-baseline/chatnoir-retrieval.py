#!/usr/bin/env python3
import click
import pandas as pd
from pathlib import Path

@click.command()
@click.option('--index', required=True, help='The ChatNoir Index to retrive from. E.g., "longeval-sci-2024-11".')
@click.option('--input-queries', help='The LongEval dataset to process. Loaded via the ir_datasets_longeval extension.')
@click.option('--num-results', help='The retrieval depth.', default=1000)
@click.option('--output', help='The output file', default=Path('run.txt'), type=Path)
@click.option('--model', help='The retrieval model in chatnoir to use.',  type=click.Choice(['bm25', 'default']), required=True)
def retrieve(index, input_queries, num_results, model, output):
    from ir_datasets_longeval import load
    from tirex_tracker import tracking, ExportFormat
    from chatnoir_pyterrier import ChatNoirRetrieve
    from pyterrier.io import write_results

    dataset = load(input_queries)
    queries = pd.DataFrame([{"qid": i.query_id, "query": i.default_text()} for i in dataset.queries_iter()])

    chatnoir = ChatNoirRetrieve(index=index, features=[], num_results=num_results, search_method=model, verbose=True)
    with tracking(export_file_path=output.parent / '.ir_metadata.yml', export_format=ExportFormat.IR_METADATA) as results:
        run = chatnoir(queries)
    write_results(run, output)


if __name__ == '__main__':
    retrieve()


