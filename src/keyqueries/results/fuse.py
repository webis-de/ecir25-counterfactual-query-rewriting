#!/usr/bin/env python3
import pandas as pd
import pyterrier as pt
pt.init()

runs_to_fuse = {
    'lag1': '../../runs/ows_ltr_all/ows_ltr_all.train_2024',
    'lag6': '../../runs/ows_ltr_all/ows_ltr_all.lag6',
    'lag8': '../../runs/ows_ltr_all/ows_ltr_all.lag8',
}

for lag, run_to_fuse in runs_to_fuse.items():
    run_to_fuse = pt.io.read_results(run_to_fuse)
    run = pt.io.read_results(f'{lag}-run.txt')
    covered_qids = set(run['qid'].unique())
    print('Have', len(run_to_fuse['qid'].unique()), 'queries')
    run_to_fuse = run_to_fuse[~run_to_fuse['qid'].isin(covered_qids)]
    print('Fusing', len(run_to_fuse['qid'].unique()), 'queries')

    run = pd.concat([run, run_to_fuse])
    pt.io.write_results(run, f'bo1-keyqueries//bo1-keyqueries.{lag}')
