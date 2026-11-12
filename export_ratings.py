"""Export consented, pseudonymous human pair-fit feedback for ML training."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path
import pandas as pd
from matcher import FEATURE_COLUMNS


def export_ratings(database: Path, output: Path):
    if not database.exists(): raise ValueError(f'No app database found at {database}')
    with closing(sqlite3.connect(database)) as connection:
        columns={row[1] for row in connection.execute('PRAGMA table_info(feedback)')}
        if 'training_consent' not in columns:
            raise ValueError('No consent-aware feedback is available; start the updated app first.')
        frame=pd.read_sql_query('''SELECT f.respondent_id,f.tutor_id,s.feature_json,f.good_fit
            FROM feedback f JOIN shown_matches s ON s.request_id=f.request_id
            AND s.tutor_id=f.tutor_id AND s.respondent_id=f.respondent_id
            WHERE f.good_fit IS NOT NULL AND f.training_consent=1 ORDER BY f.created_at''',connection)
    if frame.empty: raise ValueError('No consented positive/negative human ratings yet. Neutral ratings and demo tutors are excluded.')
    frame=frame.drop_duplicates(['respondent_id','tutor_id'],keep='last').reset_index(drop=True)
    features=pd.json_normalize(frame.pop('feature_json').map(json.loads))
    if set(FEATURE_COLUMNS)-set(features.columns) or features[FEATURE_COLUMNS].isna().any().any():
        raise ValueError('Some saved feedback uses an old feature schema; collect new ratings with the updated matcher.')
    frame=pd.concat([frame,features[FEATURE_COLUMNS]],axis=1)
    frame['label_source']='human_feedback'
    output.parent.mkdir(parents=True,exist_ok=True)
    frame[['respondent_id','tutor_id',*FEATURE_COLUMNS,'good_fit','label_source']].to_csv(output,index=False)
    return frame


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--database',type=Path,default=Path(os.environ.get('MENTORA_DATABASE',Path(__file__).parent/'instance'/'mentora.sqlite3')))
    args=parser.parse_args()
    try: frame=export_ratings(args.database,args.output)
    except ValueError as error: raise SystemExit(str(error))
    print(f'Exported {len(frame)} labeled pairs from {frame.respondent_id.nunique()} pseudonymous respondents.')


if __name__=='__main__':
    main()
