"""Train and evaluate Logistic Regression with student-grouped held-out data."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from matcher import BASELINE_WEIGHTS, FEATURE_COLUMNS, FEATURE_LABELS, MODEL_PATH, REPORT_PATH, SCHEMA_VERSION

REQUIRED_COLUMNS = ['respondent_id',*FEATURE_COLUMNS,'good_fit','label_source']


def validate_data(frame: pd.DataFrame) -> pd.DataFrame:
    missing=sorted(set(REQUIRED_COLUMNS)-set(frame.columns))
    if missing: raise ValueError('Missing required columns: '+', '.join(missing))
    frame=frame.copy()
    if frame[REQUIRED_COLUMNS].isna().any().any(): raise ValueError('Missing values must be resolved before training.')
    frame[FEATURE_COLUMNS]=frame[FEATURE_COLUMNS].apply(pd.to_numeric,errors='raise')
    features=frame[FEATURE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(features).all() or ((features<0)|(features>1)).any():
        raise ValueError('Every feature must be a finite number between 0 and 1; partial matches are supported.')
    labels=pd.to_numeric(frame['good_fit'],errors='raise')
    if not labels.isin([0,1]).all(): raise ValueError('good_fit must contain exactly 0 or 1.')
    frame['good_fit']=labels.astype(int)
    frame['respondent_id']=frame['respondent_id'].astype(str).str.strip()
    if (frame['respondent_id']=='').any(): raise ValueError('Every row needs a stable respondent ID.')
    sources=frame['label_source'].unique().tolist()
    if len(sources)!=1 or sources[0] not in {'simulated','human_feedback'}:
        raise ValueError('Use one documented label source: simulated OR human_feedback. Do not mix them.')
    if 'tutor_id' in frame and frame.duplicated(['respondent_id','tutor_id']).any():
        raise ValueError('Duplicate student–tutor pairs found; keep one current rating per pair.')
    if len(frame)<20 or frame['respondent_id'].nunique()<5:
        raise ValueError('Need at least 20 labeled pairs from 5 students; this minimum does not establish reliability.')
    if frame['good_fit'].nunique()!=2: raise ValueError('Both positive and negative labels are required.')
    return frame


def evaluate(labels, probabilities, threshold=.5):
    predicted=np.asarray(probabilities)>=threshold
    return {
        'accuracy':round(float(accuracy_score(labels,predicted)),4),
        'precision':round(float(precision_score(labels,predicted,zero_division=0)),4),
        'recall':round(float(recall_score(labels,predicted,zero_division=0)),4),
        'f1':round(float(f1_score(labels,predicted,zero_division=0)),4),
        'roc_auc':round(float(roc_auc_score(labels,probabilities)),4),
        'brier_score':round(float(brier_score_loss(labels,probabilities)),4),
        'confusion_matrix':confusion_matrix(labels,predicted,labels=[0,1]).tolist(),
        'decision_threshold':threshold,
    }


def train(csv: Path, model_path=MODEL_PATH, report_path=REPORT_PATH):
    frame=validate_data(pd.read_csv(csv))
    X,y,groups=frame[FEATURE_COLUMNS],frame['good_fit'],frame['respondent_id']
    split=GroupShuffleSplit(n_splits=1,test_size=.25,random_state=42)
    train_ids,test_ids=next(split.split(X,y,groups=groups))
    X_train,X_test=X.iloc[train_ids],X.iloc[test_ids]
    y_train,y_test=y.iloc[train_ids],y.iloc[test_ids]
    train_groups,test_groups=set(groups.iloc[train_ids]),set(groups.iloc[test_ids])
    assert not train_groups & test_groups, 'A student appeared in both splits.'
    if y_train.nunique()!=2 or y_test.nunique()!=2:
        raise ValueError('The fixed grouped split needs both labels on each side; collect more independent students.')
    make_model=lambda:make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,random_state=42))
    evaluated_model=make_model().fit(X_train,y_train)
    model_probabilities=evaluated_model.predict_proba(X_test)[:,1]
    baseline_probabilities=X_test.to_numpy() @ np.array([BASELINE_WEIGHTS[n] for n in FEATURE_COLUMNS])
    majority_probabilities=np.full(len(y_test),float(y_train.mean()))
    source=frame['label_source'].iloc[0]
    limitation=('All profiles and labels in this dataset are simulated. Metrics measure recovery of an invented pattern, not real student satisfaction or learning outcomes.' if source=='simulated' else 'Labels are subjective pair-fit feedback. A single grouped split on a small self-selected sample does not establish real tutoring outcomes or general performance.')
    final_model=make_model().fit(X,y)
    trained_at=datetime.now(timezone.utc).isoformat()
    report={
        'algorithm':'StandardScaler + LogisticRegression','schema_version':SCHEMA_VERSION,
        'training_source':source,'trained_at':trained_at,'random_seed':42,
        'dataset_file':csv.name,'dataset_sha256':hashlib.sha256(csv.read_bytes()).hexdigest(),
        'total_pairs':len(frame),'total_students':groups.nunique(),
        'training_pairs':len(train_ids),'test_pairs':len(test_ids),
        'training_students':len(train_groups),'test_students':len(test_groups),'student_overlap':0,
        'positive_labels':int(y.sum()),'negative_labels':int((y==0).sum()),
        'evaluation_method':'One fixed 75/25 split grouped by student; unseen students, potentially shared tutors.',
        'model_metrics':evaluate(y_test,model_probabilities),
        'weighted_baseline_metrics':evaluate(y_test,baseline_probabilities,threshold=.65),
        'majority_baseline_metrics':evaluate(y_test,majority_probabilities),
        'serving_model':'Refitted on all rows only after recording held-out evaluation.',
        'feature_columns':FEATURE_COLUMNS,'limitation':limitation,
        'feature_details':[{'name':n,'description':FEATURE_LABELS[n],
                            'standardized_coefficient':round(float(c),5)}
                           for n,c in zip(FEATURE_COLUMNS,final_model.named_steps['logisticregression'].coef_[0])],
        'sklearn_version':sklearn.__version__,
    }
    bundle={'model':final_model,'schema_version':SCHEMA_VERSION,'feature_columns':FEATURE_COLUMNS,
            'training_source':source,'trained_at':trained_at,'training_rows':len(frame),
            'respondents':groups.nunique(),'sklearn_version':sklearn.__version__,'evaluation':report}
    model_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(bundle,model_path,compress=3)
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv',type=Path)
    args=parser.parse_args()
    try: report=train(args.csv)
    except (ValueError,KeyError) as error: raise SystemExit(str(error))
    print(f"Source: {report['training_source']} — {report['total_pairs']} pairs, {report['total_students']} students")
    print(f"Held out: {report['test_pairs']} pairs from {report['test_students']} unseen students; student overlap = 0")
    print('Logistic regression:',json.dumps(report['model_metrics']))
    print('Weighted baseline:',json.dumps(report['weighted_baseline_metrics']))
    print(report['limitation'])
    print(f'Saved model to {MODEL_PATH} and evaluation to {REPORT_PATH}')


if __name__=='__main__':
    main()
