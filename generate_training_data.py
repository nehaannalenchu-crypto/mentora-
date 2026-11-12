"""Generate reproducible SIMULATED pairs; these are not real student ratings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from matcher import FEATURE_COLUMNS, build_pair_features

SUBJECTS = ['Mathematics', 'Physics', 'Computer Science / Programming', 'Electronics', 'AI & Machine Learning', 'English', 'French']
LANGUAGES = ['English', 'French', 'Creole', 'Hindi', 'Malayalam', 'Tamil', 'Telugu', 'Kannada']
LEVELS = ['School (K–12)', 'Undergraduate', 'Postgraduate', 'Competitive exams']
SLOTS = [f'{day} {period}' for day in ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'] for period in ['morning','afternoon','evening']]
STYLES = ['Step-by-step explanations','Practice problems','Visual explanations','Discussion and questions']


def generate(students: int = 600, pairs_per_student: int = 12, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    def sample(options, lower=1, upper=3):
        return rng.choice(options, size=int(rng.integers(lower, min(upper, len(options))+1)), replace=False).tolist()
    tutors = [{
        'subjects': sample(SUBJECTS), 'availability': sample(SLOTS,2,8),
        'languages': sample(LANGUAGES,1,4), 'grades': sample(LEVELS,1,3),
        'hourly_rate': int(rng.choice([250,350,450,500,650,800,1000,1400])),
        'gender': str(rng.choice(['female','male'])), 'formats': sample(['1-on-1','Group class'],1,2),
        'delivery_modes': sample(['Online','In person'],1,2), 'teaching_style':str(rng.choice(STYLES)),
    } for _ in range(48)]
    # An invented noisy utility model generates labels. It is NOT survey evidence
    # and intentionally differs from Mentora's hand-weighted baseline.
    simulated_coefficients = np.array([3.8,2.3,.7,.6,.4,.8,.2,.5,.6])
    rows = []
    for index in range(students):
        student = {
            'subjects':sample(SUBJECTS,1,3), 'availability':sample(SLOTS,2,6),
            'languages':sample(LANGUAGES,1,3), 'study_level':str(rng.choice(LEVELS)),
            'budget_max':int(rng.choice([300,450,500,700,1000,1500])),
            'teacher_gender':str(rng.choice(['female','male','any'],p=[.2,.2,.6])),
            'formats':sample(['1-on-1','Group class'],1,2),
            'delivery_modes':sample(['Online','In person'],1,2), 'learning_styles':sample(STYLES,1,2),
        }
        student_bias = float(rng.normal(0,.35))
        for tutor_index in rng.choice(len(tutors), size=pairs_per_student, replace=False):
            features = build_pair_features(student, tutors[int(tutor_index)])
            logit = float(np.array([features[name] for name in FEATURE_COLUMNS]) @ simulated_coefficients - 5.1 + student_bias)
            probability = 1/(1+np.exp(-logit))
            rows.append({'respondent_id':f'sim-student-{index:04d}', 'tutor_id':f'sim-tutor-{int(tutor_index):02d}',
                         **features,'good_fit':int(rng.random()<probability),'label_source':'simulated'})
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('data/simulated_pairs.csv'))
    parser.add_argument('--students',type=int,default=600)
    parser.add_argument('--pairs-per-student',type=int,default=12)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    if args.students < 5 or not 1 <= args.pairs_per_student <= 48:
        parser.error('Use at least 5 students and between 1 and 48 pairs per student.')
    frame=generate(args.students,args.pairs_per_student,args.seed)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(args.output,index=False,float_format='%.8f')
    metadata={'source':'simulated','seed':args.seed,'students':args.students,'pairs':len(frame),'tutor_profiles':48,
              'label_definition':'Bernoulli samples from an invented preference utility; no human ratings.',
              'limitation':'Evaluation measures recovery of this artificial pattern, not real matching quality.'}
    args.output.with_suffix('.metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'Generated {len(frame)} SIMULATED labeled pairs from {args.students} simulated students.')


if __name__=='__main__':
    main()
