"""Regression checks for shared features, honest evaluation and human feedback export."""
from contextlib import closing
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import matcher
from export_ratings import export_ratings
from generate_training_data import generate
from train_model import train, validate_data


class FeatureTests(unittest.TestCase):
    def test_partial_subjects_and_over_budget_remain_fractional(self):
        student={'subjects':['Mathematics','Physics'],'availability':['Monday evening','Wednesday evening'],
                 'budget_max':500,'teacher_gender':'female','languages':['English'],'study_level':'Undergraduate',
                 'formats':['1-on-1'],'delivery_modes':['Online'],'learning_styles':['Practice problems']}
        tutor={'subjects':['Mathematics'],'availability':['Wednesday evening'],'hourly_rate':1000,'gender':'male',
               'languages':['French'],'grades':['School (K–12)'],'formats':['Group class'],
               'delivery_modes':['In person'],'teaching_style':'Visual explanations'}
        f=matcher.build_pair_features(student,tutor)
        self.assertEqual(list(f),matcher.FEATURE_COLUMNS)
        self.assertEqual(f['subject_match'],.5)
        self.assertEqual(f['availability_overlap'],.5)
        self.assertEqual(f['budget_fit'],.5)
        self.assertEqual(f['gender_preference_match'],0)
        ranked=matcher.rank_tutors(student,[tutor],None)
        self.assertEqual(len(ranked),1)
        self.assertTrue(ranked[0]['tradeoffs'])

    def test_no_preferences_are_not_penalties(self):
        self.assertTrue(all(v==1 for v in matcher.build_pair_features({},{}).values()))

    def test_loaded_model_is_used_directly_for_ranking(self):
        bundle=matcher.load_trained_model()
        self.assertIsNotNone(bundle)
        student={'subjects':['Mathematics'],'availability':['Monday evening'],'budget_max':500}
        tutors=[{'subjects':['Mathematics'],'availability':['Monday evening'],'hourly_rate':450},
                {'subjects':['French'],'availability':['Saturday morning'],'hourly_rate':1000}]
        frame=pd.DataFrame([matcher.build_pair_features(student,t) for t in tutors],columns=matcher.FEATURE_COLUMNS)
        expected=sorted(bundle['model'].predict_proba(frame)[:,1],reverse=True)
        results=matcher.rank_tutors(student,tutors,bundle)
        np.testing.assert_allclose([r['score'] for r in results],expected)
        self.assertTrue(all(r['scoring_mode']=='model' for r in results))
        self.assertEqual(results[0]['subjects'],['Mathematics'])

    def test_missing_or_broken_artifact_falls_back(self):
        with tempfile.TemporaryDirectory() as root:
            target=Path(root)/'missing.joblib'
            with patch.object(matcher,'MODEL_PATH',target):
                self.assertIsNone(matcher.load_trained_model())
                target.write_bytes(b'not a model')
                with self.assertLogs('matcher',level='WARNING'):
                    self.assertIsNone(matcher.load_trained_model())


class TrainingTests(unittest.TestCase):
    def test_generation_is_reproducible_and_labels_documented(self):
        a,b=generate(30,6,42),generate(30,6,42)
        pd.testing.assert_frame_equal(a,b)
        self.assertEqual(set(a.label_source),{'simulated'})
        self.assertFalse(a.duplicated(['respondent_id','tutor_id']).any())
        self.assertEqual(len(validate_data(a)),180)

    def test_rejects_mixed_sources_invalid_labels_and_nonfinite_features(self):
        original=generate(30,6,42)
        for column,value in [('good_fit',.5),('subject_match',float('inf')),('label_source','human_feedback')]:
            bad=original.copy()
            bad[column]=bad[column].astype(object)
            bad.loc[0,column]=value
            with self.assertRaises(ValueError): validate_data(bad)

    def test_held_out_students_do_not_leak_and_report_matches_artifact(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            csv=root/'simulated.csv'
            generate(30,6,42).to_csv(csv,index=False)
            report=train(csv,root/'model.joblib',root/'evaluation.json')
            self.assertEqual(report['student_overlap'],0)
            self.assertEqual(report['training_students']+report['test_students'],30)
            self.assertEqual(report['training_pairs']+report['test_pairs'],180)
            self.assertEqual(report['training_source'],'simulated')
            self.assertEqual(json.loads((root/'evaluation.json').read_text()),report)
            self.assertTrue((root/'model.joblib').exists())


class FeedbackTests(unittest.TestCase):
    def test_export_requires_consent_and_keeps_latest_pair_rating(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);database=root/'ratings.sqlite3'
            features=json.dumps(dict.fromkeys(matcher.FEATURE_COLUMNS,.5))
            with closing(sqlite3.connect(database)) as c:
                c.executescript('''CREATE TABLE feedback(request_id,respondent_id,tutor_id,good_fit,training_consent,created_at);
                    CREATE TABLE shown_matches(request_id,respondent_id,tutor_id,feature_json);''')
                for req,student,tutor,label,consent,created in [('r1','s1','t1',0,1,'1'),('r2','s1','t1',1,1,'2'),('r3','s2','t2',1,0,'3'),('r4','s3','t3',None,1,'4')]:
                    c.execute('INSERT INTO feedback VALUES(?,?,?,?,?,?)',(req,student,tutor,label,consent,created))
                    c.execute('INSERT INTO shown_matches VALUES(?,?,?,?)',(req,student,tutor,features))
                c.commit()
            frame=export_ratings(database,root/'ratings.csv')
            self.assertEqual(len(frame),1)
            self.assertEqual(frame.iloc[0]['good_fit'],1)
            self.assertEqual(frame.iloc[0]['label_source'],'human_feedback')
            self.assertTrue(set(matcher.FEATURE_COLUMNS)<=set(frame.columns))

    def test_app_keeps_stable_participant_ids_and_requires_consent(self):
        import app as web
        with tempfile.TemporaryDirectory() as root, patch.object(web,'DATABASE_PATH',Path(root)/'app.sqlite3'):
            web.init_db()
            with web.db() as c:
                student=c.execute("INSERT INTO users(email,role,created_at) VALUES('student@example.invalid','student','now')").lastrowid
            first=web.get_respondent_id(student)
            self.assertEqual(first,web.get_respondent_id(student))
            with web.app.test_client() as client:
                with client.session_transaction() as session: session['user_id']=student
                response=client.post('/feedback',data={'rating':'4'})
                self.assertEqual(response.status_code,302)
            with web.db() as c:
                self.assertEqual(c.execute('SELECT COUNT(*) FROM feedback').fetchone()[0],0)
            with web.db() as c:
                teacher=c.execute("INSERT INTO users(email,role,created_at) VALUES('teacher@example.invalid','teacher','now')").lastrowid
            profile={'name':'Test student','subjects':['Mathematics'],'availability':['Monday evening'],'budget_max':500}
            tutor={'display_name':'Test teacher','subjects':['Mathematics'],'availability':['Monday evening'],'hourly_rate':400,
                   'languages':['English'],'grades':['Undergraduate'],'qualification':'Test only','offers_demo':False}
            web.save_profile(student,profile)
            web.save_profile(teacher,tutor,verification='approved')
            with web.app.test_client() as client:
                with client.session_transaction() as session: session['user_id']=student
                listing=client.get('/tutors')
                self.assertEqual(listing.status_code,200)
                self.assertIn(b'Logistic Regression active',listing.data)
                with web.db() as c:
                    shown=c.execute('SELECT request_id FROM shown_matches WHERE tutor_id=?',(str(teacher),)).fetchone()
                response=client.post('/feedback',data={'request_id':shown['request_id'],'tutor_id':str(teacher),'rating':'4','training_consent':'1'})
                self.assertEqual(response.status_code,302)
            exported=export_ratings(web.DATABASE_PATH,Path(root)/'export.csv')
            self.assertEqual(len(exported),1)
            self.assertEqual(exported.iloc[0]['label_source'],'human_feedback')
            self.assertEqual(exported.iloc[0]['respondent_id'],first)
            self.assertEqual(exported.iloc[0]['good_fit'],1)


if __name__=='__main__':
    unittest.main()
