import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from fda_risk import model_dashboard_data as data

ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'fda_risk/streamlit_app.py'
SECTION='模型結果與風險工作台'


class ModelDashboardViewTests(unittest.TestCase):
    def app(self):
        at=AppTest.from_file(str(APP),default_timeout=30).run()
        self.assertTrue(len(at.radio), 'F10 navigation must exist')
        at.radio(key='dashboard_section').set_value(SECTION).run()
        self.assertEqual(len(at.exception),0)
        return at

    def test_f09_default_and_f10_lazy_loading(self):
        with patch('fda_risk.model_dashboard_data.load_model_dashboard',side_effect=data.ModelDashboardError('missing F10')):
            at=AppTest.from_file(str(APP),default_timeout=30).run()
            self.assertEqual([m.value for m in at.metric],['15,150','1,336','813','114'])
            self.assertTrue(len(at.radio), 'F10 navigation must exist')
            at.radio(key='dashboard_section').set_value(SECTION).run()
            self.assertIn('missing F10',at.error[0].value)

    def test_model_result_render(self):
        at=self.app()
        self.assertEqual(at.dataframe[0].value['模型'].tolist(),
                         ['常數比例基準','Logistic Regression','Decision Tree','XGBoost'])
        metrics={m.label:m.value for m in at.metric}
        self.assertEqual(metrics['建模樣本'],'14,149')
        self.assertEqual(metrics['訓練樣本'],'10,572')
        self.assertEqual(metrics['全體 FY2023 樣本'],'3,577')
        self.assertEqual(metrics['FY2023 不利分類'],'376')
        self.assertEqual(metrics['全體 Top20 名額'],'716')
        self.assertEqual(metrics['全體期望不利數'],'188')
        self.assertEqual(metrics['全體 Top20 Recall'],'50.00%')
        self.assertEqual(metrics['全體 Top20 Lift'],'2.497905')
        text=' '.join(m.value for m in at.markdown)
        self.assertIn('Brier 越低越好',text)
        self.assertIn('不是全新未觸碰',text)
        self.assertIn('不是最終部署模型',text)
        self.assertEqual(len(at.get('vega_lite_chart')),1)
        self.assertEqual(at.dataframe[1].value.iloc[1][['TN','FP','FN','TP']].tolist(),[2245,956,105,271])

    def test_default_answers_hidden(self):
        at=self.app()
        initial=at.dataframe[2].value.copy()
        self.assertNotIn('歷史 LabClass 3 答案',initial.columns)
        self.assertEqual(at.dataframe[3].value['特徵'].tolist(),list(data.FEATURES))
        at.checkbox(key='f10_show_answers').check().run()
        self.assertIn('歷史 LabClass 3 答案',at.dataframe[2].value.columns)
        self.assertTrue(any('歷史診斷' in s.value for s in at.subheader))
        at.checkbox(key='f10_show_answers').uncheck().run()
        self.assertTrue(at.dataframe[2].value.equals(initial))
        self.assertFalse(any('歷史診斷' in s.value for s in at.subheader))

    def test_filter_and_sample_transitions(self):
        at=self.app()
        old=at.selectbox(key='f10_sample_id').value
        good=data.load_model_dashboard()
        first=good['ranked_rows'][0]
        country=next(c for c in good['countries'] if c!=first['Country'])
        at.selectbox(key='f10_country').set_value(country).run()
        expected=data.filter_ranked_rows(good['ranked_rows'],country=country)
        self.assertEqual(at.selectbox(key='f10_sample_id').value,expected[0]['sample_id'])
        self.assertNotEqual(at.selectbox(key='f10_sample_id').value,old)
        self.assertEqual(at.dataframe[2].value['全體名次'].tolist(),[r['score_rank'] for r in expected])
        code=next(code for code in good['product_codes'] if not any(r['ProdCode']==code for r in expected))
        at.selectbox(key='f10_product_code').set_value(code).run()
        self.assertTrue(any('沒有符合條件' in i.value for i in at.info))
        self.assertFalse(any(s.key=='f10_sample_id' for s in at.selectbox))
        self.assertEqual(len(at.dataframe),2)
        self.assertEqual(len(at.get('vega_lite_chart')),0)
        at.selectbox(key='f10_product_code').set_value(None).run()
        at.selectbox(key='f10_country').set_value(None).run()
        at.checkbox(key='f10_top20_only').check().run()
        self.assertEqual(len(at.dataframe[2].value),716)
        self.assertEqual({m.label:m.value for m in at.metric}['全體 Top20 Lift'],'2.497905')

    def test_errors_remove_stale_results_and_f09_still_works(self):
        at=self.app()
        at.checkbox(key='f10_show_answers').check().run()
        with patch('fda_risk.model_dashboard_data.load_model_dashboard',side_effect=data.ModelDashboardError('排序 SHA256 不一致')):
            at.run()
            self.assertEqual(len(at.exception),0)
            self.assertEqual(len(at.metric),0)
            self.assertEqual(len(at.dataframe),0)
            self.assertIn('SHA256',at.error[0].value)
            at.radio(key='dashboard_section').set_value('資料概況與探索').run()
            self.assertEqual([m.value for m in at.metric],['15,150','1,336','813','114'])
        at.radio(key='dashboard_section').set_value(SECTION).run()
        self.assertFalse(at.checkbox(key='f10_show_answers').value)

    def test_reruns_reload_without_mutation(self):
        paths=[data.DEFAULT_COMPARISON_PATH,data.DEFAULT_RANKING_REPORT_PATH,data.DEFAULT_RANKING_CSV_PATH,data.DEFAULT_SOURCE_PATH]
        hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
        actual=[]
        real=data.load_model_dashboard
        def reader(*args,**kwargs):
            result=real(*args,**kwargs); actual.append(result); return result
        with patch('fda_risk.model_dashboard_data.load_model_dashboard',side_effect=reader):
            at=self.app()
            at.checkbox(key='f10_top20_only').check().run()
        self.assertEqual(len(actual),2)
        self.assertEqual([hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],hashes)
        row=actual[-1]['ranked_rows'][0]
        values=at.dataframe[4].value['貢獻（logit）'].tolist()
        self.assertEqual(values,[row['intercept'],*[row[f'{f}_contribution'] for f in data.FEATURES]])
        self.assertTrue(any(value<0 for value in values))

    def test_fresh_process_non_root_model_entry(self):
        code="""
import pathlib,tempfile,sys,json
tempfile.tempdir=str(pathlib.Path.cwd())
from streamlit.testing.v1 import AppTest
at=AppTest.from_file(sys.argv[1],default_timeout=30).run()
if at.radio:
    at.radio(key='dashboard_section').set_value('模型結果與風險工作台').run()
print(json.dumps({'errors':[e.message for e in at.exception], 'metrics':{m.label:m.value for m in at.metric}},ensure_ascii=False))
"""
        with tempfile.TemporaryDirectory() as directory:
            env=os.environ.copy()
            env['PYTHONPATH']=os.pathsep.join(filter(None, (
                str(ROOT/'_fda_feasibility/vendor'), env.get('PYTHONPATH', ''),
            )))
            result=subprocess.run([sys.executable,'-X','utf8','-c',code,str(APP)],cwd=directory,
                                  env=env,capture_output=True,text=True,encoding='utf-8',timeout=60)
        self.assertEqual(result.returncode,0,result.stderr)
        actual=json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(actual['errors'],[])
        self.assertEqual(actual['metrics'].get('全體 FY2023 樣本'),'3,577')

    def test_visible_contribution_precision_supports_reconstruction(self):
        at=self.app()
        for index,column,expected in [(0,'AP','%.6f'),(4,'貢獻（logit）','%.15f')]:
            column_config=json.loads(at.dataframe[index].proto.columns)
            with self.subTest(column=column):
                self.assertEqual(column_config.get(column,{}).get('type_config',{}).get('format'),expected)

    def test_unseen_category_may_have_nonzero_contribution(self):
        at=self.app()
        at.selectbox(key='f10_country').set_value('Benin').run()
        self.assertGreater(at.dataframe[4].value.iloc[2]['貢獻（logit）'],0)
        text=' '.join(m.value for m in at.markdown)
        self.assertIn('未見類別可能映射至稀有類別係數',text)
        self.assertIn('即使某欄貢獻為 0',text)


if __name__=='__main__':
    unittest.main()
