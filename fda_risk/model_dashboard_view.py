"""Read-only historical model comparison and explanation workbench."""

import math
from collections.abc import Mapping
from typing import Any

import pandas as pd
import streamlit as st

from fda_risk import model_dashboard_data as data_api

MODEL_LABELS = ('常數比例基準', 'Logistic Regression', 'Decision Tree', 'XGBoost')
ANSWER_LABEL = '歷史 LabClass 3 答案'


def render_model_dashboard(data: Mapping[str, Any]) -> None:
    comparison, report, rows = data['comparison'], data['ranking_report'], data['ranked_rows']
    split, metrics = comparison['temporal_split'], report['metrics']
    st.write('範圍：PAC 04004A 建模資料；不是資料概況入口的全 PAC 15,150 個樣本。')
    st.write('訓練 FY2019–FY2022；歷史評估與排序 FY2023；訓練／測試重疊 0。')
    for column,label,value in zip(st.columns(4),
                                 ('建模樣本','訓練樣本','全體 FY2023 樣本','FY2023 不利分類'),
                                 (comparison['input_samples'],split['train_samples'],len(rows),split['test_violative'])):
        column.metric(label,f'{value:,}')
    st.caption(f"訓練集不利分類 {split['train_violative']:,}；FY2023 的答案只作歷史評估，不放進模型特徵。")
    st.header('固定設定模型比較')
    st.write('AP／ROC-AUC／Lift 越高通常越好；Brier 越低越好。三個加權模型 Brier 高於常數基準，分數不是校準市場機率。')
    st.write('未調參、早停或校準。FY2023 已在可行性階段看過，不是全新未觸碰測試集。')
    st.write('LR 是可解釋展示選擇，不是最終部署模型；不因測試指標自動選定最佳模型。')
    metric_columns = {
        'AP':'average_precision','ROC-AUC':'roc_auc','Brier':'brier',
        'Top20 期望 Precision':'top20_expected_precision',
        'Top20 期望 Recall':'top20_expected_recall','Top20 期望 Lift':'top20_expected_lift',
    }
    table={'模型':list(MODEL_LABELS)}
    for label,key in metric_columns.items():
        table[label]=[comparison['models'][name]['metrics'][key] for name in data_api.MODEL_NAMES]
    st.dataframe(pd.DataFrame(table),hide_index=True,width='stretch',
                 column_config={label:st.column_config.NumberColumn(format='%.6f') for label in metric_columns})
    with st.expander('0.5 門檻與混淆矩陣細節'):
        st.write('分數 ≥ 0.5 判為 1；矩陣定義 [[TN, FP], [FN, TP]]。TN 真負、FP 假正、FN 假負、TP 真正。這不是食品安全判定門檻。')
        detail=[]
        for label,name in zip(MODEL_LABELS,data_api.MODEL_NAMES):
            m=comparison['models'][name]['metrics']
            detail.append(dict(模型=label,Precision=m['precision_at_0_5'],Recall=m['recall_at_0_5'],
                               F1=m['f1_at_0_5'],TN=m['confusion_matrix'][0][0],FP=m['confusion_matrix'][0][1],
                               FN=m['confusion_matrix'][1][0],TP=m['confusion_matrix'][1][1]))
        st.dataframe(pd.DataFrame(detail),hide_index=True,width='stretch')
    st.header('全體 LR Top20 歷史評估')
    for column,label,value in zip(st.columns(4),
                                 ('全體 Top20 名額','全體期望不利數','全體 Top20 Recall','全體 Top20 Lift'),
                                 (f"{metrics['top20_samples']:,}",f"{metrics['top20_expected_positives']:.12g}",
                                  f"{metrics['top20_expected_recall']:.2%}",f"{metrics['top20_expected_lift']:.6f}")):
        column.metric(label,value)
    st.write('以上以全部 FY2023 樣本為分母。篩選不重算指標或 Top20；權重是同分邊界均勻抽取的期望入選權重，不是實際抽樣結果。')
    st.caption(f"本次邊界同分 {metrics['top20_boundary_tied_samples']} 個、剩餘 {metrics['top20_boundary_slots']} 名額；一般邊界權重可能介於 0 與 1。權重大於 0 的候選列數不保證等於名額數。")
    st.header('FY2023 風險排序工作台')
    # Options use full data, independently of the other filter.
    for key,options in (('f10_country',[None,*data['countries']]),
                        ('f10_product_code',[None,*data['product_codes']])):
        if st.session_state.get(key) not in options:
            st.session_state[key]=None
    country=st.selectbox('來源國 Country',[None,*data['countries']],key='f10_country',
                         format_func=lambda value:'全部' if value is None else value)
    product=st.selectbox('食品代碼 ProdCode',[None,*data['product_codes']],key='f10_product_code',
                         format_func=lambda value:'全部' if value is None else value)
    top_only=st.checkbox('只顯示全體 Top20 權重大於 0 的樣本',key='f10_top20_only',value=False)
    show_answers=st.checkbox('顯示歷史答案',key='f10_show_answers',value=False)
    filtered=data_api.filter_ranked_rows(rows,country=country,product_code=product,top20_only=top_only)
    st.metric('目前顯示樣本',f'{len(filtered):,}')
    st.caption(f'全體排序樣本 {len(rows):,}；篩選只改變顯示列，保留全體名次、顯示序號與權重。')
    if not filtered:
        st.session_state.pop('f10_sample_id',None)
        st.info('目前篩選沒有符合條件的樣本；全體模型評估不變。')
        return
    columns={'全體名次':'score_rank','顯示序號':'display_position','sample_id':'sample_id',
             'ProdCode':'ProdCode','Country':'Country','Origin':'Origin','CollMonth':'CollMonth',
             '模型分數':'risk_score','全體 Top20 權重':'top20_weight'}
    ranking={label:[r[key] for r in filtered] for label,key in columns.items()}
    if show_answers:
        ranking[ANSWER_LABEL]=[int(r['violative']) for r in filtered]
    st.dataframe(pd.DataFrame(ranking),hide_index=True,width='stretch',
                 column_config={'模型分數':st.column_config.NumberColumn(format='%.12f'),
                                '全體 Top20 權重':st.column_config.NumberColumn(format='%.6f')})
    choices=[r['sample_id'] for r in filtered]
    if st.session_state.get('f10_sample_id') not in choices:
        st.session_state['f10_sample_id']=choices[0]
    selected=st.selectbox('選取 sample_id',choices,key='f10_sample_id')
    row=next(r for r in filtered if r['sample_id']==selected)
    st.subheader(f'樣本分數解釋：{selected}')
    st.metric('模型分數（非市場機率）',f"{row['risk_score']:.12g}")
    st.dataframe(pd.DataFrame({'特徵':list(data_api.FEATURES),
                              '值':[row[f] for f in data_api.FEATURES]}),hide_index=True,width='stretch')
    st.write(f"訓練未見類別欄位：{row['unseen_features'] or '無'}")
    st.write('未見類別可能映射至稀有類別係數；即使某欄貢獻為 0，也不代表真實風險為 0。')
    terms=[row['intercept'],*[row[f'{f}_contribution'] for f in data_api.FEATURES]]
    contributions=pd.DataFrame({'項目':['截距',*data_api.FEATURES],'貢獻（logit）':terms})
    st.bar_chart(contributions,x='項目',y='貢獻（logit）',horizontal=True,sort=False,height=280)
    st.dataframe(contributions,hide_index=True,width='stretch',
                 column_config={'貢獻（logit）':st.column_config.NumberColumn(format='%.15f')})
    st.code(f"截距 + 四欄貢獻 = logit：{math.fsum(terms):.12g}\n"
            f"sigmoid(logit) = 還原分數：{row['reconstructed_score']:.12g}\n"
            f"保存的模型分數：{row['risk_score']:.12g}",language=None)
    st.write('貢獻單位是 logit，不是機率百分點、SHAP、全域重要性或因果效果；本頁不接受新樣本或假設改值預測。')
    if show_answers:
        st.subheader('歷史診斷（不是模型輸入）')
        st.write(f"{ANSWER_LABEL}：{row['violative']}；0/1 只表示當時樣本分類，不能推廣為安全結論。")
    with st.expander('成果檔案與核對界線'):
        st.code('\n'.join(f'{name}: {path}' for name,path in data['source_paths'].items()),language=None)
        st.write('核對來源與排序 SHA256、樣本鍵與四特徵、歷史答案、排名／權重、分解還原及跨報告一致性。雜湊不是第三方簽章；本頁不重訓模型，也不重新估計 AP／ROC-AUC。')


def main() -> None:
    st.title('FDA 農藥殘留｜模型結果與風險工作台')
    st.warning('僅供 FDA 目標式抽驗資料內的歷史回顧與相對風險展示；不是新樣本預測，不代表整體市場違規機率，也不能判定食品安全。')
    try:
        data=data_api.load_model_dashboard()
    except data_api.ModelDashboardError as exc:
        for key in list(st.session_state):
            if key.startswith('f10_'):
                del st.session_state[key]
        st.error(f'無法顯示模型結果：{exc}')
        st.stop()
    render_model_dashboard(data)
