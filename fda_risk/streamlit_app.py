"""F09: read-only Streamlit overview and descriptive exploration."""

from collections.abc import Mapping
from pathlib import Path
import sys
from typing import Any

# Streamlit executes this file as a script, adding its directory rather than
# the project root. Resolve package imports even in a fresh non-root process.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from fda_risk import dashboard_data

RATE_LABEL = '資料內 LabClass 3 比例'
LIMITATION = (
    'FDA 目標式抽驗不是隨機市場調查。以下比例只描述這份抽驗資料，'
    '不代表整體市場違規率，也不能直接判定食品安全或不安全。'
)


def _show_ranking(title: str, items: list[dict[str, Any]], *, rate: bool) -> None:
    st.subheader(title)
    if not items:
        st.info('目前門檻沒有符合條件的群組')
        return
    table = pd.DataFrame({
        '名稱': [item['name'] for item in items],
        '樣本數': [item['samples'] for item in items],
        'LabClass 3 樣本數': [item['violative'] for item in items],
        RATE_LABEL: [f"{item['rate']:.2%}" for item in items],
    })
    axis = RATE_LABEL if rate else '樣本數'
    chart = pd.DataFrame({
        '名稱': table['名稱'],
        axis: [item['rate'] for item in items] if rate else table['樣本數'],
    })
    st.bar_chart(chart, x='名稱', y=axis, horizontal=True, sort=False, height=380)
    if rate:
        st.caption('圖表使用 0–1 原始比例；表格以百分比顯示。僅保留達最小樣本數門檻的群組。')
    st.dataframe(table, hide_index=True, width='stretch')


def render_dashboard(data: Mapping[str, Any]) -> None:
    """Render only after both formal analyses have succeeded."""
    overview, analysis = data['overview'], data['analysis']
    for column, label, value in zip(st.columns(4),
                                   ('全部樣本', 'LabClass 3 樣本', '食品群組', '來源國'),
                                   (overview['sample_count'], data['labclass3_count'],
                                    analysis['products']['group_count'],
                                    analysis['countries']['group_count'])):
        column.metric(label, f'{value:,}')
    st.caption('範圍：全部 PAC 的樣本層級資料，不是 PAC 04004A 建模子集。')
    st.header('年度概況')
    yearly = pd.DataFrame({
        '年度': [r['year'] for r in overview['yearly']],
        '樣本數': [r['samples'] for r in overview['yearly']],
        'LabClass 3 樣本數': [r['labclass3'] for r in overview['yearly']],
        RATE_LABEL: [f"{r['rate']:.2%}" for r in overview['yearly']],
    })
    st.dataframe(yearly, hide_index=True, width='stretch')
    st.bar_chart(yearly, x='年度', y='樣本數', sort=False)
    for name, key in (('食品', 'products'), ('來源國', 'countries')):
        st.header(f'{name}探索分析')
        _show_ranking(f'{name}樣本量 Top 10', analysis[key]['volume_top'], rate=False)
        _show_ranking(f'{name} {RATE_LABEL} Top 10', analysis[key]['rate_top'], rate=True)
    with st.expander('定義、資料品質與方法'):
        st.write('來源：FDA FY2019–FY2023 農藥殘留抽驗資料；本頁讀取 F02 正式處理後 CSV。')
        st.code(data['source_path'], language=None)
        st.write('Year + SplNo 識別一個樣本；原始一列是分析紀錄，CSV 一列才是一個樣本。')
        st.write('LabClass == 3 是樣本層級不利分類；VioType N/X 是分析列層級資訊，不能互相取代。')
        st.write('analytical_rows 只保留來源分析列數，不是風險權重或殘留次數；本頁不含原始濃度細節。')
        st.write('F03 提供年度與品質概況，F04 按 ProdName／Country 分組。各樣本計數一次。')
        st.write('比例排行依原始比例、樣本數降序，再依名稱升序；樣本量排行依數量降序、名稱升序。')
        st.write('門檻只改變比例排行，不改變資料範圍、年度表或樣本量排行。每次互動重新讀取並驗證，來源唯讀。')
        st.dataframe(pd.DataFrame({
            '欄位': list(overview['fieldnames']),
            '空白': [overview['missing'][name] for name in overview['fieldnames']],
            'Unknown': [overview['unknown'][name] for name in overview['fieldnames']],
        }), hide_index=True, width='stretch')


def main() -> None:
    st.set_page_config(page_title='FDA 抽驗資料概況與探索', layout='wide')
    section = st.sidebar.radio('功能入口', ('資料概況與探索', '模型結果與風險工作台'),
                               key='dashboard_section')
    if section == '模型結果與風險工作台':
        from fda_risk.model_dashboard_view import main as model_main
        model_main()
        return
    st.title('FDA 農藥殘留｜專案概況與探索分析')
    st.write('FY2019–FY2023 · 樣本層級描述統計 · 作品集與模擬實務探索入口')
    st.warning(LIMITATION)
    threshold = st.number_input('比例排行最小樣本數', min_value=1, value=50,
                                step=1, key='min_rate_samples')
    try:
        data = dashboard_data.load_dashboard(min_rate_samples=threshold)
    except dashboard_data.DashboardDataError as exc:
        st.error(f'無法顯示分析結果：{exc}')
        st.stop()
    render_dashboard(data)


if __name__ == '__main__':
    main()
