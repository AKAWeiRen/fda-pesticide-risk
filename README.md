# FDA 農藥殘留｜歷史資料探索與可解釋風險排序

使用 FDA FY2019–FY2023 農藥殘留資料，展示样本層級探索、固定模型比較與可還原的 Logistic Regression 排序。這是作品集與模擬實務工具，不是食品安全判定服務。

網站使用範圍：僅本機。公開部署狀態：尚未公開部署。
本版本已保存於 [Private GitHub 儲存庫](https://github.com/AKAWeiRen/fda-pesticide-risk)，上傳與遠端 clone 驗收見[交付證據](docs/delivery/local-delivery.md)。F01–F10 已完成；F11 私有版本交付已驗收，整體 F11 維持進行中，雲端部署暫緩。

## 安裝與啟動

在專案根目錄，使用自己已選擇並啟用的 Python 環境。已驗證 Python 3.12.14；其他版本需另行驗證。

```powershell
python -m pip install -r requirements-ui.txt
python -m streamlit run fda_risk/streamlit_app.py --server.address 127.0.0.1
```

開啟終端機顯示的本機網址，Ctrl+C 停止。展示只需 Streamlit／pandas 及間接相依，不需要模型訓練套件，不下載資料、不重訓、不提供新樣本預測。

## 兩個入口

| 入口 | 範圍 | 核對數字 |
|---|---|---|
| 資料概況與探索（預設） | 全 PAC 的樣本資料 | 15,150 樣本、1,336 個 LabClass 3、813 食品群組、114 來源國 |
| 模型結果與風險工作台 | PAC 04004A 建模資料及 FY2023 LR 排序 | 14,149 建模樣本；訓練 10,572／933；FY2023 3,577／376；重疊 0 |

F09 將比例門檻 50→100000→50：只有比例排行改變，年度表與總數不變。
F10 的全體 Top20 權重和 716、期望不利數 188、Recall 0.5、Lift 約 2.497905。Country／ProdCode 以 AND 篩選；Top20 只顯示權重大於 0 的候選，篩選不改全體分數、名次、權重或指標。歷史答案預設隱藏且不是輸入特徵。

選取 sample_id 可檢查截距加四欄貢獻還原 logit，再以 sigmoid 還原分數；容差 logit 1e-10、score 1e-12。空篩選會清除舊樣本詳情。未見類別可能映射至稀有類別係數；Benin 的 Country 貢獻並非一定為 0。

## 正式分析與資料

原始一列是分析紀錄，依 Year + SplNo 聚合為一個樣本一列。violative 定義為 LabClass == 3；VioType 是分析紀錄診斷資訊，N/X 不可代替樣本答案。analytical_rows 僅記錄來源分析列數，不是殘留次數或風險。

特徵僅 ProdCode、Country、Origin、CollMonth；sample_id 識別、Year 時間切分、violative 答案，不放入 X。來源五年 77,455 分析列聚合為 15,150 樣本。
參見[資料字典](docs/references/fda/FDA_農藥殘留資料字典_zh-TW.md)、[標籤判讀](docs/references/fda/FDA_LabClass與VioType判讀注意事項.txt)與[官方手冊](docs/references/fda/FDA_Pesticide_Residue_Monitoring_Database_User_Manual_FY2023.pdf)。

原始輸入保存在 _fda_feasibility/extracted/2019 至 2023；data/processed 保存兩份正式 CSV；data/results 保存基準、比較與排序成果。來源及資料對外散布授權未在本次確認，不自行附加開源授權或改成公開儲存庫。

## 完整測試與重現界線

展示安裝與完整模型測試是不同層級。完整測試額外安裝模型套件：

```powershell
python -m pip install -r requirements-models.txt
python -m unittest discover -s tests -v
python -m fda_risk.feasibility
```

測試會擬合臨時 fixture 模型並核對保存資料，不代表完整資料模型已跨環境重訓。正式 F02/F05/F06/F07/F08 指令可能覆寫結果，請先在獨立副本另行規劃，不以展示啟動成功當成重訓驗證。

## 結論與限制

- 結果僅支持 FDA 目標式抽驗資料內相對風險，不代表整體市場違規率，不能直接判定食品安全；未觀察到 LabClass 3 不等於安全。
- FY2023 已用於可行性分析，不是全新未觸碰測試集；沒有未來年度或新市場的外部驗證。
- 固定模型未調參、早停或校準。加權模型 Brier 高於常數基準，分數未校準，不是可靠市場違規機率。
- LR 是可解釋展示選擇，不是最終部署模型。貢獻為 logit 單位，不是機率百分點、SHAP、全域重要性或因果效果。
- 本頁 AP/ROC-AUC 是保存報告核對，不是重新計算完整評估；SHA-256 用於一致性，不是第三方可信簽章。
- 保存資料中的絕對路徑是原產出環境 metadata，程式不依賴該環境路徑；沒有為美化而改寫成果。
- 本機驗證不保證任意 Windows 鎖定／斷電復原或雲端／Linux 相容。

## 文件與維護

fda_risk 是正式模組，tests 是正式自動測試；docs/features/F01.html 至 F11.html 說明功能與驗收。
參見[正式進度](PROGRESS.md)、[交付清單](docs/delivery/manifest.json)、[更新流程](docs/delivery/maintenance.md)、[F09](docs/features/F09.html)、[F10](docs/features/F10.html)、[F11](docs/features/F11.html)。
