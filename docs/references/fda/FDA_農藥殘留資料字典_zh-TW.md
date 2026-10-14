# FDA 農藥殘留資料字典

## 文件用途

這份資料字典說明 FDA `SampleData2023.txt` 的欄位、代碼與分析注意事項，供本專案進行資料清理、探索分析與違規風險模型建置時查閱。

## 官方來源

- [FDA Pesticide Residue Monitoring Database User Manual FY 2023](https://www.fda.gov/media/190128/download)
- [FDA Pesticide Residue Monitoring Report and Data for FY 2023](https://www.fda.gov/food/pesticides/pesticide-residue-monitoring-report-and-data-fy-2023)

FDA 官網列出的 `SampleData2023.txt` 共有 21,462 筆分析紀錄，與本專案讀取結果一致。檔案採 Tab 分隔文字格式。

## 資料粒度

`SampleData2023.txt` 的一列是「某個樣本的一筆分析或農藥殘留紀錄」，不是一個完整樣本。每個樣本可能使用多種分析方法，也可能驗出多種農藥，因此同一個 `SplNo` 會出現多次。

進行樣本層級分析時，應先以 `Year + SplNo` 作為樣本識別鍵，再將多筆分析紀錄彙整成每個樣本一列。

## 欄位說明

| 欄位 | 英文全名 | 中文說明 | 分析角色 |
|---|---|---|---|
| `Year` | FDA Fiscal Year | 樣本採集所屬的 FDA 會計年度；FDA 會計年度為前一年 10 月 1 日至該年 9 月 30 日 | 時間欄位 |
| `SplNo` | Sample Number | 樣本編號；需搭配 `Year` 形成跨年度唯一識別鍵 | 識別欄位 |
| `SplDescr` | Sample Description | 採樣人員填寫的敘述性樣本說明 | 抽驗前資訊，但屬非結構化文字 |
| `ProdCode` | Product Code | 五碼英數商品分類碼 | 抽驗前預測特徵 |
| `ProdName` | Product Name | FDA 依商品代碼給予的商品名稱 | 抽驗前預測特徵 |
| `Origin` | Sample Origin | 樣本來源；`D` 為美國國內、`I` 為進口、`R` 為已進入美國國內流通的進口品 | 抽驗前預測特徵 |
| `Country` | Country of Origin | 商品實際生產國家，不一定是採樣所在地 | 抽驗前預測特徵 |
| `State` | Collection State | 採樣所在州的兩碼郵政縮寫 | 抽驗前資訊 |
| `PAC` | Program Assignment Code | 樣本所屬的 FDA 計畫或任務代碼 | 分析範圍欄位 |
| `CollDist` | Collection District | 負責採樣的 FDA 區域或單位代碼 | 抽驗流程資訊 |
| `CollDate` | Collection Date | 採樣日期，文字格式為 `YYYYMMDD` | 抽驗前時間特徵 |
| `CmplDate` | Completion Date | 實驗室完成樣本分析的日期，格式為 `YYYYMMDD` | 檢驗後資訊，不可用來預測同一樣本 |
| `LabClass` | Laboratory Classification | 樣本的最終實驗室合規分類 | 模型答案欄位 |
| `SubNo` | Subsample Number | 子樣本代碼；通常為空白，否則為數字或標準代碼 | 樣本結構欄位 |
| `Portion` | Sample Portion | 接受分析的樣本部位或處理狀態 | 分析流程欄位 |
| `Type` | Analysis Type | 分析類型，例如原始分析、複驗、額外分析或重複分析 | 檢驗流程資訊 |
| `VioType` | Violation Type | 單筆農藥殘留的違規類型 | 檢驗後結果，不可作為預測特徵 |
| `ResCode` | Residue Code | 三碼農藥殘留代碼；`000` 表示未發現殘留，`RES` 表示殘留結果記錄於其他分析 | 檢驗後結果 |
| `ResName` | Residue Name | 農藥殘留名稱 | 檢驗後結果 |
| `Trace` | Trace Indicator | `T` 表示殘留量低於定量極限或低於 0.01 ppm，取兩者較高者 | 檢驗後結果 |
| `Found` | Residue Level Found | 實際測得的農藥殘留數值 | 檢驗後結果 |
| `Unit` | Residue Unit | 殘留量單位；`M` 表示 ppm，`B` 表示 ppb | 檢驗後結果 |
| `ExtCode` | Extraction Method Code | 萃取方法的三碼代碼，詳細說明見 `ExtCode.txt` | 檢驗方法資訊 |
| `DetCode` | Determination Method Code | 檢測方法的兩碼代碼，詳細說明見 `DetCode.txt` | 檢驗方法資訊 |
| `Lab` | Analytical Laboratory | 執行分析的 FDA 實驗室代碼 | 檢驗流程資訊 |
| `Remarks` | Remarks | 關於方法或樣本的備註 | 檢驗後非結構化資訊 |

## 重要代碼

### Origin 樣本來源

| 代碼 | 意思 |
|---|---|
| `D` | 美國國內樣本 |
| `I` | 進口樣本 |
| `R` | 已通過海關並進入美國國內流通的進口樣本 |

### PAC 計畫代碼

| 代碼 | 意思 |
|---|---|
| `04004A` | 食品中的農藥與工業化學物質，涵蓋美國國內及進口食品 |
| `04R800` | 監管後續或稽核樣本 |
| `71003A` | 動物飼料污染物計畫 |

本專案的食品違規風險模型以 `PAC == 04004A` 為主要範圍。

### LabClass 樣本分類

| 代碼 | 意思 |
|---|---|
| `1` | 合規 |
| `2` | 不需要採取監管行動 |
| `3` | 不利發現 |
| `4` | 不需要分類 |
| `5` | 未分析或未審查 |

本專案將 `LabClass == 3` 定義為樣本違規結果。這是樣本層級的答案，不能作為模型輸入特徵。

### VioType 殘留違規類型

| 代碼 | 意思 |
|---|---|
| `N` | 驗出的農藥或農藥與商品組合沒有美國容許量 |
| `X` | 殘留量超過美國既定容許量 |
| `A` | 殘留量超過 FDA 行動標準 |
| 空白 | 未驗出殘留，或驗出值未超過容許量或行動標準 |

`VioType` 是殘留紀錄層級的檢驗後資訊，不能直接作為同一樣本的預測特徵，也不應直接取代樣本層級的 `LabClass`。

### Portion 分析部位

| 代碼 | 意思 |
|---|---|
| `B` | 底部 |
| `C` | 熟製部分 |
| `E` | 可食部分 |
| `P` | 包裝材料 |
| `R` | 生鮮部分 |
| `T` | 頂部 |
| `W` | 完整產品 |
| `X` | 其他 |
| 空白 | 將完整產品混合後分析 |

### Type 分析類型

| 代碼 | 意思 |
|---|---|
| `O` | 原始分析 |
| `C` | 複驗分析 |
| `A` | 額外分析 |
| `D` | 重複分析 |
| `I` | 僅確認殘留物身分 |

## 建模使用原則

### 可考慮作為抽驗前預測特徵

- `ProdCode`
- `ProdName`
- `Origin`
- `Country`
- 從 `CollDate` 衍生的月份
- 僅使用過去年度計算的歷史統計

### 不可作為同一樣本的預測特徵

- `LabClass`
- `VioType`
- `ResCode`
- `ResName`
- `Trace`
- `Found`
- `Unit`
- `CmplDate`
- 任何由本次檢驗結果計算出的統計量

這些欄位在實驗室完成分析後才會產生。如果將它們放入模型，模型等於事先看見答案，會造成資料洩漏。

## 參考檔案

FDA 另外提供下列代碼表，可用來補充主資料：

- `ProdCode.txt`：商品代碼與商品名稱
- `Country.txt`：國家名稱
- `Chemical.txt`：農藥殘留代碼與名稱
- `ExtCode.txt`：萃取方法代碼與說明
- `DetCode.txt`：檢測方法代碼與說明
- `MethodScope2023.txt`：各萃取及檢測方法涵蓋的農藥範圍

最後查核日期：2026-09-17。
