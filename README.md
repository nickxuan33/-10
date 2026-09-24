# TF-IDF 關鍵字分析

透過搜尋 API 取得指定地區（預設是加拿大亞伯達）Google 搜尋前幾名的網頁，抓下頁面文字，再用 TF-IDF 找出這些頁面共同強調的詞和兩字詞組，作為規劃內容的參考。

推廣博弈相關內容前，請先對照[亞伯達內容合規檢查表](docs/alberta-ad-compliance-checklist.md)。

專案裡的亞伯達娛樂城評論網站放在 [`site/`](site/)，使用方式見 [site/README.md](site/README.md)。

## 環境需求

- Python 3.10 以上
- [SerpApi](https://serpapi.com/) 帳號和 API key，註冊後可以在 Dashboard 找到。每搜尋一頁（10 筆結果）會用掉 1 次額度；免費方案每月的次數有上限，詳見 SerpApi 的定價頁。
- 網路連線

## 安裝

```bash
# 建立並啟用虛擬環境
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux

# 安裝套件
pip install -r requirements.txt
```

NLTK 的英文停用詞（stopwords）會在第一次執行時自動下載。

## 設定 API key

設定只對目前的終端機視窗有效，開新視窗要重新設定。

```bash
$env:SERPAPI_API_KEY="你的 API key"      # Windows PowerShell
set SERPAPI_API_KEY=你的 API key         # Windows 命令提示字元
export SERPAPI_API_KEY="你的 API key"    # macOS / Linux
```

## 使用方式

```bash
python tf-idf.py "alberta online casino"
python tf-idf.py "alberta online casino" "legal online casino alberta" --results 20
```

一次可以分析多個關鍵字，含空白的關鍵字要加引號。

| 參數 | 預設值 | 說明 |
| --- | --- | --- |
| `--results` | `10` | 每個關鍵字分析幾筆搜尋結果；每 10 筆用掉 1 次 SerpApi 額度 |
| `--top` | `20` | 終端機上每個關鍵字顯示幾個詞 |
| `--location` | `Alberta, Canada` | 模擬從哪裡搜尋，例如 `Calgary, Alberta, Canada` |
| `--gl` | `ca` | Google 國家代碼 |
| `--hl` | `en` | Google 介面語言 |
| `--google-domain` | `google.ca` | 使用的 Google 網域 |
| `--out` | `output` | 輸出資料夾 |

## 輸出

每次執行都會覆寫 `output` 資料夾裡的兩個檔案。檔案是 UTF-8（含 BOM），可以直接用 Excel 開啟。

**`serp.csv`**：每個關鍵字的搜尋結果

| 欄位 | 內容 |
| --- | --- |
| `keyword` | 關鍵字 |
| `rank` | 排名 |
| `title` | 頁面標題 |
| `url` | 網址 |
| `fetched` | 是否成功抓到頁面文字 |

**`terms.csv`**：每個關鍵字的詞彙分析

| 欄位 | 內容 |
| --- | --- |
| `keyword` | 關鍵字 |
| `term` | 詞或兩字詞組 |
| `average_tfidf` | 在所有頁面的平均 TF-IDF |
| `max_tfidf` | 在單一頁面的最高 TF-IDF |
| `frequency` | 用到這個詞的頁面比例（%） |

- 只保留至少出現在 2 個頁面的詞。
- 含數字的詞（金額、年份、紅利數字）會被排除。
- 終端機會依 `max_tfidf` 排序，顯示每個關鍵字的前幾名。

## 已知限制

- 有些網站會擋程式抓取，或內容是由 JavaScript 產生的。這些頁面會標成 `fetched=False`，或只抓到很少文字。
- 一個關鍵字至少要有 2 個頁面成功抓到，而且頁面之間有共同詞彙，才會產生分析結果，否則會顯示 skipped。
- 停用詞只有英文，斷詞也是以空白為準，所以不適合分析中文。
- 搜尋途中出錯時（例如 API 額度用完）程式會停止，但已經完成的關鍵字仍會存檔。
