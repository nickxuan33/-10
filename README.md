# TF-IDF 關鍵字分析

輸入一個關鍵字，程式會抓取 Google 搜尋前幾名的網頁內容，用 TF-IDF 算出這些頁面裡最具代表性的詞彙並輸出成檔案，方便觀察排名靠前的頁面都在寫什麼。

## 環境需求

- Python 3.10 以上
- 網路連線：執行時會連到 Google 搜尋、搜尋結果中的各個網頁，以及 NLTK 的資料下載站

## 安裝

```bash
# 建立並啟用虛擬環境
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux

# 安裝套件
pip install -r requirements.txt
```

NLTK 的英文停用詞（stopwords）會在第一次執行時自動下載，不需要另外安裝。

## 使用方式

```bash
python tf-idf.py
```

搜尋關鍵字直接寫在 `tf-idf.py` 裡（目前是 `'百家樂賺錢'`），出現在 `google_results(...)` 和 `tf_idf_analysis(...)` 兩個呼叫中，換關鍵字時兩處都要改。

## 輸出

執行後會在目前的資料夾產生兩個檔案：

| 檔案 | 內容 |
| --- | --- |
| `google.csv` | Google 搜尋結果的網址，一行一個 |
| `myfile.csv` | 依 `max_tfidf` 排序的前 10 個詞，欄位以空白分隔、沒有標題列 |

`myfile.csv` 的欄位依序為：

- `word`：詞彙
- `average_tfidf`：在所有網頁中的平均 TF-IDF
- `max_tfidf`：在單一網頁中最高的 TF-IDF
- `frequency`：出現這個詞的網頁比例（%）

同樣的結果也會連同欄位名稱印在終端機上。

## 已知限制

- **Google 搜尋結果可能抓不到**：`google_results()` 直接解析 Google 搜尋結果頁的 HTML（`div.ZINbbc`）。Google 改版或擋下非瀏覽器的請求時會抓不到網址，這時 `google.csv` 是空的，分析步驟會出現 `ValueError: empty vocabulary`。成功抓到內容的網頁少於 2 個時同樣會失敗（`min_df=2`）。
- **中文沒有斷詞**：程式使用 scikit-learn 預設的斷詞方式，只會在空白和標點處切開，停用詞也只有英文。分析中文網頁時，兩個標點之間的一整串中文會被當成一個「詞」。如果要正確分析中文內容，需要另外加入中文斷詞（例如 [jieba](https://github.com/fxsjy/jieba)）。
