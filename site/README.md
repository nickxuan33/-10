# 亞伯達線上娛樂城評論網站

用 Python 產生的靜態網站。內容用 Markdown 撰寫，建置時會自動產生所有 SEO 標記，並檢查合規與 SEO 問題。設計採用 theme-factory 的 **Ocean Depths** 主題。

## 快速開始

```bash
cd site
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt

python build.py --drafts                      # 預覽版，輸出到 dist-preview/
python -m http.server 8000 -d dist-preview    # 用瀏覽器打開 http://localhost:8000
```

- **預覽版**（`--drafts`）：包含草稿評論，每頁都設成 noindex，問題只顯示為警告。預覽版**不要**上線。
- **正式版**（`python build.py`）：輸出到 `dist/`，任何錯誤都會讓建置失敗。失敗時，`dist/` 會保持上一次成功的版本，所以不會誤把壞掉的網站部署上去。

## 上線前必須完成

1. **台灣法律風險**：人在台灣推廣國外合法的博弈網站，是否可能觸犯刑法第 268 條、佣金怎麼報稅，都要先請律師確認。
2. **`config.yaml`**：填好品牌名稱、網域（`base_url`）、聯絡信箱，以及作者的真實姓名和簡介。還有預留值時，正式建置不會執行。
3. **「How we review」要符合你實際的做法**：這頁寫明會用真實帳號存款、提款、聯絡客服。你必須真的這樣測試，否則就是不實陳述；做法不同的話，修改 `content/pages/how-we-review.md`。
4. **隱私權政策**：`content/pages/privacy.md` 只是基本範本，請找律師看過。如果之後加入 Google Analytics 之類的工具，要更新這一頁，並處理 cookie 同意。
5. **換成真實評論**：刪除範例業者 `example-casino`，改成你實際測試過的業者（見下方）。
6. **AiGC 標誌**：有些業者要求在他們的標誌旁放 AiGC 官方標誌。向 AiGC 取得官方檔案，放在 `static/img/`，再到 `config.yaml` 設定 `aigc_logo`。
7. **對照合規檢查表**：[`docs/alberta-ad-compliance-checklist.md`](../docs/alberta-ad-compliance-checklist.md)。

## 目錄結構

| 路徑 | 內容 |
| --- | --- |
| `config.yaml` | 網站名稱、網域、作者、選單 |
| `data/operators.yaml` | 業者資料（每個欄位的說明都寫在檔案開頭） |
| `content/home.md` | 首頁標題與介紹 |
| `content/guides/` | 指南文章，一篇一個檔案 |
| `content/reviews/` | 評論內文，檔名要和業者的 `slug` 相同 |
| `content/pages/` | 關於我們、評測方法、聯盟揭露、負責任博弈、聯絡、隱私權 |
| `templates/` | HTML 模板 |
| `assets/site.css` | 樣式，建置時會直接內嵌到每一頁 |
| `static/` | 原樣複製到網站根目錄的檔案，例如 favicon 和圖片 |

## 新增一篇評論

1. 到 AiGC 的[合法網站名單](https://www.abigaming.ca/players/approved-igaming-sites)確認業者在名單上。
2. 在 `data/operators.yaml` 新增一筆資料，`status` 先設成 `draft`。
3. 新增 `content/reviews/<slug>.md`，寫下你實際測試的結果。
4. 用 `python build.py --drafts` 預覽。沒問題後把 `status` 改成 `published`。

正式建置時，業者有以下任一情況都會被擋下：
- `aigc_listed` 不是 `true`
- 超過 35 天沒有重新確認 AiGC 名單（`listing_checked_on`）
- 網址不是 https
- 評分不在 0 到 5 之間

聯盟連結會自動變成 `/go/<slug>/` 的轉址頁，並加上 `rel="sponsored nofollow"`。

## 撰寫指南文章

在 `content/guides/` 新增 `.md` 檔，開頭放 front matter：

```yaml
---
title: How to Set a Deposit Limit       # 頁面標題（H1）
seo_title: ...                          # 選填，<title>，建議 30 到 60 字元
description: ...                        # 必填，搜尋結果摘要，建議 70 到 160 字元
date: 2026-10-01                        # 發布日期
updated: 2026-10-15                     # 選填，更新日期
author: editor                          # config.yaml 裡的作者 id
order: 5                                # 選填，列表排序
faq:                                    # 選填，會產生 FAQ 區塊和 FAQPage 結構化資料
  - q: 問題
    a: 答案
sources:                                # 選填，參考資料（https 連結）
  - title: 來源名稱
    url: https://...
compliance_allow: [bonus]               # 選填，刻意使用的高風險用語（見下方）
---
```

- 內文標題從 `##` 開始，因為 `#`（H1）由模板自動產生。
- 站內連結用 `/` 開頭，例如 `/guides/house-edge-explained/`。
- 可以用 `{{ site.site_name }}`、`{{ site.contact_email }}` 插入設定值。
- 值裡面有「冒號加空格」（`: `）時，整個值要加上雙引號。

## 建置時的自動檢查

**合規**
- 頁面出現高風險用語就會被擋下，例如 make money、income、investment、guaranteed、risk-free、win back、bonus、free spins、no deposit、cashback、promo code。
- 完整清單在 `build.py` 的 `HIGH_RISK_PHRASES`。
- 如果是刻意使用（例如「賭博不是賺錢的方法」），在該頁的 `compliance_allow` 列出對應標籤。

**業者**：上方「新增一篇評論」列出的條件。

**SEO**
- 擋下：缺少標題或描述、兩頁的標題或描述重複、H1 不是剛好一個、圖片沒有 alt、站內斷鏈、相對路徑連結。
- 警告：標題或描述的長度超出建議範圍。

**設定**：`config.yaml` 還有預留值時會擋下。

## 已經內建的 SEO

**技術 SEO**
- 每頁都有唯一的 `<title>`、meta description 和 canonical 網址。
- 語意化 HTML，以及 Open Graph、Twitter 分享標記。
- `sitemap.xml`（含 lastmod）和 `robots.txt`。
- 乾淨網址，以及 404 頁面。
- 安全標頭 `_headers`，Cloudflare Pages 和 Netlify 會自動讀取。

**結構化資料（JSON-LD）**
- 全站：Organization、WebSite、BreadcrumbList。
- 指南：Article，有 FAQ 的頁面另加 FAQPage。
- 評論：Review，含評分。
- 列表頁：CollectionPage 和 ItemList。
- 關於我們、聯絡頁：AboutPage、ContactPage。

**內部連結**：麵包屑、指南目錄錨點、相關指南，以及頁尾的信任頁面。

**聯盟連結**：一律經過 `/go/` 轉址頁。
- 連結本身標記 `rel="sponsored nofollow"`。
- 轉址頁設成 noindex，`robots.txt` 也排除 `/go/`。

**速度**
- 不用 JavaScript。
- CSS 直接內嵌在頁面裡。
- 使用系統字型，不需下載網路字型。

**行動版與無障礙**：響應式版面；文字對比符合 WCAG AA；提供「跳到內容」連結。

**Lighthouse 本機測試結果**：首頁、指南、評論、負責任博弈四頁的效能、無障礙、最佳實務、SEO 都是 100 分。

## 部署

- **任何靜態主機都可以**：把 `dist/` 資料夾上傳即可。
- **Cloudflare Pages 或 Netlify**：可以直接連 GitHub 自動建置。
  - 根目錄設為 `site`
  - 建置指令設為 `pip install -r requirements.txt && python build.py`
  - 輸出資料夾設為 `dist`
- **先看主機的使用條款**：有些主機限制博弈內容。GitHub Pages 的規範不允許以線上商業為主要用途的網站，所以不建議使用。
- **網域**：.ca 網域要符合加拿大居留資格（CIRA 的 Canadian Presence Requirements），人在台灣通常要用 .com 之類的網域。
- **上線後**：
  - 在 Google Search Console 驗證網域，提交 `https://你的網域/sitemap.xml`。
  - Bing Webmaster Tools 也照樣做一次。

## 之後的 SEO 工作

- **找內容缺口**：用專案根目錄的 `tf-idf.py` 分析目標關鍵字的前幾名頁面，找出你的內容還缺哪些主題，例如：
  ```bash
  python ../tf-idf.py "alberta online casino" "legal online casino alberta"
  ```
- **搶新業者上線的時機**：新業者上線時，盡快發布實測評論，這是新網站比較有機會排名的長尾內容。
- **定期更新**：每月確認 AiGC 名單，更新評論和指南的 `updated` 日期。
