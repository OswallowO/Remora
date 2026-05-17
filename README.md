# REMORA — 台灣當沖量化交易平台

> **目前版本:v2.0.3**  ·  上一個正式版:v1.9.13  ·  涵蓋 1.9.14 → 2.0.3 全部變動

REMORA 是一套針對台股當沖設計的量化交易平台,提供 **即時量能監控、族群連動分析、自動進場與停損、回測與策略最佳化、跨平台推播** 等完整工具鏈。v2.0.3 為自 v1.9.13 以來首次正式釋出的新版本,涵蓋兩年的開發成果,包含完整的 UI 重構、雙券商整合、自動更新機制、伺服器端 Discord/Web Portal 推播,以及經過 v2.0.4 大規模搜尋驗證的新策略配置。

---

## 功能總覽

### 交易與選股
- 即時量能監控與族群共振進場(支援漲停、拉高、高到低反轉 三條觸發路徑)
- 動態族群過濾(時序前進統計,自動排除近期勝率低的族群)
- 完整風控:每日進場上限、停損次數熔斷、大盤防線、止損到漲停 tick 防嘎漲停
- 觸價委託單自動掛單與雙路徑備援(SDK + 本地)

### 介面與體驗(v2.0.0+)
- TradingView 風格單視窗多分頁(Ctrl + 1 至 9 切分頁、Ctrl + B 開回測、Ctrl + L 開即時 K 線)
- 三模式外觀(淺色 / 深色 / 自動依時段切換),1.5 秒漸變動畫
- iOS 風切換按鈕、帶值滑桿、自繪 K 線與標記
- 通知中心(右鍵增刪改 / 降冪排序 / 持久化註解)
- 盤後分析 2×2 卡片(最適相似度門檻、計算平均過高、族群連動分析、利潤矩陣優化)
- 績效與風控合併 Dashboard v2,每 5 秒自動刷新

### 自動更新與安裝
- 安裝程式可選擇「最新版」或「指定版本」,直接從 GitHub Releases 取得
- 每次啟動背景檢查更新,有新版會彈出對話框
- 用戶端與伺服器端互相綁定:更新用戶端必一起更新伺服器端

### 外部服務(伺服器端,可選)
- Discord Bot(16 個斜線指令:查狀態、查持倉、查損益等)
- Flask Web Portal(管理員後台、訂閱管理、推播範本)
- Telegram Bot(11 個斜線指令,含每小時彙總)
- Cloudflare Tunnel(自動下載,免設定即可對外發布 Web Portal)

---

## 系統需求

- **作業系統**:Windows 10 / 11(x64)
- **券商帳號**(擇一或皆有):
  - 永豐金 Shioaji API(實際下單通道)
  - 玉山 Esun API(看盤數據源)
- 約 1 GB 磁碟空間(含 Python runtime 與相依套件)
- **不需要** 預先安裝 Python — 安裝程式已包含完整環境

---

## 安裝方式

1. 到 [Releases](https://github.com/OswallowO/Remora/releases) 下載最新的 `Remora_Suite_Setup_v2.0.3.exe`(約 195 MB)
2. 雙擊執行安裝程式
3. 輸入安裝密碼(請聯絡管理員取得)
4. 選擇要安裝的元件:
   - **用戶端**:主交易與回測程式
   - **伺服器端**:Discord Bot + Web Portal(可選)
   - **完整安裝**:兩者皆裝(建議)
5. 選擇版本來源(預設「最新版 from GitHub」,可改「指定版本」或「本機附帶」)
6. 安裝完成後首次啟動會跳「設定精靈」,填入:
   - 玉山 API Key / Secret / 帳號 / 憑證
   - 永豐 Shioaji 模擬與正式 API Key / Secret / CA 憑證
   - 憑證會自動複製到程式安裝目錄,無需手動拉檔

---

## 啟動方式

- **只開用戶端** → 雙擊桌面「Remora 用戶端」捷徑
- **同時開伺服器端 + 用戶端** → 雙擊桌面「Remora 伺服器端」捷徑(會自動順帶開用戶端)
- 要停止伺服器端 → 直接關閉伺服器端 console 視窗

---

## 更新方式

- 啟動時程式會自動檢查 GitHub 是否有新版,有的話彈對話框詢問
- 確認更新後會自動下載、替換 exe、重啟,使用者完全不需手動操作
- 更新時用戶資料(quant_data.db / config.ini / 匯出設定)完整保留

---

## 資料儲存位置

- 程式檔:`C:\Program Files\Remora\`(或安裝時自選的路徑)
- 用戶資料:`%APPDATA%\Remora\`(quant_data.db、log、設定匯出、籌碼快取)
- 憑證:程式安裝目錄根(設定精靈會自動複製)

---

## 授權

商業機密,不對外開源。安裝程式僅提供給授權使用者。

---

## Screenshots

<img width="1919" height="988" alt="主介面" src="https://github.com/user-attachments/assets/29bf3ec0-0929-4482-8ce9-6aa665a31130" />
<img width="1885" height="400" alt="盤後分析" src="https://github.com/user-attachments/assets/ccef3100-b3e0-4029-85ee-11667032da25" />
<img width="1856" height="381" alt="儀表板" src="https://github.com/user-attachments/assets/660bf3cb-7e75-4bab-88d5-98e00d94859b" />
<img width="1373" height="883" alt="K 線視窗" src="https://github.com/user-attachments/assets/92aed3d5-7818-4de5-ba36-eb1e10576bf8" />
