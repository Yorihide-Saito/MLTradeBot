# MLTradeBot

BitFlyer Lightning を対象とした BTC/JPY 自動売買ボット。
LightGBM + TA-Lib テクニカル指標によるシグナル生成と、Optuna Walk-forward CV による定期再学習を組み合わせたシステム。

---

## 特徴

- **クリーンアーキテクチャ (Hexagonal)** — 取引所・モデル・状態管理をすべてインターフェース越しに注入。テストが書きやすく、取引所の差し替えが容易
- **BitFlyer 専用** — FX_BTC_JPY (Lightning FX) をターゲット。WebSocket でリアルタイム OHLCV を構築
- **再学習パイプライン** — Binance の長期データ (BTCUSDT × USDJPY 換算) を使って学習データ不足を解決
- **複数 bot 同時稼働** — bot_id ごとに異なる ATR 係数でエントリー閾値を変え、リスク分散

---

## ディレクトリ構成

```
MLTradeBot/
├── bot/                          # 取引ボット
│   ├── src/
│   │   └── main.py               # Composition Root (起動エントリーポイント)
│   └── unit_test/                # ボット・インフラ層の単体テスト
│       ├── conftest.py
│       ├── test_auth.py
│       ├── test_bot_agent.py
│       ├── test_lot_allocator.py
│       └── test_maintenance_scheduler.py
│
├── training/                     # 再学習パイプライン (ボットと独立)
│   ├── src/
│   │   ├── main.py               # 再学習 CLI
│   │   ├── pipeline.py           # パイプライン統合
│   │   ├── binance_data_fetcher.py  # Binance BTCUSDT × USDJPY → BTCJPY
│   │   ├── data_fetcher.py       # BitFlyer REST (最大 31 日)
│   │   ├── label_generator.py    # ATR ベースラベル生成
│   │   ├── model_trainer.py      # Optuna + Walk-forward CV
│   │   └── model_evaluator.py    # 評価・JSON レポート出力
│   └── unit_test/
│       └── test_label_generator.py
│
├── back_test/                    # バックテスト (将来実装)
│   ├── src/
│   └── unit_test/
│
├── src/                          # 共有レイヤー
│   ├── domain/
│   │   ├── entities/             # Order, Execution, Position (frozen dataclass)
│   │   └── ports/                # ExchangePort, SignalGeneratorPort, ... (ABC)
│   ├── application/
│   │   └── services/             # BotAgent, BotOrchestrator, LotAllocator, MaintenanceScheduler
│   ├── infrastructure/
│   │   ├── exchange/bitflyer/    # 認証・HTTP・ExchangeAdapter・DataProvider
│   │   ├── exchange/gmo/         # GMO アダプタ (参考実装、非使用)
│   │   ├── feature_engineering/  # TALibFeatureCalculator, FeatureSelector
│   │   ├── ml/                   # LightGBMSignalGenerator
│   │   └── persistence/          # PickleStateRepository, JoblibModelRepository
│   └── config/                   # pydantic-settings
│
├── model_buy/                    # buy_<bot_id>.xz (.gitkeep のみ追跡)
├── model_sell/                   # sell_<bot_id>.xz (.gitkeep のみ追跡)
├── docker/Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── .env.example
└── .gitignore
```

**レイヤー依存ルール**: `domain` ← `application` ← `infrastructure` ← `bot/src` / `training/src` / `back_test/src`

---

## 必要なもの

| ソフトウェア | バージョン |
|---|---|
| Python | 3.12 以上 |
| TA-Lib C ライブラリ | 0.4.0 |
| Docker / Docker Compose | (推奨) |

BitFlyer の API キーは [Lightning 開発者ページ](https://lightning.bitflyer.com/developer) から取得してください。
必要な権限: **取引**, **残高確認**, **注文照会**

---

## セットアップ

### 1. リポジトリをクローン

```bash
git clone git@github.com:Yorihide-Saito/MLTradeBot.git
cd MLTradeBot
```

### 2. 環境変数を設定

```bash
cp .env.example .env
# .env を編集して API キーを入力
```

`.env` の主な設定項目:

```dotenv
BITFLYER_API_KEY=your_key
BITFLYER_SECRET_KEY=your_secret

SYMBOL=FX_BTC_JPY          # FX証拠金 (推奨) or BTC_JPY (スポット)
AVAILABLE_MARGIN=0.5        # 証拠金の何割をボットに使うか (0.0〜0.85)
CANDLE_INTERVAL_MINUTES=15
ATR_PERIOD=14
PIPS=1
```

### 3. モデルと特徴量リストを配置

```
model_buy/   buy_<bot_id>.xz    (例: buy_0p186.xz)
model_sell/  sell_<bot_id>.xz   (例: sell_0p186.xz)
features/    features_default.pkl
```

モデルがない場合は先に「再学習」を実行してください。

---

## ボット起動

### Docker (推奨)

```bash
docker compose up bot          # 起動
docker compose up -d bot       # バックグラウンド
docker compose logs -f bot     # ログ確認
docker compose stop bot        # 停止
```

### ローカル

```bash
# TA-Lib C ライブラリのインストール (macOS)
brew install ta-lib

# TA-Lib C ライブラリのインストール (Ubuntu / WSL2)
wget https://sourceforge.net/projects/ta-lib/files/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz && cd ta-lib && ./configure --prefix=/usr && make && sudo make install

pip install -e ".[dev]"

python -m bot.src.main
```

---

## 再学習

**BitFlyer REST は過去 31 日分しか取得できない**ため、長期データには Binance を使います。

### パターン 1: Binance (推奨)

Binance BTCUSDT を USDJPY 換算して学習データを生成します。認証不要。

```bash
# 全 bot を 1 年分のデータで再学習
python -m training.src.main --binance --days 365 --trials 100

# 特定 bot のみ
python -m training.src.main --binance --days 365 --bot-id 0p186 --trials 50

# 2 年分 (最大; yfinance の 1h 足制限)
python -m training.src.main --binance --days 730 --trials 100
```

Docker:

```bash
# デフォルトで --binance --days 365 --trials 100
docker compose --profile retrain run retrain

# オプション変更
docker compose --profile retrain run retrain \
  python -m training.src.main --binance --days 730 --trials 200
```

### パターン 2: CSV ブートストラップ (BitFlyer データ)

[cryptodatadownload.com](https://www.cryptodatadownload.com/data/bitflyer/) から BitFlyer の CSV をダウンロード後:

```bash
python -m training.src.main --csv /path/to/BitFlyer_BTCJPY_1h.csv --trials 100
```

### パターン 3: BitFlyer REST のみ (直近 30 日)

データが少なく過学習しやすいため、動作確認用途のみ推奨。

```bash
python -m training.src.main --trials 50
```

### 再学習オプション一覧

| オプション | デフォルト | 説明 |
|---|---|---|
| `--binance` | False | Binance + USDJPY 換算モードを有効化 |
| `--days N` | 365 | 取得日数 (Binance: 最大 730, BitFlyer: 最大 30) |
| `--trials N` | 50 | Optuna ハイパラ探索の試行回数 |
| `--bot-id ID` | (全bot) | 特定 bot のみ再学習 |
| `--csv PATH` | None | BitFlyer CSV ファイルパス |

評価レポートは `work/eval/eval_<bot_id>_<date>.json` に出力されます。

---

## モデルファイルの命名規則

```
model_buy/buy_<bot_id>.xz
model_sell/sell_<bot_id>.xz
```

`bot_id` は ATR 係数を `p` で小数点を表した文字列。

| bot_id | ATR 係数 |
|---|---|
| `0p186` | 0.186 |
| `0p5` | 0.5 |
| `1p0` | 1.0 |

複数の bot_id を用意することで、異なる閾値の bot を同時稼働できます。

---

## 特徴量 pkl ファイル

`features/features_default.pkl` に使用する特徴量名のリスト (`List[str]`) が保存されています。

```python
import pickle
with open("features/features_default.pkl", "rb") as f:
    feature_names = pickle.load(f)  # List[str]
```

**モデルと pkl は必ずペアで管理してください。** pkl を更新したら必ず再学習が必要です。

---

## 再学習パイプラインの仕組み

```
データ取得
  Binance BTCUSDT × USDJPY (推奨, 最大 730 日)
  or BitFlyer REST (最大 30 日)
  or CSV (cryptodatadownload.com 形式)
        ↓
TALibFeatureCalculator  (~50 テクニカル指標: RSI, MACD, BBands, ATR, ...)
        ↓
LabelGenerator          (ATR × coeff 以上の値動き → y_buy / y_sell)
        ↓
ModelTrainer            (Optuna TPE × TimeSeriesSplit 5-fold Walk-forward CV)
        ↓
ModelEvaluator          (precision / recall / profit_factor / Sharpe)
        ↓
JoblibModelRepository   (model_buy/buy_<bot_id>.xz として保存)
```

---

## 開発

```bash
pip install -e ".[dev]"

# テスト (pyproject.toml で testpaths 設定済み)
pytest
pytest bot/unit_test/test_bot_agent.py  # 単ファイル
pytest -x                               # 最初の失敗で停止

# Lint
ruff check src/ bot/ training/ back_test/
ruff format src/ bot/ training/ back_test/

# 型チェック
mypy src/ bot/ training/ back_test/
```

---

## 動作環境

| 項目 | 値 |
|---|---|
| Python | 3.12 |
| 取引所 | BitFlyer Lightning (FX_BTC_JPY) |
| 時間足 | 15 分 |
| モデル | LightGBM (binary classification) |
| ハイパラ探索 | Optuna TPESampler |
| 交差検証 | TimeSeriesSplit (Walk-forward, n_splits=5) |
| テクニカル指標 | TA-Lib (~50 特徴量) |
