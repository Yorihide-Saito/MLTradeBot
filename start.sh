#!/bin/bash
set -e

cd "$(dirname "$0")"

# .env チェック
if [ ! -f .env ]; then
  echo "ERROR: .env が見つかりません。.env.example をコピーして API キーを設定してください。"
  exit 1
fi

if grep -q "YOUR_BITFLYER_API_KEY" .env; then
  echo "ERROR: .env の API キーが設定されていません。"
  exit 1
fi

# モデルファイルチェック
if [ -z "$(ls model_buy/*.xz 2>/dev/null)" ]; then
  echo "ERROR: model_buy/ にモデルがありません。先に再学習を実行してください:"
  echo "  docker compose --profile retrain run --rm retrain"
  exit 1
fi

echo "=== MLTradeBot 起動 ==="
echo "Symbol: $(grep SYMBOL .env | cut -d= -f2)"
echo "Margin: $(grep AVAILABLE_MARGIN .env | cut -d= -f2)"
echo ""

docker compose up bot
