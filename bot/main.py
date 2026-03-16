"""
bot/main.py — Trading Bot Composition Root

取引ボットの起動エントリーポイント。
このファイルだけが全レイヤーを import して依存関係を組み立てる。

起動方法:
  python -m bot.main
  docker compose up bot
"""
from __future__ import annotations

from typing import List

from loguru import logger

from src.application.services.bot_agent import BotAgent
from src.application.services.bot_orchestrator import BotOrchestrator
from src.application.services.lot_allocator import LotAllocator
from src.application.services.maintenance_scheduler import MaintenanceScheduler
from src.config.logging_config import configure_logging
from src.config.settings import Settings
from src.infrastructure.exchange.bitflyer.bitflyer_auth import BitFlyerAuthenticator
from src.infrastructure.exchange.bitflyer.bitflyer_data_provider import BitFlyerDataProvider
from src.infrastructure.exchange.bitflyer.bitflyer_exchange_adapter import BitFlyerExchangeAdapter
from src.infrastructure.exchange.bitflyer.bitflyer_http_client import BitFlyerHttpClient
from src.infrastructure.feature_engineering.feature_selector import FeatureSelector
from src.infrastructure.feature_engineering.talib_feature_calculator import TALibFeatureCalculator
from src.infrastructure.ml.lightgbm_signal_generator import LightGBMSignalGenerator
from src.infrastructure.persistence.bitflyer_ohlcv_repository import BitFlyerOHLCVRepository
from src.infrastructure.persistence.joblib_model_repository import JoblibModelRepository
from src.infrastructure.persistence.pickle_state_repository import PickleStateRepository


def build_application(settings: Settings) -> BotOrchestrator:
    """全依存関係を BitFlyer のみで組み立てる。"""

    # 1. BitFlyer HTTP クライアント
    auth = BitFlyerAuthenticator(settings.bitflyer_api_key, settings.bitflyer_secret_key)
    http = BitFlyerHttpClient(auth)

    # 2. Exchange adapter
    exchange = BitFlyerExchangeAdapter(http)

    # 3. OHLCV: WebSocket でリアルタイム収集 + REST で補完
    data_provider = BitFlyerDataProvider(
        http=http,
        data_dir=settings.data_dir,
        product_code=settings.symbol,
        interval_minutes=settings.candle_interval_minutes,
    )
    data_provider.start_websocket()
    ohlcv_repo = BitFlyerOHLCVRepository(data_provider)

    # 4. State / Model repositories
    state_repo = PickleStateRepository(settings.cache_dir)
    model_repo = JoblibModelRepository(settings.model_buy_dir, settings.model_sell_dir)

    # 5. Feature engineering
    feature_calc = TALibFeatureCalculator()
    feature_sel = FeatureSelector(settings.feature_pkl_path)
    feature_names = feature_sel.load_feature_names()
    logger.info(f"Loaded {len(feature_names)} feature names")

    # 6. BotAgent をモデルペアの数だけ生成
    agents: List[BotAgent] = []
    for bot_id in model_repo.list_bot_ids():
        model_buy, model_sell = model_repo.load_model_pair(bot_id)
        atr_coeff = model_repo.get_atr_coeff(bot_id)
        agent = BotAgent(
            bot_id=bot_id,
            atr_coeff=atr_coeff,
            lot=0.01,  # 起動直後に update_lots() で上書きされる
            atr_period=settings.atr_period,
            symbol=settings.symbol,
            pips=settings.pips,
            exchange=exchange,
            signal_gen_buy=LightGBMSignalGenerator(model_buy),
            signal_gen_sell=LightGBMSignalGenerator(model_sell),
            state_repo=state_repo,
            feature_names=feature_names,
        )
        agents.append(agent)
        logger.info(f"BotAgent: {bot_id} (atr_coeff={atr_coeff})")

    logger.info(f"Total bots: {len(agents)}")

    return BotOrchestrator(
        agents=agents,
        exchange=exchange,
        ohlcv_repo=ohlcv_repo,
        feature_calculator=feature_calc,
        feature_selector=feature_sel,
        lot_allocator=LotAllocator(settings.available_margin),
        maintenance_scheduler=MaintenanceScheduler(),
        symbol=settings.symbol,
        interval_minutes=settings.candle_interval_minutes,
    )


def main() -> None:
    settings = Settings()
    configure_logging(level=settings.log_level, debug=settings.debug)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting bot: symbol={settings.symbol}")
    build_application(settings).run()


if __name__ == "__main__":
    main()
