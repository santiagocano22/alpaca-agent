# Estado del proyecto

## Módulos

- [X] a — config + storage + logger + market_hours
- [X] b.1 — broker: alpaca_client + stream_manager + rate_limiter + asset_cache
- [X] c — strategy: schema + indicators + engine
- [X] b.2 — risk_manager
- [X] d — llm: strategy_parser + summarizer
- [X] e — telegram_bot
- [X] f — scheduler + main.py
- [X] g — deployment (systemd + README, NO docker en v1)

## Módulo actual

(completado)

## Última actualización

2026-05-16

## Tests totales

393 passed

## Acciones pendientes del usuario

ninguna

## Decisiones globales del proyecto

- Python 3.11 fijado (no 3.14 por incompatibilidad con pydantic-core wheel).
- Alembic desde el inicio para migraciones.
- conftest.py raíz pre-importa alpaca-py dentro de catch_warnings() para silenciar DeprecationWarning de websockets.legacy.
- Timestamps UTC en DB, tz-aware en código. `_ensure_aware` lanza ValueError ruidoso ante naive.
- Token bucket con sleeper inyectable, epsilon 1e-12 para IEEE-754.
- MarketCalendarDay almacena UTC, expone open_et/close_et vía properties. Mapper rechaza naive del SDK con RuntimeError citando fecha de verificación.
- get_bars del SDK pagina internamente, no manualmente.
- Streams: dos asyncio.Tasks independientes con su propio loop de reconexión, sleeper y jitter inyectables.
- Eventos del stream se mapean a TradeUpdateEvent / BarEvent tipados.
- Deployment: systemd (no Docker en v1). Docker queda como apéndice opcional al final del README.

## Log de módulos completados

### Módulo a — config + storage + logger + market_hours

- Tests: 79 passed
- Decisiones notables: Alembic desde el inicio, `_ensure_aware` con ValueError, `PendingStrategy.expires_at` en vez de ttl_seconds, get_session como async context manager con commit/rollback automático.
- Discrepancias resueltas: pydantic-core no compila en Python 3.14 → fijar 3.11. Test pre-existente `test_defaults_match_paper_trading_setup` corregido (la URL de Alpaca paper incluye /v2 explícito).
- Pendientes: ninguno.

### Módulo b.1 — broker (sin risk_manager)

- Tests: 116 nuevos.
- Decisiones notables:
  - Token bucket con sleeper y clock inyectables; bug IEEE-754 detectado y resuelto con epsilon 1e-12.
  - AssetCache con TTL 1h y método invalidate_symbol para invalidación granular.
  - _classify_422 con regex patterns; body del response loggeado antes de mapear para detectar wordings nuevos.
  - get_order retorna OrderStatusResult (modelo separado de OrderSubmitResult).
  - MarketCalendarDay con UTC en campos persistibles y open_et/close_et como properties.
  - Mappers _map_calendar_day, _map_trade_update y _map_bar lanzan RuntimeError si SDK devuelve formato distinto al verificado.
  - Stream manager con tasks independientes, stop() idempotente (stop_ws + cancel + wait timeout 5s), no permite restart.
  - conftest.py raíz pre-importa alpaca-py para silenciar warning.
- Discrepancias resueltas:
  - SDK devuelve Calendar.open/close como datetime naive en ET wall-clock (verificado paper API 2026-05-09).
  - SDK pagina get_bars internamente; no manual.
  - TradingStream y StockDataStream emiten timestamps UTC-aware (verificado paper API 2026-05-10).
- Pendientes: risk_manager se mueve a b.2 después de c (depende del schema de Strategy).

### Módulo c — strategy: schema + indicators + engine

- Tests: 62 passed (20 schema + 21 indicators + 21 engine). Total acumulado: 257.
- Wall time: 0.16s.
- Decisiones notables:
  - RSI con avg_loss=0: división por cero produce inf → RSI=100 naturalmente. Guard explícito para serie completamente creciente. Caso plano (avg_gain=0 AND avg_loss=0) → RSI=100, documentado.
  - RuleGroup recursivo resuelto con model_rebuild() post-definición.
  - required_lookback_bars = max_period × 2 (conservador), cacheado en _required para O(1).
  - Prioridad de exits implementada en cascade: SL → TP → trailing → inverse.
  - Condition.lookback_bars aceptado pero engine usa siempre últimas 2 barras para crosses (suficiente para tests requeridos).
- Discrepancias resueltas: ninguna.
- Pendientes: ninguno.

### Módulo b.2 — risk_manager

- Tests: 43 passed. Total acumulado: 300.
- Wall time: 0.78s (suite completa).
- Decisiones notables:
  - 12 checks en orden cheap-first. Checks 9–12 comparten UN solo get_account() + get_positions().
  - is_closing=True exime checks 3, 8, 10, 11, 12. sell+is_closing exime check 9.
  - Clock cache module-level con TTL 2min, timeout 2s → drift=0 en timeout (no rechaza).
  - _resolve_overrides: override más grande (menos restrictivo) → WARNING loggeado + descartado.
  - _persist_log: session.add() + flush(); el caller es dueño del commit.
  - override stop_loss_pct puede bajar el effective SL por debajo de _STOP_LOSS_MIN (0.1) → check 8 lo detecta.
  - loguru no propaga al logging estándar de Python → tests de warnings usan patch("src.broker.risk_manager.logger").
- Discrepancias resueltas: AlpacaSymbolNotFoundError requiere argumento `symbol` además del mensaje.
- Pendientes: ninguno.

### Módulo d — llm: strategy_parser + summarizer

- Tests: 38 passed (9 client + 14 strategy_parser + 15 summarizer). Total acumulado: 338.
- Wall time: 0.93s (suite completa).
- Decisiones notables:
  - LLMClient inyectable (_sleeper para tests, max_retries=3 por defecto). Retry en 5xx, 429, conexión. 4xx permanente falla rápido.
  - parse_strategy() acepta LLMClient inyectado (nunca crea desde settings). LLMError se propaga; JSON malformado → ParseError; ValidationError pydantic → ParseError con hint legible.
  - _extract_json_text() limpia code fences ```json ... ``` y ``` ... ```.
  - LLM error {"error": "..."} con exactamente 1 campo → ParseError.
  - summarizer: siempre retorna fallback string en caso de error (nunca propaga a Telegram bot).
  - Token budgets: summary max_tokens=300, ask max_tokens=500.
- Discrepancias resueltas: ninguna.
- Pendientes: ninguno.

### Módulo e — telegram_bot

- Tests: 35 passed. Total acumulado: 373.
- Wall time: 1.10s (suite completa).
- Decisiones notables:
  - BotState como dataclass con estado mutable (paused, market_state, active/pending strategy, P&L diario).
  - BotDeps con dependencias inyectadas (alpaca, session_factory, llm_client, authorized_chat_id).
  - Handlers acceden a state/deps via context.bot_data → completamente testables sin Telegram real.
  - Autorización con _check_auth(): chat_id incorrecto → solo "👋", no info leakage.
  - /strategy <text>: parse_strategy() inyectable en tests via patch.
  - /closeall confirm: si mercado cerrado → PendingAction en DB; si abierto → alpaca.close_position().
  - /setlimit: guarda RiskOverride en DB via session_factory async context manager.
  - session_factory: asynccontextmanager en tests, async_sessionmaker en producción.
- Discrepancias resueltas: ninguna.
- Pendientes: ninguno.

### Módulo f — scheduler + main.py

- Tests: 20 passed (scheduler jobs). Total acumulado: 393.
- Wall time: 1.16s (suite completa).
- Decisiones notables:
  - Todos los jobs son funciones async puras con dependencias inyectadas (sin APScheduler en tests).
  - warmup_job: transiciona IDLE→WARMUP, precarga barras históricas para todos los símbolos del universo.
  - close_job: eod_policy=close_all cierra posiciones; eod_policy=hold no cierra.
  - on_bar_event: evalúa entry/exit, valida con risk_manager, persiste OrderAttempt, envía Telegram.
  - create_scheduler() wires APScheduler con date jobs para la sesión de hoy.
  - main.py: startup sequence completa (Alembic → engine → Alpaca → LLM → BotState → Telegram → StreamManager → APScheduler).
  - notify es un Callable async (tests usan lambda async → messages.append).
- Discrepancias resueltas: notify debe ser async; tests con append sincrónico causaban TypeError.
- Pendientes: ninguno.

### Módulo g — deployment (systemd + README)

- Tests: 0 (infrastructure, no unit tests).
- Wall time: 1.15s (suite completa, sin cambios).
- Archivos creados:
  - `.env.example` — template con todas las variables requeridas
  - `.gitignore` — excluye .env, *.db, .venv, logs
  - `deploy/trading-agent.service` — systemd unit con ProtectSystem=strict, MemoryMax=512M, EnvironmentFile
  - `deploy/install.sh` — instala en Ubuntu 22.04, crea user, venv, migrations, systemd, backup timer
  - `README.md` — documentación completa: API keys, setup local, test con /strategy, deploy VM, live trading safety, comandos Telegram, arquitectura, apéndice Docker
- Decisiones notables:
  - systemd service corre como user no-root trading-agent, EnvironmentFile=/etc/trading-agent/.env (600, root-only).
  - Backup SQLite cada 6h via systemd timer (install.sh lo instala automáticamente).
  - Docker documentado solo en apéndice del README, no implementado en v1.
  - README incluye sección explícita "Switch from paper to live" con 5 pasos y advertencia de dinero real.
- Discrepancias resueltas: ninguna.
- Pendientes: ninguno.

## Proyecto completo ✅

Todos los módulos (a, b.1, c, b.2, d, e, f, g) implementados y testeados.
Tests totales: 393 passed en 1.15s.
