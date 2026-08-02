Quiero que construyas un bot de trading automatizado en Python que opera en Alpaca y se controla vía Telegram. Lee este prompt completo antes de empezar y hazme preguntas si algo no queda claro. NO escribas código todavía: primero propón la estructura de archivos, dependencias con versiones y decisiones de diseño dudosas, y espera mi confirmación.

## Objetivo

Un bot que:

1. Recibe una "estrategia" en lenguaje natural vía Telegram (ej: "compra QQQ cuando el RSI 14 esté bajo 30 en velas de 15min, vende cuando suba sobre 70, máximo 10% del capital por operación, stop-loss 2%").
2. Usa Claude API (claude-sonnet-4-5) UNA SOLA VEZ para parsear esa estrategia a un JSON estructurado con reglas deterministas.
3. Ejecuta esas reglas automáticamente contra datos de Alpaca, sin volver a llamar al LLM en cada decisión.
4. Notifica por Telegram cada operación y manda un resumen diario al cierre de mercado.
5. Permite consultar posiciones, P&L, estrategia activa, pausar/reanudar el bot, todo desde Telegram.
6. Solo opera durante horario de mercados principales (no 24/7).

## Arquitectura y restricciones

- **Lenguaje:** Python 3.11+
- **Inicio:** Paper trading de Alpaca. La URL base se lee de variable de entorno; debe ser trivial cambiar a live después.
- **Deployment:** Primero local, luego una VM. Dockerizable (incluye Dockerfile y docker-compose.yml).
- **Eficiencia de tokens — MUY IMPORTANTE:** Claude API se llama solamente en estos casos:
  (a) Parseo de estrategia nueva (cuando el usuario manda /strategy `<texto>` en Telegram) → claude-sonnet-4-5.
  (b) Resumen diario al cierre de mercado → claude-haiku-4-5.
  (c) Consultas en Telegram que requieran lenguaje natural libre, ej: /ask "¿por qué vendiste TSLA?" → claude-haiku-4-5.
  NUNCA llames al LLM dentro del loop de decisión de trading.
- **El motor de trading es 100% determinista**, basado en el JSON de reglas parseado.
- **Zona horaria:** todo el código interno trabaja en UTC o America/New_York usando `zoneinfo`. Nunca en zona local de la máquina.

## Estructura de archivos sugerida (propónmela y refínala antes de codear)

trading-agent/
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── src/
│   ├──  **init** .py
│   ├── main.py                    # entrypoint, levanta todos los servicios
│   ├── config.py                  # carga .env, settings con pydantic-settings
│   ├── llm/
│   │   ├── client.py              # wrapper de anthropic con control de costos y reintentos
│   │   ├── strategy_parser.py     # parsea NL → JSON usando Sonnet
│   │   └── summarizer.py          # resumen diario y respuestas a /ask con Haiku
│   ├── strategy/
│   │   ├── schema.py              # pydantic models de la estrategia parseada
│   │   ├── engine.py              # evalúa reglas contra datos de mercado
│   │   └── indicators.py          # wrapper de pandas-ta (RSI, EMA, MACD, etc.)
│   ├── broker/
│   │   ├── alpaca_client.py       # wrapper de alpaca-py (REST + stream)
│   │   └── risk_manager.py        # valida tamaño de posición, stop-loss, exposición máx
│   ├── telegram_bot/
│   │   ├── bot.py                 # python-telegram-bot, handlers
│   │   └── commands.py            # /start /status /positions /strategy /pause /resume /ask /closeall /setlimit /confirm /cancel /help
│   ├── scheduler/
│   │   └── jobs.py                # apscheduler: warmup, apertura, cierre, resumen diario, healthcheck
│   ├── storage/
│   │   ├── db.py                  # sqlite con sqlalchemy
│   │   └── models.py              # Trade, StrategyVersion, AlertLog
│   └── utils/
│       ├── logger.py              # loguru
│       └── market_hours.py        # estado de mercado, calendario Alpaca
└── tests/
├── test_strategy_parser.py
├── test_engine.py
├── test_risk_manager.py
└── test_market_hours.py


## Comandos de Telegram requeridos

- `/start` — bienvenida y estado actual.
- `/status` — bot activo/pausado, estado de mercado (IDLE/WARMUP/ACTIVE), próxima apertura/cierre, estrategia vigente, P&L del día, equity.
- `/positions` — posiciones abiertas con P&L unrealized.
- `/strategy` — muestra la estrategia activa en formato legible.
- `/strategy <texto largo>` — reemplaza la estrategia. Antes de aplicar, parsea con LLM y muestra al usuario el JSON resultante para que confirme con `/confirm` o `/cancel`.
- `/pause` y `/resume` — detiene/reanuda la ejecución de nuevas órdenes (no cierra posiciones existentes).
- `/closeall` — cierra todas las posiciones (pide confirmación). Si está fuera de horario, avisa que las órdenes quedarán encoladas para la apertura.
- `/ask <pregunta>` — consulta libre que pasa por Haiku con contexto de operaciones recientes y posiciones.
- `/setlimit <param> <valor>` — overrides puntuales (ej: max_position_pct, stop_loss_pct).
- `/help` — lista comandos.
- Solo responde a un `chat_id` autorizado leído de variable de entorno (TELEGRAM_AUTHORIZED_CHAT_ID). Cualquier otro chat_id recibe un mensaje neutro y se loggea el intento.

## Esquema de la estrategia (pydantic)

Diseña un schema flexible pero estricto. Debe incluir como mínimo:

- `name`: string corto identificador.
- `universe`: lista de tickers o filtros (ej: "ETFs sectoriales US").
- `timeframe`: "1Min" | "5Min" | "15Min" | "1H" | "1D".
- `entry_rules`: lista de condiciones combinables con AND/OR (RSI, EMA cross, MACD, breakout de N días, volumen, etc.).
- `exit_rules`: take_profit_pct, stop_loss_pct, trailing_stop_pct, indicador inverso.
- `position_sizing`: max_position_pct, max_total_exposure_pct, max_concurrent_positions.
- `session`: "regular" | "extended" | "24/7" (este último solo si Alpaca lo permite para el activo).
- `horizon`: "intraday" | "swing" | "position".
- `eod_policy`: "close_all" | "hold". Si el usuario no lo especifica, el parser infiere un default razonable basado en `horizon` (intraday → close_all, swing/position → hold) y lo muestra en la confirmación.

El parser debe rechazar (con mensaje claro al usuario) cualquier cosa que no pueda mapear a este schema, en vez de inventar reglas.

## Loop de trading

- Al iniciar y al cambiar de estrategia: suscribirse al stream de Alpaca para los símbolos del universe.
- En cada barra completa (no en cada tick) del timeframe configurado: evaluar entry_rules y exit_rules.
- Antes de mandar cualquier orden, pasar por risk_manager (chequea exposición, stop loss, capital disponible, órdenes duplicadas).
- Cada orden ejecutada: log a sqlite + alerta a Telegram con detalles (símbolo, lado, qty, precio, motivo en texto generado por la regla, no por LLM).
- Reconexión automática si el WebSocket se cae.

## Horario de operación (IMPORTANTE)

El bot NO opera 24/7. Solo evalúa reglas y manda órdenes durante el horario regular de los mercados principales (NYSE/NASDAQ por defecto: 9:30–16:00 ET, lunes a viernes, excluyendo holidays oficiales).

Requisitos:

- Usa el endpoint `GET /v2/calendar` de Alpaca para obtener los días hábiles y horarios reales (incluye días de cierre temprano como víspera de Thanksgiving). Cachea el calendario diariamente.
- En `src/utils/market_hours.py` implementa funciones puras: `is_market_open(now)`, `next_open(now)`, `next_close(now)`, `minutes_to_close(now)`. Usa zona horaria America/New_York con `zoneinfo`.
- El loop de trading tiene tres estados: IDLE (fuera de horario), WARMUP (5 min antes de apertura, abre stream y precalienta indicadores con datos históricos vía REST), ACTIVE (evaluando reglas).
- **Warmup de indicadores:** durante WARMUP el bot debe descargar las últimas N barras históricas necesarias para que los indicadores configurados estén listos al primer tick (ej: si la estrategia usa EMA-50 en 15min, descargar al menos 50 barras de 15min previas). Esto es crítico, no se debe saltar.
- Apscheduler programa: warmup pre-apertura, transición a active en la apertura, transición a idle en el cierre, resumen diario 5 min post-cierre.
- Pre-market y after-hours desactivados por defecto. Solo se activan si la estrategia parseada incluye `session: "extended"`. Si está activo, el bot opera de 4:00 a 20:00 ET.
- Política de fin de día (`eod_policy`):
  - "close_all": N minutos antes del cierre (configurable, default 5 min) cierra todas las posiciones con órdenes de mercado.
  - "hold": deja las posiciones abiertas durante la noche; solo deja de mandar órdenes nuevas al cierre.
- Notificaciones automáticas de Telegram (texto fijo, sin LLM):
  - Al pasar a WARMUP: "⏰ Mercado abre en 5 min. Preparando estrategia: `<nombre>`."
  - Al pasar a ACTIVE: "🟢 Mercado abierto. Bot operativo."
  - Al pasar a IDLE: "🔴 Mercado cerrado. Generando resumen del día…"
  - En holidays: al inicio del día mandar "📅 Hoy `<fecha>` es día no hábil (`<motivo>`). Bot en standby."
- Los comandos de Telegram siguen respondiendo siempre, incluso fuera de horario. Solo la ejecución de órdenes se pausa. `/status` debe mostrar el estado actual (IDLE/WARMUP/ACTIVE) y la próxima apertura/cierre.

## Resumen diario

- Job de apscheduler que corre 5 min después del cierre del mercado regular US.
- Recopila: trades del día, P&L realizado, P&L no realizado, posiciones abiertas, mejor/peor operación, % de aciertos.
- Pasa esos datos como contexto compacto a Haiku con un prompt template fijo, máximo 500 tokens de input + 300 de output. NO mandes el log completo, solo agregados.
- Envía el texto resultante por Telegram.

## Seguridad y robustez

- Todas las claves en `.env` (Alpaca API key, secret, base_url, Anthropic key, Telegram token, chat_id autorizado).
- Modo "dry run" activable por flag para probar la lógica sin mandar órdenes reales.
- Manejo de errores: si Alpaca o Anthropic fallan, alerta a Telegram, no rompas el proceso. Reintentos con backoff exponencial.
- Logs estructurados en archivo + stdout (loguru).
- Protección contra órdenes duplicadas (idempotencia por client_order_id).
- Rate limiter activo en el cliente de Alpaca.
- Antes de cualquier orden con dinero real (cuando cambie a live), exige confirmación explícita de un flag en .env (LIVE_TRADING_CONFIRMED=true). Si está en false y la base_url apunta a live, el bot se niega a arrancar.

## Lo que necesito que entregues

1. **Primero, antes de codear:** tu propuesta de estructura de archivos, dependencias exactas con versiones y cualquier decisión de diseño donde tengas duda. Pregúntame.
2. **Después de mi confirmación:** implementa por módulos en este orden, mostrándome cada uno y esperando mi visto bueno antes de seguir:
   a. config + storage + logger + market_hours (con sus tests)
   b. broker (alpaca_client + risk_manager) con tests
   c. strategy (schema + indicators + engine) con tests
   d. llm (strategy_parser + summarizer)
   e. telegram_bot
   f. scheduler + main.py
   g. Dockerfile + docker-compose.yml + README con instrucciones de instalación local y en VM (systemd o docker-compose).
3. Un README con: cómo conseguir cada API key, cómo correr en local, cómo probar con `/strategy` de ejemplo, cómo deployar en VM, cómo cambiar de paper a live de forma segura.

Empieza ahora con el paso 1: estructura propuesta y preguntas.
