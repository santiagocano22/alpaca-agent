# Investigación de estrategia para ETFs estadounidenses

Fecha de corte: 2026-07-31. Rama analizada: `claude/zen-austin-586c0a`.

## Conclusión ejecutiva

La recomendación provisional para *paper trading* es `breakout_20`: comprar al cierre de una ruptura del máximo de las 20 sesiones anteriores, sólo si el cierre está por encima de SMA 200; la señal se ejecuta en la siguiente apertura. Usa stop de 7%, trailing stop de 10%, salida por cruce bajo EMA 20, 15% máximo por ETF, 75% total y cinco posiciones.

No se recomienda por ser la de mayor retorno. Se seleccionó **antes de mirar la prueba final**, mediante un score de entrenamiento y validación que combina Sharpe, Calmar, actividad y penalización por drawdown/frecuencia fuera de objetivo. En la prueba final completamente fuera de muestra (2024-07-01 a 2026-07-31) obtuvo +5,22%, Sharpe 0,43, drawdown −8,45% y 3,52 entradas/mes. SPY buy-and-hold obtuvo +40,16% en el mismo periodo. La diferencia es un coste de oportunidad material.

La evidencia no permite llamarla “rentable” ni “la mejor”. El resultado es una simulación histórica sobre un universo pequeño, correlacionado y elegido con conocimiento actual. Se propone sólo una prueba operativa de dos a cuatro semanas, sin activación automática y sin tocar trading real.

Selección final:

- Recomendada: `breakout_20`, por equilibrio entre simplicidad, estabilidad local, actividad y drawdown.
- Conservadora: `breakout_50`, con menos entradas (2,34/mes en toda la muestra), menor drawdown (−7,85%) y mejor prueba final (+8,03%) aunque no fue elegida usando ese dato.
- Mayor frecuencia/riesgo observado: `pullback_below_ema20`, 3,11 entradas/mes, 87,17% de días con alguna posición y drawdown −15,30%.

## 1. Auditoría de la implementación

### Capacidades de `StrategyEngine`

El esquema acepta RSI, EMA, SMA, MACD, Bollinger Bands, ATR, VWAP, promedio de volumen, precio OHLC y breakout. Permite `<`, `<=`, `>`, `>=`, `==`, `crosses_above` y `crosses_below`, con grupos AND/OR anidados. Los cruces pueden buscar una transición en las últimas N barras mediante `lookback_bars`.

No puede expresar:

- aritmética entre indicadores, por ejemplo ATR/precio o retorno de 126 días;
- percentiles o z-scores;
- ranking entre símbolos;
- rebalanceo mensual de cartera;
- filtros de régimen de SPY aplicados a otro ETF;
- selección por volatilidad o momentum relativo;
- prioridad cuantitativa cuando llegan más señales simultáneas que plazas.

Por ello, `relative_momentum_126` fue investigada pero está marcada `deployable=false`: requiere un evaluador de cartera que la arquitectura actual no tiene. No se simuló como si fuera ejecutable por el engine.

### Hallazgos del backtester y motor

1. **Breakout imposible, corregido.** El máximo/mínimo móvil incluía la barra actual. Como `close <= high` por definición, `close > breakout.high_n` no podía cumplirse. El nivel ahora usa sólo las N barras anteriores y tiene prueba de regresión.
2. **Ajustes corporativos, corregido para backtest.** `get_bars` no permitía pedir barras ajustadas y el backtester consumía el default `raw`. Esto puede interpretar dividendos/splits como pérdidas o gaps. El cliente acepta ahora un `adjustment` opcional y `src/backtest.py` pide `Adjustment.ALL`; el warmup operativo conserva el default raw.
3. **Look-ahead.** No se encontró ejecución al cierre de la señal. Las señales se forman con la barra diaria completada y los fills se realizan en la siguiente apertura; los tests sintéticos verifican señal, gap y slippage.
4. **Timezone.** Las barras diarias de Alpaca llegan a medianoche de Nueva York (04:00/05:00 UTC). `_bars_frame` convierte a `America/New_York` antes de extraer la fecha, por lo que DST no cambia el día de negociación.
5. **Gaps.** El modelo usa la apertura observada del día siguiente; un gap atraviesa el stop y se ejecuta peor, sin asumir fill al precio teórico del stop.
6. **Stops diarios.** Stop y trailing se evalúan al cierre y se ejecutan en la apertura siguiente. No modelan un stop intradía. Esto es coherente con el loop 1D actual, pero puede subestimar o sobrestimar el resultado frente a una orden stop real.
7. **Trailing stop.** El backtester mantiene el máximo sólo desde la entrada. El runtime también lo hace en memoria, pero `BotState.position_highs` no se persiste: un reinicio con una posición abierta puede olvidar el máximo anterior y aflojar el trailing. Debe validarse en paper antes de live.
8. **Posiciones simultáneas y equity.** Se cierran pendientes antes de abrir nuevas; la valoración usa cierres diarios, efectivo y acciones fraccionarias. Los límites se comprueban antes del fill. Si faltara una barra se conservaría el último mark; no ocurrió en estos ETFs.
9. **Sesgo por orden del universo.** Cuando hay más señales simultáneas que plazas, las estrategias desplegables heredan el orden del universo. No hay ranking de desempate. Puede favorecer SPY/QQQ/IWM/XLK.
10. **Datos incompletos en consulta múltiple.** Una consulta Alpaca para los nueve símbolos omitió parte del universo por la paginación/límite global. El downloader de investigación pide y valida cada ETF por separado.

Pruebas iniciales: 418/418. Pruebas específicas añadidas: breakout anterior, barras ajustadas, next-open con gap/slippage y límite de exposición.

## 2. Diagnóstico de la estrategia anterior

En julio de 2026 hubo 198 combinaciones símbolo-día:

| Condición | Veces cumplida | Tasa |
| --- | ---: | ---: |
| Precio > SMA 200 | 183 | 92,4% |
| EMA 20 > EMA 50 | 167 | 84,3% |
| RSI 14 cruza 40 en una barra | 5 | 2,5% |
| Precio > EMA 20 | 114 | 57,6% |
| AND completo | 0 | 0,0% |

El cruce fue claramente el cuello de botella. Aun cuando apareció cinco veces, nunca coincidió en la misma barra con las otras tres condiciones. En 2020–2026 el baseline sólo produjo cinco entradas: 0,06/mes, 75 de 79 meses sin entradas y 2,72% del tiempo con alguna exposición.

Ampliar el cruce no lo solucionó de forma robusta: lookback 3 dio 0,58 entradas/mes y retorno total −2,86%; lookback 5 dio 0,99 entradas/mes y +5,23%, pero perdió en validación y prueba. La zona RSI 40–55 elevó la actividad a 2,72/mes, pero fue muy sensible al slippage y perdió −1,32% en prueba.

## 3. Metodología

- Fuente: Alpaca Market Data, feed SIP, `adjustment=all`, barras 1D por ETF.
- Datos descargados: 2018-01-02 a 2026-07-31, 2.156 sesiones por símbolo. 2018–2019 se usa para warmup; la evaluación abarca 2020-01-02 a 2026-07-31 (1.653 sesiones).
- Entrenamiento: 2020-01-02 a 2022-12-30.
- Validación: 2023-01-03 a 2024-06-28.
- Prueba final cerrada: 2024-07-01 a 2026-07-31.
- Walk-forward anclado: selección con datos anteriores y aplicación al año siguiente, 2022–2026.
- Capital inicial: USD 100.000; acciones fraccionarias; sin apalancamiento.
- Fill: siguiente apertura, compra con `open*(1+slippage)` y venta con `open*(1-slippage)`.
- Slippage: 5 bps base; estrés 10 y 20 bps.
- Benchmark: SPY ajustado, buy-and-hold desde primera apertura a último cierre.
- Métricas diarias: retorno, CAGR, volatilidad 252 días, Sharpe con rf=0, Sortino, maximum drawdown y Calmar.
- Métricas de trades: profit factor, expectancy porcentual, win rate, ganancias/pérdidas medias, mejor/peor trade, entradas/salidas y turnover.
- Actividad: entradas/mes, meses sin entrada, porcentaje de días con posición y exposición bruta media.
- Selección: `(score_train + 2*score_validation)/3`; el score premia Sharpe/Calmar y actividad y penaliza desviación respecto a 4 entradas/mes y drawdown sobre 25%. No utiliza prueba final.

“Riesgo medio-alto” se trató como **presupuesto**, no como obligación de perder 15–25%: 15% por posición, 75% máximo, cinco posiciones, stop explícito. La recomendada sólo realizó −9,00% de drawdown histórico y 36,65% de exposición bruta media. Por comportamiento realizado es más cercana a riesgo medio; la variante pullback utilizó mejor el presupuesto y alcanzó −15,30%.

## 4. Familias evaluadas y resultados completos

Las cifras son para toda la muestra, 5 bps. `Ret@20` es el retorno total bajo estrés de 20 bps. `Exp` es expectancy por operación; `T.mkt` son días con alguna posición; `Exp.avg` es exposición bruta media. La tabla canónica con todas las columnas y splits está en `research/results/all_metrics.csv`.

| Candidato | Ret% | CAGR% | Vol% | Sharpe | Sortino | MDD% | Calmar | PF | Exp% | Win% | Mejor% | Peor% | E/S | E/mes | Meses 0 | T.mkt% | Exp.avg% | Turnover/año | Ret@20% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_rsi_cross_1 | -1,29 | -0,20 | 0,31 | -0,64 | -0,74 | -1,53 | -0,13 | 0,00 | -2,15 | 0,00 | -0,61 | -3,40 | 5/5 | 0,06 | 75 | 2,72 | 0,33 | 0,18x | -1,46 |
| rsi_cross_3 | -2,86 | -0,44 | 2,36 | -0,18 | -0,23 | -8,23 | -0,05 | 0,69 | -0,40 | 23,91 | 9,29 | -3,66 | 46/46 | 0,58 | 56 | 22,08 | 6,37 | 2,10x | -4,84 |
| rsi_cross_5 | 5,23 | 0,78 | 3,41 | 0,24 | 0,34 | -6,90 | 0,11 | 1,31 | 0,47 | 29,49 | 28,09 | -7,50 | 78/78 | 0,99 | 47 | 37,57 | 12,47 | 3,55x | 1,60 |
| rsi_zone_40_55 | 11,00 | 1,60 | 7,08 | 0,26 | 0,35 | -11,21 | 0,14 | 1,20 | 0,36 | 29,91 | 30,79 | -7,50 | 215/214 | 2,72 | 19 | 79,98 | 38,53 | 9,80x | 0,77 |
| pullback_below_ema20 | 46,45 | 5,97 | 8,61 | 0,72 | 1,02 | -15,30 | 0,39 | 1,81 | 1,14 | 47,52 | 42,81 | -11,42 | 246/242 | 3,11 | 24 | 87,17 | 48,07 | 11,14x | 30,72 |
| pullback_reclaim_ema20 | 17,31 | 2,46 | 6,91 | 0,39 | 0,54 | -10,52 | 0,23 | 1,34 | 0,59 | 34,24 | 39,68 | -7,50 | 185/184 | 2,34 | 21 | 77,43 | 38,51 | 8,42x | 7,65 |
| breakout_20 | 33,32 | 4,47 | 6,43 | 0,71 | 0,95 | -9,00 | 0,50 | 1,60 | 0,86 | 42,59 | 30,53 | -10,41 | 219/216 | 2,77 | 12 | 80,10 | 36,65 | 10,00x | 22,23 |
| breakout_50 | 31,15 | 4,21 | 6,03 | 0,72 | 0,96 | -7,85 | 0,54 | 1,73 | 0,97 | 43,72 | 25,95 | -6,31 | 185/183 | 2,34 | 18 | 73,26 | 32,22 | 8,44x | 20,05 |
| relative_momentum_126* | 27,43 | 3,75 | 9,39 | 0,44 | 0,59 | -17,11 | 0,22 | 1,39 | 1,65 | 44,44 | 49,07 | -18,56 | 93/90 | 1,18 | 21 | 93,22 | 52,19 | 4,20x | 22,70 |
| mean_reversion_rsi_35 | 10,09 | 1,47 | 5,05 | 0,32 | 0,42 | -8,02 | 0,18 | 1,35 | 0,82 | 66,67 | 9,43 | -13,20 | 83/81 | 1,05 | 47 | 33,03 | 10,64 | 3,78x | 6,94 |
| mean_reversion_bbands | 21,71 | 3,03 | 4,98 | 0,63 | 0,90 | -7,22 | 0,42 | 1,95 | 0,88 | 75,66 | 8,74 | -9,49 | 157/152 | 1,99 | 29 | 32,79 | 10,74 | 7,09x | 11,77 |
| simple_trend | 57,79 | 7,18 | 9,72 | 0,76 | 1,06 | -14,04 | 0,51 | 1,86 | 1,34 | 48,35 | 30,72 | -11,45 | 247/242 | 3,13 | 17 | 94,98 | 56,13 | 11,14x | 42,17 |

\* No desplegable por el engine actual.

SPY buy-and-hold retornó +153,32% en toda la muestra. Ningún candidato se acerca a ese retorno absoluto; la comparación relevante es la reducción de exposición, volatilidad y drawdown, no “alpha” demostrada.

## 5. Entrenamiento, validación y prueba final

Cada celda muestra `retorno / MDD / Sharpe / entradas por mes`.

| Candidato | Entrenamiento | Validación | Prueba final |
| --- | --- | --- | --- |
| baseline_rsi_cross_1 | -0,74 / -0,98 / -0,61 / 0,08 | -0,41 / -0,41 / -1,08 / 0,06 | -0,15 / -0,22 / -0,50 / 0,04 |
| rsi_cross_3 | 1,12 / -4,33 / 0,13 / 0,67 | -2,50 / -2,57 / -1,47 / 0,50 | -1,48 / -2,75 / -0,47 / 0,52 |
| rsi_cross_5 | 10,50 / -4,20 / 0,78 / 1,17 | -3,61 / -3,68 / -1,74 / 0,67 | -1,20 / -2,70 / -0,20 / 0,96 |
| rsi_zone_40_55 | 9,30 / -9,39 / 0,44 / 2,36 | 4,45 / -7,64 / 0,47 / 3,33 | -1,32 / -7,59 / -0,06 / 2,96 |
| pullback_below_ema20 | 14,72 / -12,36 / 0,56 / 2,94 | 12,07 / -5,89 / 1,01 / 3,56 | 5,02 / -9,57 / 0,32 / 3,72 |
| pullback_reclaim_ema20 | 6,87 / -8,16 / 0,35 / 2,14 | 8,89 / -6,81 / 0,90 / 2,61 | 3,35 / -5,85 / 0,27 / 2,56 |
| breakout_20 | 11,21 / -9,00 / 0,54 / 2,56 | 12,22 / -3,23 / 1,47 / 2,39 | 5,22 / -8,45 / 0,43 / 3,52 |
| breakout_50 | 9,05 / -5,73 / 0,48 / 2,28 | 9,66 / -4,10 / 1,27 / 2,00 | 8,03 / -7,79 / 0,64 / 2,84 |
| relative_momentum_126* | 5,72 / -14,28 / 0,23 / 1,22 | 11,36 / -7,33 / 0,95 / 0,89 | 6,01 / -10,26 / 0,38 / 1,44 |
| mean_reversion_rsi_35 | 3,97 / -8,02 / 0,26 / 1,00 | 8,57 / -3,16 / 1,48 / 1,06 | -2,47 / -7,04 / -0,21 / 1,12 |
| mean_reversion_bbands | 9,82 / -7,22 / 0,57 / 1,94 | 5,59 / -1,68 / 1,12 / 1,72 | 4,96 / -4,76 / 0,52 / 2,24 |
| simple_trend | 19,25 / -11,92 / 0,62 / 3,00 | 12,23 / -7,01 / 0,95 / 3,72 | 18,50 / -7,05 / 0,91 / 3,20 |

## 6. Walk-forward, sensibilidad y robustez

Selección walk-forward usando sólo datos anteriores:

| Año OOS | Seleccionada previamente | Retorno | MDD | Sharpe | Entradas/mes |
| ---: | --- | ---: | ---: | ---: | ---: |
| 2022 | pullback_below_ema20 | -9,42% | -10,31% | -1,62 | 2,67 |
| 2023 | simple_trend | 6,31% | -7,01% | 0,74 | 3,92 |
| 2024 | breakout_20 | 1,78% | -5,91% | 0,35 | 3,17 |
| 2025 | pullback_below_ema20 | 1,51% | -8,20% | 0,23 | 3,08 |
| 2026 parcial | pullback_below_ema20 | -1,31% | -6,24% | -0,19 | 5,00 |

El walk-forward es mixto y desaconseja extrapolar el promedio histórico. La estrategia recomendada por años produjo +12,95%, +1,68%, −2,61%, +8,15%, +1,78%, −1,13% y +11,00% parcial entre 2020 y 2026.

Sensibilidad local de breakout:

- Grid: lookback 15/20/25, stop 6/7/8%, trailing 9/10/11%; 27 combinaciones.
- Las 27 fueron positivas en entrenamiento, validación y prueba.
- Prueba: retorno +5,22% a +8,02%; Sharpe 0,43 a 0,63; MDD −8,45% a −9,29%; 3,16 a 3,72 entradas/mes.
- El punto exacto 20/7/10 fue el peor retorno del grid en prueba, una señal favorable contra cherry-picking: la familia funciona alrededor, pero también recuerda que no se eligió usando test.
- Slippage 5/10/20 bps para breakout 20: +33,32% / +29,02% / +22,23%; MDD −9,00% / −9,37% / −10,11%.

Por ETF, todas las sleeves de breakout 20 tuvieron expectancy positiva salvo XLE (26 trades, −0,18% medio, P&L realizado −USD 919). QQQ+XLK explicaron 50,3% del P&L realizado: existe concentración tecnológica aunque cada ticker esté limitado a 15%.

Por régimen, los días con SPY sobre SMA 200 acumularon +38,70%; los días con SPY en/bajo SMA 200, −3,88%. El filtro individual no elimina el riesgo de régimen bajista.

Bootstrap de 10.000 remuestreos de operaciones, ponderando cada retorno por el 15% objetivo: retorno terminal p5/mediana/p95 = +8,38%/+30,93%/+59,98%; drawdown mediano −5,23%, percentil 5 adverso −9,36%. Es diagnóstico, no intervalo de confianza: ignora dependencia temporal, trades solapados y correlación entre ETFs.

## 7. Fuentes consultadas

- Alpaca, Historical Bars y ajustes: https://docs.alpaca.markets/us/reference/stockbars
- Alpaca Python SDK, `StockBarsRequest`: https://alpaca.markets/sdks/python/api_reference/data/stock/requests.html
- Faber, *A Quantitative Approach to Tactical Asset Allocation*: https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID2403936_code649342.pdf?abstractid=962461&mirid=1
- Moskowitz, Ooi y Pedersen, *Time Series Momentum*, JFE: https://pages.stern.nyu.edu/~lpederse/papers/TimeSeriesMomentum.pdf
- Hurst, Ooi y Pedersen, *A Century of Evidence on Trend-Following Investing*: https://www.aqr.com/insights/research/journal-article/a-century-of-evidence-on-trend-following-investing
- Jegadeesh y Titman, *Returns to Buying Winners and Selling Losers*: https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1993.tb04702.x
- Brock, Lakonishok y LeBaron, moving averages y trading-range breakouts: https://www.technicalanalysis.org.uk/moving-averages/BrLL92.pdf
- George y Hwang, *The 52-Week High and Momentum Investing*: https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2004.00695.x
- Poterba y Summers, *Mean Reversion in Stock Prices*: https://www.nber.org/papers/w2343

Estas fuentes motivan familias, no parámetros. Los números 20/50/126, RSI y stops se validaron localmente y no se copiaron como promesa de retorno.

## 8. JSON y reproducibilidad

JSONs validados por `Strategy.model_validate()`:

- `strategies/etf_breakout_20_recommended.json`
- `strategies/etf_breakout_50_conservative.json`
- `strategies/etf_pullback_higher_frequency.json`

Comandos exactos:

```bash
git switch claude/zen-austin-586c0a
.venv/bin/python -m research.run_etf_research --download
.venv/bin/python -m research.run_etf_research --run
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from src.strategy.schema import Strategy
for path in Path('strategies').glob('*.json'):
    Strategy.model_validate(json.loads(path.read_text()))
    print('OK', path)
PY
.venv/bin/python -m pytest -q
```

El backtester operativo puede reproducir ventanas de hasta 730 días:

```bash
.venv/bin/python -m src.backtest_cli \
  --strategy strategies/etf_breakout_20_recommended.json \
  --days 730 --initial-cash 100000 --slippage-bps 5
```

Los artefactos numéricos están en `research/results/`: `all_metrics.csv`, `candidate_comparison.csv`, `walk_forward.csv`, `sensitivity_grid.csv`, `recommended_by_symbol.csv`, `recommended_by_regime.csv`, `recommended_by_year.csv`, `recommended_fills.csv`, `bootstrap.json` y `baseline_last_month.json`.

## 9. Resumen para Telegram

```text
🧪 ETF research (2018–2026; OOS 2024-07→2026-07)
Recomendada para paper, NO activada: Breakout 20 + SMA200
Entrada: close > máximo previo 20d y > SMA200; fill próxima apertura
Salida: SL 7%, trailing 10%, cruce bajo EMA20
Riesgo: 15%/ETF, 75% total, máx. 5

OOS: +5.22% | SPY +40.16% | MDD -8.45% | Sharpe 0.43
Actividad OOS: 3.52 entradas/mes | 2/25 meses sin entradas
Full: +33.32%, MDD -9.00%, 2.77 entradas/mes
20 bps: +22.23%, MDD -10.11%
Robustez: 27/27 vecinos positivos en train/val/test

⚠️ No garantía. Bajo SMA200 de SPY perdió; QQQ+XLK = 50% del P&L.
⚠️ Stops diarios ejecutan próxima apertura; gaps pueden exceder 7%.
Siguiente paso: 2–4 semanas paper con criterios operativos, sin live.
```

## 10. Riesgos, limitaciones y condiciones de parada

- Universo retrospectivo y fijo; no elimina survivorship/selection bias. Los ETFs existían en toda la muestra, pero elegirlos hoy sigue usando conocimiento ex post.
- Correlación alta: SPY, QQQ, XLK y XLY no son riesgos independientes. Cinco posiciones pueden caer juntas.
- IEX operativo y SIP histórico no son idénticos. IEX tiene menor cobertura de trades; debe compararse la señal paper con SIP/engine.
- Barras ajustadas aproximan retorno total; no modelan pago/retención de dividendos, impuestos ni lotes fiscales.
- Slippage fijo no sustituye bid-ask, profundidad, retraso de red, rechazos, subastas de apertura ni partial fills.
- Las barras diarias no muestran el orden intradía de high/low. Trailing y stop sólo actúan al cierre/next-open.
- No se modela interés sobre efectivo, comisiones regulatorias, borrow ni impacto. Sólo se opera long.
- El periodo incluye COVID, 2022 y varios años alcistas, pero sigue siendo una muestra corta para estimar colas.
- Detener paper y revisar si: exposición >75%, >5 posiciones, orden duplicada, fill sin señal previa, señal con barra incompleta, discrepancia de indicador >1e-6, pérdida de pico trailing tras reinicio, slippage >20 bps en dos fills, pérdida de una posición >2% del equity por gap, drawdown paper >6%, o tres errores de datos/scheduler en cinco sesiones.
- No considerar live si la lógica pierde en paper por causas no explicadas, aunque el P&L sea positivo.

## 11. Plan de paper trading de dos a cuatro semanas

1. **Semana 1, shadow mode.** Cargar el JSON sólo para cálculo/diagnóstico sin órdenes; comparar diariamente breakout, SMA200, timestamp y señal contra el runner. Probar un reinicio con posición simulada para verificar el trailing peak.
2. **Semanas 2–3, paper orders.** Si shadow tiene cero discrepancias, permitir únicamente cuenta paper. Registrar hora de señal, próxima apertura, precio teórico, fill, slippage, rechazos, exposición y máximo desde entrada.
3. **Semana 4 o extensión.** Repetir hasta observar al menos dos entradas válidas y una salida. Si hay menos de dos oportunidades, extender dos semanas; no fabricar operaciones.
4. **Aprobación operativa.** Cero órdenes reales/duplicadas, 100% señales basadas en barra completa, 100% fills posteriores a la señal, límites respetados, mediana de slippage <=10 bps, ningún fill >20 bps sin explicación de subasta/gap, trailing consistente tras reinicio y cero errores silenciosos.
5. **Aprobación cuantitativa provisional.** Actividad observada compatible con 2–8 entradas/mes o explicada por ausencia real de rupturas; drawdown <=6% en paper; ninguna pérdida >2% del equity por posición; concentración QQQ+XLK <=40% de exposición instantánea.
6. **Decisión.** Paper aprobado sólo habilita más paper. El paso a live requiere una decisión separada, revisión de persistencia del trailing, modelo de órdenes/stops intradía y nueva autorización explícita.
