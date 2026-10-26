# Quantum Ends (QE)

Python port of the **QUANTUM ENDS-IN-OUT** Deriv Bot, built on ER's
infrastructure (Deriv client, journal, settlement, restart safety, `/stats`).
Deploys exactly like ER: Dockerfile + `python run.py` on Railway.

## What was ported from the XML

| XML | QE |
|---|---|
| 1HZ90V, Ends Between (EXPIRYRANGE), 2 min | `SYMBOLS`, `DURATION_MINUTES` |
| Buy when EMA(7) crosses **below** EMA(14) on the tick stream, last tick is a fall, last 60s candle is black | `app/strategy/signals.py` (tick-driven, fires on the exact tick) |
| Barriers +41.457 / -41.31 | `FIXED_BARRIERS` (rounded to 2dp: Deriv rejects more on some symbols) |
| Martingale: first stake 1, factor 1.5 (stake x2.5), level 5, start after loss 1, reset after level | `STAKING_MODE=xml_martingale`, ladder 1 / 2.5 / 6.25 / 15.62 / 39.06 / 97.65 / back to 1 |
| Target profit 62, stop loss 200 | `TARGET_PROFIT`, `STOP_LOSS` (day P&L from the journal) |
| One contract at a time | `MAX_CONCURRENT_CONTRACTS=1` |

## What was added (each can be switched off)

1. **Regime gate** (`USE_REGIME_GATE`): an Ends Between pays when price stays put, while an EMA cross is a trend-start signal. Trades only in range-like regimes (`RANGE_REGIMES`).
2. **Monte Carlo + calibration + live proposal + edge gate** (`USE_EDGE_GATE`, `MIN_EDGE`, `MIN_EV`): prices the probability of finishing inside the barriers, compares it with what Deriv charges, and learns from settled trades.
3. **Barrier width search** (`BARRIER_SCALES`): prices a few widths around the XML barriers live and takes the best EV. `1.0` = XML only.
4. **Volatility-scaled barriers** (`BARRIER_MODE=auto|vol|fixed`): the XML's 41 points only make sense for 1HZ90V (it is ~1 sigma of a 2-minute move there). Other symbols get `BARRIER_VOL_MULTIPLE x sigma x sqrt(minutes)`.
5. **Safety rails**: `STAKING_MAXIMUM_STAKE` (the XML had none), session halt that lasts until 00:00 UTC instead of stopping forever, DB retry/fallback, restart-safe ladder.

With `USE_REGIME_GATE=false USE_EDGE_GATE=false BARRIER_SCALES=1.0 STAKING_MAXIMUM_STAKE=100` it is the plain XML strategy.

## Railway
Variables: see `.env.example`. Start with `DERIV_ACCOUNT_MODE=DEMO`, `SHADOW_MODE=true`, `DB_SCHEMA=qe`.
Logs to look for: `STAKE LADDER`, `tick stream subscribed`, `SIGNAL DOWN`, `TRADE EXECUTED` / `SHADOW TRADE`, `SETTLED`.
`/stats` shows session P&L, loss streak, next stake, win rate vs predicted probability.

## Known limits
- The XML's "candle index 1" is implemented as the last **completed** 60s candle (`CANDLE_REFERENCE=forming` for the live one). Check this against how your XML behaved.
- Tick streaming is new to this client (ER polls). It routes by subscription id with a symbol fallback, but verify `SIGNAL` lines appear on demo.
- An EMA cross has no demonstrated relationship with whether price stays in range. Judge it by `/stats` (win rate vs `avg_implied`) over a few hundred shadow trades.
