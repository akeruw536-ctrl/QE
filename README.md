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
3. **Win-probability floor** (`MIN_WIN_PROBABILITY`, default 0.80): the XML's +/-41 points is about 1 sigma, so price finishes inside only ~62-70% of the time. Candidates below the floor are rejected, which pushes the search to wider barriers (higher win rate, lower payout). `0` disables it.
4. **Adaptive barriers** (`BARRIER_SCALES`, `BARRIER_SELECTION`): on every signal, 7 widths (1.0x to 2.5x the XML width) are quoted live in parallel; every one that passes the gates is a candidate and the best is bought. `BARRIER_SELECTION=edge` (default) picks the highest edge, `ev` the highest expected profit per stake. `BARRIER_SCALES=1.0` = XML barriers only.
5. **Volatility-scaled barriers** (`BARRIER_MODE=auto|vol|fixed`): the XML's 41 points only make sense for 1HZ90V (it is ~1 sigma of a 2-minute move there). Other symbols get `BARRIER_VOL_MULTIPLE x sigma x sqrt(minutes)`.
6. **Safety rails**: `STAKING_MAXIMUM_STAKE` (the XML had none), session halt that lasts until 00:00 UTC instead of stopping forever, DB retry/fallback, restart-safe ladder.

With `USE_REGIME_GATE=false USE_EDGE_GATE=false MIN_WIN_PROBABILITY=0 BARRIER_SCALES=1.0 STAKING_MAXIMUM_STAKE=100` it is the plain XML strategy.

## Railway
Variables: see `.env.example`. Start with `DERIV_ACCOUNT_MODE=DEMO`, `SHADOW_MODE=true`, `DB_SCHEMA=qe`.
Logs to look for: `STAKE LADDER`, `tick stream subscribed`, `SIGNAL DOWN`, `TRADE EXECUTED` / `SHADOW TRADE`, `SETTLED`.
`/stats` shows session P&L, loss streak, next stake, win rate vs predicted probability.

## How the bot learns
- **Calibration** (`app/models/calibration.py`): a logistic recalibration with two slopes, `calibrated = sigmoid(a + b_level*m + b_width*(logit(raw)-m) + regime offset)`, where `m` is the mean logit over all widths priced for the signal. Width (within a signal) is reliable, slope ~1; the market level (between signals) is not -- after a calm spell volatility tends to jump -- so a single shared slope flattened the width response and made tight barriers look better than they are. Fitted on all outcomes at once, fitted on *all* outcomes at once, so a systematic over-optimism is learned from tens of outcomes, not hundreds. A ridge prior keeps it at "no correction" until there is evidence; recent outcomes count more (`CALIBRATION_HALFLIFE_HOURS`).
- **Probes** (`LEARN_FROM_REJECTED`): every signal prices several barrier widths but buys one. The others are saved and resolved against the real expiry tick, giving 3-6x more labelled outcomes per hour and removing the bias of learning only from traded setups. Signals that arrive while a trade is open, or after a session limit, are also priced and learned from (they just cannot trade).
- Everything is rebuilt from the database at startup (`CALIBRATION loaded: ...`), so a persistent `DATABASE_URL` keeps what it learned. `/stats` -> `calibration` and the 15-minute `CALIBRATION` log line show the fit and whether it beats the raw model (`brier_calibrated` vs `brier_raw`).

## Reliability
- **Liveness check:** after every connect/reconnect the client pings; a socket that opens but never answers is dropped and re-handshaken (fresh OTP), up to 3 times.
- **Data recovery loop:** a failed tick subscription or empty candle history is retried every 20s instead of crashing the bot; if no symbol ticks for 90s the connection is forced to reconnect.
- **Restart policy:** `railway.toml` uses `ALWAYS`, so Railway never gives up after a few crashes.

## Known limits
- The XML's "candle index 1" is implemented as the last **completed** 60s candle (`CANDLE_REFERENCE=forming` for the live one). Check this against how your XML behaved.
- Tick streaming is new to this client (ER polls). It routes by subscription id with a symbol fallback, but verify `SIGNAL` lines appear on demo.
- An EMA cross has no demonstrated relationship with whether price stays in range. Judge it by `/stats` (win rate vs `avg_implied`) over a few hundred shadow trades.

## Persistent database (keeps your journal and calibration across redeploys)
Without `DATABASE_URL` the bot uses SQLite on Railway's ephemeral disk, which is wiped on every deploy.
1. Add a Postgres database to the Railway project (Railway Postgres is the simplest; Supabase also works).
2. **Before redeploying**, copy the journal out of the running container: `python tools/import_journal.py --print-export-command`, run that inside `railway ssh`, save the BEGIN..END output to `journal.b64`.
3. Import it from your machine: `DATABASE_URL=<postgres PUBLIC url> python tools/import_journal.py journal.b64 --schema qe`.
4. On the QE service set `DATABASE_URL=${{Postgres.DATABASE_URL}}` and `DB_SCHEMA=qe`, then deploy. Open trades are re-attached on start; calibration and stats continue.
