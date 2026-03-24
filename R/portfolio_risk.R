#!/usr/bin/env Rscript
# portfolio_risk.R
# Called by portfolio_builder.py via subprocess.
#
# Reads:
#   /tmp/pb_returns.csv  — daily log-returns, columns = tickers, index = dates
#   /tmp/pb_weights.csv  — one row: optimal weights from Python (MPT or LSTM)
#
# Writes:
#   /tmp/pb_metrics.csv  — 8 PerformanceAnalytics risk metrics

library(PerformanceAnalytics)
library(xts)

# ── Load inputs ────────────────────────────────────────────────────────────────
returns_raw <- read.csv("/tmp/pb_returns.csv", row.names = 1, check.names = FALSE)
weights_raw <- read.csv("/tmp/pb_weights.csv", check.names = FALSE)

# Build xts return series
dates   <- as.Date(rownames(returns_raw))
ret_xts <- xts(as.matrix(returns_raw), order.by = dates)

# Portfolio weights as numeric vector
w <- as.numeric(weights_raw[1, ])
tickers <- colnames(weights_raw)

# ── Portfolio return series ────────────────────────────────────────────────────
port_ret <- Return.portfolio(ret_xts, weights = w, rebalance_on = NA)

# ── 8 risk metrics ─────────────────────────────────────────────────────────────
ann_return  <- as.numeric(Return.annualized(port_ret, scale = 252))
ann_vol     <- as.numeric(StdDev.annualized(port_ret, scale = 252))
sharpe      <- as.numeric(SharpeRatio.annualized(port_ret, Rf = 0))
sortino     <- as.numeric(SortinoRatio(port_ret))
max_dd      <- as.numeric(maxDrawdown(port_ret))
calmar      <- if (max_dd > 0) ann_return / max_dd else NA
var_95      <- as.numeric(VaR(port_ret, p = 0.95, method = "historical"))
cvar_95     <- as.numeric(ES(port_ret,  p = 0.95, method = "historical"))

metrics <- data.frame(
  Metric  = c("Ann. Return", "Ann. Volatility", "Sharpe Ratio",
              "Sortino Ratio", "Max Drawdown", "Calmar Ratio",
              "VaR (95%)", "CVaR / ES (95%)"),
  Value   = c(ann_return, ann_vol, sharpe, sortino,
              max_dd, calmar, var_95, cvar_95),
  stringsAsFactors = FALSE
)

write.csv(metrics, "/tmp/pb_metrics.csv", row.names = FALSE)
cat("portfolio_risk.R: metrics written to /tmp/pb_metrics.csv\n")
