# global.R - DSR Portfolio Optimizer Dashboard
# Load data, source helpers, define constants

# Required packages
library(shiny)
library(shinydashboard)
library(ggplot2)
library(xts)
library(zoo)
library(PerformanceAnalytics)
library(tidyr)
library(scales)
library(DT)
library(yaml)

# Source existing helper functions (relative to shiny/ directory)
source("../R/metrics.R")
source("../R/report_utils.R")
source("../R/efficient_frontier.R")
source("../R/transaction_costs.R")

# Load pre-computed results ONCE at startup
results <- readRDS("../results/backtest_results.rds")
config <- read_yaml("../config.yaml")

# Extract commonly used data
combined_returns <- results$combined_returns
weights_history <- results$dsr$weights_history
test_period <- results$test_period
tickers <- results$dsr$tickers

# Pre-compute metrics for faster dashboard loading
all_metrics <- compute_all_metrics(combined_returns)

# Define color scheme (consistent with existing plots)
strategy_colors <- c(
"DSR_Constrained" = "#2E86AB",
"DSR_Unconstrained" = "#E8871E",
"EqualWeight" = "#A23B72"
)

asset_colors <- c(
"AAPL" = "#66c2a5",
"MSFT" = "#fc8d62",
"GOOGL" = "#8da0cb",
"AMZN" = "#e78ac3",
"META" = "#a6d854"
)

# Dashboard title
app_title <- "DSR Portfolio Optimizer"

# Helper: Format percentage
fmt_pct <- function(x) sprintf("%.2f%%", x * 100)

# Helper: Format currency
fmt_currency <- function(x) sprintf("$%.2f", x)

# Helper: Format number
fmt_num <- function(x, digits = 3) sprintf(paste0("%.", digits, "f"), x)
