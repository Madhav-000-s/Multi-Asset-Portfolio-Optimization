# Transaction Cost Model for DSR Portfolio Optimizer
# Tracks turnover and applies proportional trading costs

library(xts)

#' Compute Portfolio Turnover
#'
#' Calculate turnover as half the sum of absolute weight changes.
#' Turnover = 0.5 * sum(|w_new - w_old|)
#'
#' This represents the fraction of portfolio traded (one-way).
#'
#' @param old_weights Previous period weights
#' @param new_weights Current period weights
#' @return Turnover fraction (0 to 1)
compute_turnover <- function(old_weights, new_weights) {
  if (is.null(old_weights) || length(old_weights) == 0) {
    # First period - assume starting from cash, full investment
    return(1.0)
  }

  # Ensure same length
  if (length(old_weights) != length(new_weights)) {
    stop("Weight vectors must have same length")
  }

  # Turnover = half of total absolute changes
  # (buying and selling are both counted, so divide by 2)
  turnover <- sum(abs(new_weights - old_weights)) / 2

  return(turnover)
}

#' Compute Transaction Cost for a Single Rebalance
#'
#' @param old_weights Previous weights
#' @param new_weights New weights
#' @param cost_bps Transaction cost in basis points (default 10)
#' @return Transaction cost as fraction of portfolio value
compute_transaction_cost <- function(old_weights, new_weights, cost_bps = 10) {
  turnover <- compute_turnover(old_weights, new_weights)

  # Cost = turnover * 2 * cost_bps (both buy and sell incur cost)
  # Actually: cost = sum(|w_new - w_old|) * cost_bps
  # Since turnover = sum/2, cost = turnover * 2 * cost_bps
  cost <- turnover * 2 * (cost_bps / 10000)

  return(cost)
}

#' Apply Transaction Costs to Portfolio Returns
#'
#' Reduces portfolio returns by transaction costs at each rebalance date.
#'
#' @param returns xts object of daily portfolio returns
#' @param weights_history xts object of weights at rebalance dates
#' @param cost_bps Transaction cost in basis points (default 10)
#' @return xts object with transaction-cost-adjusted returns
apply_transaction_costs <- function(returns, weights_history, cost_bps = 10) {
  if (is.null(weights_history) || nrow(weights_history) == 0) {
    return(returns)
  }

  # Get rebalance dates
  rebalance_dates <- zoo::index(weights_history)

  # Create adjusted returns
  adjusted_returns <- returns

  # Track previous weights
  prev_weights <- NULL

  for (i in seq_along(rebalance_dates)) {
    date <- rebalance_dates[i]

    # Get current weights
    current_weights <- as.numeric(weights_history[date, ])

    # Compute transaction cost
    tc <- compute_transaction_cost(prev_weights, current_weights, cost_bps)

    # Deduct cost from return on rebalance date
    if (as.character(date) %in% as.character(zoo::index(adjusted_returns))) {
      # Find matching date in returns
      date_idx <- which(zoo::index(adjusted_returns) == date)
      if (length(date_idx) > 0) {
        adjusted_returns[date_idx] <- adjusted_returns[date_idx] - tc
      }
    }

    prev_weights <- current_weights
  }

  return(adjusted_returns)
}

#' Compute Turnover Metrics for Full Backtest
#'
#' Calculate summary statistics for portfolio turnover.
#'
#' @param weights_history xts object of weights at rebalance dates
#' @return List with turnover metrics
compute_turnover_metrics <- function(weights_history) {
  if (is.null(weights_history) || nrow(weights_history) < 2) {
    return(list(
      total_turnover = NA,
      avg_turnover = NA,
      max_turnover = NA,
      num_rebalances = 0,
      annualized_turnover = NA
    ))
  }

  n_rebalances <- nrow(weights_history)
  turnovers <- numeric(n_rebalances)

  # First rebalance - full investment
  turnovers[1] <- 1.0

  for (i in 2:n_rebalances) {
    old_w <- as.numeric(weights_history[i - 1, ])
    new_w <- as.numeric(weights_history[i, ])
    turnovers[i] <- compute_turnover(old_w, new_w)
  }

  # Calculate metrics
  total_turnover <- sum(turnovers)
  avg_turnover <- mean(turnovers[-1])  # Exclude first (full investment)
  max_turnover <- max(turnovers[-1])

  # Annualized turnover (assuming 52 weeks per year for weekly rebalancing)
  dates <- zoo::index(weights_history)
  years <- as.numeric(difftime(max(dates), min(dates), units = "days")) / 365.25
  annualized_turnover <- if (years > 0) total_turnover / years else total_turnover

  list(
    total_turnover = total_turnover,
    avg_turnover = avg_turnover,
    max_turnover = max_turnover,
    num_rebalances = n_rebalances,
    annualized_turnover = annualized_turnover,
    turnovers = turnovers
  )
}

#' Compute Total Transaction Costs
#'
#' @param weights_history xts object of weights
#' @param cost_bps Transaction cost in basis points
#' @return Total transaction costs as fraction of portfolio
compute_total_transaction_costs <- function(weights_history, cost_bps = 10) {
  metrics <- compute_turnover_metrics(weights_history)

  # Total cost = total turnover * 2 * cost_bps (both sides)
  total_cost <- metrics$total_turnover * 2 * (cost_bps / 10000)

  return(total_cost)
}

#' Print Transaction Cost Summary
#'
#' @param weights_history xts object of weights
#' @param cost_bps Transaction cost in basis points
print_transaction_cost_summary <- function(weights_history, cost_bps = 10) {
  metrics <- compute_turnover_metrics(weights_history)
  total_cost <- compute_total_transaction_costs(weights_history, cost_bps)

  cat("\nTransaction Cost Summary:\n")
  cat(sprintf("  Rebalances: %d\n", metrics$num_rebalances))
  cat(sprintf("  Total turnover: %.2f (%.0f%%)\n",
              metrics$total_turnover, metrics$total_turnover * 100))
  cat(sprintf("  Avg turnover per rebalance: %.2f%%\n", metrics$avg_turnover * 100))
  cat(sprintf("  Max turnover: %.2f%%\n", metrics$max_turnover * 100))
  cat(sprintf("  Annualized turnover: %.2f\n", metrics$annualized_turnover))
  cat(sprintf("  Total transaction costs: %.2f%% (at %d bps)\n",
              total_cost * 100, cost_bps))
}

#' Compare Returns Before and After Transaction Costs
#'
#' @param gross_returns xts of returns before costs
#' @param net_returns xts of returns after costs
#' @return List with comparison metrics
compare_gross_net_returns <- function(gross_returns, net_returns) {
  # Annualize
  scale <- 252

  gross_ann <- as.numeric(PerformanceAnalytics::Return.annualized(gross_returns, scale = scale))
  net_ann <- as.numeric(PerformanceAnalytics::Return.annualized(net_returns, scale = scale))

  gross_sharpe <- as.numeric(PerformanceAnalytics::SharpeRatio.annualized(gross_returns, scale = scale))
  net_sharpe <- as.numeric(PerformanceAnalytics::SharpeRatio.annualized(net_returns, scale = scale))

  list(
    gross_return = gross_ann,
    net_return = net_ann,
    return_drag = gross_ann - net_ann,
    gross_sharpe = gross_sharpe,
    net_sharpe = net_sharpe,
    sharpe_drag = gross_sharpe - net_sharpe
  )
}
