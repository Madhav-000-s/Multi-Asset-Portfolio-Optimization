# Portfolio Weight Constraints for DSR Portfolio Optimizer
# Applies max/min weight caps, CVaR constraints, and normalization

library(PerformanceAnalytics)

#' Apply Minimum Weight Constraint
#'
#' Zeros out positions below minimum threshold and redistributes.
#'
#' @param weights Named numeric vector of portfolio weights
#' @param min_weight Minimum weight threshold (default 0.02 = 2%)
#' @return Adjusted weights vector
apply_min_weight_constraint <- function(weights, min_weight = 0.02) {
  # Zero out positions below threshold
  weights[weights < min_weight] <- 0

  # Renormalize if any weights remain
  if (sum(weights) > 0) {
    weights <- weights / sum(weights)
  } else {
    # Fallback to equal weight if all zeroed
    weights <- rep(1 / length(weights), length(weights))
    names(weights) <- names(weights)
  }

  return(weights)
}

#' Apply Maximum Weight Constraint
#'
#' Caps positions at maximum and redistributes excess proportionally.
#'
#' @param weights Named numeric vector of portfolio weights
#' @param max_weight Maximum weight per position (default 0.30 = 30%)
#' @return Adjusted weights vector
apply_max_weight_constraint <- function(weights, max_weight = 0.30) {
  # Iterative capping - may need multiple passes
  max_iterations <- 10
  for (i in seq_len(max_iterations)) {
    excess_mask <- weights > max_weight
    if (!any(excess_mask)) break

    # Calculate total excess
    total_excess <- sum(weights[excess_mask] - max_weight)

    # Cap the excess positions
    weights[excess_mask] <- max_weight

    # Redistribute excess to non-capped positions proportionally
    non_capped <- !excess_mask & weights > 0
    if (any(non_capped)) {
      redistribute <- weights[non_capped] / sum(weights[non_capped]) * total_excess
      weights[non_capped] <- weights[non_capped] + redistribute
    }
  }

  # Final normalization to ensure sum = 1
  weights <- weights / sum(weights)

  return(weights)
}

#' Compute Portfolio VaR
#'
#' Calculate Value at Risk using historical simulation.
#'
#' @param weights Portfolio weights vector
#' @param returns xts object of asset returns
#' @param confidence Confidence level (default 0.95)
#' @return VaR as positive number (loss)
compute_portfolio_var <- function(weights, returns, confidence = 0.95) {
  # Compute portfolio returns
  portfolio_returns <- as.numeric(returns %*% weights)

  # VaR is the negative of the quantile at (1 - confidence)
  var <- -quantile(portfolio_returns, probs = 1 - confidence, na.rm = TRUE)

  return(as.numeric(var))
}

#' Compute Portfolio CVaR (Conditional VaR / Expected Shortfall)
#'
#' Calculate CVaR using historical simulation.
#'
#' @param weights Portfolio weights vector
#' @param returns xts object of asset returns
#' @param confidence Confidence level (default 0.95)
#' @return CVaR as positive number (expected loss beyond VaR)
compute_portfolio_cvar <- function(weights, returns, confidence = 0.95) {
  # Compute portfolio returns
  portfolio_returns <- as.numeric(returns %*% weights)

  # VaR threshold
  var_threshold <- quantile(portfolio_returns, probs = 1 - confidence, na.rm = TRUE)

  # CVaR is mean of returns below VaR threshold
  tail_returns <- portfolio_returns[portfolio_returns <= var_threshold]

  if (length(tail_returns) == 0) {
    return(as.numeric(-var_threshold))
  }

  cvar <- -mean(tail_returns)

  return(as.numeric(cvar))
}

#' Simple Weight Optimization with Box Constraints
#'
#' Adjust weights to reduce CVaR while staying within box constraints.
#' Uses a heuristic approach rather than full optimization solver.
#'
#' @param weights Initial weights from LSTM
#' @param returns Historical returns matrix
#' @param max_weight Maximum weight per asset
#' @param target_cvar Target CVaR (optional, NULL means just compute)
#' @return Optimized weights
optimize_weights_simple <- function(
  weights,
  returns,
  max_weight = 0.30,
  target_cvar = NULL
) {
  # Start with constrained weights
  opt_weights <- apply_max_weight_constraint(weights, max_weight)

  # If no target CVaR, just return box-constrained weights
  if (is.null(target_cvar)) {
    return(opt_weights)
  }

  # Simple heuristic: reduce weight on high-volatility assets
  # to lower CVaR if needed
  current_cvar <- compute_portfolio_cvar(opt_weights, returns)

  if (current_cvar <= target_cvar) {
    return(opt_weights)
  }

  # Calculate asset volatilities
  asset_vols <- apply(returns, 2, sd, na.rm = TRUE)

  # Iteratively shift weight from high-vol to low-vol assets
  max_iterations <- 20
  step_size <- 0.02

  for (i in seq_len(max_iterations)) {
    current_cvar <- compute_portfolio_cvar(opt_weights, returns)
    if (current_cvar <= target_cvar) break

    # Find highest and lowest vol assets with positive weight
    positive_mask <- opt_weights > 0.01
    if (sum(positive_mask) < 2) break

    vols_active <- asset_vols
    vols_active[!positive_mask] <- NA

    high_vol_idx <- which.max(vols_active)
    low_vol_idx <- which.min(vols_active)

    if (high_vol_idx == low_vol_idx) break

    # Shift weight
    shift <- min(step_size, opt_weights[high_vol_idx] - 0.01)
    opt_weights[high_vol_idx] <- opt_weights[high_vol_idx] - shift
    opt_weights[low_vol_idx] <- opt_weights[low_vol_idx] + shift

    # Re-apply max constraint
    opt_weights <- apply_max_weight_constraint(opt_weights, max_weight)
  }

  return(opt_weights)
}

#' Main Constraint Pipeline
#'
#' Apply all constraints in sequence: min → max → CVaR optimization.
#'
#' @param raw_weights Raw weights from LSTM model
#' @param returns_history Historical returns for CVaR calculation
#' @param config Configuration list with constraint parameters
#' @return List with constrained weights and constraint info
constrain_weights <- function(raw_weights, returns_history, config) {
  # Extract config parameters with defaults
  min_weight <- config$constraints$min_position_weight %||% 0.02
  max_weight <- config$constraints$max_position_weight %||% 0.30
  use_cvar <- config$constraints$use_cvar_optimization %||% FALSE
  target_cvar <- config$constraints$target_cvar  # Can be NULL
  cvar_confidence <- config$constraints$cvar_confidence %||% 0.95

  # Store original weights
  original_weights <- raw_weights

  # Step 1: Apply minimum weight constraint
  weights <- apply_min_weight_constraint(raw_weights, min_weight)

  # Step 2: Apply maximum weight constraint
  weights <- apply_max_weight_constraint(weights, max_weight)

  # Step 3: CVaR optimization (if enabled and we have history)
  if (use_cvar && !is.null(returns_history) && nrow(returns_history) > 20) {
    weights <- optimize_weights_simple(
      weights,
      returns_history,
      max_weight = max_weight,
      target_cvar = target_cvar
    )
  }

  # Compute metrics for the constrained portfolio
  cvar_value <- NA
  var_value <- NA
  if (!is.null(returns_history) && nrow(returns_history) > 20) {
    cvar_value <- compute_portfolio_cvar(weights, returns_history, cvar_confidence)
    var_value <- compute_portfolio_var(weights, returns_history, cvar_confidence)
  }

  # Return results
  list(
    weights = weights,
    original_weights = original_weights,
    cvar = cvar_value,
    var = var_value,
    max_weight_applied = max(weights),
    min_weight_applied = min(weights[weights > 0]),
    num_positions = sum(weights > 0)
  )
}

#' Check if Weights are Valid
#'
#' @param weights Weight vector
#' @param tolerance Numerical tolerance for sum check
#' @return TRUE if valid, FALSE otherwise
is_valid_portfolio <- function(weights, tolerance = 1e-6) {
  # Check sum to 1
  if (abs(sum(weights) - 1) > tolerance) return(FALSE)

  # Check all non-negative (long-only)
  if (any(weights < -tolerance)) return(FALSE)

  return(TRUE)
}

#' Null coalescing operator
#'
#' @param x Value to check
#' @param y Default value if x is NULL
#' @return x if not NULL, otherwise y
`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}

#' Print Constraint Summary
#'
#' @param constraint_result Result from constrain_weights()
print_constraint_summary <- function(constraint_result) {
  cat("Constraint Summary:\n")
  cat(sprintf("  Positions: %d\n", constraint_result$num_positions))
  cat(sprintf("  Max weight: %.2f%%\n", constraint_result$max_weight_applied * 100))
  cat(sprintf("  Min weight: %.2f%%\n", constraint_result$min_weight_applied * 100))
  if (!is.na(constraint_result$cvar)) {
    cat(sprintf("  CVaR@95%%: %.2f%%\n", constraint_result$cvar * 100))
    cat(sprintf("  VaR@95%%: %.2f%%\n", constraint_result$var * 100))
  }
}
