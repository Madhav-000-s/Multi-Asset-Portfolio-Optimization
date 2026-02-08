# Walk-Forward Backtesting for DSR Portfolio Optimizer
# Calls Python predict.py via reticulate to get portfolio weights
# Supports constrained and unconstrained portfolios

library(yaml)

# Get script directory robustly
get_script_dir <- function() {
  # Try different methods to find script location
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("--file=", args, value = TRUE)

  if (length(file_arg) > 0) {
    # Running via Rscript --file=
    return(dirname(normalizePath(sub("--file=", "", file_arg))))
  }

  # Try sys.frame for sourced files
  tryCatch({
    frame_file <- sys.frame(1)$ofile
    if (!is.null(frame_file)) {
      return(dirname(normalizePath(frame_file)))
    }
  }, error = function(e) NULL)

  # Fallback: assume we're in project root
  if (file.exists("R/utils.R")) {
    return("R")
  }

  # Last resort
  return(".")
}

script_dir <- get_script_dir()
source(file.path(script_dir, "utils.R"))
source(file.path(script_dir, "constraints.R"))
source(file.path(script_dir, "transaction_costs.R"))

# Null coalescing operator (if not already defined)
if (!exists("%||%")) {
  `%||%` <- function(x, y) if (is.null(x)) y else x
}

#' Get Project Root
#'
#' @return Project root directory path
get_project_root <- function() {
  # Try to find project root from current working directory
  if (file.exists("config.yaml")) {
    return(normalizePath("."))
  }

  # Try parent of R directory
  if (file.exists("../config.yaml")) {
    return(normalizePath(".."))
  }

  # Fallback
  return(getwd())
}

#' Initialize Backtest Environment
#'
#' Loads Python environment and sources predict module.
#'
#' @return Invisible NULL
init_backtest <- function() {
  # Initialize Python environment
  init_python_env("dsr-portfolio")

  # Source Python predict module into global environment
  project_root <- get_project_root()
  reticulate::source_python(
    file.path(project_root, "python/predict.py"),
    envir = globalenv()
  )

  message("Backtest environment initialized")
  invisible(NULL)
}

#' Load Configuration
#'
#' @return Configuration list
load_backtest_config <- function() {
  project_root <- get_project_root()
  config_path <- file.path(project_root, "config.yaml")
  yaml::read_yaml(config_path)
}

#' Load Price Data for Backtesting
#'
#' Reads parquet file and converts to xts format.
#'
#' @return xts object with Close prices for each asset
load_prices_for_backtest <- function() {
  library(arrow)
  library(xts)

  project_root <- get_project_root()
  prices_path <- file.path(project_root, "data/raw/prices.parquet")

  # Read parquet
  df <- arrow::read_parquet(prices_path)

  # Convert to data.frame if tibble
  df <- as.data.frame(df)

  # Find date column (could be index or named column)
  if ("Date" %in% names(df)) {
    dates <- as.Date(df$Date)
    df$Date <- NULL
  } else if ("...1" %in% names(df)) {
    dates <- as.Date(df$`...1`)
    df$`...1` <- NULL
  } else {
    # Try to use row names
    dates <- as.Date(rownames(df))
  }

  # Extract Close prices only
  close_cols <- grep("_Close$", names(df), value = TRUE)

  if (length(close_cols) == 0) {
    stop("No Close price columns found in data")
  }

  # Create matrix of close prices
  price_matrix <- as.matrix(df[, close_cols])

  # Clean column names (remove _Close suffix)
  colnames(price_matrix) <- gsub("_Close$", "", colnames(price_matrix))

  # Create xts object
  prices_xts <- xts::xts(price_matrix, order.by = dates)

  return(prices_xts)
}

#' Compute Simple Returns
#'
#' @param prices xts object with prices
#' @return xts object with simple returns
compute_returns <- function(prices) {
  # Simple returns: (P_t - P_{t-1}) / P_{t-1}
  returns <- diff(prices) / stats::lag(prices, 1)
  return(returns)
}

#' Get Rebalance Dates
#'
#' Generate dates for portfolio rebalancing.
#'
#' @param dates Vector of all trading dates
#' @param frequency "daily", "weekly", or "monthly"
#' @return Vector of rebalance dates
get_rebalance_dates <- function(dates, frequency = "weekly") {
  library(xts)

  # Create temporary xts for endpoint calculation
  temp_xts <- xts::xts(rep(1, length(dates)), order.by = dates)

  if (frequency == "daily") {
    return(dates)
  } else if (frequency == "weekly") {
    # Last trading day of each week
    ep <- xts::endpoints(temp_xts, on = "weeks")
    ep <- ep[ep > 0]  # Remove first 0
    return(dates[ep])
  } else if (frequency == "monthly") {
    ep <- xts::endpoints(temp_xts, on = "months")
    ep <- ep[ep > 0]
    return(dates[ep])
  }
  stop("Invalid frequency. Use 'daily', 'weekly', or 'monthly'")
}

#' Run Walk-Forward Backtest with DSR Model
#'
#' @param prices xts object with Close prices
#' @param start_date Start of backtest period
#' @param end_date End of backtest period
#' @param lookback Lookback window for features (default from config)
#' @param frequency Rebalance frequency (default "weekly")
#' @param use_constraints Whether to apply weight constraints (default from config)
#' @return List with portfolio returns, weights history, etc.
run_dsr_backtest <- function(
  prices,
  start_date,
  end_date,
  lookback = NULL,
  frequency = "weekly",
  use_constraints = NULL
) {
  library(xts)

  config <- load_backtest_config()
  if (is.null(lookback)) {
    lookback <- as.integer(config$model$lookback_window)
  }
  if (is.null(use_constraints)) {
    use_constraints <- config$backtest$use_constraints %||% FALSE
  }

  cat("Running DSR walk-forward backtest...\n")
  cat(sprintf("  Period: %s to %s\n", start_date, end_date))
  cat(sprintf("  Rebalance frequency: %s\n", frequency))
  cat(sprintf("  Constraints: %s\n", ifelse(use_constraints, "ENABLED", "disabled")))

  # Get all dates
  all_dates <- zoo::index(prices)

  # Get dates in test period
  start_date <- as.Date(start_date)
  end_date <- as.Date(end_date)
  valid_dates <- all_dates[all_dates >= start_date & all_dates <= end_date]

  if (length(valid_dates) == 0) {
    stop("No valid dates in specified range")
  }

  # Get rebalance dates
  rebalance_dates <- get_rebalance_dates(valid_dates, frequency)
  cat(sprintf("  Rebalance dates: %d\n", length(rebalance_dates)))

  # Compute returns
  returns <- compute_returns(prices)

  # Get tickers from Python
  tickers <- get_tickers()
  n_assets <- length(tickers)

  # Initialize storage
  portfolio_returns <- xts::xts(
    rep(NA_real_, length(valid_dates)),
    order.by = valid_dates
  )
  names(portfolio_returns) <- "DSR"

  weights_history <- xts::xts(
    matrix(NA_real_, nrow = length(rebalance_dates), ncol = n_assets),
    order.by = rebalance_dates
  )
  colnames(weights_history) <- tickers

  # Also track raw (unconstrained) weights if constraints enabled
  raw_weights_history <- NULL
  if (use_constraints) {
    raw_weights_history <- xts::xts(
      matrix(NA_real_, nrow = length(rebalance_dates), ncol = n_assets),
      order.by = rebalance_dates
    )
    colnames(raw_weights_history) <- tickers
  }

  # Track constraint info
  constraint_info <- list()

  # Start with equal weights
  current_weights <- rep(1/n_assets, n_assets)
  names(current_weights) <- tickers

  # Walk-forward loop
  cat("\nProcessing dates:\n")
  rebal_idx <- 1

  for (i in seq_along(valid_dates)) {
    current_date <- valid_dates[i]

    # Check if rebalance day
    if (current_date %in% rebalance_dates) {
      date_str <- as.character(current_date)

      tryCatch({
        # Get weights from Python model
        raw_weights <- predict_at_date(date_str, lookback)

        if (length(raw_weights) == n_assets && all(is.finite(raw_weights))) {
          raw_weights <- as.numeric(raw_weights)
          names(raw_weights) <- tickers

          # Store raw weights if tracking
          if (!is.null(raw_weights_history)) {
            raw_weights_history[current_date, ] <- raw_weights
          }

          # Apply constraints if enabled
          if (use_constraints) {
            # Get historical returns for CVaR calculation (last 252 days)
            hist_end_idx <- which(all_dates == current_date)
            hist_start_idx <- max(1, hist_end_idx - 252)
            hist_returns <- returns[hist_start_idx:hist_end_idx, tickers]

            # Apply constraints
            constraint_result <- constrain_weights(raw_weights, hist_returns, config)
            current_weights <- constraint_result$weights
            names(current_weights) <- tickers

            # Store constraint info
            constraint_info[[date_str]] <- constraint_result
          } else {
            current_weights <- raw_weights
          }

          weights_history[current_date, ] <- current_weights

          if (rebal_idx %% 10 == 1 || rebal_idx == length(rebalance_dates)) {
            cat(sprintf("  [%d/%d] %s: %s\n",
                        rebal_idx, length(rebalance_dates), date_str,
                        paste(sprintf("%s=%.2f", tickers, current_weights), collapse=", ")))
          }
        }
      }, error = function(e) {
        cat(sprintf("  Warning: %s - %s (keeping previous weights)\n",
                    date_str, e$message))
      })

      rebal_idx <- rebal_idx + 1
    }

    # Compute portfolio return for this day
    if (!is.na(returns[current_date, 1])) {
      daily_rets <- as.numeric(returns[current_date, tickers])
      if (all(is.finite(daily_rets))) {
        portfolio_returns[current_date] <- sum(current_weights * daily_rets)
      }
    }
  }

  # Remove NA values
  portfolio_returns <- na.omit(portfolio_returns)
  weights_history <- na.omit(weights_history)
  if (!is.null(raw_weights_history)) {
    raw_weights_history <- na.omit(raw_weights_history)
  }

  cat(sprintf("\nBacktest complete: %d trading days\n", length(portfolio_returns)))

  result <- list(
    returns = portfolio_returns,
    weights_history = weights_history,
    rebalance_dates = rebalance_dates,
    tickers = tickers,
    use_constraints = use_constraints
  )

  # Add constraint-specific data if enabled

  if (use_constraints) {
    result$raw_weights_history <- raw_weights_history
    result$constraint_info <- constraint_info
  }

  result
}

#' Compute Equal Weight Benchmark
#'
#' @param prices xts object with prices
#' @param start_date Start date
#' @param end_date End date
#' @return xts object with EW portfolio returns
compute_equal_weight_benchmark <- function(prices, start_date, end_date) {
  library(xts)

  # Compute returns
  returns <- compute_returns(prices)

  # Filter to date range
  start_date <- as.Date(start_date)
  end_date <- as.Date(end_date)

  returns <- returns[zoo::index(returns) >= start_date & zoo::index(returns) <= end_date]

  # Equal weight
  n_assets <- ncol(returns)
  weights <- rep(1/n_assets, n_assets)

  # Portfolio return = sum of weighted returns
  portfolio_returns <- xts::xts(
    rowSums(returns * matrix(weights, nrow = nrow(returns), ncol = n_assets, byrow = TRUE)),
    order.by = zoo::index(returns)
  )
  names(portfolio_returns) <- "EqualWeight"

  na.omit(portfolio_returns)
}

#' Run Complete Backtest (DSR + Benchmarks)
#'
#' Runs both constrained and unconstrained DSR backtests for comparison,
#' applies transaction costs, and computes benchmarks.
#'
#' @param test_split_ratio Fraction of data for test period (from end)
#' @param frequency Rebalance frequency
#' @return List with all results
run_backtest <- function(test_split_ratio = 0.15, frequency = "weekly") {
  # Initialize environment
  init_backtest()

  # Load prices
  prices <- load_prices_for_backtest()
  cat(sprintf("Loaded prices: %d days, %d assets\n", nrow(prices), ncol(prices)))

  # Load config for split ratios
  config <- load_backtest_config()
  train_ratio <- config$training$train_ratio
  val_ratio <- config$training$val_ratio
  use_constraints <- config$backtest$use_constraints %||% FALSE
  apply_tc <- config$backtest$apply_transaction_costs %||% FALSE
  tc_bps <- config$backtest$transaction_cost_bps %||% 10

  # Determine test period
  all_dates <- zoo::index(prices)
  n_dates <- length(all_dates)

  test_start_idx <- floor(n_dates * (train_ratio + val_ratio)) + 1
  test_start <- all_dates[test_start_idx]
  test_end <- all_dates[n_dates]

  cat(sprintf("\nTest period: %s to %s\n", test_start, test_end))
  cat(sprintf("Constraints: %s\n", ifelse(use_constraints, "ENABLED", "disabled")))
  cat(sprintf("Transaction costs: %s (%d bps)\n",
              ifelse(apply_tc, "ENABLED", "disabled"), tc_bps))

  # Run DSR backtest (unconstrained for comparison if constraints enabled)
  cat("\n--- Running Unconstrained DSR Backtest ---\n")
  dsr_unconstrained <- run_dsr_backtest(
    prices,
    start_date = as.character(test_start),
    end_date = as.character(test_end),
    frequency = frequency,
    use_constraints = FALSE
  )

  # Run constrained DSR backtest if enabled
  dsr_constrained <- NULL
  if (use_constraints) {
    cat("\n--- Running Constrained DSR Backtest ---\n")
    dsr_constrained <- run_dsr_backtest(
      prices,
      start_date = as.character(test_start),
      end_date = as.character(test_end),
      frequency = frequency,
      use_constraints = TRUE
    )
  }

  # Compute Equal Weight benchmark
  ew_returns <- compute_equal_weight_benchmark(
    prices,
    start_date = as.character(test_start),
    end_date = as.character(test_end)
  )

  # Apply transaction costs if enabled
  dsr_unconstrained_gross <- dsr_unconstrained$returns
  dsr_unconstrained_net <- dsr_unconstrained_gross
  dsr_constrained_gross <- NULL
  dsr_constrained_net <- NULL

  if (apply_tc) {
    cat("\nApplying transaction costs...\n")
    dsr_unconstrained_net <- apply_transaction_costs(
      dsr_unconstrained_gross,
      dsr_unconstrained$weights_history,
      tc_bps
    )
    colnames(dsr_unconstrained_net) <- "DSR_Unconstrained"

    if (!is.null(dsr_constrained)) {
      dsr_constrained_gross <- dsr_constrained$returns
      dsr_constrained_net <- apply_transaction_costs(
        dsr_constrained_gross,
        dsr_constrained$weights_history,
        tc_bps
      )
      colnames(dsr_constrained_net) <- "DSR_Constrained"
    }

    # Print transaction cost summary
    print_transaction_cost_summary(dsr_unconstrained$weights_history, tc_bps)
    if (!is.null(dsr_constrained)) {
      cat("\nConstrained portfolio:\n")
      print_transaction_cost_summary(dsr_constrained$weights_history, tc_bps)
    }
  }

  # Prepare combined returns for comparison
  if (use_constraints && !is.null(dsr_constrained)) {
    # Rename columns for clarity
    unconstrained_ret <- dsr_unconstrained_net
    colnames(unconstrained_ret) <- "DSR_Unconstrained"

    constrained_ret <- dsr_constrained_net
    colnames(constrained_ret) <- "DSR_Constrained"

    combined_returns <- merge(
      constrained_ret,
      unconstrained_ret,
      ew_returns,
      join = "inner"
    )

    # Primary DSR is the constrained version
    dsr_primary <- dsr_constrained
    dsr_primary$returns <- constrained_ret
    colnames(dsr_primary$returns) <- "DSR"
  } else {
    # No constraints - just use unconstrained
    colnames(dsr_unconstrained_net) <- "DSR"
    combined_returns <- merge(dsr_unconstrained_net, ew_returns, join = "inner")
    dsr_primary <- dsr_unconstrained
    dsr_primary$returns <- dsr_unconstrained_net
  }

  # Build result
  result <- list(
    dsr = dsr_primary,
    ew_returns = ew_returns,
    combined_returns = combined_returns,
    prices = prices,
    test_period = c(test_start, test_end),
    frequency = frequency,
    use_constraints = use_constraints,
    apply_transaction_costs = apply_tc,
    transaction_cost_bps = tc_bps
  )

  # Add detailed constraint comparison data
  if (use_constraints) {
    result$dsr_unconstrained <- dsr_unconstrained
    result$dsr_constrained <- dsr_constrained
    result$turnover_unconstrained <- compute_turnover_metrics(dsr_unconstrained$weights_history)
    result$turnover_constrained <- compute_turnover_metrics(dsr_constrained$weights_history)
  }

  result
}

#' Save Backtest Results
#'
#' @param results Results from run_backtest()
#' @param output_dir Directory for output files
save_backtest_results <- function(results, output_dir = "results") {
  project_root <- get_project_root()
  output_path <- file.path(project_root, output_dir)

  # Create output directory
  if (!dir.exists(output_path)) {
    dir.create(output_path, recursive = TRUE)
  }

  # Save RDS
  saveRDS(results, file.path(output_path, "backtest_results.rds"))
  cat(sprintf("Saved results to %s/backtest_results.rds\n", output_path))

  # Save weights history as CSV (primary strategy)
  weights_df <- as.data.frame(results$dsr$weights_history)
  weights_df$Date <- zoo::index(results$dsr$weights_history)
  write.csv(weights_df, file.path(output_path, "weights_history.csv"), row.names = FALSE)
  cat(sprintf("Saved weights to %s/weights_history.csv\n", output_path))

  # Save constrained vs unconstrained comparison if available
  if (!is.null(results$dsr_constrained) && !is.null(results$dsr_unconstrained)) {
    # Constrained weights
    constrained_df <- as.data.frame(results$dsr_constrained$weights_history)
    constrained_df$Date <- zoo::index(results$dsr_constrained$weights_history)
    write.csv(constrained_df, file.path(output_path, "weights_constrained.csv"), row.names = FALSE)

    # Unconstrained weights
    unconstrained_df <- as.data.frame(results$dsr_unconstrained$weights_history)
    unconstrained_df$Date <- zoo::index(results$dsr_unconstrained$weights_history)
    write.csv(unconstrained_df, file.path(output_path, "weights_unconstrained.csv"), row.names = FALSE)

    cat(sprintf("Saved constrained/unconstrained weights comparison\n"))
  }

  invisible(results)
}

# Run if executed directly
if (!interactive()) {
  cat("=", rep("=", 60), "\n", sep = "")
  cat("DSR Portfolio Optimizer - Backtest\n")
  cat(rep("=", 61), "\n", sep = "")

  # Run backtest
  results <- run_backtest()

  # Save results
  save_backtest_results(results)

  # Source metrics and generate report
  metrics_path <- file.path(get_project_root(), "R/metrics.R")
  source(metrics_path)
  generate_report(results)

  cat("\nBacktest complete!\n")
}
