# Performance Metrics and Visualization for DSR Portfolio Optimizer
# Uses PerformanceAnalytics for standard metrics
# Includes CVaR, VaR, and turnover metrics for Iteration 3

library(PerformanceAnalytics)
library(xts)
library(ggplot2)

# Null coalescing operator
if (!exists("%||%")) {
  `%||%` <- function(x, y) if (is.null(x)) y else x
}

#' Compute Annualized Return
#'
#' @param returns xts object with returns
#' @param scale Trading days per year (default 252)
#' @return Annualized return
compute_annualized_return <- function(returns, scale = 252) {
  PerformanceAnalytics::Return.annualized(returns, scale = scale)
}

#' Compute Annualized Volatility
#'
#' @param returns xts object with returns
#' @param scale Trading days per year (default 252)
#' @return Annualized volatility
compute_annualized_volatility <- function(returns, scale = 252) {
  PerformanceAnalytics::StdDev.annualized(returns, scale = scale)
}

#' Compute Sharpe Ratio
#'
#' @param returns xts object with returns
#' @param Rf Risk-free rate (default 0)
#' @param scale Trading days per year (default 252)
#' @return Annualized Sharpe Ratio
compute_sharpe_ratio <- function(returns, Rf = 0, scale = 252) {
  PerformanceAnalytics::SharpeRatio.annualized(returns, Rf = Rf, scale = scale)
}

#' Compute Maximum Drawdown
#'
#' @param returns xts object with returns
#' @return Maximum drawdown (as positive value for display)
compute_max_drawdown <- function(returns) {
  PerformanceAnalytics::maxDrawdown(returns)
}

#' Compute Sortino Ratio
#'
#' @param returns xts object with returns
#' @param MAR Minimum acceptable return (default 0)
#' @return Sortino Ratio
compute_sortino_ratio <- function(returns, MAR = 0) {
  PerformanceAnalytics::SortinoRatio(returns, MAR = MAR)
}

#' Compute Value at Risk (VaR)
#'
#' @param returns xts object with returns
#' @param confidence Confidence level (default 0.95)
#' @param method "historical" or "gaussian"
#' @return VaR as positive value
compute_var <- function(returns, confidence = 0.95, method = "historical") {
  var_result <- PerformanceAnalytics::VaR(
    returns,
    p = confidence,
    method = method
  )
  abs(as.numeric(var_result))
}

#' Compute Conditional Value at Risk (CVaR / Expected Shortfall)
#'
#' @param returns xts object with returns
#' @param confidence Confidence level (default 0.95)
#' @param method "historical" or "gaussian"
#' @return CVaR as positive value
compute_cvar <- function(returns, confidence = 0.95, method = "historical") {
  es_result <- PerformanceAnalytics::ES(
    returns,
    p = confidence,
    method = method
  )
  abs(as.numeric(es_result))
}

#' Compute Calmar Ratio
#'
#' @param returns xts object with returns
#' @param scale Annualization factor
#' @return Calmar Ratio (Ann. Return / Max Drawdown)
compute_calmar_ratio <- function(returns, scale = 252) {
  ann_ret <- as.numeric(PerformanceAnalytics::Return.annualized(returns, scale = scale))
  max_dd <- as.numeric(PerformanceAnalytics::maxDrawdown(returns))
  if (max_dd > 0) ann_ret / max_dd else 0
}

#' Compute All Performance Metrics
#'
#' @param returns xts object with one or more return series
#' @param Rf Risk-free rate (default 0)
#' @param confidence Confidence level for VaR/CVaR (default 0.95)
#' @return Data frame with all metrics
compute_all_metrics <- function(returns, Rf = 0, confidence = 0.95) {
  n_strategies <- ncol(returns)

  metrics <- data.frame(
    Strategy = colnames(returns),
    stringsAsFactors = FALSE
  )

  # Compute metrics for each strategy
  ann_ret <- numeric(n_strategies)
  ann_vol <- numeric(n_strategies)
  sharpe <- numeric(n_strategies)
  sortino <- numeric(n_strategies)
  max_dd <- numeric(n_strategies)
  var_95 <- numeric(n_strategies)
  cvar_95 <- numeric(n_strategies)
  calmar <- numeric(n_strategies)

  for (i in 1:n_strategies) {
    r <- returns[, i]
    ann_ret[i] <- as.numeric(compute_annualized_return(r))
    ann_vol[i] <- as.numeric(compute_annualized_volatility(r))
    sharpe[i] <- as.numeric(compute_sharpe_ratio(r, Rf = Rf))
    sortino[i] <- as.numeric(compute_sortino_ratio(r))
    max_dd[i] <- as.numeric(compute_max_drawdown(r))
    var_95[i] <- compute_var(r, confidence = confidence)
    cvar_95[i] <- compute_cvar(r, confidence = confidence)
    calmar[i] <- compute_calmar_ratio(r)
  }

  metrics$Ann_Return <- ann_ret
  metrics$Ann_Volatility <- ann_vol
  metrics$Sharpe_Ratio <- sharpe
  metrics$Sortino_Ratio <- sortino
  metrics$Max_Drawdown <- max_dd
  metrics$VaR_95 <- var_95
  metrics$CVaR_95 <- cvar_95
  metrics$Calmar_Ratio <- calmar

  # Add formatted percentage columns
  metrics$Ann_Return_Pct <- sprintf("%.2f%%", metrics$Ann_Return * 100)
  metrics$Ann_Volatility_Pct <- sprintf("%.2f%%", metrics$Ann_Volatility * 100)
  metrics$Max_Drawdown_Pct <- sprintf("%.2f%%", metrics$Max_Drawdown * 100)
  metrics$VaR_95_Pct <- sprintf("%.2f%%", metrics$VaR_95 * 100)
  metrics$CVaR_95_Pct <- sprintf("%.2f%%", metrics$CVaR_95 * 100)

  metrics
}

#' Compute Equity Curve
#'
#' @param returns xts object with return series
#' @param initial_value Starting portfolio value (default 1)
#' @return xts object with cumulative wealth
compute_equity_curve <- function(returns, initial_value = 1) {
  # Cumulative product of (1 + returns)
  equity <- cumprod(1 + returns) * initial_value
  equity
}

#' Plot Equity Curves
#'
#' @param returns xts object with one or more return series
#' @param title Plot title
#' @param save_path Optional path to save plot
#' @return ggplot object
plot_equity_curves <- function(
  returns,
  title = "Equity Curve: DSR vs Equal Weight",
  save_path = NULL
) {
  # Compute equity curves
  equity <- compute_equity_curve(returns)

  # Convert to data frame for ggplot
  df <- data.frame(
    Date = zoo::index(equity),
    coredata(equity),
    check.names = FALSE
  )

  # Reshape to long format
  df_long <- tidyr::pivot_longer(
    df,
    cols = -Date,
    names_to = "Strategy",
    values_to = "Value"
  )

  # Define colors - support multiple strategies
  strategy_colors <- c(
    "DSR" = "#2E86AB",
    "DSR_Constrained" = "#2E86AB",
    "DSR_Unconstrained" = "#E8871E",
    "EqualWeight" = "#A23B72"
  )

  # Use default palette for any unrecognized strategies
  unique_strategies <- unique(df_long$Strategy)
  missing <- setdiff(unique_strategies, names(strategy_colors))
  if (length(missing) > 0) {
    extra_colors <- scales::hue_pal()(length(missing))
    names(extra_colors) <- missing
    strategy_colors <- c(strategy_colors, extra_colors)
  }

  # Plot
  p <- ggplot(df_long, aes(x = Date, y = Value, color = Strategy)) +
    geom_line(linewidth = 1) +
    labs(
      title = title,
      x = "Date",
      y = "Portfolio Value (Starting = 1)",
      color = "Strategy"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      legend.position = "bottom",
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      panel.grid.minor = element_blank()
    ) +
    scale_color_manual(values = strategy_colors) +
    scale_y_continuous(labels = scales::dollar_format(prefix = "$"))

  if (!is.null(save_path)) {
    ggsave(save_path, p, width = 10, height = 6, dpi = 150)
    cat(sprintf("Saved equity curve to %s\n", save_path))
  }

  p
}

#' Plot Drawdown
#'
#' @param returns xts object with return series
#' @param title Plot title
#' @param save_path Optional path to save plot
#' @return ggplot object
plot_drawdown <- function(
  returns,
  title = "Drawdown Analysis",
  save_path = NULL
) {
  # Compute drawdowns
  dd <- PerformanceAnalytics::Drawdowns(returns)

  # Convert to data frame
  df <- data.frame(
    Date = zoo::index(dd),
    coredata(dd),
    check.names = FALSE
  )

  # Reshape
  df_long <- tidyr::pivot_longer(
    df,
    cols = -Date,
    names_to = "Strategy",
    values_to = "Drawdown"
  )

  # Colors
  strategy_colors <- c("DSR" = "#2E86AB", "EqualWeight" = "#A23B72")

  p <- ggplot(df_long, aes(x = Date, y = Drawdown * 100, color = Strategy)) +
    geom_line(linewidth = 0.8) +
    geom_area(aes(fill = Strategy), alpha = 0.3, position = "identity") +
    labs(
      title = title,
      x = "Date",
      y = "Drawdown (%)",
      color = "Strategy",
      fill = "Strategy"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      legend.position = "bottom",
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      panel.grid.minor = element_blank()
    ) +
    scale_color_manual(values = strategy_colors) +
    scale_fill_manual(values = strategy_colors) +
    scale_y_continuous(labels = function(x) paste0(x, "%"))

  if (!is.null(save_path)) {
    ggsave(save_path, p, width = 10, height = 6, dpi = 150)
    cat(sprintf("Saved drawdown plot to %s\n", save_path))
  }

  p
}

#' Plot Weight Allocation Over Time
#'
#' @param weights_history xts object with weight history
#' @param title Plot title
#' @param save_path Optional path to save plot
#' @return ggplot object
plot_weight_allocation <- function(
  weights_history,
  title = "Portfolio Weight Allocation Over Time",
  save_path = NULL
) {
  # Convert to data frame
  df <- data.frame(
    Date = zoo::index(weights_history),
    coredata(weights_history),
    check.names = FALSE
  )

  # Reshape to long format
  df_long <- tidyr::pivot_longer(
    df,
    cols = -Date,
    names_to = "Asset",
    values_to = "Weight"
  )

  # Stacked area plot
  p <- ggplot(df_long, aes(x = Date, y = Weight, fill = Asset)) +
    geom_area(alpha = 0.8) +
    labs(
      title = title,
      x = "Date",
      y = "Weight",
      fill = "Asset"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      legend.position = "bottom",
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      panel.grid.minor = element_blank()
    ) +
    scale_y_continuous(labels = scales::percent_format()) +
    scale_fill_brewer(palette = "Set2")

  if (!is.null(save_path)) {
    ggsave(save_path, p, width = 10, height = 6, dpi = 150)
    cat(sprintf("Saved weight allocation to %s\n", save_path))
  }

  p
}

#' Print Performance Summary
#'
#' @param returns xts object with return series
#' @param include_risk Whether to include VaR/CVaR (default TRUE)
print_performance_summary <- function(returns, include_risk = TRUE) {
  metrics <- compute_all_metrics(returns)

  cat("\n")
  cat(strrep("=", 85), "\n")
  cat("                    PERFORMANCE SUMMARY\n")
  cat(strrep("=", 85), "\n\n")

  # Basic metrics table
  print_df <- metrics[, c("Strategy", "Ann_Return_Pct", "Ann_Volatility_Pct",
                           "Sharpe_Ratio", "Sortino_Ratio", "Max_Drawdown_Pct")]
  colnames(print_df) <- c("Strategy", "Ann.Return", "Ann.Vol", "Sharpe", "Sortino", "Max DD")
  print_df$Sharpe <- sprintf("%.3f", print_df$Sharpe)
  print_df$Sortino <- sprintf("%.3f", print_df$Sortino)

  print(print_df, row.names = FALSE)

  if (include_risk) {
    cat("\n")
    cat(strrep("-", 85), "\n")
    cat("Risk Metrics (95% confidence):\n\n")

    risk_df <- metrics[, c("Strategy", "VaR_95_Pct", "CVaR_95_Pct", "Calmar_Ratio")]
    colnames(risk_df) <- c("Strategy", "VaR@95%", "CVaR@95%", "Calmar")
    risk_df$Calmar <- sprintf("%.3f", risk_df$Calmar)

    print(risk_df, row.names = FALSE)
  }

  cat("\n")
  cat(strrep("=", 85), "\n")

  invisible(metrics)
}

#' Generate Full Report
#'
#' @param results List from run_backtest()
#' @param output_dir Directory for output files
generate_report <- function(results, output_dir = "results") {
  # Get project root - try to find config.yaml
  if (file.exists("config.yaml")) {
    project_root <- normalizePath(".")
  } else if (file.exists("../config.yaml")) {
    project_root <- normalizePath("..")
  } else {
    project_root <- getwd()
  }
  output_path <- file.path(project_root, output_dir)

  # Create output directory
  if (!dir.exists(output_path)) {
    dir.create(output_path, recursive = TRUE)
  }

  combined <- results$combined_returns
  use_constraints <- results$use_constraints %||% FALSE

  cat("\nGenerating report...\n")

  # 1. Compute and save metrics
  metrics <- compute_all_metrics(combined)
  write.csv(metrics, file.path(output_path, "metrics.csv"), row.names = FALSE)
  cat(sprintf("Saved metrics to %s/metrics.csv\n", output_path))

  # 2. Print performance summary
  print_performance_summary(combined, include_risk = TRUE)

  # 3. Equity curve plot
  title_suffix <- if (use_constraints) " (Constrained vs Unconstrained)" else ""
  plot_equity_curves(
    combined,
    title = sprintf("DSR Portfolio Comparison%s\n(%s to %s)",
                    title_suffix,
                    format(results$test_period[1], "%Y-%m-%d"),
                    format(results$test_period[2], "%Y-%m-%d")),
    save_path = file.path(output_path, "equity_curve.png")
  )

  # 4. Drawdown plot
  plot_drawdown(
    combined,
    title = "Drawdown Analysis",
    save_path = file.path(output_path, "drawdown.png")
  )

  # 5. Weight allocation plot
  if (!is.null(results$dsr$weights_history)) {
    weight_title <- if (use_constraints) "DSR Constrained Weight Allocation" else "DSR Portfolio Weight Allocation"
    plot_weight_allocation(
      results$dsr$weights_history,
      title = weight_title,
      save_path = file.path(output_path, "weight_allocation.png")
    )
  }

  # 6. Turnover comparison (if constraints enabled)
  if (use_constraints && !is.null(results$turnover_constrained)) {
    cat("\n--- Turnover Comparison ---\n")
    cat(sprintf("Unconstrained: %.2f total (%.2f%% avg per rebalance)\n",
                results$turnover_unconstrained$total_turnover,
                results$turnover_unconstrained$avg_turnover * 100))
    cat(sprintf("Constrained:   %.2f total (%.2f%% avg per rebalance)\n",
                results$turnover_constrained$total_turnover,
                results$turnover_constrained$avg_turnover * 100))
  }

  # 7. Efficient frontier plot
  tryCatch({
    # Source efficient frontier module
    ef_path <- file.path(project_root, "R/efficient_frontier.R")
    if (file.exists(ef_path)) {
      source(ef_path)
      cat("\nGenerating efficient frontier...\n")
      generate_efficient_frontier_plot(results, output_dir)
    }
  }, error = function(e) {
    cat(sprintf("Note: Could not generate efficient frontier: %s\n", e$message))
  })

  cat(sprintf("\nReport generated in %s/\n", output_path))

  invisible(metrics)
}

#' Generate Full HTML Report via R Markdown
#'
#' Creates a comprehensive HTML report with all analytics.
#'
#' @param results List from run_backtest()
#' @param output_format "html" or "pdf" (default "html")
#' @param output_dir Directory for output (default "reports")
#' @return Path to generated report
generate_full_report <- function(results, output_format = "html", output_dir = "reports") {
  # Get project root
  if (file.exists("config.yaml")) {
    project_root <- normalizePath(".")
  } else if (file.exists("../config.yaml")) {
    project_root <- normalizePath("..")
  } else {
    project_root <- getwd()
  }

  # Create output directory
  output_path <- file.path(project_root, output_dir)
  if (!dir.exists(output_path)) {
    dir.create(output_path, recursive = TRUE)
  }

  # Save results for report to access
  data_path <- file.path(output_path, "report_data.rds")
  saveRDS(results, data_path)
  cat(sprintf("Saved report data to %s\n", data_path))

  # Find report template
  rmd_path <- file.path(project_root, "reports/report.Rmd")
  if (!file.exists(rmd_path)) {
    stop("Report template not found at: ", rmd_path)
  }

  # Determine output file
  output_file <- if (output_format == "pdf") "report.pdf" else "report.html"

  cat(sprintf("\nRendering %s report...\n", toupper(output_format)))

  # Render the report
  tryCatch({
    rmarkdown::render(
      input = rmd_path,
      output_format = if (output_format == "pdf") "pdf_document" else "html_document",
      output_file = output_file,
      output_dir = output_path,
      params = list(results_path = data_path),
      quiet = FALSE
    )

    report_path <- file.path(output_path, output_file)
    cat(sprintf("\nFull report generated: %s\n", report_path))

    invisible(report_path)
  }, error = function(e) {
    cat(sprintf("\nError rendering report: %s\n", e$message))
    cat("Falling back to basic report generation...\n")
    generate_report(results, "results")
    invisible(NULL)
  })
}

# Helper to check if tidyr is available, install if not
ensure_tidyr <- function() {
  if (!requireNamespace("tidyr", quietly = TRUE)) {
    install.packages("tidyr")
  }
  library(tidyr)
}

# Helper to check if scales is available
ensure_scales <- function() {
  if (!requireNamespace("scales", quietly = TRUE)) {
    install.packages("scales")
  }
  library(scales)
}

# Load dependencies on source
tryCatch({
  ensure_tidyr()
  ensure_scales()
}, error = function(e) {
  message("Note: Install tidyr and scales for full visualization: install.packages(c('tidyr', 'scales'))")
})
