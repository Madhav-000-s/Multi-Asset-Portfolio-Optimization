# Report Utility Functions for DSR Portfolio Optimizer
# Helper functions for R Markdown report visualizations

library(ggplot2)
library(xts)
library(zoo)
library(PerformanceAnalytics)

# Null coalescing operator
if (!exists("%||%")) {
`%||%` <- function(x, y) if (is.null(x)) y else x
}

#' Compute Monthly Returns
#'
#' Aggregate daily returns to monthly.
#'
#' @param returns xts object with daily returns
#' @return xts object with monthly returns
compute_monthly_returns <- function(returns) {
  if (nrow(returns) == 0) return(returns)

  # Handle each column separately and combine
  ep <- endpoints(returns, on = "months")

  if (length(ep) <= 1) return(xts())  # No complete months

  # For multi-column xts, apply to each column
  if (ncol(returns) > 1) {
    monthly_list <- lapply(seq_len(ncol(returns)), function(i) {
      r <- returns[, i]
      m <- period.apply(r, INDEX = ep, FUN = function(x) {
        prod(1 + x, na.rm = TRUE) - 1
      })
      m
    })

    # Combine columns
    monthly <- do.call(merge, monthly_list)
    colnames(monthly) <- colnames(returns)
  } else {
    monthly <- period.apply(returns, INDEX = ep, FUN = function(x) {
      prod(1 + x, na.rm = TRUE) - 1
    })
  }

  monthly
}

#' Create Monthly Returns Data Frame
#'
#' Transform monthly returns into year × month matrix for heatmap.
#'
#' @param returns xts object with daily returns
#' @param strategy_name Name of the strategy
#' @return Data frame ready for heatmap plotting
prepare_monthly_heatmap_data <- function(returns, strategy_name = "Strategy") {
  monthly <- compute_monthly_returns(returns)

  # Extract year and month
  dates <- zoo::index(monthly)
  years <- format(dates, "%Y")
  months <- format(dates, "%b")
  month_nums <- as.numeric(format(dates, "%m"))

  df <- data.frame(
    Year = years,
    Month = months,
    MonthNum = month_nums,
    Return = as.numeric(monthly) * 100,  # Convert to percentage
    Strategy = strategy_name
  )

  # Order months correctly
  df$Month <- factor(df$Month, levels = month.abb)

  df
}

#' Plot Monthly Returns Heatmap
#'
#' @param returns xts object with one or more return series
#' @param title Plot title
#' @return ggplot object
plot_monthly_heatmap <- function(returns, title = "Monthly Returns Heatmap") {
  # Handle multiple strategies
  if (ncol(returns) > 1) {
    # Use first strategy for single heatmap, or combine
    all_data <- data.frame()
    for (i in seq_len(ncol(returns))) {
      strategy_name <- colnames(returns)[i]
      df <- prepare_monthly_heatmap_data(returns[, i], strategy_name)
      all_data <- rbind(all_data, df)
    }

    p <- ggplot(all_data, aes(x = Month, y = Year, fill = Return)) +
      geom_tile(color = "white", linewidth = 0.5) +
      geom_text(aes(label = sprintf("%.1f", Return)), size = 2.5) +
      scale_fill_gradient2(
        low = "#d73027", mid = "white", high = "#1a9850",
        midpoint = 0, name = "Return (%)"
      ) +
      facet_wrap(~Strategy, ncol = 1) +
      labs(title = title, x = "Month", y = "Year") +
      theme_minimal(base_size = 10) +
      theme(
        plot.title = element_text(hjust = 0.5, face = "bold"),
        axis.text.x = element_text(angle = 45, hjust = 1),
        panel.grid = element_blank()
      )
  } else {
    df <- prepare_monthly_heatmap_data(returns[, 1], colnames(returns)[1])

    p <- ggplot(df, aes(x = Month, y = Year, fill = Return)) +
      geom_tile(color = "white", linewidth = 0.5) +
      geom_text(aes(label = sprintf("%.1f", Return)), size = 3) +
      scale_fill_gradient2(
        low = "#d73027", mid = "white", high = "#1a9850",
        midpoint = 0, name = "Return (%)"
      ) +
      labs(title = title, x = "Month", y = "Year") +
      theme_minimal(base_size = 11) +
      theme(
        plot.title = element_text(hjust = 0.5, face = "bold"),
        axis.text.x = element_text(angle = 45, hjust = 1),
        panel.grid = element_blank()
      )
  }

  p
}

#' Compute Rolling Sharpe Ratio
#'
#' @param returns xts object with returns
#' @param window Rolling window size (default 252 days)
#' @param Rf Risk-free rate (default 0)
#' @param scale Annualization factor (default 252)
#' @return xts object with rolling Sharpe ratios
compute_rolling_sharpe <- function(returns, window = 252, Rf = 0, scale = 252) {
  # Compute rolling mean and sd
  rolling_mean <- rollapply(returns, width = window, FUN = mean, align = "right", fill = NA)
  rolling_sd <- rollapply(returns, width = window, FUN = sd, align = "right", fill = NA)

  # Annualize
  ann_mean <- rolling_mean * scale
  ann_sd <- rolling_sd * sqrt(scale)

  # Sharpe ratio
  sharpe <- (ann_mean - Rf) / ann_sd

  sharpe
}

#' Compute Rolling Volatility
#'
#' @param returns xts object with returns
#' @param window Rolling window size (default 252 days)
#' @param scale Annualization factor (default 252)
#' @return xts object with rolling volatility
compute_rolling_volatility <- function(returns, window = 252, scale = 252) {
  rolling_sd <- rollapply(returns, width = window, FUN = sd, align = "right", fill = NA)
  ann_vol <- rolling_sd * sqrt(scale)
  ann_vol
}

#' Plot Rolling Metrics
#'
#' @param returns xts object with one or more return series
#' @param metric "sharpe" or "volatility"
#' @param window Rolling window size
#' @param title Plot title
#' @return ggplot object
plot_rolling_metrics <- function(returns, metric = "sharpe", window = 252, title = NULL) {
  if (metric == "sharpe") {
    rolling_data <- compute_rolling_sharpe(returns, window)
    y_label <- "Rolling Sharpe Ratio"
    if (is.null(title)) title <- sprintf("Rolling %d-Day Sharpe Ratio", window)
  } else {
    rolling_data <- compute_rolling_volatility(returns, window)
    y_label <- "Rolling Volatility (%)"
    if (is.null(title)) title <- sprintf("Rolling %d-Day Volatility", window)
  }

  # Convert to data frame
  df <- data.frame(
    Date = zoo::index(rolling_data),
    coredata(rolling_data),
    check.names = FALSE
  )

  # Reshape to long format
  df_long <- tidyr::pivot_longer(
    df,
    cols = -Date,
    names_to = "Strategy",
    values_to = "Value"
  )

  # Colors
  strategy_colors <- c(
    "DSR_Constrained" = "#2E86AB",
    "DSR_Unconstrained" = "#E8871E",
    "EqualWeight" = "#A23B72",
    "DSR" = "#2E86AB"
  )

  p <- ggplot(df_long, aes(x = Date, y = Value, color = Strategy)) +
    geom_line(linewidth = 0.8) +
    labs(title = title, x = "Date", y = y_label, color = "Strategy") +
    theme_minimal(base_size = 11) +
    theme(
      legend.position = "bottom",
      plot.title = element_text(hjust = 0.5, face = "bold")
    ) +
    scale_color_manual(values = strategy_colors, na.value = "gray50")

  if (metric == "volatility") {
    p <- p + scale_y_continuous(labels = function(x) paste0(x * 100, "%"))
  }

  p
}

#' Plot Underwater Chart
#'
#' Shows time spent underwater (below high-water mark).
#'
#' @param returns xts object with returns
#' @param title Plot title
#' @return ggplot object
plot_underwater <- function(returns, title = "Underwater Chart") {
  # Compute drawdowns
  dd <- PerformanceAnalytics::Drawdowns(returns)

  # Convert to data frame
  df <- data.frame(
    Date = zoo::index(dd),
    coredata(dd) * 100,
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
  strategy_colors <- c(
    "DSR_Constrained" = "#2E86AB",
    "DSR_Unconstrained" = "#E8871E",
    "EqualWeight" = "#A23B72",
    "DSR" = "#2E86AB"
  )

  p <- ggplot(df_long, aes(x = Date, y = Drawdown, fill = Strategy)) +
    geom_area(alpha = 0.6, position = "identity") +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray40") +
    labs(
      title = title,
      x = "Date",
      y = "Drawdown (%)",
      fill = "Strategy"
    ) +
    theme_minimal(base_size = 11) +
    theme(
      legend.position = "bottom",
      plot.title = element_text(hjust = 0.5, face = "bold")
    ) +
    scale_fill_manual(values = strategy_colors, na.value = "gray50") +
    scale_y_continuous(labels = function(x) paste0(x, "%"))

  p
}

#' Identify High Volatility Regimes
#'
#' @param returns xts object with returns (single series)
#' @param threshold Annualized volatility threshold (default 0.25 = 25%)
#' @param window Rolling window for volatility calculation
#' @return Data frame with regime periods
identify_high_vol_regimes <- function(returns, threshold = 0.25, window = 20) {
  # Use equal-weight portfolio if multiple columns
  if (ncol(returns) > 1) {
    returns <- xts(rowMeans(returns, na.rm = TRUE), order.by = zoo::index(returns))
  }

  # Compute rolling volatility (annualized)
  rolling_vol <- rollapply(returns, width = window, FUN = sd, align = "right", fill = NA) * sqrt(252)

  # Identify high-vol periods
  high_vol <- rolling_vol > threshold
  high_vol[is.na(high_vol)] <- FALSE

  # Find regime start/end dates
  dates <- zoo::index(returns)
  regimes <- data.frame()

  in_regime <- FALSE
  regime_start <- NULL

  for (i in seq_along(dates)) {
    if (high_vol[i] && !in_regime) {
      in_regime <- TRUE
      regime_start <- dates[i]
    } else if (!high_vol[i] && in_regime) {
      in_regime <- FALSE
      regimes <- rbind(regimes, data.frame(
        Start = regime_start,
        End = dates[i - 1]
      ))
    }
  }

  # Handle regime still active at end

  if (in_regime) {
    regimes <- rbind(regimes, data.frame(
      Start = regime_start,
      End = dates[length(dates)]
    ))
  }

  regimes
}

#' Compute Performance by Regime
#'
#' @param returns xts object with returns
#' @param regimes Data frame with Start/End columns
#' @return Data frame with performance metrics by regime
compute_regime_performance <- function(returns, regimes) {
  if (nrow(regimes) == 0) {
    return(data.frame(Message = "No high-volatility regimes identified"))
  }

  results <- data.frame()

  for (i in seq_len(nrow(regimes))) {
    start <- regimes$Start[i]
    end <- regimes$End[i]

    # Subset returns
    regime_returns <- returns[paste0(start, "/", end)]

    if (nrow(regime_returns) < 5) next

    for (j in seq_len(ncol(returns))) {
      strategy <- colnames(returns)[j]
      r <- regime_returns[, j]

      results <- rbind(results, data.frame(
        Regime = i,
        Start = as.character(start),
        End = as.character(end),
        Days = nrow(r),
        Strategy = strategy,
        Total_Return = as.numeric(prod(1 + r, na.rm = TRUE) - 1) * 100,
        Volatility = as.numeric(sd(r, na.rm = TRUE) * sqrt(252)) * 100,
        Max_Drawdown = as.numeric(maxDrawdown(r)) * 100
      ))
    }
  }

  results
}

#' Compute Herfindahl-Hirschman Index (HHI)
#'
#' Measures portfolio concentration. HHI = sum(w_i^2)
#' Range: 1/N (equal weight) to 1 (single asset)
#'
#' @param weights Weight vector or matrix
#' @return HHI value(s)
compute_hhi <- function(weights) {
  if (is.matrix(weights) || is.data.frame(weights)) {
    apply(weights, 1, function(w) sum(w^2))
  } else {
    sum(weights^2)
  }
}

#' Compute Weight Statistics
#'
#' @param weights_history xts object with weight history
#' @return Data frame with weight statistics per asset
compute_weight_stats <- function(weights_history) {
  assets <- colnames(weights_history)

  stats <- data.frame(
    Asset = assets,
    Mean_Weight = colMeans(weights_history) * 100,
    Std_Weight = apply(weights_history, 2, sd) * 100,
    Min_Weight = apply(weights_history, 2, min) * 100,
    Max_Weight = apply(weights_history, 2, max) * 100
  )

  # Add average HHI
  hhi_values <- compute_hhi(weights_history)
  avg_hhi <- mean(hhi_values, na.rm = TRUE)

  attr(stats, "avg_hhi") <- avg_hhi
  attr(stats, "min_hhi") <- 1 / length(assets)  # Equal weight HHI

  stats
}

#' Plot Weight Distribution
#'
#' @param weights_history xts object with weight history
#' @param title Plot title
#' @return ggplot object
plot_weight_distribution <- function(weights_history, title = "Average Weight Distribution") {
  stats <- compute_weight_stats(weights_history)

  p <- ggplot(stats, aes(x = reorder(Asset, -Mean_Weight), y = Mean_Weight, fill = Asset)) +
    geom_bar(stat = "identity", alpha = 0.8) +
    geom_errorbar(
      aes(ymin = Mean_Weight - Std_Weight, ymax = Mean_Weight + Std_Weight),
      width = 0.3
    ) +
    labs(
      title = title,
      subtitle = sprintf("Avg HHI: %.3f (Equal weight: %.3f)", attr(stats, "avg_hhi"), attr(stats, "min_hhi")),
      x = "Asset",
      y = "Average Weight (%)"
    ) +
    theme_minimal(base_size = 11) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5),
      legend.position = "none"
    ) +
    scale_fill_brewer(palette = "Set2")

  p
}

#' Compute Drawdown Statistics
#'
#' @param returns xts object with returns
#' @return Data frame with drawdown statistics
compute_drawdown_stats <- function(returns) {
  stats <- data.frame()

  for (i in seq_len(ncol(returns))) {
    r <- returns[, i]
    strategy <- colnames(returns)[i]

    # Get drawdown table from PerformanceAnalytics
    dd_table <- table.Drawdowns(r, top = 5)

    # Max drawdown
    max_dd <- maxDrawdown(r)

    # Average drawdown
    dd <- Drawdowns(r)
    avg_dd <- mean(dd[dd < 0], na.rm = TRUE)

    # Time underwater (percentage of days in drawdown)
    pct_underwater <- sum(dd < 0, na.rm = TRUE) / length(dd) * 100

    stats <- rbind(stats, data.frame(
      Strategy = strategy,
      Max_Drawdown = max_dd * 100,
      Avg_Drawdown = abs(avg_dd) * 100,
      Pct_Underwater = pct_underwater
    ))
  }

  stats
}

#' Format Metrics Table for Report
#'
#' @param metrics Data frame from compute_all_metrics()
#' @return Formatted data frame for display
format_metrics_table <- function(metrics) {
  display_df <- data.frame(
    Strategy = metrics$Strategy,
    `Ann. Return` = metrics$Ann_Return_Pct,
    `Ann. Vol` = metrics$Ann_Volatility_Pct,
    Sharpe = sprintf("%.3f", metrics$Sharpe_Ratio),
    Sortino = sprintf("%.3f", metrics$Sortino_Ratio),
    Calmar = sprintf("%.3f", metrics$Calmar_Ratio),
    `Max DD` = metrics$Max_Drawdown_Pct,
    `VaR@95%` = metrics$VaR_95_Pct,
    `CVaR@95%` = metrics$CVaR_95_Pct,
    check.names = FALSE
  )

  display_df
}

#' Generate Report Entry Point
#'
#' Renders the R Markdown report.
#'
#' @param results Backtest results from run_backtest()
#' @param output_format "html" or "pdf"
#' @param output_dir Output directory
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
  saveRDS(results, file.path(output_path, "report_data.rds"))

  # Render report
  rmd_path <- file.path(project_root, "reports/report.Rmd")

  if (!file.exists(rmd_path)) {
    stop("Report template not found at: ", rmd_path)
  }

  output_file <- if (output_format == "pdf") "report.pdf" else "report.html"

  rmarkdown::render(
    input = rmd_path,
    output_format = if (output_format == "pdf") "pdf_document" else "html_document",
    output_file = output_file,
    output_dir = output_path,
    params = list(results_path = file.path(output_path, "report_data.rds")),
    quiet = FALSE
  )

  cat(sprintf("\nReport generated: %s/%s\n", output_path, output_file))

  invisible(file.path(output_path, output_file))
}

# ===========================================
# SHINY DASHBOARD HELPER FUNCTIONS
# ===========================================

#' Plot Current Allocation Pie Chart
#'
#' Creates a pie chart showing the latest portfolio allocation.
#'
#' @param weights_history xts object with weight history
#' @param title Plot title
#' @return ggplot object
plot_current_allocation <- function(weights_history, title = "Current Allocation") {
  # Get latest weights
  latest <- tail(weights_history, 1)

  df <- data.frame(
    Asset = colnames(latest),
    Weight = as.numeric(latest)
  )

  # Sort by weight descending for better visualization
  df <- df[order(-df$Weight), ]
  df$Asset <- factor(df$Asset, levels = df$Asset)

  # Calculate label positions
  df$ypos <- cumsum(df$Weight) - 0.5 * df$Weight

  ggplot(df, aes(x = "", y = Weight, fill = Asset)) +
    geom_bar(stat = "identity", width = 1, color = "white", linewidth = 0.5) +
    coord_polar("y", start = 0) +
    geom_text(
      aes(y = ypos, label = sprintf("%.1f%%", Weight * 100)),
      color = "white",
      fontface = "bold",
      size = 4
    ) +
    labs(title = title, fill = "Asset") +
    theme_void(base_size = 12) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
      legend.position = "right",
      legend.title = element_text(face = "bold")
    ) +
    scale_fill_brewer(palette = "Set2")
}

#' Compute Rolling VaR
#'
#' @param returns xts object with returns
#' @param window Rolling window size
#' @param confidence Confidence level (default 0.95)
#' @return xts object with rolling VaR
compute_rolling_var <- function(returns, window = 252, confidence = 0.95) {
  rollapply(returns, width = window,
            FUN = function(x) -quantile(x, 1 - confidence, na.rm = TRUE),
            align = "right", fill = NA)
}

#' Compute Rolling CVaR
#'
#' @param returns xts object with returns
#' @param window Rolling window size
#' @param confidence Confidence level (default 0.95)
#' @return xts object with rolling CVaR
compute_rolling_cvar <- function(returns, window = 252, confidence = 0.95) {
  rollapply(returns, width = window,
            FUN = function(x) {
              threshold <- quantile(x, 1 - confidence, na.rm = TRUE)
              -mean(x[x <= threshold], na.rm = TRUE)
            },
            align = "right", fill = NA)
}

#' Plot Rolling VaR and CVaR
#'
#' Creates a time series plot of rolling VaR and CVaR.
#'
#' @param returns xts object with returns (single or multiple series)
#' @param window Rolling window size (default 252 days)
#' @param confidence Confidence level (default 0.95)
#' @param title Plot title
#' @return ggplot object
plot_rolling_risk <- function(returns, window = 252, confidence = 0.95,
                               title = "Rolling VaR / CVaR") {
  # If multiple columns, use first one (or DSR_Constrained if available)
  if (ncol(returns) > 1) {
    col_name <- if ("DSR_Constrained" %in% colnames(returns)) {
      "DSR_Constrained"
    } else {
      colnames(returns)[1]
    }
    returns <- returns[, col_name]
  } else {
    col_name <- colnames(returns)[1]
  }

  # Compute rolling metrics
  var_data <- compute_rolling_var(returns, window, confidence)
  cvar_data <- compute_rolling_cvar(returns, window, confidence)

  # Combine into data frame
  df <- data.frame(
    Date = zoo::index(var_data),
    VaR = as.numeric(var_data) * 100,
    CVaR = as.numeric(cvar_data) * 100
  )

  # Remove NAs

  df <- df[complete.cases(df), ]

  if (nrow(df) == 0) {
    # Not enough data for rolling window
    return(
      ggplot() +
        annotate("text", x = 0.5, y = 0.5,
                 label = paste("Insufficient data for", window, "day rolling window"),
                 size = 5) +
        theme_void() +
        labs(title = title)
    )
  }

  # Reshape to long format
  df_long <- tidyr::pivot_longer(
    df,
    cols = c("VaR", "CVaR"),
    names_to = "Metric",
    values_to = "Value"
  )

  # Plot
  ggplot(df_long, aes(x = Date, y = Value, color = Metric)) +
    geom_line(linewidth = 0.8) +
    labs(
      title = title,
      subtitle = paste0("Strategy: ", col_name, " | Window: ", window, " days | Confidence: ", confidence * 100, "%"),
      x = "Date",
      y = "Risk (%)",
      color = "Metric"
    ) +
    theme_minimal(base_size = 11) +
    theme(
      legend.position = "bottom",
      plot.title = element_text(hjust = 0.5, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5, size = 9, color = "gray50")
    ) +
    scale_color_manual(values = c("VaR" = "#E8871E", "CVaR" = "#C73E1D")) +
    scale_y_continuous(labels = function(x) paste0(x, "%"))
}

#' Plot Weight Allocation Over Time (Stacked Area)
#'
#' Creates a stacked area chart of portfolio weights.
#'
#' @param weights_history xts object with weight history
#' @param title Plot title
#' @return ggplot object
plot_weight_allocation <- function(weights_history, title = "Weight Allocation Over Time") {
  # Convert to data frame
  weights_df <- data.frame(
    Date = zoo::index(weights_history),
    coredata(weights_history),
    check.names = FALSE
  )

  # Reshape to long format
  weights_long <- tidyr::pivot_longer(
    weights_df,
    cols = -Date,
    names_to = "Asset",
    values_to = "Weight"
  )

  # Plot
  ggplot(weights_long, aes(x = Date, y = Weight, fill = Asset)) +
    geom_area(alpha = 0.8, position = "stack") +
    labs(
      title = title,
      x = "Date",
      y = "Weight",
      fill = "Asset"
    ) +
    theme_minimal(base_size = 11) +
    theme(
      legend.position = "bottom",
      plot.title = element_text(hjust = 0.5, face = "bold")
    ) +
    scale_y_continuous(labels = scales::percent_format()) +
    scale_fill_brewer(palette = "Set2")
}
