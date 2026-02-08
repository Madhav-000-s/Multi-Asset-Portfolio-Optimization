# Efficient Frontier Analysis for DSR Portfolio Optimizer
# Computes mean-variance frontier and plots portfolio positions

library(ggplot2)
library(xts)

#' Compute Portfolio Statistics
#'
#' @param weights Portfolio weight vector
#' @param returns xts object of asset returns
#' @param scale Annualization factor (default 252)
#' @return List with annualized return and volatility
compute_portfolio_stats <- function(weights, returns, scale = 252) {
  # Ensure weights match returns columns
  if (length(weights) != ncol(returns)) {
    stop("Weight length must match number of assets")
  }

  # Compute portfolio returns
  port_returns <- as.numeric(returns %*% weights)

  # Annualized return
  mean_ret <- mean(port_returns, na.rm = TRUE)
  ann_return <- mean_ret * scale

  # Annualized volatility
  std_ret <- sd(port_returns, na.rm = TRUE)
  ann_vol <- std_ret * sqrt(scale)

  list(
    return = ann_return,
    volatility = ann_vol,
    sharpe = if (ann_vol > 0) ann_return / ann_vol else 0
  )
}

#' Generate Random Portfolios for Frontier Approximation
#'
#' @param returns xts object of asset returns
#' @param n_portfolios Number of random portfolios to generate
#' @param max_weight Maximum weight per asset (default 1.0 = no limit)
#' @return Data frame with return, volatility, and weights for each portfolio
generate_random_portfolios <- function(returns, n_portfolios = 1000, max_weight = 1.0) {
  n_assets <- ncol(returns)
  results <- data.frame(
    Return = numeric(n_portfolios),
    Volatility = numeric(n_portfolios),
    Sharpe = numeric(n_portfolios)
  )

  # Store weights
  weight_matrix <- matrix(0, nrow = n_portfolios, ncol = n_assets)
  colnames(weight_matrix) <- colnames(returns)

  for (i in seq_len(n_portfolios)) {
    # Generate random weights (Dirichlet-like)
    weights <- runif(n_assets)

    # Apply max weight constraint
    if (max_weight < 1.0) {
      # Iteratively cap weights
      for (j in 1:10) {
        weights <- pmin(weights, max_weight)
        weights <- weights / sum(weights)
      }
    } else {
      weights <- weights / sum(weights)
    }

    # Compute stats
    stats <- compute_portfolio_stats(weights, returns)
    results$Return[i] <- stats$return
    results$Volatility[i] <- stats$volatility
    results$Sharpe[i] <- stats$sharpe
    weight_matrix[i, ] <- weights
  }

  results$weights <- weight_matrix
  results
}

#' Compute Efficient Frontier Points
#'
#' Uses optimization to find efficient portfolios across risk levels.
#'
#' @param returns xts object of asset returns
#' @param n_points Number of frontier points
#' @param max_weight Maximum weight per asset
#' @return Data frame with frontier portfolios
compute_efficient_frontier <- function(returns, n_points = 50, max_weight = 0.30) {
  n_assets <- ncol(returns)

  # First generate random portfolios to estimate return range
  random_ports <- generate_random_portfolios(returns, 500, max_weight)

  min_ret <- min(random_ports$Return)
  max_ret <- max(random_ports$Return)

  # Target returns across the range

  target_returns <- seq(min_ret, max_ret, length.out = n_points)

  frontier <- data.frame(
    Return = numeric(n_points),
    Volatility = numeric(n_points),
    Sharpe = numeric(n_points)
  )

  weight_matrix <- matrix(0, nrow = n_points, ncol = n_assets)
  colnames(weight_matrix) <- colnames(returns)

  # For each target return, find minimum variance portfolio
  for (i in seq_len(n_points)) {
    target <- target_returns[i]

    # Simple grid search approach (more robust than quadprog for small N)
    best_vol <- Inf
    best_weights <- rep(1/n_assets, n_assets)

    # Generate candidates near target return
    for (j in 1:200) {
      weights <- runif(n_assets)
      for (k in 1:10) {
        weights <- pmin(weights, max_weight)
        weights <- weights / sum(weights)
      }

      stats <- compute_portfolio_stats(weights, returns)

      # Accept if return is close to target and vol is lower
      if (abs(stats$return - target) < 0.05 && stats$volatility < best_vol) {
        best_vol <- stats$volatility
        best_weights <- weights
      }
    }

    stats <- compute_portfolio_stats(best_weights, returns)
    frontier$Return[i] <- stats$return
    frontier$Volatility[i] <- stats$volatility
    frontier$Sharpe[i] <- stats$sharpe
    weight_matrix[i, ] <- best_weights
  }

  frontier$weights <- weight_matrix

  # Sort by volatility and remove dominated points
  frontier <- frontier[order(frontier$Volatility), ]

  # Keep only efficient points (higher return for same or higher vol)
  efficient_mask <- rep(TRUE, nrow(frontier))
  max_return_so_far <- -Inf
  for (i in seq_len(nrow(frontier))) {
    if (frontier$Return[i] >= max_return_so_far) {
      max_return_so_far <- frontier$Return[i]
    } else {
      efficient_mask[i] <- FALSE
    }
  }

  frontier[efficient_mask, ]
}

#' Locate Portfolio on Risk-Return Space
#'
#' @param weights Portfolio weights
#' @param returns Asset returns
#' @param name Portfolio name
#' @return Data frame row with portfolio stats
locate_portfolio <- function(weights, returns, name = "Portfolio") {
  stats <- compute_portfolio_stats(weights, returns)

  data.frame(
    Name = name,
    Return = stats$return,
    Volatility = stats$volatility,
    Sharpe = stats$sharpe
  )
}

#' Plot Efficient Frontier with Portfolios
#'
#' @param returns Asset returns
#' @param portfolios List of portfolios to mark, each with $weights and $name
#' @param n_points Number of frontier points
#' @param max_weight Max weight constraint
#' @param save_path Optional path to save plot
#' @return ggplot object
plot_efficient_frontier <- function(
  returns,
  portfolios = list(),
  n_points = 50,
  max_weight = 0.30,
  save_path = NULL
) {
  cat("Computing efficient frontier...\n")

  # Compute frontier
  frontier <- compute_efficient_frontier(returns, n_points, max_weight)

  # Generate random portfolios for background
  random_ports <- generate_random_portfolios(returns, 500, max_weight)

  # Locate specified portfolios
  portfolio_points <- data.frame()
  for (port in portfolios) {
    if (!is.null(port$weights) && !is.null(port$name)) {
      point <- locate_portfolio(port$weights, returns, port$name)
      portfolio_points <- rbind(portfolio_points, point)
    }
  }

  # Add equal weight for reference
  n_assets <- ncol(returns)
  ew_weights <- rep(1/n_assets, n_assets)
  ew_point <- locate_portfolio(ew_weights, returns, "Equal Weight")
  portfolio_points <- rbind(portfolio_points, ew_point)

  # Create plot
  p <- ggplot() +
    # Random portfolios (feasible region)
    geom_point(
      data = random_ports,
      aes(x = Volatility * 100, y = Return * 100),
      color = "lightgray",
      alpha = 0.3,
      size = 1
    ) +
    # Efficient frontier line
    geom_line(
      data = frontier,
      aes(x = Volatility * 100, y = Return * 100),
      color = "#2E86AB",
      linewidth = 1.5
    ) +
    geom_point(
      data = frontier,
      aes(x = Volatility * 100, y = Return * 100),
      color = "#2E86AB",
      size = 2
    ) +
    # Mark specific portfolios
    geom_point(
      data = portfolio_points,
      aes(x = Volatility * 100, y = Return * 100, color = Name),
      size = 4,
      shape = 18
    ) +
    geom_text(
      data = portfolio_points,
      aes(x = Volatility * 100, y = Return * 100, label = Name, color = Name),
      vjust = -1,
      hjust = 0.5,
      size = 3.5,
      fontface = "bold"
    ) +
    labs(
      title = "Efficient Frontier",
      subtitle = sprintf("Max weight per asset: %.0f%%", max_weight * 100),
      x = "Annualized Volatility (%)",
      y = "Annualized Return (%)",
      color = "Portfolio"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      legend.position = "bottom",
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5, size = 10),
      panel.grid.minor = element_blank()
    ) +
    scale_color_brewer(palette = "Set1")

  if (!is.null(save_path)) {
    ggsave(save_path, p, width = 10, height = 8, dpi = 150)
    cat(sprintf("Saved efficient frontier plot to %s\n", save_path))
  }

  p
}

#' Generate Efficient Frontier from Backtest Results
#'
#' @param results Results from run_backtest()
#' @param output_dir Directory for output
#' @return ggplot object
generate_efficient_frontier_plot <- function(results, output_dir = "results") {
  # Get returns for frontier calculation
  # Use the full test period returns
  returns <- compute_returns(results$prices)
  test_start <- results$test_period[1]
  test_end <- results$test_period[2]
  returns <- returns[zoo::index(returns) >= test_start & zoo::index(returns) <= test_end]

  # Prepare portfolios to mark
  portfolios <- list()

  # Add DSR (primary - constrained if available)
  if (!is.null(results$dsr$weights_history)) {
    # Use average weights over the period
    avg_weights <- colMeans(results$dsr$weights_history)
    portfolios[[length(portfolios) + 1]] <- list(
      weights = avg_weights,
      name = if (results$use_constraints %||% FALSE) "DSR (Constrained)" else "DSR"
    )
  }

  # Add unconstrained DSR if we have it
  if (!is.null(results$dsr_unconstrained)) {
    avg_weights <- colMeans(results$dsr_unconstrained$weights_history)
    portfolios[[length(portfolios) + 1]] <- list(
      weights = avg_weights,
      name = "DSR (Unconstrained)"
    )
  }

  # Get max weight from config
  config <- yaml::read_yaml(file.path(get_project_root(), "config.yaml"))
  max_weight <- config$constraints$max_position_weight %||% 0.30

  # Get project root for save path
  project_root <- get_project_root()
  save_path <- file.path(project_root, output_dir, "efficient_frontier.png")

  # Plot
  p <- plot_efficient_frontier(
    returns,
    portfolios,
    n_points = config$efficient_frontier$n_points %||% 50,
    max_weight = max_weight,
    save_path = save_path
  )

  p
}

# Helper: null coalescing
if (!exists("%||%")) {
  `%||%` <- function(x, y) if (is.null(x)) y else x
}

# Helper: get project root (if not already defined)
if (!exists("get_project_root")) {
  get_project_root <- function() {
    if (file.exists("config.yaml")) {
      return(normalizePath("."))
    }
    if (file.exists("../config.yaml")) {
      return(normalizePath(".."))
    }
    getwd()
  }
}

# Helper: compute returns (if not already defined)
if (!exists("compute_returns")) {
  compute_returns <- function(prices) {
    diff(prices) / stats::lag(prices, 1)
  }
}
