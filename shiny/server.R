# server.R - DSR Portfolio Optimizer Dashboard Server Logic

function(input, output, session) {

  # ===========================================
  # REACTIVE DATA FILTERING
  # ===========================================

  # Filtered returns based on date range and strategy selection
  filtered_returns <- reactive({
    req(input$date_range, input$strategies)

    start <- input$date_range[1]
    end <- input$date_range[2]

    # Filter by date range
    subset_returns <- combined_returns[paste0(start, "/", end)]

    # Filter by selected strategies
    if (length(input$strategies) > 0) {
      cols <- intersect(input$strategies, colnames(subset_returns))
      if (length(cols) > 0) {
        subset_returns <- subset_returns[, cols, drop = FALSE]
      }
    }

    subset_returns
  })

  # Filtered weights
  filtered_weights <- reactive({
    req(input$date_range)

    start <- input$date_range[1]
    end <- input$date_range[2]

    weights_history[paste0(start, "/", end)]
  })

  # Computed metrics for filtered data
  filtered_metrics <- reactive({
    req(filtered_returns())
    compute_all_metrics(filtered_returns())
  })

  # ===========================================
  # PAGE 1: PORTFOLIO OVERVIEW
  # ===========================================

  # Metric Cards
  output$box_return <- renderValueBox({
    req(filtered_metrics())
    metrics <- filtered_metrics()
    # Use DSR_Constrained as primary, or first available
    primary <- if ("DSR_Constrained" %in% metrics$Strategy) "DSR_Constrained" else metrics$Strategy[1]
    val <- metrics$Ann_Return[metrics$Strategy == primary]
    valueBox(
      value = sprintf("%.1f%%", val * 100),
      subtitle = "Annualized Return",
      icon = icon("arrow-trend-up"),
      color = if (val > 0) "green" else "red"
    )
  })

  output$box_volatility <- renderValueBox({
    req(filtered_metrics())
    metrics <- filtered_metrics()
    primary <- if ("DSR_Constrained" %in% metrics$Strategy) "DSR_Constrained" else metrics$Strategy[1]
    val <- metrics$Ann_Volatility[metrics$Strategy == primary]
    valueBox(
      value = sprintf("%.1f%%", val * 100),
      subtitle = "Annualized Volatility",
      icon = icon("chart-line"),
      color = "blue"
    )
  })

  output$box_sharpe <- renderValueBox({
    req(filtered_metrics())
    metrics <- filtered_metrics()
    primary <- if ("DSR_Constrained" %in% metrics$Strategy) "DSR_Constrained" else metrics$Strategy[1]
    val <- metrics$Sharpe_Ratio[metrics$Strategy == primary]
    valueBox(
      value = sprintf("%.3f", val),
      subtitle = "Sharpe Ratio",
      icon = icon("star"),
      color = if (val > 1) "green" else if (val > 0) "yellow" else "red"
    )
  })

  output$box_maxdd <- renderValueBox({
    req(filtered_metrics())
    metrics <- filtered_metrics()
    primary <- if ("DSR_Constrained" %in% metrics$Strategy) "DSR_Constrained" else metrics$Strategy[1]
    val <- metrics$Max_Drawdown[metrics$Strategy == primary]
    valueBox(
      value = sprintf("%.1f%%", val * 100),
      subtitle = "Maximum Drawdown",
      icon = icon("arrow-trend-down"),
      color = if (abs(val) < 0.1) "green" else if (abs(val) < 0.2) "yellow" else "red"
    )
  })

  # Equity Curve Plot
  output$equity_curve <- renderPlot({
    req(filtered_returns())
    plot_equity_curves(filtered_returns(), title = "Cumulative Portfolio Performance")
  })

  # Metrics Table
  output$metrics_table <- DT::renderDataTable({
    req(filtered_metrics())
    metrics <- filtered_metrics()

    display_df <- data.frame(
      Strategy = metrics$Strategy,
      `Ann. Return` = sprintf("%.2f%%", metrics$Ann_Return * 100),
      `Ann. Volatility` = sprintf("%.2f%%", metrics$Ann_Volatility * 100),
      `Sharpe` = sprintf("%.3f", metrics$Sharpe_Ratio),
      `Sortino` = sprintf("%.3f", metrics$Sortino_Ratio),
      `Max DD` = sprintf("%.2f%%", metrics$Max_Drawdown * 100),
      `VaR (95%)` = sprintf("%.2f%%", metrics$VaR_95 * 100),
      `CVaR (95%)` = sprintf("%.2f%%", metrics$CVaR_95 * 100),
      `Calmar` = sprintf("%.3f", metrics$Calmar_Ratio),
      check.names = FALSE
    )

    DT::datatable(
      display_df,
      options = list(
        dom = 't',
        pageLength = 10,
        ordering = FALSE
      ),
      rownames = FALSE
    )
  })

  # ===========================================
  # PAGE 2: WEIGHT ALLOCATION
  # ===========================================

  # Current Allocation Pie Chart
  output$current_allocation_pie <- renderPlot({
    req(filtered_weights())
    plot_current_allocation(filtered_weights(), title = "Latest Portfolio Allocation")
  })

  # HHI Value Box
  output$box_hhi <- renderValueBox({
    req(filtered_weights())
    hhi <- mean(compute_hhi(filtered_weights()), na.rm = TRUE)
    n_assets <- ncol(filtered_weights())
    min_hhi <- 1 / n_assets

    valueBox(
      value = sprintf("%.4f", hhi),
      subtitle = paste0("Avg HHI (min: ", sprintf("%.4f", min_hhi), ")"),
      icon = icon("chart-pie"),
      color = if (hhi < 0.3) "green" else if (hhi < 0.5) "yellow" else "red"
    )
  })

  # Max Weight Value Box
  output$box_max_weight <- renderValueBox({
    req(filtered_weights())
    max_weight <- max(filtered_weights(), na.rm = TRUE)

    valueBox(
      value = sprintf("%.1f%%", max_weight * 100),
      subtitle = "Max Single Position",
      icon = icon("maximize"),
      color = if (max_weight <= 0.30) "green" else "yellow"
    )
  })

  # Weight Stats Table
  output$weight_stats_table <- DT::renderDataTable({
    req(filtered_weights())
    stats <- compute_weight_stats(filtered_weights())

    display_df <- data.frame(
      Asset = stats$Asset,
      `Mean (%)` = sprintf("%.1f", stats$Mean_Weight),
      `Std (%)` = sprintf("%.1f", stats$Std_Weight),
      `Min (%)` = sprintf("%.1f", stats$Min_Weight),
      `Max (%)` = sprintf("%.1f", stats$Max_Weight),
      check.names = FALSE
    )

    DT::datatable(
      display_df,
      options = list(dom = 't', pageLength = 10, ordering = FALSE),
      rownames = FALSE
    )
  })

  # Weight Allocation Stacked Area
  output$weight_allocation <- renderPlot({
    req(filtered_weights())
    plot_weight_allocation(filtered_weights(), title = "Portfolio Weight Allocation Over Time")
  })

  # ===========================================
  # PAGE 3: ANALYTICS
  # ===========================================

  # Monthly Heatmap
  output$monthly_heatmap <- renderPlot({
    req(filtered_returns())
    plot_monthly_heatmap(filtered_returns(), title = "Monthly Returns by Strategy (%)")
  })

  # Rolling Sharpe
  output$rolling_sharpe <- renderPlot({
    req(filtered_returns(), input$rolling_window)
    plot_rolling_metrics(
      filtered_returns(),
      metric = "sharpe",
      window = input$rolling_window,
      title = paste0("Rolling ", input$rolling_window, "-Day Sharpe Ratio")
    )
  })

  # Rolling Volatility
  output$rolling_volatility <- renderPlot({
    req(filtered_returns(), input$rolling_window)
    plot_rolling_metrics(
      filtered_returns(),
      metric = "volatility",
      window = input$rolling_window,
      title = paste0("Rolling ", input$rolling_window, "-Day Volatility")
    )
  })

  # Regime Analysis Table
  output$regime_table <- DT::renderDataTable({
    req(filtered_returns())

    # Use EqualWeight as market proxy if available
    ew_col <- grep("Equal", colnames(filtered_returns()), value = TRUE)
    if (length(ew_col) > 0) {
      regimes <- identify_high_vol_regimes(filtered_returns()[, ew_col], threshold = 0.25, window = 20)
    } else {
      regimes <- identify_high_vol_regimes(filtered_returns()[, 1], threshold = 0.25, window = 20)
    }

    if (nrow(regimes) > 0) {
      regime_perf <- compute_regime_performance(filtered_returns(), regimes)

      if (nrow(regime_perf) > 0) {
        display_df <- data.frame(
          Regime = regime_perf$Regime,
          Strategy = regime_perf$Strategy,
          `Period` = paste(regime_perf$Start, "to", regime_perf$End),
          `Days` = regime_perf$Days,
          `Total Return` = sprintf("%.1f%%", regime_perf$Total_Return),
          `Volatility` = sprintf("%.1f%%", regime_perf$Volatility),
          `Max DD` = sprintf("%.1f%%", regime_perf$Max_Drawdown),
          check.names = FALSE
        )
      } else {
        display_df <- data.frame(Message = "No regime performance data available")
      }
    } else {
      display_df <- data.frame(Message = "No high-volatility regimes (>25% annualized) identified")
    }

    DT::datatable(
      display_df,
      options = list(dom = 't', pageLength = 20, ordering = TRUE),
      rownames = FALSE
    )
  })

  # ===========================================
  # PAGE 4: RISK MONITOR
  # ===========================================

  # Risk Value Boxes
  output$box_var <- renderValueBox({
    req(filtered_metrics())
    metrics <- filtered_metrics()
    primary <- if ("DSR_Constrained" %in% metrics$Strategy) "DSR_Constrained" else metrics$Strategy[1]
    val <- metrics$VaR_95[metrics$Strategy == primary]
    valueBox(
      value = sprintf("%.2f%%", val * 100),
      subtitle = "VaR (95%)",
      icon = icon("exclamation-triangle"),
      color = "yellow"
    )
  })

  output$box_cvar <- renderValueBox({
    req(filtered_metrics())
    metrics <- filtered_metrics()
    primary <- if ("DSR_Constrained" %in% metrics$Strategy) "DSR_Constrained" else metrics$Strategy[1]
    val <- metrics$CVaR_95[metrics$Strategy == primary]
    valueBox(
      value = sprintf("%.2f%%", val * 100),
      subtitle = "CVaR (95%)",
      icon = icon("exclamation-circle"),
      color = "orange"
    )
  })

  output$box_calmar <- renderValueBox({
    req(filtered_metrics())
    metrics <- filtered_metrics()
    primary <- if ("DSR_Constrained" %in% metrics$Strategy) "DSR_Constrained" else metrics$Strategy[1]
    val <- metrics$Calmar_Ratio[metrics$Strategy == primary]
    valueBox(
      value = sprintf("%.3f", val),
      subtitle = "Calmar Ratio",
      icon = icon("balance-scale"),
      color = if (val > 1) "green" else "blue"
    )
  })

  output$box_sortino <- renderValueBox({
    req(filtered_metrics())
    metrics <- filtered_metrics()
    primary <- if ("DSR_Constrained" %in% metrics$Strategy) "DSR_Constrained" else metrics$Strategy[1]
    val <- metrics$Sortino_Ratio[metrics$Strategy == primary]
    valueBox(
      value = sprintf("%.3f", val),
      subtitle = "Sortino Ratio",
      icon = icon("shield-alt"),
      color = if (val > 0.2) "green" else "blue"
    )
  })

  # Underwater Chart
  output$underwater_chart <- renderPlot({
    req(filtered_returns())
    plot_underwater(filtered_returns(), title = "Drawdown (Underwater) Chart")
  })

  # Efficient Frontier
  output$efficient_frontier <- renderPlot({
    # Use full price history for frontier calculation
    tryCatch({
      plot_efficient_frontier(
        results$prices,
        results$dsr$weights_history,
        title = "Efficient Frontier"
      )
    }, error = function(e) {
      # Fallback: generate simple frontier
      generate_efficient_frontier_plot(results)
    })
  })

  # Rolling VaR/CVaR
  output$rolling_risk <- renderPlot({
    req(filtered_returns(), input$rolling_window)
    plot_rolling_risk(
      filtered_returns(),
      window = input$rolling_window,
      confidence = 0.95,
      title = paste0("Rolling ", input$rolling_window, "-Day VaR/CVaR (95%)")
    )
  })

  # ===========================================
  # DOWNLOAD HANDLER
  # ===========================================

  output$download_report <- downloadHandler(
    filename = function() {
      paste0("dsr_portfolio_report_", format(Sys.Date(), "%Y%m%d"), ".html")
    },
    content = function(file) {
      # Serve pre-generated report
      report_path <- "../reports/report.html"
      if (file.exists(report_path)) {
        file.copy(report_path, file)
      } else {
        # Generate fresh report if not available
        tryCatch({
          generate_full_report(results, output_format = "html", output_dir = dirname(file))
          file.rename(file.path(dirname(file), "report.html"), file)
        }, error = function(e) {
          # Create simple error message file
          writeLines(paste("Report generation failed:", e$message), file)
        })
      }
    }
  )
}
