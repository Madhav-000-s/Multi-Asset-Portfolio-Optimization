# ui.R - DSR Portfolio Optimizer Dashboard UI

dashboardPage(
  skin = "blue",

  # Header
  dashboardHeader(
    title = "DSR Portfolio Optimizer",
    titleWidth = 280
  ),

# Sidebar
dashboardSidebar(
  width = 280,
  sidebarMenu(
    id = "tabs",
    menuItem("Portfolio Overview", tabName = "overview", icon = icon("chart-line")),
    menuItem("Weight Allocation", tabName = "weights", icon = icon("chart-pie")),
    menuItem("Analytics", tabName = "analytics", icon = icon("chart-bar")),
    menuItem("Risk Monitor", tabName = "risk", icon = icon("shield-alt"))
  ),

  hr(),

  # Date Range Filter
  dateRangeInput(
    "date_range",
    "Date Range:",
    start = test_period[1],
    end = test_period[2],
    min = test_period[1],
    max = test_period[2],
    format = "yyyy-mm-dd"
  ),

  # Strategy Selector
  checkboxGroupInput(
    "strategies",
    "Strategies:",
    choices = c(
      "DSR (Constrained)" = "DSR_Constrained",
      "DSR (Unconstrained)" = "DSR_Unconstrained",
      "Equal Weight" = "EqualWeight"
    ),
    selected = c("DSR_Constrained", "EqualWeight")
  ),

  # Rolling Window Slider
  sliderInput(
    "rolling_window",
    "Rolling Window (days):",
    min = 21,
    max = 252,
    value = 126,
    step = 21
  ),

  hr(),

  # Download Button
  downloadButton("download_report", "Download Report", class = "btn-primary btn-block"),

  br(),

  # Info text
  div(
    style = "padding: 10px; font-size: 11px; color: #888;",
    p("Data: Dec 2023 - Dec 2024"),
    p("Rebalance: Weekly"),
    p("Assets: AAPL, MSFT, GOOGL, AMZN, META")
  )
),

# Body
dashboardBody(
  # Custom CSS
  tags$head(
    tags$style(HTML("
      .content-wrapper { background-color: #f4f6f9; }
      .box { border-radius: 5px; }
      .small-box { border-radius: 5px; }
      .value-box-title { font-size: 14px; }
      .plot-container { min-height: 400px; }
      .info-box { min-height: 90px; }
      .nav-tabs-custom > .tab-content { padding: 15px; }
    "))
  ),

  tabItems(
    # ===========================================
    # PAGE 1: PORTFOLIO OVERVIEW
    # ===========================================
    tabItem(
      tabName = "overview",

      # Metric Cards Row
      fluidRow(
        valueBoxOutput("box_return", width = 3),
        valueBoxOutput("box_volatility", width = 3),
        valueBoxOutput("box_sharpe", width = 3),
        valueBoxOutput("box_maxdd", width = 3)
      ),

      # Equity Curve
      fluidRow(
        box(
          title = "Cumulative Portfolio Performance",
          status = "primary",
          solidHeader = TRUE,
          width = 12,
          plotOutput("equity_curve", height = "400px")
        )
      ),

      # Metrics Table
      fluidRow(
        box(
          title = "Performance Metrics Comparison",
          status = "info",
          solidHeader = TRUE,
          width = 12,
          DT::dataTableOutput("metrics_table")
        )
      )
    ),

    # ===========================================
    # PAGE 2: WEIGHT ALLOCATION
    # ===========================================
    tabItem(
      tabName = "weights",

      # Current Allocation + HHI
      fluidRow(
        box(
          title = "Current Allocation",
          status = "primary",
          solidHeader = TRUE,
          width = 6,
          plotOutput("current_allocation_pie", height = "350px")
        ),
        box(
          title = "Portfolio Concentration",
          status = "info",
          solidHeader = TRUE,
          width = 6,
          fluidRow(
            valueBoxOutput("box_hhi", width = 6),
            valueBoxOutput("box_max_weight", width = 6)
          ),
          DT::dataTableOutput("weight_stats_table")
        )
      ),

      # Stacked Area Chart
      fluidRow(
        box(
          title = "Weight Allocation Over Time",
          status = "primary",
          solidHeader = TRUE,
          width = 12,
          plotOutput("weight_allocation", height = "400px")
        )
      )
    ),

    # ===========================================
    # PAGE 3: ANALYTICS
    # ===========================================
    tabItem(
      tabName = "analytics",

      # Monthly Heatmap
      fluidRow(
        box(
          title = "Monthly Returns Heatmap",
          status = "primary",
          solidHeader = TRUE,
          width = 12,
          plotOutput("monthly_heatmap", height = "500px")
        )
      ),

      # Rolling Metrics Row
      fluidRow(
        box(
          title = "Rolling Sharpe Ratio",
          status = "info",
          solidHeader = TRUE,
          width = 6,
          plotOutput("rolling_sharpe", height = "300px")
        ),
        box(
          title = "Rolling Volatility",
          status = "info",
          solidHeader = TRUE,
          width = 6,
          plotOutput("rolling_volatility", height = "300px")
        )
      ),

      # Regime Analysis
      fluidRow(
        box(
          title = "High Volatility Regime Analysis",
          status = "warning",
          solidHeader = TRUE,
          width = 12,
          DT::dataTableOutput("regime_table")
        )
      )
    ),

    # ===========================================
    # PAGE 4: RISK MONITOR
    # ===========================================
    tabItem(
      tabName = "risk",

      # Risk Metric Cards
      fluidRow(
        valueBoxOutput("box_var", width = 3),
        valueBoxOutput("box_cvar", width = 3),
        valueBoxOutput("box_calmar", width = 3),
        valueBoxOutput("box_sortino", width = 3)
      ),

      # Drawdown Chart
      fluidRow(
        box(
          title = "Drawdown (Underwater) Chart",
          status = "danger",
          solidHeader = TRUE,
          width = 12,
          plotOutput("underwater_chart", height = "350px")
        )
      ),

      # Efficient Frontier + Rolling Risk
      fluidRow(
        box(
          title = "Efficient Frontier",
          status = "primary",
          solidHeader = TRUE,
          width = 6,
          plotOutput("efficient_frontier", height = "400px")
        ),
        box(
          title = "Rolling VaR / CVaR (95%)",
          status = "warning",
          solidHeader = TRUE,
          width = 6,
          plotOutput("rolling_risk", height = "400px")
        )
      )
    )
  )
)
)
