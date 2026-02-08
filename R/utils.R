# R Utilities for DSR Portfolio Optimizer
# This file contains helper functions and reticulate configuration

library(reticulate)

# Set conda path for reticulate
CONDA_PATH <- file.path(Sys.getenv("HOME"), "miniconda3", "bin", "conda")
options(reticulate.conda_binary = CONDA_PATH)

#' Initialize Python Environment
#'
#' Sets up the conda environment for reticulate to use.
#' Call this function at the start of any R script that needs Python.
#'
#' @param env_name Name of the conda environment (default: "dsr-portfolio")
#' @return Invisible NULL
init_python_env <- function(env_name = "dsr-portfolio") {
  # Set conda binary path
  conda_binary(CONDA_PATH)

  # Try to use the conda environment
  tryCatch({
    use_condaenv(env_name, conda = CONDA_PATH, required = TRUE)
    message(paste("Using conda environment:", env_name))
  }, error = function(e) {
    warning(paste("Could not find conda environment:", env_name))
    warning("Please create it with: conda create -n dsr-portfolio python=3.10")
    warning("Error:", e$message)
  })

  invisible(NULL)
}

#' Get Python Configuration
#'
#' Returns information about the current Python configuration.
#'
#' @return List with Python configuration details
get_python_config <- function() {
  config <- py_config()
  list(
    python = config$python,
    version = config$version,
    numpy = py_module_available("numpy"),
    pandas = py_module_available("pandas"),
    torch = py_module_available("torch")
  )
}

#' Source Python Module
#'
#' Sources a Python file and returns it as an R object.
#'
#' @param module_path Path to the Python file
#' @return Python module as R object
source_python_module <- function(module_path) {
  source_python(module_path)
}

#' Get Project Root Directory
#'
#' Returns the absolute path to the project root directory.
#'
#' @return Character string with project root path
get_project_root <- function() {
  # Assuming this file is in R/ subdirectory
  normalizePath(file.path(dirname(sys.frame(1)$ofile), ".."))
}
