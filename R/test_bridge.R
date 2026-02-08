# Test R-Python Bridge via reticulate
# This script verifies that R can successfully call Python functions

# Load utilities
source("R/utils.R")

#' Run Bridge Tests
#'
#' Executes a series of tests to verify R-Python communication.
#'
#' @return Logical indicating if all tests passed
run_bridge_tests <- function() {
  cat("=== R-Python Bridge Test ===\n\n")

  # Initialize Python environment
  cat("1. Initializing Python environment...\n")
  init_python_env("dsr-portfolio")

  # Check Python config
  cat("\n2. Checking Python configuration...\n")
  config <- get_python_config()
  cat(paste("   Python:", config$python, "\n"))
  cat(paste("   Version:", config$version, "\n"))
  cat(paste("   NumPy available:", config$numpy, "\n"))
  cat(paste("   Pandas available:", config$pandas, "\n"))
  cat(paste("   PyTorch available:", config$torch, "\n"))

  # Source the test module
  cat("\n3. Sourcing Python test module...\n")
  source_python("python/test_bridge.py")

  # Test 1: Get equal weights
  cat("\n4. Testing get_test_weights()...\n")
  weights <- get_test_weights(5L)
  cat(paste("   Weights:", paste(round(weights, 4), collapse = ", "), "\n"))
  cat(paste("   Sum:", sum(weights), "\n"))

  test1_pass <- abs(sum(weights) - 1.0) < 1e-6
  cat(paste("   Test 1 PASSED:", test1_pass, "\n"))

  # Test 2: Get random weights
  cat("\n5. Testing get_random_weights()...\n")
  random_weights <- get_random_weights(5L, 42L)
  cat(paste("   Weights:", paste(round(random_weights, 4), collapse = ", "), "\n"))
  cat(paste("   Sum:", sum(random_weights), "\n"))

  test2_pass <- abs(sum(random_weights) - 1.0) < 1e-6
  cat(paste("   Test 2 PASSED:", test2_pass, "\n"))

  # Test 3: Array addition
  cat("\n6. Testing add_arrays()...\n")
  a <- c(1, 2, 3)
  b <- c(4, 5, 6)
  result <- add_arrays(a, b)
  expected <- c(5, 7, 9)
  cat(paste("   Input a:", paste(a, collapse = ", "), "\n"))
  cat(paste("   Input b:", paste(b, collapse = ", "), "\n"))
  cat(paste("   Result:", paste(result, collapse = ", "), "\n"))

  test3_pass <- all(result == expected)
  cat(paste("   Test 3 PASSED:", test3_pass, "\n"))

  # Test 4: Bridge info
  cat("\n7. Testing bridge_info()...\n")
  info <- bridge_info()
  cat(paste("   Python version:", info$python_version, "\n"))
  cat(paste("   NumPy version:", info$numpy_version, "\n"))

  test4_pass <- !is.null(info$python_version) && !is.null(info$numpy_version)
  cat(paste("   Test 4 PASSED:", test4_pass, "\n"))

  # Summary
  all_pass <- all(c(test1_pass, test2_pass, test3_pass, test4_pass))
  cat("\n=== Test Summary ===\n")
  cat(paste("Tests passed: ", sum(c(test1_pass, test2_pass, test3_pass, test4_pass)), "/4\n"))

  if (all_pass) {
    cat("\nSUCCESS: R-Python bridge is working correctly!\n")
  } else {
    cat("\nFAILURE: Some tests failed. Check the output above.\n")
  }

  return(all_pass)
}

# Run tests if executed directly
if (!interactive()) {
  success <- run_bridge_tests()
  quit(status = if (success) 0 else 1)
}
