#' easyml: Easily build and evaluate machine learning models. 
#'
#' @docType package
#' @name easyml
NULL

.onAttach <- function(libname, pkgname) {
  packageStartupMessage("Loaded easyml 0.1.0. Also loading ggplot2.")
  requireNamespace("ggplot2")
}
