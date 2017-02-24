#' Measure mean squared error.
#' 
#' Given the ground truth (correct) target values and the estimated target 
#' values, calculates the mean squared error metric.
#'
#' @param y_true A numeric vector; the ground truth (correct) target values.
#' @param y_pred A numeric vector; the estimated target values.
#' @return A numeric vector of length one; the mean squared error metric.
#' @family measure
#' @export
measure_mean_squared_error <- scorer::mean_squared_error

#' Measure R^2 score.
#' 
#' Given the ground truth (correct) target values and the estimated target 
#' values, calculates the the R^2 metric.
#'
#' @param y_true A numeric vector; the ground truth (correct) target values.
#' @param y_pred A numeric vector; the estimated target values.
#' @return A numeric vector of length one; the R^2 metric.
#' @family measure
#' @export
measure_r2_score <- scorer::r2_score

#' Measure area under the curve.
#' 
#' Given the ground truth (correct) target values and the estimated target 
#' values, calculates the the AUC metric.
#'
#' @param y_true A numeric vector; the ground truth (correct) target values.
#' @param y_pred A numeric vector; the estimated target values.
#' @return A numeric vector of length one; the AUC metric.
#' @family measure
#' @export
measure_area_under_curve <- function(y_true, y_pred) {
  as.numeric(pROC::roc(as.numeric(y_true), as.numeric(y_pred))$auc)
}
