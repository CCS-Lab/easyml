#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
measure_mean_squared_error <- function(y_true, y_pred) {
  scorer::mean_squared_error(y_true, y_pred)
}

#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
measure_r2_score <- scorer::r2_score

#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
measure_area_under_curve <- function(y_true, y_pred) {
  as.numeric(pROC::roc(y_true, y_pred)$auc)
}
