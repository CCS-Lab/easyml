#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @param y_true Ground truth (correct) target values.
#' @param y_pred Estimated target values.
#' @return TO BE EDITED.
#' @export
plot_roc_auc <- function(y_true, y_pred) {
  results <- pROC::roc(y_true, y_pred)
  auc <- as.numeric(results$auc)
  auc_label <- paste("AUC = ", auc, sep = "")
  df <- data.frame(sensitivities = results$sensitivities, 
                   one_minus_specificities = 1 - results$specificities, 
                   stringsAsFactors = FALSE)
  g <- ggplot2::ggplot(df, ggplot2::aes(x = sensitivities, y = one_minus_specificities)) +
    ggplot2::geom_path(alpha = 1, size = 1) +
    ggplot2::geom_segment(ggplot2::aes(x = 0, y = 0, xend = 1, yend = 1) , linetype = "dashed") +
    ggplot2::scale_x_continuous("1 - Specificity") + 
    ggplot2::scale_y_continuous("Sensitivity") + 
    ggplot2::ggtitle("ROC Curve") + 
    ggplot2::theme_bw()
  return(g)
}

#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @param aucs TO BE EDITED.
#' @return TO BE EDITED.
#' @export
plot_auc_histogram <- function(aucs) {
  df <- data.frame(aucs = aucs)
  g <- ggplot2::ggplot(df, ggplot2::aes(x = aucs)) +
    ggplot2::geom_histogram() + 
    ggplot2::scale_x_continuous("AUC") + 
    ggplot2::scale_y_continuous("Frequency", label = scales::comma) + 
    ggplot2::ggtitle("Distribution of AUCS") + 
    ggplot2::theme_bw()
  return(g)
}
