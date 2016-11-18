#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @param y_true Ground truth (correct) target values.
#' @param y_pred Estimated target values.
#' @return TO BE EDITED.
#' @export
plot_betas <- function(betas) {
  g <- 
    ggplot2::ggplot(betas, ggplot2::aes(x = reorder(predictor, -(1:nrow(betas))), 
                                        y = mean, colour = dotColor)) +
    ggplot2::geom_errorbar(ggplot2::aes(ymin = lb, ymax = ub), width = 0.1) +
    ggplot2::geom_line() +
    ggplot2::geom_point() +
    ggplot2::scale_x_discrete("Predictors") +
    ggplot2::scale_y_continuous("Beta estimates") + 
    ggplot2::scale_color_manual("", values = c("0" = "grey", "2" = "black"), 
                                labels = c("0" = "Insignificant", "2" = "Significant")) + 
    ggplot2::ggtitle("Beta estimates of predictors") + 
    ggplot2::theme_bw() + 
    ggplot2::coord_flip()
  
  g
}

#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @param y_true Ground truth (correct) target values.
#' @param y_pred Estimated target values.
#' @return TO BE EDITED.
#' @export
plot_roc_curve <- function(y_true, y_pred) {
  results <- pROC::roc(y_true, y_pred)
  auc <- as.numeric(results$auc)
  auc_label <- paste("AUC = ", round(auc, digits = 3), sep = "")
  df <- data.frame(sensitivities = results$sensitivities, 
                   one_minus_specificities = 1 - results$specificities, 
                   stringsAsFactors = FALSE)
  
  g <- 
    ggplot2::ggplot(df, ggplot2::aes(x = one_minus_specificities, y = sensitivities)) +
    ggplot2::geom_path(alpha = 1, size = 1) +
    ggplot2::geom_segment(ggplot2::aes(x = 0, y = 0, xend = 1, yend = 1) , linetype = "dashed") + 
    ggplot2::annotate("text", label = auc_label, x = 0.6, y = 0.1, size = 8) + 
    ggplot2::scale_x_continuous("1 - Specificity") + 
    ggplot2::scale_y_continuous("Sensitivity") + 
    ggplot2::ggtitle("ROC Curve") + 
    ggplot2::theme_bw()
  
  g
}

#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @param aucs TO BE EDITED.
#' @return TO BE EDITED.
#' @export
plot_auc_histogram <- function(aucs) {
  df <- data.frame(aucs = aucs, stringsAsFactors = FALSE)
  g <- 
    ggplot2::ggplot(df, ggplot2::aes(x = aucs)) +
    ggplot2::geom_histogram(binwidth = 0.02) + 
    ggplot2::scale_x_continuous("AUC", limits = c(0, 1)) + 
    ggplot2::scale_y_continuous("Frequency", label = scales::comma) + 
    ggplot2::ggtitle("Distribution of AUCS") + 
    ggplot2::theme_bw()
  
  g
}
