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
plot_gaussian_predictions <- function(y_true, y_pred) {
  df <- data.frame(y_true = y_true, y_pred = y_pred, stringsAsFactors = FALSE)
  
  g <- 
    ggplot2::ggplot(df, ggplot2::aes(x = y_pred, y = y_true)) +
    ggplot2::geom_point() +
    ggplot2::scale_x_continuous("Predicted y values") + 
    ggplot2::scale_y_continuous("True y values") + 
    ggplot2::ggtitle("") + 
    ggplot2::theme_bw()
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
    ggplot2::annotate("text", label = auc_label, x = 0.85, y = 0.025, size = 8) + 
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
#' @param mses TO BE EDITED.
#' @return TO BE EDITED.
#' @export
plot_mse_histogram <- function(mses) {
  mean_mse <- mean(mses)
  mse_label <- paste("Mean MSE = ", round(mean_mse, digits = 3), sep = "")
  df <- data.frame(mses = mses, stringsAsFactors = FALSE)
  g <- 
    ggplot2::ggplot(df, ggplot2::aes(x = mses)) +
    ggplot2::geom_histogram(binwidth = 0.02) + 
    ggplot2::geom_vline(xintercept = mean_mse, linetype = "dotted") + 
    ggplot2::annotate("text", label = mse_label, x = 0.2, y = 0.2, size = 8) + 
    ggplot2::scale_x_continuous("MSE", limits = c(0, 1)) + 
    ggplot2::scale_y_continuous("Frequency", label = scales::comma) + 
    ggplot2::ggtitle("Distribution of MSEs") + 
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
  mean_auc <- mean(aucs)
  auc_label <- paste("Mean AUC = ", round(mean_auc, digits = 3), sep = "")
  df <- data.frame(aucs = aucs, stringsAsFactors = FALSE)
  g <- 
    ggplot2::ggplot(df, ggplot2::aes(x = aucs)) +
    ggplot2::geom_histogram(binwidth = 0.02) + 
    ggplot2::geom_vline(xintercept = mean_auc, linetype = "dotted") + 
    ggplot2::annotate("text", label = auc_label, x = 0.2, y = 0.2, size = 8) + 
    ggplot2::scale_x_continuous("AUC", limits = c(0, 1)) + 
    ggplot2::scale_y_continuous("Frequency", label = scales::comma) + 
    ggplot2::ggtitle("Distribution of AUCs") + 
    ggplot2::theme_bw()
  g
}
