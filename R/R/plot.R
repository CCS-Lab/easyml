#' Plot processed coefficients.
#'
#' @param coefficients_processed TO BE EDITED.
#' @return TO BE EDITED.
#' @family plot
#' @export
plot_coefficients_processed <- function(coefficients_processed) {
  if (nrow(coefficients_processed) > 20) 
    warning("Number of predictors exceeds 20; plot may not render as nicely.")
  
  g <- 
    ggplot2::ggplot(coefficients_processed, ggplot2::aes_string(x = "predictor", y = "mean", colour = "dot_color")) +
    ggplot2::geom_errorbar(ggplot2::aes_string(ymin = "lower_bound", ymax = "upper_bound"), width = 0.1) + 
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

#' Plot gaussian predictions.
#'
#' @param y_true Ground truth (correct) target values.
#' @param y_pred Estimated target values.
#' @return TO BE EDITED.
#' @family plot
#' @export
plot_predictions_gaussian <- function(y_true, y_pred) {
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

#' Plot binomial predictions.
#'
#' @param y_true Ground truth (correct) target values.
#' @param y_pred Estimated target values.
#' @return TO BE EDITED.
#' @family plot
#' @export
plot_predictions_binomial <- function(y_true, y_pred) {
  results <- pROC::roc(y_true, y_pred)
  auc <- as.numeric(results$auc)
  auc_label <- paste("AUC = ", round(auc, digits = 3), sep = "")
  df <- data.frame(sensitivities = results$sensitivities, 
                   one_minus_specificities = 1 - results$specificities, 
                   stringsAsFactors = FALSE)
  
  g <- 
    ggplot2::ggplot(df, ggplot2::aes_string(x = "one_minus_specificities", y = "sensitivities")) +
    ggplot2::geom_path(alpha = 1, size = 1) +
    ggplot2::geom_segment(ggplot2::aes(x = 0, y = 0, xend = 1, yend = 1) , linetype = "dashed") + 
    ggplot2::annotate("text", label = auc_label, x = 0.85, y = 0.025, size = 8) + 
    ggplot2::scale_x_continuous("1 - Specificity") + 
    ggplot2::scale_y_continuous("Sensitivity") + 
    ggplot2::ggtitle("ROC Curve") + 
    ggplot2::theme_bw()
  
  g
}

#' Plot mean squared error metrics.
#'
#' @param mses TO BE EDITED.
#' @return TO BE EDITED.
#' @family plot
#' @export
plot_metrics_gaussian_mean_squared_error <- function(mses) {
  mean_mse <- mean(mses)
  mse_label <- paste("Mean MSE = ", round(mean_mse, digits = 3), sep = "")
  df <- data.frame(mses = mses, stringsAsFactors = FALSE)
  
  g <- 
    ggplot2::ggplot(df, ggplot2::aes(x = mses)) +
    ggplot2::geom_histogram(binwidth = 0.02) + 
    ggplot2::geom_vline(xintercept = mean_mse, linetype = "dotted") + 
    ggplot2::annotate("text", label = mse_label, x = 0.2, y = 0.2, size = 8) + 
    ggplot2::scale_x_continuous("MSE") + 
    ggplot2::scale_y_continuous("Frequency", label = scales::comma) + 
    ggplot2::ggtitle("Distribution of MSEs") + 
    ggplot2::theme_bw()
  
  g
}

#' Plot R^2 metrics.
#'
#' @param r2_scores TO BE EDITED.
#' @return TO BE EDITED.
#' @family plot
#' @export
plot_metrics_gaussian_r2_score <- function(r2_scores) {
  mean_r2_score <- mean(r2_scores)
  r2_score_label <- paste("Mean R^2 Score = ", round(mean_r2_score, digits = 3), sep = "")
  df <- data.frame(r2_scores = r2_scores, stringsAsFactors = FALSE)
  
  g <- 
    ggplot2::ggplot(df, ggplot2::aes(x = r2_scores)) +
    ggplot2::geom_histogram(binwidth = 0.02) + 
    ggplot2::geom_vline(xintercept = mean_r2_score, linetype = "dotted") + 
    ggplot2::annotate("text", label = r2_score_label, x = 0.2, y = 0.2, size = 8) + 
    ggplot2::scale_x_continuous("R^2 Score", limits = c(0, 1)) + 
    ggplot2::scale_y_continuous("Frequency", label = scales::comma) + 
    ggplot2::ggtitle("Distribution of R^2 Scores") + 
    ggplot2::theme_bw()
  
  g
}

#' Plot AUC metrics.
#'
#' @param aucs TO BE EDITED.
#' @return TO BE EDITED.
#' @family plot
#' @export
plot_metrics_binomial_area_under_curve <- function(aucs) {
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
