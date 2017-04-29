#' Plot penalized regression coefficients.
#' 
#' When calling \code{\link{easy_glmnet}}, coefficients from the 
#' \code{\link{generate_coefficients}} output are processed by the 
#' \code{\link{process_coefficients}}  function and generated into 
#' a plot. This plot tells us the direction, magitude, and statistical
#' significance of each coefficient. Be careful using this plotting
#' method with datasets containing more than 20 variables as the plot 
#' may not render as nicely.
#' 
#' @param coefficients_processed A data.frame, the output of the function \code{\link{process_coefficients}}.
#' @return A ggplot object. This plot may be rendered by outputting it to the command line or modified using ggplot semantics.
#' @family plot
#' @export
plot_coefficients_processed <- function(coefficients_processed) {
  if (nrow(coefficients_processed) > 20) 
    warning("Number of predictors exceeds 20; plot may not render as nicely.")
  
  g <- 
    ggplot2::ggplot(coefficients_processed, ggplot2::aes_string(x = "predictor", y = "mean", colour = "dot_color")) +
    ggplot2::geom_errorbar(ggplot2::aes_string(ymin = "lower_bound", ymax = "upper_bound"), width = 0.1) + 
    ggplot2::geom_point() +
    ggplot2::scale_x_discrete("Predictors") +
    ggplot2::scale_y_continuous("Coefficient estimates") + 
    ggplot2::scale_color_manual("", values = c("0" = "grey", "2" = "black"), 
                                labels = c("0" = "Insignificant", "2" = "Significant")) + 
    ggplot2::ggtitle("Estimates of weights") + 
    ggplot2::theme_bw() + 
    ggplot2::coord_flip()
  
  g
}

#' Plot random forest variable importances scores.
#' 
#' When calling \code{\link{easy_random_forest}}, variable importances scores from the 
#' \code{\link{generate_variable_importances}} output are processed by the 
#' \code{\link{process_variable_importances}}  function and generated into 
#' a plot. Importance scores for each predictor were estimated using the increase in 
#' node impurity. Node impuirty measures the change in residual squared error 
#' that is attributable to the predictor across all trees. Unlike the 
#' \code{\link{easy_glmnet}} coefficients, random forest importance scores 
#' do not indicate directional effects, but instead represent the magnitude 
#' of the effect that the predictor has 
#' on overall prediction accuracy. Be careful using this plotting
#' method with datasets containing more than 20 variables as the plot 
#' may not render as nicely.
#'
#' @param variable_importances_processed A data.frame, the output of the function \code{\link{process_variable_importances}}.
#' @return A ggplot object. This plot may be rendered by outputting it to the command line or modified using ggplot semantics.
#' @family plot
#' @export
plot_variable_importances_processed <- function(variable_importances_processed) {
  if (nrow(variable_importances_processed) > 20) 
    warning("Number of variables exceeds 20; plot may not render as nicely.")
  
  g <- 
    ggplot2::ggplot(variable_importances_processed, 
                    ggplot2::aes_string(x = "predictor", y = "mean")) +
    ggplot2::geom_bar(stat = "identity") + 
    ggplot2::geom_errorbar(ggplot2::aes_string(ymin = "lower_bound", ymax = "upper_bound"), width = 0.1) + 
    ggplot2::scale_x_discrete("Predictors") +
    ggplot2::scale_y_continuous("Variable Importance (Mean Decrease in Gini Index)") + 
    ggplot2::ggtitle("Variable Importance") + 
    ggplot2::theme_bw() + 
    ggplot2::coord_flip()
  
  g
}

#' Plot gaussian predictions.
#'
#' @param y_true Ground truth (correct) target values.
#' @param y_pred Estimated target values.
#' @return A ggplot object. This plot may be rendered by outputting it to the command line or modified using ggplot semantics.
#' @family plot
#' @export
plot_predictions_gaussian <- function(y_true, y_pred) {
  df <- data.frame(y_true = y_true, y_pred = y_pred, stringsAsFactors = FALSE)
  cor_score <- measure_cor_score(y_true, y_pred)
  cor_score_label <- paste("Correlation Score = ", round(cor_score, digits = 3), sep = "")
  
  g <- 
    ggplot2::ggplot(df, ggplot2::aes(x = y_pred, y = y_true)) +
    ggplot2::geom_point() + 
    ggplot2::geom_smooth(method = 'lm') + 
    ggplot2::scale_x_continuous("Predicted y values") + 
    ggplot2::scale_y_continuous("True y values") + 
    ggplot2::ggtitle(paste0("Actual vs. Predicted y values (", cor_score_label, ")")) + 
    ggplot2::theme_bw()
  
  g
}

#' Plot binomial predictions.
#'
#' @param y_true Ground truth (correct) target values.
#' @param y_pred Estimated target values.
#' @return A ggplot object. This plot may be rendered by outputting it to the command line or modified using ggplot semantics.
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
    ggplot2::scale_x_continuous("1 - Specificity", breaks = seq(0, 1, 0.05), 
                                minor_breaks = seq(0, 1, 0.01)) + 
    ggplot2::scale_y_continuous("Sensitivity", breaks = seq(0, 1, 0.05), 
                                minor_breaks = seq(0, 1, 0.01)) + 
    ggplot2::ggtitle(paste0("ROC Curve (", auc_label, ")")) + 
    ggplot2::theme_bw()
  
  g
}

#' Plot mean squared error metrics.
#'
#' @param mses A vector, the mean squared error metrics to be plotted as a histogram.
#' @return A ggplot object. This plot may be rendered by outputting it to the command line or modified using ggplot semantics.
#' @family plot
#' @export
plot_metrics_gaussian_mean_squared_error <- function(mses) {
  mean_mse <- mean(mses)
  mse_label <- paste("Mean MSE = ", round(mean_mse, digits = 3), sep = "")
  df <- data.frame(mses = mses, stringsAsFactors = FALSE)
  
  g <- 
    ggplot2::ggplot(df, ggplot2::aes(x = mses)) +
    ggplot2::geom_histogram(binwidth = 0.01) + 
    ggplot2::geom_vline(xintercept = mean_mse, linetype = "dotted") + 
    ggplot2::scale_x_continuous("MSE") + 
    ggplot2::scale_y_continuous("Frequency", label = scales::comma) + 
    ggplot2::ggtitle(paste0("Distribution of MSEs (", mse_label, ")")) + 
    ggplot2::theme_bw()
  
  g
}

#' Plot coefficient of determination (R^2) metrics.
#'
#' @param r2_scores A vector, the coefficient of determination (R^2) metrics to be plotted as a histogram.
#' @return A ggplot object. This plot may be rendered by outputting it to the command line or modified using ggplot semantics.
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
    ggplot2::scale_x_continuous("R^2 Score", limits = c(0, 1), 
                                breaks = seq(0, 1, 0.05), 
                                minor_breaks = seq(0, 1, 0.01)) + 
    ggplot2::scale_y_continuous("Frequency", label = scales::comma) + 
    ggplot2::ggtitle(paste0("Distribution of R^2 Scores (", r2_score_label, ")")) + 
    ggplot2::theme_bw()
  
  g
}

#' Plot correlation coefficient metrics.
#'
#' @param cor_scores A vector, the correlation coefficient metrics to be plotted as a histogram.
#' @return A ggplot object. This plot may be rendered by outputting it to the command line or modified using ggplot semantics.
#' @family plot
#' @export
plot_metrics_gaussian_cor_score <- function(cor_scores) {
  mean_cor_score <- mean(cor_scores)
  cor_score_label <- paste("Mean Correlation Score = ", round(mean_cor_score, digits = 3), sep = "")
  df <- data.frame(cor_scores = cor_scores, stringsAsFactors = FALSE)
  
  g <- 
    ggplot2::ggplot(df, ggplot2::aes(x = cor_scores)) +
    ggplot2::geom_histogram(binwidth = 0.01, boundary = 0) + 
    ggplot2::geom_vline(xintercept = mean_cor_score, linetype = "dotted") + 
    ggplot2::annotate("text", label = cor_score_label, x = 0.2, y = 0.2, size = 8) + 
    ggplot2::scale_x_continuous("Correlation Score", limits = c(0, 1), 
                                breaks = seq(0, 1, 0.05), 
                                minor_breaks = seq(0, 1, 0.01)) + 
    ggplot2::scale_y_continuous("Frequency", label = scales::comma) + 
    ggplot2::ggtitle("Distribution of Correlation Scores") + 
    ggplot2::theme_bw()
  
  g
}

#' Plot area under the curve (AUC) metrics.
#'
#' @param aucs A vector, the area under the curve (AUC) metrics to be plotted as a histogram.
#' @return A ggplot object. This plot may be rendered by outputting it to the command line or modified using ggplot semantics.
#' @family plot
#' @export
plot_metrics_binomial_area_under_curve <- function(aucs) {
  mean_auc <- mean(aucs)
  auc_label <- paste("Mean AUC = ", round(mean_auc, digits = 3), sep = "")
  df <- data.frame(aucs = aucs, stringsAsFactors = FALSE)
  
  g <- 
    ggplot2::ggplot(df, ggplot2::aes(x = aucs)) +
    ggplot2::geom_histogram(binwidth = 0.01, boundary = 0) + 
    ggplot2::geom_vline(xintercept = mean_auc, linetype = "dotted") + 
    ggplot2::scale_x_continuous("AUC", limits = c(0, 1), 
                                breaks = seq(0, 1, 0.05), 
                                minor_breaks = seq(0, 1, 0.01)) + 
    ggplot2::scale_y_continuous("Frequency", label = scales::comma) + 
    ggplot2::ggtitle(paste0("Distribution of AUCs (", auc_label, ")")) + 
    ggplot2::theme_bw()
  
  g
}
