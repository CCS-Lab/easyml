#' Plot penalized regression coefficients.
#' 
#' When calling \code{\link{easy_glmnet}}, coefficients from the 
#' \code{\link{generate_coefficients}} output are processed by the 
#' \code{\link{process_coefficients}}  function and generated into 
#' a plot. This plot tells us the direction, magnitude, and statistical
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
  
  p <- coefficients_processed
  p$predictor <- factor(p$predictor, levels = p$predictor[order(p$mean)])  # sort beta coefficients
  
  g <- 
    ggplot2::ggplot(p, ggplot2::aes_string(x = "predictor", y = "mean", colour = "dot_color")) +
    ggplot2::geom_errorbar(ggplot2::aes_string(ymin = "lower_bound", ymax = "upper_bound"), width = 0.1) + 
    ggplot2::geom_point() +
    ggplot2::scale_x_discrete("Predictors") +
    ggplot2::scale_y_continuous("Coefficient estimates") + 
    ggplot2::scale_color_manual("", values = c("0" = "#D3D3D3", "2" = "#000000"), 
                                labels = c("0" = "Insignificant", "2" = "Significant")) + 
    ggplot2::ggtitle("Estimates of coefficients") + 
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
#' node impurity. Node impurity measures the change in residual squared error 
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
  
  p <- variable_importances_processed
  p$predictor <- factor(p$predictor, levels = p$predictor[order(p$mean)])  # sort variable importances
  
  g <- 
    ggplot2::ggplot(p, 
                    ggplot2::aes_string(x = "predictor", y = "mean")) +
    ggplot2::geom_bar(stat = "identity") + 
    ggplot2::geom_errorbar(ggplot2::aes_string(ymin = "lower_bound", ymax = "upper_bound"), width = 0.1) + 
    ggplot2::scale_x_discrete("Predictors") +
    ggplot2::scale_y_continuous("Variable Importance (Mean Decrease in Gini Index)") + 
    ggplot2::ggtitle("Variable Importances") + 
    ggplot2::theme_bw() + 
    ggplot2::coord_flip()
  
  g
}

#' Plot gaussian predictions.
#' 
#' Plots a scatter plot of the ground truth (correct) target values 
#' and the estimated target values.
#'
#' @param y_true Ground truth (correct) target values.
#' @param y_pred Estimated target values.
#' @return A ggplot object. This plot may be rendered by outputting it to the command line or modified using ggplot semantics.
#' @family plot
#' @export
plot_predictions_gaussian <- function(y_true, y_pred) {
  df <- data.frame(y_true = y_true, y_pred = y_pred, 
                   stringsAsFactors = FALSE)
  cor_score <- round(measure_correlation_score(y_true, y_pred), digits = 2)
  msg <- "Actual vs. Predicted y values (Correlation Score = "
  .title <- paste0(msg, cor_score, ")")
  g <- 
    ggplot2::ggplot(df, ggplot2::aes(x = y_pred, y = y_true)) +
    ggplot2::geom_point() + 
    ggplot2::geom_smooth(method = 'lm') + 
    ggplot2::scale_x_continuous("Predicted y values") + 
    ggplot2::scale_y_continuous("True y values") + 
    ggplot2::ggtitle(.title) + 
    ggplot2::theme_bw()
  
  g
}

#' Plot binomial predictions.
#' 
#' Plots a logistic plot of the ground truth (correct) target values 
#' and the estimated target values.
#'
#' @param y_true Ground truth (correct) target values.
#' @param y_pred Estimated target values.
#' @return A ggplot object. This plot may be rendered by outputting it to the command line or modified using ggplot semantics.
#' @family plot
#' @export
plot_predictions_binomial <- function(y_true, y_pred) {
  df <- data.frame(y_true = y_true, y_pred = y_pred, 
                   stringsAsFactors = FALSE)
  cor_score <- round(measure_correlation_score(y_true, y_pred), digits = 2)
  msg <- "Actual vs. Predicted y values (Correlation Score = "
  .title <- paste0(msg, cor_score, ")")
  g <- 
    ggplot2::ggplot(df, ggplot2::aes(x = y_pred, y = y_true)) + 
    ggplot2::geom_point() + 
    ggplot2::stat_smooth(method="glm", method.args = list(family = "binomial"), se=FALSE) + 
    ggplot2::scale_x_continuous("Predicted y values", limits = c(0, 1), 
                       breaks = seq(0, 1, 0.05), minor_breaks = seq(0, 1, 0.01)) + 
    ggplot2::scale_y_continuous("True y values", limits = c(0, 1), 
                       breaks = seq(0, 1, 0.05), minor_breaks = seq(0, 1, 0.01)) + 
    ggplot2::ggtitle(.title) + 
    ggplot2::theme_bw()
  
  g
}

#' Plot ROC Curve.
#' 
#' Given the ground truth (correct) target values and the estimated 
#' target values will plot an ROC curve.
#'
#' @param y_true Ground truth (correct) target values.
#' @param y_pred Estimated target values.
#' @return A ggplot object. This plot may be rendered by outputting it to the command line or modified using ggplot semantics.
#' @family plot
#' @export
plot_roc_curve <- function(y_true, y_pred) {
  results <- pROC::roc(y_true, y_pred)
  auc <- as.numeric(results$auc)
  auc_label <- paste("AUC Score = ", round(auc, digits = 2), sep = "")
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

#' Plot histogram of measures of model performance.
#' 
#' Given a numeric vector will plot a histogram 
#' for any number of measures of model performance.
#'
#' @param x A vector, the mean squared error metrics to be plotted as a histogram.
#' @param name A character vector of length one, the name of the metric.
#' @return A ggplot object. This plot may be rendered by outputting it to the command line or modified using ggplot semantics.
#' @family plot
#' @export
plot_model_performance_histogram <- function(x, name) {
  mean_x <- round(mean(x), digits = 2)
  label <- paste0("Mean ", name, " Score = ", mean_x)
  .title <- paste0("Distribution of ", name, " Scores (", label, ")")
  df <- data.frame(x = x, stringsAsFactors = FALSE)
  
  g <- 
    ggplot2::ggplot(df, ggplot2::aes(x = x)) +
    ggplot2::geom_histogram(binwidth = 0.01, boundary = 0) + 
    ggplot2::geom_vline(xintercept = mean_x, linetype = "dotted") + 
    ggplot2::scale_y_continuous("Frequency", label = scales::comma) + 
    ggplot2::ggtitle(.title) + 
    ggplot2::theme_bw()
  
  g
}

#' Plot histogram of the mean squared error metrics.
#' 
#' This function plots a histogram of the mean squared error metrics.
#'
#' @param x A vector, the mean squared error metrics to be plotted as a histogram.
#' @return A ggplot object. This plot may be rendered by outputting it to the command line or modified using ggplot semantics.
#' @family plot
#' @export
plot_model_performance_gaussian_mse_score <- function(x) {
  name <- "MSE"
  g <- 
    plot_model_performance_histogram(x, name) + 
    ggplot2::scale_x_continuous(paste0(name, " Score"))
  
  g
}

#' Plot histogram of the correlation coefficient metrics.
#' 
#' This function plots a histogram of the correlation coefficient metrics.
#'
#' @param x A vector, the correlation coefficient metrics to be plotted as a histogram.
#' @return A ggplot object. This plot may be rendered by outputting it to the command line or modified using ggplot semantics.
#' @family plot
#' @export
plot_model_performance_gaussian_correlation_score <- function(x) {
  name <- "Correlation"
  g <- 
    plot_model_performance_histogram(x, name) + 
    ggplot2::scale_x_continuous(paste0(name, " Score"), limits = c(0, 1), 
                                breaks = seq(0, 1, 0.05), 
                                minor_breaks = seq(0, 1, 0.01))
  
  g
}

#' Plot histogram of the coefficient of determination (R^2) metrics.
#' 
#' This function plots a histogram of the coefficient of determination (R^2) metrics.
#'
#' @param x A vector, the coefficient of determination (R^2) metrics to be plotted as a histogram.
#' @return A ggplot object. This plot may be rendered by outputting it to the command line or modified using ggplot semantics.
#' @family plot
#' @export
plot_model_performance_gaussian_r2_score <- function(x) {
  name <- "R^2"
  g <- 
    plot_model_performance_histogram(x, name) + 
    ggplot2::scale_x_continuous(paste0(name, " Score"), limits = c(0, 1), 
                                breaks = seq(0, 1, 0.05), 
                                minor_breaks = seq(0, 1, 0.01))
  
  g
}

#' Plot histogram of the area under the curve (AUC) metrics.
#' 
#' This function plots a histogram of the area under the curve (AUC) metrics.
#'
#' @param x A vector, the area under the curve (AUC) metrics to be plotted as a histogram.
#' @return A ggplot object. This plot may be rendered by outputting it to the command line or modified using ggplot semantics.
#' @family plot
#' @export
plot_model_performance_binomial_auc_score <- function(x) {
  name <- "AUC"
  g <- 
    plot_model_performance_histogram(x, name) + 
    ggplot2::scale_x_continuous(paste0(name, " Score"), limits = c(0, 1), 
                                breaks = seq(0, 1, 0.05), 
                                minor_breaks = seq(0, 1, 0.01))
  
  g
}
