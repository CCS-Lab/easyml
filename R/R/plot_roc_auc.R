#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @param y_true Ground truth (correct) target values.
#' @param y_pred Estimated target values.
#' @return TO BE EDITED.
#' @export
plot_roc_auc <- function(y_true, y_pred) {
  return(1L)
}

# ### Drawing
# auc_figure_tmp = pROC::roc(Actual ~ Predicted, data = dat_min)
# auc_figure = as.numeric( auc_figure_tmp$auc )
# auc_figure_digit = prettyNum(auc_figure, digits=3, nsmall=3,width=5, format="fg")
# auc_dat = data.frame(Sens = auc_figure_tmp$sensitivities, OneMinusSpec = 1 - auc_figure_tmp$specificities)
# 
# # Draw a ROC curve (test set)
# x11()
# h1 = ggplot(auc_dat, aes(x=OneMinusSpec, y=Sens)) +
#   geom_path(alpha=1, size=1, colour = plot_color) +
#   ggtitle(paste0("ROC Curve ", ggtitle_v) ) +
#   annotate("text", label = paste("AUC = ", auc_figure_digit, sep=""), x = 0.6, y = 0.1, size = 15, colour = "black") +
#   theme(plot.title=element_text(size=30)) +
#   theme(axis.title = element_text(size = 30) ) +
#   geom_segment(aes(x = 0, y = 0, xend = 1, yend = 1) , linetype="dashed") +
#   theme(axis.text = element_text(size = 20, colour="black")) +   # for black tick label color
#   xlab("1 - Specificity") + ylab("Sensitivity")
# print(h1)
# 
# # training set
# auc_figure_tmp_t = pROC::roc(Actual ~ Predicted, data = dat_min_t)
# auc_figure_t = as.numeric( auc_figure_tmp_t$auc )
# auc_figure_digit_t = prettyNum(auc_figure_t, digits=3, nsmall=3, width=5, format="fg")
# auc_dat_t = data.frame(Sens = auc_figure_tmp_t$sensitivities, OneMinusSpec = 1 - auc_figure_tmp_t$specificities)
# 
# x11()
# h2 = ggplot(auc_dat_t, aes(x=OneMinusSpec, y=Sens)) +
#   geom_path(alpha=1, size=1, colour = plot_color) +
#   ggtitle(paste0("ROC Curve ",  ggtitle_t) ) +
#   annotate("text", label = paste("AUC = ", auc_figure_digit_t, sep=""), x = 0.6, y = 0.1, size = 15, colour = "black") +
#   theme(plot.title=element_text(size=30)) +
#   theme(axis.title = element_text(size = 30) ) +
#   geom_segment(aes(x = 0, y = 0, xend = 1, yend = 1) , linetype="dashed") +
#   theme(axis.text = element_text(size = 20, colour="black")) +   # for black tick label color
#   xlab("1 - Specificity") + ylab("Sensitivity")
# print(h2)
