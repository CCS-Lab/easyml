#' Quickly and easily run penalized linear or logistic regression on a dataset.
#'
#' @param data An object of class "DataFrame".
#' @param dependent_variable A character string, the name of the dependent variable.
#' @param family A character string, the response type.
#' @param n_iterations An integer, the number of iterations. Defaults to 1000.
#' @param out_of_sample A boolean, indicating if out-of-sample predictions be generated. Defaults to TRUE.
#' @param proportion Percentage of data to use as a training set. Defaults to 0.667.
#' @param alpha  A numeric, the value to use for elasticnet. Use 0 for Ridge or 1 for LASSO. Defaults to 1.
#' @param n_folds An integer, number of folds to use for cross-validation. Defaults to 5.
#' @param survival_threshold A numeric between 0 and 1.0, the threshold cutoff for variables to be included in the beta matrix. Defaults to 0.05.
#' @return TO BE EDITED.
#' @export
# easy_glmnet <- function(data, dependent_variable = NULL, 
#                         family = c("gaussian", "binomial", "poisson", "multinomial", "cox", "mgaussian"), 
#                         n_iterations = 1000, out_of_sample = TRUE, proportion = 0.667, alpha = 1.0, 
#                         n_folds  = 5, survival_threshold = 0.05) { 
# 
#   cat("Generating out-of-sample predictions? ", outOfSample, "\n")
#   
#   # Read (raw) data
#   raw_data <- data
# 
#   numSubjs = length(depVar) # number of participants (i.e., n)
#   numPredictors = dim(allDat)[2] + 1  # number of features (i.e, p). +1 because of intercept
#   cat("# of participants=", numSubjs, ", # of measures=", numPredictors, "\n", sep="")
#   cat(nFolds, "-folds cross-validation, ", "% of data for training=", round((1-1/splitBy)*100,1),"%\n", sep="")
#   cat("Alpha=", myAlpha, ". Note: Alpha=0 --> ridge, Alpha=1 --> LASSO, 0< Alpha <1 --> Elastic net \n")
#   # lassoDat --> a matrix: 1st column=dependent variable, the other columns=independent variables
#   lassoDat = cbind(depVar, allDat)
#   
#   #####################################################
#   ### Divided lassoDat into trainDat and testDat    ###
#   #####################################################
#   ## 3 sequences
#   ## Decide which one to use for prediction
#   
#   allSeq = 1:numSubjs
#   subjSeq1 = seq(whichSeq, numSubjs, by = splitBy)  # e.g., 1, 4, 7, ..., 82 --> N = 28
#   
#   if (outOfSample) { # if yes, use 2/3 of data as the training set and 1/3 of data as the test (validation) set
#     vSeq = subjSeq1
#     tSeq = allSeq[-vSeq]
#     ggtitle_t = "(Training Set)"
#     ggtitle_v = "(Test Set)"
#   } else {  # then, use all data as the training and test sets
#     vSeq = allSeq
#     tSeq = allSeq
#     ggtitle_t = "(No out-of-sample)"
#     ggtitle_v = "(No out-of-sample)"
#   }
#   
#   # validation(test) set
#   testDat = lassoDat[ vSeq, ]  # matrix including the dependent variable
#   testVar = testDat[, -which(colnames(testDat)=="depVar")] # matrix without the dependent variable
#   # train set
#   trainDat = lassoDat[ tSeq, ] # matrix including the dependent variable
#   trainVar = trainDat[, -which(colnames(trainDat)=="depVar")] # matrix without the dependent variable
#   
#   # dependent variable in training and test sets
#   depVar_t = depVar[tSeq]  # '_t' --> training set
#   depVar_v = depVar[vSeq]  # '_v' --> test (validation) set
#   # number of participants in each set
#   numSubjs_t = length(depVar_t)
#   numSubjs_v = length(depVar_v)
#   
#   #####################################################
#   ### Implement a penalized logistic regression     ###
#   #####################################################
#   
#   
#   ###############################################################
#   ### compute mean values of multiple(e.g., 1,000) iterations ###
#   ###############################################################
#   
#   # predicted depVar on the test set (w/ min lambda)
#   preddepVar_min = apply(all_predictedVar_min, 1, mean)
#   
#   # predicted depVar on the training set (w/ min lambda)
#   preddepVar_min_t = apply(all_predictedVar_min_t, 1, mean)
#   
#   # To plot ROC curves (test set) w/ ggplot2
#   dat_min = data.frame(Actual = as.numeric(depVar_v), Predicted = as.numeric(preddepVar_min))
#   # To plot ROC curves (train set) w/ ggplot2
#   dat_min_t = data.frame(Actual = as.numeric(depVar_t), Predicted = as.numeric(preddepVar_min_t))
#   
#   #############################################
#   ### mean beta coefficients of regressors  ###
#   ### (only using the training set)         ###
#   #############################################
#   
#   # Calculate survival rate  (w/ min lambda)
#   mean_survivalRate = apply(all_survivalRate_min, 1, mean)
#   # if survival rate < cutoff (5% = 0.05), set its mean to zero
#   mean_survivalRate_cutoff = ( mean_survivalRate > survivalRate_cutoff ) * apply(all_beta_min, 1, mean)
#   # Calculate survival rate  (w/ +1se lambda)
#   mean_survivalRate_1se = apply(all_survivalRate_1se, 1, mean)
#   mean_survivalRate_cutoff_1se = ( mean_survivalRate_1se > survivalRate_cutoff ) * apply(all_beta_1se, 1, mean)
#   
#   # beta coefficients of regressors
#   # beta w/ min lambda
#   bounds_min = apply(all_beta_min, 1, quantile, probs = c(0.025, 0.975))  # 95% confidence interval
#   rownames(bounds_min) = c("lb", "ub")
#   beta_min = data.frame(mean=apply(all_beta_min, 1, mean), lb = bounds_min["lb",], ub = bounds_min["ub", ], survival=mean_survivalRate)
#   rownames(beta_min) = rownames(tmp_beta)
#   # beta_min_cutoff --> remove variables w/ less than 5% survival rate
#   beta_min_cutoff = data.frame(mean=mean_survivalRate_cutoff, lb = bounds_min["lb",], ub = bounds_min["ub", ], survival=mean_survivalRate)
#   rownames(beta_min_cutoff) = rownames(tmp_beta)
#   
#   # beta w/ +1se lambda
#   bounds_1se = apply(all_beta_1se, 1, quantile, probs = c(0.025, 0.975))  # 95% confidence interval
#   rownames(bounds_1se) = c("lb", "ub")
#   beta_1se = data.frame(mean=apply(all_beta_1se, 1, mean), lb = bounds_1se["lb",], ub = bounds_1se["ub", ], survival=mean_survivalRate_1se)
#   rownames(beta_1se) = rownames(tmp_beta)
#   # beta_min_cutoff --> remove variables w/ less than 5% survival rate
#   beta_1se_cutoff = data.frame(mean=mean_survivalRate_cutoff, lb = bounds_min["lb",], ub = bounds_min["ub", ], survival=mean_survivalRate_cutoff_1se)
#   rownames(beta_1se_cutoff) = rownames(tmp_beta)
#   
#   # return output w/
#   return( list(dat_min, dat_min_t, beta_min_cutoff, beta_1se_cutoff) )
#   
#   cat("\n All done! \n")
# }
