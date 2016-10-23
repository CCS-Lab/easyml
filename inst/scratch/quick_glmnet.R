quickGlmnet <- function(dat_path = NULL,              # path to a data file
                        dependentVar = NULL,          # dependent variable (DV)
                        depCate = NULL,               # category of DV: "binary" or "continuous"
                        excludeVar = NULL,            # any variables to exlcude? (e.g., excludeVar = c("subject") )
                        categoricalVar = NULL,        # any categorical variables? (e.g., categoricalVar=c("Male) )
                        numIterations = 1000,         # Default=1000. Number of iterations
                        header = T,                   # Does the data file have a header? T or F
                        outOfSample = T,              # out-of-sample predictions? T or F
                        splitBy  = 3,                 # Default=3. Use 1/splitBy (=33%) of data as a test set
                        whichSeq = 1,                 # Default=1. Which sequence to use for a test set? 1, 2, or 3
                        myAlpha = 1,                  # Default=1. myAlpha=1 for LASSO
                        nFolds  = 5,                  # Default=5. k-fold
                        survivalRate_cutoff = 0.05) { # cutoff to be included in the beta matrix
  # To quickly and easily run glmnet (penalized linear or logistic regression) on a dataset

  # Programmed by Woo-Young Ahn (u.osu.edu/ccsl), Feb 2016
  # ** To cite this code in publication use:
  # Ahn, W.-Y.∗, Ramesh∗, D., Moeller, F. G., & Vassileva, J. (2016)
  #   Utility of machine learning approaches to identify behavioral markers
  #   for substance use disorders: Impulsivity dimensions as predictors of
  #   current cocaine dependence. Frontiers in Psychiatry.
  #
  # e.g.,
  # setwd("/Users/myID/myFolder)  # go to a folder that contains a data file
  # fit = quickGlmnet("cocaineData_frontiers.txt", depCate="binary", "DIAGNOSIS", c("subject"), c("Male"), numIterations=100, outOfSample = T)
  #
  # fit = a list of length 4
  # fit[[1]] --> observed (column#1) and predicted (column#2) outcomes (test set, w/ min lambda)
  # fit[[2]] --> observed (column#1) and predicted (column#2) outcomes (training set, w/ min lambda)
  # fit[[3]] --> mean beta coefficients and survival rate (w/ min lambda)
  # fit[[4]] --> mean beta coefficients and survival rate (w/ +1se lambda)
  #
  # Check if required packages are installed (from John Kruschke's code)
  # If not, install them
  reqPackages = c("glmnet", "ggplot2", "pROC", "corrplot")
  have = reqPackages %in% rownames(installed.packages())
  if ( any(!have) ) {
    cat("** Some required packages cannot be found. They are being installed now **\n")
    install.packages( reqPackages[!have] )
  }

  library(glmnet)   # penalized regression
  library(ggplot2)  # paper-ready figures
  library(pROC)     # ROC curves
  library(corrplot) # Correlation plots
  ##############################################################################

  # Detect OS
  if ( Sys.info()["sysname"] == "Darwin" ) {  # if Mac OS X --> use 'quartz'
    x11 <- function( ... ) quartz( ... )
  }

  plotColor = c("black")     # Use your favorite color for your ROC curve
  cat("Generating out-of-sample predictions? ", outOfSample, "\n")

  # Read (raw) data
  rawDat = read.table( dat_path, header=T)

  # DV category - binary or continuous?
  if (depCate == "binary") {
    glmnetDist = "binomial"
  } else if (depCate == "continuous") {
    glmnetDist = "gaussian"
  } else {
    stop("A dependent variable should be either a binary or continuous variable\n")
  }
  # exclude variables (if there is any)
  if ( !is.null(excludeVar)) {
    rawDat = eval(parse(text = paste0( "subset(rawDat, select = -c(", noquote(paste( excludeVar, collapse=", ")), "))" ) ) )
  }
  # specify dependent variable.
  depVar = eval(parse(text = paste("rawDat$", dependentVar, sep="")))
  # to make sure depVar is an integer...
  depVar = as.numeric(depVar)
  # indepVar --> independent variables
  indepVar = eval(parse(text = paste0( "subset(rawDat, select = -c(", dependentVar, "))") ) )

  # categorical variables --> no z-scoring.
  if ( !is.null(categoricalVar)) {
    cateVar = eval(parse(text = paste0( "indepVar[, c(", paste(shQuote(categoricalVar), collapse=", ") , ")]") ) )
    # remove categorical variables from indepVar
    contVar = eval(parse(text = paste0( "subset(indepVar, select = -c(", noquote(paste( categoricalVar, collapse=", ")), "))" ) ) )
    # if there is only one categorical variable, change its name to categoricalVar
    if (length(categoricalVar) ==1) {
      cateVar = data.frame(cateVar)
      colnames(cateVar) = categoricalVar
    }
  } else {
    contVar = indepVar
  }

  # z-score continuous variables
  contVar = scale(contVar)

  if ( is.null(categoricalVar)) {
    # combine raw categorical and z-scored continuous (independent) variables
    allDat = as.matrix( contVar )
  } else {
    allDat = as.matrix( data.frame(cateVar, contVar) )
  }
  numSubjs = length(depVar) # number of participants (i.e., n)
  numPredictors = dim(allDat)[2] + 1  # number of features (i.e, p). +1 because of intercept
  cat("# of participants=", numSubjs, ", # of measures=", numPredictors, "\n", sep="")
  cat(nFolds, "-folds cross-validation, ", "% of data for training=", round((1-1/splitBy)*100,1),"%\n", sep="")
  cat("Alpha=", myAlpha, ". Note: Alpha=0 --> ridge, Alpha=1 --> LASSO, 0< Alpha <1 --> Elastic net \n")
  # lassoDat --> a matrix: 1st column=dependent variable, the other columns=independent variables
  lassoDat = cbind(depVar, allDat)

  #####################################################
  ### Divided lassoDat into trainDat and testDat    ###
  #####################################################
  ## 3 sequences
  ## Decide which one to use for prediction

  allSeq = 1:numSubjs
  subjSeq1 = seq(whichSeq, numSubjs, by = splitBy)  # e.g., 1, 4, 7, ..., 82 --> N = 28

  if (outOfSample) { # if yes, use 2/3 of data as the training set and 1/3 of data as the test (validation) set
    vSeq = subjSeq1
    tSeq = allSeq[-vSeq]
    ggtitle_t = "(Training Set)"
    ggtitle_v = "(Test Set)"
  } else {  # then, use all data as the training and test sets
    vSeq = allSeq
    tSeq = allSeq
    ggtitle_t = "(No out-of-sample)"
    ggtitle_v = "(No out-of-sample)"
  }

  # validation(test) set
  testDat = lassoDat[ vSeq, ]  # matrix including the dependent variable
  testVar = testDat[, -which(colnames(testDat)=="depVar")] # matrix without the dependent variable
  # train set
  trainDat = lassoDat[ tSeq, ] # matrix including the dependent variable
  trainVar = trainDat[, -which(colnames(trainDat)=="depVar")] # matrix without the dependent variable

  # dependent variable in training and test sets
  depVar_t = depVar[tSeq]  # '_t' --> training set
  depVar_v = depVar[vSeq]  # '_v' --> test (validation) set
  # number of participants in each set
  numSubjs_t = length(depVar_t)
  numSubjs_v = length(depVar_v)

  #####################################################
  ### Implement a penalized logistic regression     ###
  #####################################################

  # To save values for the test (validation) set.
  # for min lambda
  all_predictedVar_min = array(NA, c(numSubjs_v, numIterations) )    # predicted depVar (on the test set)
  all_beta_min = array(NA, c(numPredictors, numIterations) )         # fitted beta coefficients (w/ train set)
  all_survivalRate_min = array(NA, c(numPredictors, numIterations) ) # survival rate (w/ train set)
  # for +1se lambda
  all_predictedVar_1se = array(NA, c(numSubjs_v, numIterations) )    # predicted depVar (on the test set)
  all_beta_1se = array(NA, c(numPredictors, numIterations) )         # fitted beta coefficients (w/ train set)
  all_survivalRate_1se = array(NA, c(numPredictors, numIterations) ) # survival rate (w/ train set)

  # To save values for the train set (w/ min lambda)
  all_predictedVar_min_t = array(NA, c(numSubjs_t, numIterations) )  # predicted depVar (on the train set)

  # A text based progress bar
  progressBar = txtProgressBar(min=1, max=numIterations, style=3)
  cat("Running ", numIterations, " iterations.\n")

  for (rIdx in 1:numIterations) {
    # fit LASSO with the training set
    lasso_glmnet = glmnet(x=trainVar, y=depVar_t, family=glmnetDist, standardize=F, alpha=myAlpha, maxit=10^6)
    lasso_cv_glmnet = cv.glmnet(x=trainVar, y=depVar_t, family=glmnetDist, standardize=F, alpha=myAlpha, nfolds=nFolds, maxit=10^6)

    ## test predictions on the test set (with min lambda)
    tmp_preddepVar_min = predict(lasso_glmnet, newx = testVar, s = lasso_cv_glmnet$lambda.min , type="response")

    # test predictions on the training set (with min lambda)
    tmp_preddepVar_min_t = predict(lasso_glmnet, newx = trainVar, s = lasso_cv_glmnet$lambda.min, type="response")

    ## test predictions on the test set (with +1se lambda)
    tmp_preddepVar_1se = predict(lasso_glmnet, newx = testVar, s = lasso_cv_glmnet$lambda.1se, type="link" )

    # extract beta coefficients with min lambda
    tmp_beta = predict(lasso_glmnet, s = lasso_cv_glmnet$lambda.min, type="coefficient" )
    # extract beta coefficients with min lambda
    tmp_beta_1se = predict(lasso_glmnet, s = lasso_cv_glmnet$lambda.1se, type="coefficient" )

    # save predictions made on the test set (w/ min lambda)
    all_predictedVar_min[, rIdx] = tmp_preddepVar_min
    all_beta_min[, rIdx] = as.matrix(tmp_beta)
    all_survivalRate_min[, rIdx] = as.numeric(abs(tmp_beta) > 0)

    # save predictions made on the test set (w/ +1se lambda)
    all_predictedVar_1se[, rIdx] = tmp_preddepVar_1se
    all_beta_1se[, rIdx] = as.matrix(tmp_beta_1se)
    all_survivalRate_1se[, rIdx] = as.numeric(abs(tmp_beta_1se) > 0)

    # save predictions made on the train set (w/ min lambda)
    all_predictedVar_min_t[, rIdx] = tmp_preddepVar_min_t

    setTxtProgressBar(progressBar, rIdx)
  }

  ###############################################################
  ### compute mean values of multiple(e.g., 1,000) iterations ###
  ###############################################################

  # predicted depVar on the test set (w/ min lambda)
  preddepVar_min = apply(all_predictedVar_min, 1, mean)

  # predicted depVar on the training set (w/ min lambda)
  preddepVar_min_t = apply(all_predictedVar_min_t, 1, mean)




  # To plot ROC curves (test set) w/ ggplot2
  dat_min = data.frame(Actual = as.numeric(depVar_v), Predicted = as.numeric(preddepVar_min))
  # To plot ROC curves (train set) w/ ggplot2
  dat_min_t = data.frame(Actual = as.numeric(depVar_t), Predicted = as.numeric(preddepVar_min_t))

  if (depCate == "binary") { # if binary classification --> draw ROC curves
    ### Drawing
    auc_figure_tmp = roc(Actual ~ Predicted, data = dat_min)
    auc_figure = as.numeric( auc_figure_tmp$auc )
    auc_figure_digit = prettyNum(auc_figure, digits=3, nsmall=3,width=5, format="fg")
    auc_dat = data.frame(Sens = auc_figure_tmp$sensitivities, OneMinusSpec = 1 - auc_figure_tmp$specificities)

    # Draw a ROC curve (test set)
    x11()
    h1 = ggplot(auc_dat, aes(x=OneMinusSpec, y=Sens)) +
      geom_path(alpha=1, size=1, colour = plotColor) +
      ggtitle(paste0("ROC Curve ", ggtitle_v) ) +
      annotate("text", label = paste("AUC = ", auc_figure_digit, sep=""), x = 0.6, y = 0.1, size = 15, colour = "black") +
      theme(plot.title=element_text(size=30)) +
      theme(axis.title = element_text(size = 30) ) +
      geom_segment(aes(x = 0, y = 0, xend = 1, yend = 1) , linetype="dashed") +
      theme(axis.text = element_text(size = 20, colour="black")) +   # for black tick label color
      xlab("1 - Specificity") + ylab("Sensitivity")
    print(h1)

    # training set
    auc_figure_tmp_t = roc(Actual ~ Predicted, data = dat_min_t)
    auc_figure_t = as.numeric( auc_figure_tmp_t$auc )
    auc_figure_digit_t = prettyNum(auc_figure_t, digits=3, nsmall=3, width=5, format="fg")
    auc_dat_t = data.frame(Sens = auc_figure_tmp_t$sensitivities, OneMinusSpec = 1 - auc_figure_tmp_t$specificities)

    x11()
    h2 = ggplot(auc_dat_t, aes(x=OneMinusSpec, y=Sens)) +
      geom_path(alpha=1, size=1, colour = plotColor) +
      ggtitle(paste0("ROC Curve ",  ggtitle_t) ) +
      annotate("text", label = paste("AUC = ", auc_figure_digit_t, sep=""), x = 0.6, y = 0.1, size = 15, colour = "black") +
      theme(plot.title=element_text(size=30)) +
      theme(axis.title = element_text(size = 30) ) +
      geom_segment(aes(x = 0, y = 0, xend = 1, yend = 1) , linetype="dashed") +
      theme(axis.text = element_text(size = 20, colour="black")) +   # for black tick label color
      xlab("1 - Specificity") + ylab("Sensitivity")
    print(h2)

  } else if (depCate == "continuous") { # if linear regression --> draw correlation plots

    # Draw a ROC curve (test set)
    cor_figure = cor.test(~ Actual + Predicted, data = dat_min)
    num = cor_figure$p.value
    r_figure = prettyNum(cor_figure$estimate, digits=3, width=5, format="fg")
    pValue_figure = ifelse(num==0, "< 2.200e-16" , paste0("= ", format(num, scientific=T, digits=4)) )

    x11()
    h1 = ggplot(dat_min, aes(x=Actual, y=Predicted, color=plotColor)) +
      geom_point(shape=19, alpha=0.8, size=3) +
      geom_smooth(method=lm, aes(group=1), colour="black") +
      ggtitle( paste0("Correlation between actual and predicted values: ", dependentVar, " (Test Set)") ) +
      annotate("text", label = paste("r = ", r_figure, ", p ", pValue_figure, sep=""),
               x=Inf, y=Inf, size = 6, colour = plotColor, vjust=1, hjust=1)
    print(h1)

    # training set
    # Draw a ROC curve (test set)
    cor_figure_t = cor.test(~ Actual + Predicted, data = dat_min_t)
    num_t = cor_figure_t$p.value
    r_figure_t = prettyNum(cor_figure_t$estimate, digits=3, width=5, format="fg")
    pValue_figure_t = ifelse(num_t==0, "< 2.200e-16" , paste0("= ", format(num_t, scientific=T, digits=4)) )

    x11()
    h2 = ggplot(dat_min_t, aes(x=Actual, y=Predicted, color=plotColor)) +
      geom_point(shape=19, alpha=0.8, size=3) +
      geom_smooth(method=lm, aes(group=1), colour="black") +
      ggtitle( paste0("Correlation between actual and predicted values: ", dependentVar, " (Training Set)") ) +
      annotate("text", label = paste("r = ", r_figure_t, ", p ", pValue_figure_t, sep=""),
               x=Inf, y=Inf, size = 6, colour = plotColor, vjust=1, hjust=1)
    print(h2)
  }
  #############################################
  ### mean beta coefficients of regressors  ###
  ### (only using the training set)         ###
  #############################################

  # Calculate survival rate  (w/ min lambda)
  mean_survivalRate = apply(all_survivalRate_min, 1, mean)
  # if survival rate < cutoff (5% = 0.05), set its mean to zero
  mean_survivalRate_cutoff = ( mean_survivalRate > survivalRate_cutoff ) * apply(all_beta_min, 1, mean)
  # Calculate survival rate  (w/ +1se lambda)
  mean_survivalRate_1se = apply(all_survivalRate_1se, 1, mean)
  mean_survivalRate_cutoff_1se = ( mean_survivalRate_1se > survivalRate_cutoff ) * apply(all_beta_1se, 1, mean)

  # beta coefficients of regressors
  # beta w/ min lambda
  bounds_min = apply(all_beta_min, 1, quantile, probs = c(0.025, 0.975))  # 95% confidence interval
  rownames(bounds_min) = c("lb", "ub")
  beta_min = data.frame(mean=apply(all_beta_min, 1, mean), lb = bounds_min["lb",], ub = bounds_min["ub", ], survival=mean_survivalRate)
  rownames(beta_min) = rownames(tmp_beta)
  # beta_min_cutoff --> remove variables w/ less than 5% survival rate
  beta_min_cutoff = data.frame(mean=mean_survivalRate_cutoff, lb = bounds_min["lb",], ub = bounds_min["ub", ], survival=mean_survivalRate)
  rownames(beta_min_cutoff) = rownames(tmp_beta)

  # beta w/ +1se lambda
  bounds_1se = apply(all_beta_1se, 1, quantile, probs = c(0.025, 0.975))  # 95% confidence interval
  rownames(bounds_1se) = c("lb", "ub")
  beta_1se = data.frame(mean=apply(all_beta_1se, 1, mean), lb = bounds_1se["lb",], ub = bounds_1se["ub", ], survival=mean_survivalRate_1se)
  rownames(beta_1se) = rownames(tmp_beta)
  # beta_min_cutoff --> remove variables w/ less than 5% survival rate
  beta_1se_cutoff = data.frame(mean=mean_survivalRate_cutoff, lb = bounds_min["lb",], ub = bounds_min["ub", ], survival=mean_survivalRate_cutoff_1se)
  rownames(beta_1se_cutoff) = rownames(tmp_beta)

  # return output w/
  return( list(dat_min, dat_min_t, beta_min_cutoff, beta_1se_cutoff) )

  cat("\n All done! \n")
}
