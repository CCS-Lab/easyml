# # To save values for the test (validation) set.
# # for min lambda
# all_predictedVar_min = array(NA, c(numSubjs_v, numIterations) )    # predicted depVar (on the test set)
# all_beta_min = array(NA, c(numPredictors, numIterations) )         # fitted beta coefficients (w/ train set)
# all_survivalRate_min = array(NA, c(numPredictors, numIterations) ) # survival rate (w/ train set)
# # for +1se lambda
# all_predictedVar_1se = array(NA, c(numSubjs_v, numIterations) )    # predicted depVar (on the test set)
# all_beta_1se = array(NA, c(numPredictors, numIterations) )         # fitted beta coefficients (w/ train set)
# all_survivalRate_1se = array(NA, c(numPredictors, numIterations) ) # survival rate (w/ train set)
# 
# # To save values for the train set (w/ min lambda)
# all_predictedVar_min_t = array(NA, c(numSubjs_t, numIterations) )  # predicted depVar (on the train set)
# 
# # A text based progress bar
# progressBar = txtProgressBar(min=1, max=numIterations, style=3)
# cat("Running ", numIterations, " iterations.\n")
# 
# for (rIdx in 1:numIterations) {
#   # fit LASSO with the training set
#   lasso_glmnet = glmnet(x=trainVar, y=depVar_t, family=glmnetDist, standardize=F, alpha=myAlpha, maxit=10^6)
#   lasso_cv_glmnet = cv.glmnet(x=trainVar, y=depVar_t, family=glmnetDist, standardize=F, alpha=myAlpha, nfolds=nFolds, maxit=10^6)
#   
#   ## test predictions on the test set (with min lambda)
#   tmp_preddepVar_min = predict(lasso_glmnet, newx = testVar, s = lasso_cv_glmnet$lambda.min , type="response")
#   
#   # test predictions on the training set (with min lambda)
#   tmp_preddepVar_min_t = predict(lasso_glmnet, newx = trainVar, s = lasso_cv_glmnet$lambda.min, type="response")
#   
#   ## test predictions on the test set (with +1se lambda)
#   tmp_preddepVar_1se = predict(lasso_glmnet, newx = testVar, s = lasso_cv_glmnet$lambda.1se, type="link" )
#   
#   # extract beta coefficients with min lambda
#   tmp_beta = predict(lasso_glmnet, s = lasso_cv_glmnet$lambda.min, type="coefficient" )
#   # extract beta coefficients with min lambda
#   tmp_beta_1se = predict(lasso_glmnet, s = lasso_cv_glmnet$lambda.1se, type="coefficient" )
#   
#   # save predictions made on the test set (w/ min lambda)
#   all_predictedVar_min[, rIdx] = tmp_preddepVar_min
#   all_beta_min[, rIdx] = as.matrix(tmp_beta)
#   all_survivalRate_min[, rIdx] = as.numeric(abs(tmp_beta) > 0)
#   
#   # save predictions made on the test set (w/ +1se lambda)
#   all_predictedVar_1se[, rIdx] = tmp_preddepVar_1se
#   all_beta_1se[, rIdx] = as.matrix(tmp_beta_1se)
#   all_survivalRate_1se[, rIdx] = as.numeric(abs(tmp_beta_1se) > 0)
#   
#   # save predictions made on the train set (w/ min lambda)
#   all_predictedVar_min_t[, rIdx] = tmp_preddepVar_min_t
#   
#   setTxtProgressBar(progressBar, rIdx)
# }
