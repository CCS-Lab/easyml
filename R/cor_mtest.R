#' TO BE EDITED.
#'
#' TO BE EDITED.
#'
#' @param mat TO BE EDITED.
#' @param conf_level TO BE EDITED.
#' @return TO BE EDITED.
#' @export
cor_mtest <- function(mat, conf_level = 0.95){
  mat <- as.matrix(mat)
  n <- ncol(mat)
  p_mat <- lowCI_mat <- uppCI_mat <- matrix(NA, n, n)
  diag(p_mat) <- 0
  diag(lowCI_mat) <- diag(uppCI_mat) <- 1
  
  for(i in 1:(n-1)){
    for(j in (i+1):n){
      tmp <- stats::cor.test(mat[,i], mat[,j], conf.level = conf_level)
      p_mat[i,j] <- p_mat[j,i] <- tmp$p.value
      lowCI_mat[i,j] <- lowCI_mat[j,i] <- tmp$conf.int[1]
      uppCI_mat[i,j] <- uppCI_mat[j,i] <- tmp$conf.int[2]
    }
  }
  
  list(p_mat, lowCI_mat, uppCI_mat)
}
