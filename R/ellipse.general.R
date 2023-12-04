#' @title Draw a confidence ellipse using R
#' @description Draw a confidence ellipse using R
#' @param mu the mean vector
#' @param s the covariance matrix
#' @param c a constant
#' @return some points of the confidence ellipse
#' @examples
#' \dontrun{
#'     mu <- c(2.5, 2.5)  # 均值向量
#'     s <- matrix(c(2, 1, 1, 2), ncol = 2)  # 协方差矩阵
#'     c <- 1
#'     X <- ellipse.general(mu = mu, s = s, c = c)
#' }
#' @export
ellipse.general <- function(mu, s, c){
  ellipse.simple <- function(s1, s2, c){
    a <- s1*c
    b <- s2*c 
    x <- seq(from = -a, to = a, length.out = 400)
    points <- data.frame(
      x1 = c(-x, x),
      x2 = NA
    )
    points$x2[1:400] <- sqrt(((a*b)^2-(b*x)^2)/a^2)
    points$x2[401:800] <- -sqrt(((a*b)^2-(b*x)^2)/a^2)
    return(points)
  }
  lambda <- diag(eigen(s)$values)  # 特征值
  P <- eigen(s)$vectors            # 特征向量
  Y <- ellipse.simple(s1 = sqrt(lambda[1,1]), s2 = sqrt(lambda[2,2]), c = c)  # 中心在原点，没有倾斜的椭圆的坐标
  X <- t(P%*%t(Y) + mu)  # 对坐标旋转位移
  X <- as.data.frame(X)
  colnames(X) <- c('x1', 'x2')
  return(X)
}

#' @title Rcpp functions.
#' @name Rcpp
#' @description Rcpp package
#' @useDynLib SA23204156
#' @import Rcpp
NULL