## ----eval=FALSE---------------------------------------------------------------
#  library(ggplot2)
#  library(Rcpp)

## ----eval=FALSE---------------------------------------------------------------
#  ellipse.general <- function(mu, s, c){
#    ellipse.simple <- function(s1, s2, c){
#      a <- s1*c
#      b <- s2*c
#      x <- seq(from = -a, to = a, length.out = 400)
#      points <- data.frame(
#        x1 = c(-x, x),
#        x2 = NA
#      )
#      points$x2[1:400] <- sqrt(((a*b)^2-(b*x)^2)/a^2)
#      points$x2[401:800] <- -sqrt(((a*b)^2-(b*x)^2)/a^2)
#      return(points)
#    }
#    lambda <- diag(eigen(s)$values)  # 特征值
#    P <- eigen(s)$vectors            # 特征向量
#    Y <- ellipse.simple(s1 = sqrt(lambda[1,1]), s2 = sqrt(lambda[2,2]), c = c)
#    X <- t(P%*%t(Y) + mu)  # 对坐标旋转位移
#    X <- as.data.frame(X)
#    colnames(X) <- c('x1', 'x2')
#    return(X)
#  }

## ----eval=FALSE---------------------------------------------------------------
#  # 设置参数
#  mu <- c(2.5, 2.5)  # 均值向量
#  s <- matrix(c(2, 1, 1, 2), ncol = 2)  # 协方差矩阵
#  c <- 1
#  # 计算坐标
#  X <- ellipse.general(mu = mu, s = s, c = c)
#  ggplot(
#    data = X,
#    mapping = aes(x = x1, y = x2)
#  ) +
#    geom_path() +
#    geom_hline(yintercept = 0,lty = 2) +
#    geom_vline(xintercept = 0, lty = 2)

## ----eval=FALSE---------------------------------------------------------------
#  cppFunction(' double maxcol(NumericMatrix x){
#     int nr = x.nrow();
#     int nc = x.ncol();
#     double y, yl;
#     y = 0.0;
#     NumericVector ycol(nr);
#     for(int j = 0; j < nc; j++){
#       ycol = x.column(j);
#       yl = sum(ycol * ycol);
#       if(yl > y) y = yl;
#     }
#     y = sqrt(y);
#     return y;
#   }')
#  X <- matrix(rnorm(100*100), nrow = 100, ncol = 100)
#  a <- maxcol(X)
#  a

