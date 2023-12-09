## ----eval=FALSE---------------------------------------------------------------
#  library(Rcpp)
#  library(microbenchmark)

## ----eval=FALSE---------------------------------------------------------------
#  d <- data.frame(x = c(1), y = c(1), z = c(1))
#  d <- d[-1,]
#  d

## ----eval=FALSE---------------------------------------------------------------
#  d <- data.frame(x = c(1,2,3))
#  d <- d[,-1]
#  d

## ----eval=FALSE---------------------------------------------------------------
#  scale01 <- function(x) {
#    rng <- range(x, na.rm = TRUE)
#    (x - rng[1]) / (rng[2] - rng[1])
#  }

## ----eval=FALSE---------------------------------------------------------------
#  d1 <- data.frame(x = c(1,2,3), y = c(2,3,6))
#  lapply(d1, scale01)

## ----eval=FALSE---------------------------------------------------------------
#  d2 <- data.frame(x = c(1,2,3), y = c(2,3,6), z = c("a", "b", "c"))
#  lapply(d2[,which(unlist(lapply(d2, is.numeric)))], scale01)   #对每一列先应用is.numeric函数,再对合适的列用scale01函数

## ----eval=FALSE---------------------------------------------------------------
#  d1 <- data.frame(x = c(1,2,3), y = c(2,3,6))
#  vapply(d1, sd, c(sd = 0))

## ----eval=FALSE---------------------------------------------------------------
#  d2 <- data.frame(x = c(1,2,3), y = c(2,3,6), z = c("a", "b", "c"))
#  vapply(d2[,which(unlist(vapply(d2, is.numeric, c(sd=0)))==1)], sd, c(sd=0))   #对每一列先应用is.numeric函数,再对合适的列用scale01函数

## ----eval=FALSE---------------------------------------------------------------
#  fr <- function(a, b, n, N){
#    X <- matrix(0, N, 2)
#    X[1, ] <- c(1, 1/2)
#    for (i in 2:N) {
#      y <- X[i-1, 2]
#      X[i,1] <- rbinom(1, n, y)   #更新一个变量
#      x <- X[i,1]
#      X[i,2] <- rbeta(1, a, b)    #更新另一个变量
#    }
#    return(X)
#  }

## ----eval=FALSE---------------------------------------------------------------
#  cppFunction('NumericMatrix fc(float a, float b, int n, int N){;
#              NumericMatrix X(N,2);
#              X[1,1] = 1;
#              X[1,2] = 1/2;
#              for(int i = 2; i <= N; i++){
#                float y = X[i-1,2];
#                NumericVector x = rbinom(1, n, y);
#                X[i,1] = x[1];
#                NumericVector z = rbeta(1, a, b);
#                X[i,2] = z[1];
#              }
#              return(X);
#  }')

## ----eval=FALSE---------------------------------------------------------------
#  ts <- microbenchmark(fR = fr(2, 2, 10, 1000), fC = fc(2, 2, 10, 1000))
#  summary(ts)[,c(1,3,5,6)]

