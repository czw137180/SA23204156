## ----eval=FALSE---------------------------------------------------------------
#  m <- 1e6
#  theta.hat <- se <- numeric(2)
#  g <- function(x){
#    x^2/sqrt(2*pi)*exp(-x^2/2)
#  }
#  f1 <- function(x){
#    exp(-x+1)
#  }
#  x <- 1 + rexp(m)
#  theta.hat[1] <- mean(g(x)/f1(x))
#  se[1] <- sd(g(x)/f1(x))

## ----eval=FALSE---------------------------------------------------------------
#  f2 <- function(x){
#    sqrt(2/pi)*exp(-(x-1)^2/2)
#  }
#  x <- 1 + abs(rnorm(m, 1, 1) - 1)
#  theta.hat[2] <- mean(g(x)/f2(x))
#  se[2] <- sd(g(x)/f2(x))

## ----eval=FALSE---------------------------------------------------------------
#  integrate(g, 1, Inf)

## ----eval=FALSE---------------------------------------------------------------
#  rbind(theta.hat, se)

## ----eval=FALSE---------------------------------------------------------------
#  m <- 1e4
#  k <- 5
#  theta.hat2 <- numeric(5)
#  se2 <- numeric(5)
#  g2 <- function(x){
#    exp(-x)/(1+x^2)
#  }
#  f3 <- function(x,j){
#    exp(-x)/(exp(-(j-1)/5)-exp(-j/5))
#  }
#  for (j in 1:5) {
#    u <- runif(m/k)
#    x <- -log(exp(-(j-1)/5) - u*(exp(-(j-1)/5)-exp(-j/5)))
#    theta.hat2[j] <- mean(g2(x)/f3(x,j))
#    se2[j] <- sd(g2(x)/f3(x,j))
#  }
#  theta.hat3 <- sum(theta.hat2)
#  se3 <- sqrt(1/(m*k)*sum(se2^2))

## ----eval=FALSE---------------------------------------------------------------
#  rbind(theta.hat3, se3)

## ----eval=FALSE---------------------------------------------------------------
#  n = 20
#  alpha <- 0.05
#  UCL <- LCL <- numeric(1000)
#  for (j in 1:1000) {
#    x <- rchisq(n, df = 2)
#    UCL[j] <- mean(x) + sd(x)/sqrt(n)*qt(1 - alpha/2, df = n-1)
#    LCL[j] <- mean(x) - sd(x)/sqrt(n)*qt(1 - alpha/2, df = n-1)
#  }
#  mean(UCL > 2 & LCL < 2)

## ----eval=FALSE---------------------------------------------------------------
#  error1 <- numeric(3)
#  n = 20
#  alpha <- 0.05
#  T1 <- numeric(10000)
#  for (j in 1:10000) {
#    x <- rchisq(n, df = 1)
#    T1[j] <- sqrt(n)*(mean(x) - 1)/sd(x)
#  }
#  error1[1] <- mean(abs(T1) > qt(1 - alpha/2, df = n-1))

## ----eval=FALSE---------------------------------------------------------------
#  n = 20
#  alpha <- 0.05
#  T2 <- numeric(10000)
#  for (j in 1:10000) {
#    x <- runif(n, 0, 2)
#    T2[j] <- sqrt(n)*(mean(x) - 1)/sd(x)
#  }
#  error1[2] <- mean(abs(T2) > qt(1 - alpha/2, df = n-1))

## ----eval=FALSE---------------------------------------------------------------
#  n = 20
#  alpha <- 0.05
#  T3 <- numeric(10000)
#  for (j in 1:10000) {
#    x <- rexp(n)
#    T3[j] <- sqrt(n)*(mean(x) - 1)/sd(x)
#  }
#  error1[3] <- mean(abs(T3) > qt(1 - alpha/2, df = n-1))

## ----eval=FALSE---------------------------------------------------------------
#  rbind(error1)

