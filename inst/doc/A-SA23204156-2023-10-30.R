## ----eval=FALSE---------------------------------------------------------------
#  f1 <- function(x, y){
#    mean(y <= x)
#  }
#  
#  cvm <- function(x, y){
#    W <- 0
#    n <- length(x)
#    m <- length(y)
#    for (i in 1:n) {
#      W <- W + (f1(x[i], x) - f1(x[i], y))^2
#    }
#    for (j in 1:m) {
#      W <- W + (f1(y[j], x) - f1(y[j], y))^2
#    }
#    m*n/(m+n)^2*W
#  }

## ----eval=FALSE---------------------------------------------------------------
#  x <- c(158 ,171 ,193 ,199 ,230 ,243 ,248 ,248 ,250 ,267 ,271 ,316 ,327 ,329)
#  y <- c(141 ,148 ,169 ,181 ,203 ,213 ,229 ,244 ,257 ,260 ,271 ,309)
#  T.hat <- cvm(x,y)
#  B <- 1000
#  z <- c(x,y)
#  Tstar <- numeric(B)
#  for (i in 1:B) {
#    k <- sample(1:length(z), size = length(x), replace = F)
#    x1 <- z[k]
#    y1 <- z[-k]
#    Tstar[i] <- cvm(x1, y1)
#  }
#  p <- mean(c(T.hat, Tstar) >= T.hat)
#  p

## ----eval=FALSE---------------------------------------------------------------
#  count5test <- function(x, y){
#    X <- x - mean(x)
#    Y <- y - mean(y)
#    outx <- sum(X > max(Y)) + sum(X < min(Y))
#    outy <- sum(Y > max(X)) + sum(Y < min(X))
#    return(max(c(outx, outy)))
#  }
#  n1 <- 20
#  n2 <- 30
#  mu1 <- mu2 <- 0
#  sigma1 <- sigma2 <- 1
#  m <- 10000
#  x <- rnorm(n1, mu1, sigma1)
#  y <- rnorm(n2, mu2, sigma2)
#  count.hat <- count5test(x,y)
#  z <- c(x,y)
#  countstar <- numeric(m)
#  for (i in 1:m) {
#    k <- sample(1:length(z), size = length(x), replace = F)
#    x1 <- z[k]
#    y1 <- z[-k]
#    countstar[i] <- count5test(x1,y1)
#  }
#  p <- mean(c(count.hat, countstar) >= count.hat)
#  p

