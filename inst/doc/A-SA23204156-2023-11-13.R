## ----eval=FALSE---------------------------------------------------------------
#  library(boot)

## ----eval=FALSE---------------------------------------------------------------
#  u <- c(11,8,27,13,16,0,23,10,24,2)
#  t <- sum(u)
#  f <- function(x) -t-10*exp(-x)/(exp(-x)-1)
#  s1 <- uniroot(f, interval = c(0.0001,10))
#  s1$root

## ----eval=FALSE---------------------------------------------------------------
#  lambda0 <- 0
#  lambda1 <- 1
#  k <- 0
#  while (lambda1 - lambda0 > 0.00001 | k<10000) {
#    lambda0 <- lambda1
#    lambda1 <- 10/(t+10/lambda0+10*exp(-lambda0)/(exp(-lambda0)-1))
#      k <- k+1
#  }
#  lambda1

## ----eval=FALSE---------------------------------------------------------------
#  solve.game <- function(A) {
#  min.A <- min(A)
#  A <- A - min.A
#  max.A <- max(A)
#  A <- A / max(A)
#  m <- nrow(A)
#  n <- ncol(A)
#  it <- n^3
#  a <- c(rep(0, m), 1)
#  A1 <- -cbind(t(A), rep(-1, n))
#  b1 <- rep(0, n)
#  A3 <- t(as.matrix(c(rep(1, m), 0)))
#  b3 <- 1
#  sx <- simplex(a=a, A1=A1, b1=b1, A3=A3, b3=b3,
#  maxi=TRUE, n.iter=it)
#  
#  a <- c(rep(0, n), 1)
#  A1 <- cbind(A, rep(-1, m))
#  b1 <- rep(0, m)
#  A3 <- t(as.matrix(c(rep(1, n), 0)))
#  b3 <- 1
#  sy <- simplex(a=a, A1=A1, b1=b1, A3=A3, b3=b3,
#  maxi=FALSE, n.iter=it)
#  soln <- list("A" = A * max.A + min.A,
#  "x" = sx$soln[1:m],
#  "y" = sy$soln[1:n],
#  "v" = sx$soln[m+1] * max.A + min.A)
#  soln
#  }

## ----eval=FALSE---------------------------------------------------------------
#  A <- matrix(c( 0,-2,-2,3,0,0,4,0,0,
#                 2,0,0,0,-3,-3,4,0,0,
#                 2,0,0,3,0,0,0,-4,-4,
#                 -3,0,-3,0,4,0,0,5,0,
#                 0,3,0,-4,0,-4,0,5,0,
#                 0,3,0,0,4,0,-5,0,-5,
#                 -4,-4,0,0,0,5,0,0,6,
#                 0,0,4,-5,-5,0,0,0,6,
#                 0,0,4,0,0,5,-6,-6,0), 9, 9)
#  s <- solve.game(A+3)
#  round(cbind(s$x, s$y), 7)

## ----eval=FALSE---------------------------------------------------------------
#  s2 <- solve.game(A)
#  rbind(A = s2$v,B = s$v)

