## ----eval=FALSE---------------------------------------------------------------
#  l <- c(0.5,0.8,1)
#  d <- 1
#  m <- 1e4
#  v <- numeric(3)
#  pihat <- numeric(1e6)
#  for (i in 1:3) {
#    for (j in 1:1e6) {
#      x <- runif(m,0,d/2)
#      y <- runif(m,0,pi/2)
#      pihat[j] <- 2*l[i]/d/mean(l[i]/2*sin(y)>x)
#    }
#    v[i] <- var(pihat)
#  }
#  v

## ----eval=FALSE---------------------------------------------------------------
#  m <- 1e4
#  U <- runif(m)
#  T1 <- exp(U)
#  T2 <- (exp(U) + exp(1-U))/2
#  mean(T1)
#  mean(T2)
#  (var(T1)-var(T2))/var(T1)

