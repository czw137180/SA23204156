## ----eval=FALSE---------------------------------------------------------------
#  library("coda")

## ----eval=FALSE---------------------------------------------------------------
#  f <- function(N, b1, b2, b3, f0){
#    x1 <- rpois(N, 1); x2 <- rexp(N); x3 <- rbinom(N, 1, 0.5)
#    g <- function(alpha){
#      tmp <- exp(-alpha-b1*x1-b2*x2-b3*x3); p <- 1/(1+tmp)
#      mean(p) - f0
#    }
#   solution <- uniroot(g, c(-20,0))
#   solution$root
#  }
#  
#  f0 <- c(0.1, 0.01, 0.001, 0.0001)
#  N <- 1e6; b1 <- 0; b2 <- 1; b3 <- -1
#  a <- numeric(4)
#  for (i in 1:4) {
#    a[i] <- f(N, b1, b2, b3, f0[i])
#  }
#  

## ----eval=FALSE---------------------------------------------------------------
#  rbind(f0,a)

## ----eval=FALSE---------------------------------------------------------------
#  plot(-log(f0), a, pch = 19)

## ----eval=FALSE---------------------------------------------------------------
#  dlaplace <- function(x){
#    if(x >=0){
#      exp(-x)/2
#    } else {
#      exp(x)/2
#    }
#  }
#  
#  rw.Metropolis <- function(sigma, x0, N){
#    x <- numeric(N)
#    x[1] <- x0
#    u <- runif(N)
#    k <- 0
#    for (i in 2:N) {
#      y <- rnorm(1, x[i-1], sigma)
#      if (u[i] > (dlaplace(y)/dlaplace(x[i-1])))
#        x[i] <- x[i-1]
#      else {
#        x[i] <- y
#        k <- k + 1
#      }
#    }
#    return(list(x=x, k=k))
#  }

## ----eval=FALSE---------------------------------------------------------------
#  N <- 2000
#  Sigma <- c(.05, .5, 2, 16)
#  x0 <- 20
#  rw1 <- rw.Metropolis(Sigma[1], x0, N)
#  rw2 <- rw.Metropolis(Sigma[2], x0, N)
#  rw3 <- rw.Metropolis(Sigma[3], x0, N)
#  rw4 <- rw.Metropolis(Sigma[4], x0, N)
#  rbind(Sigma, c(rw1$k, rw2$k, rw3$k, rw4$k)/N)

## ----eval=FALSE---------------------------------------------------------------
#  rw <- cbind(rw1$x, rw2$x, rw3$x, rw4$x)
#  for (j in 1:4) {
#    plot(rw[,j], type = "l",
#         xlab = bquote(sigma == .(round(Sigma[j],3))),
#         ylab="X", ylim = range(rw[,j]))
#  }

## ----eval=FALSE---------------------------------------------------------------
#  N <- 4000
#  burn <- 1500
#  X <- matrix(0, N, 2)
#  rho <- 0.9;mu <- c(0,0);Sigma <- c(1,1)
#  s <- sqrt(1-rho^2)*Sigma
#  X[1, ] <- c(mu[1], mu[2])
#  for (i in 2:N) {
#  x2 <- X[i-1, 2]
#  m1 <- mu[1] + rho * (x2 - mu[2]) * Sigma[1]/Sigma[2]
#  X[i, 1] <- rnorm(1, m1, s[1])
#  x1 <- X[i, 1]
#  m2 <- mu[2] + rho * (x1 - mu[1]) * Sigma[2]/Sigma[1]
#  X[i, 2] <- rnorm(1, m2, s[2])
#  }
#  b <- burn + 1
#  x <- X[b:N, ]

## ----eval=FALSE---------------------------------------------------------------
#  plot(x[,1],type='l',col=1,lwd=2,xlab='Index',ylab='Random numbers')
#  lines(x[,2],col=2,lwd=2)
#  legend('bottomright',c(expression(X[1]),expression(X[2])),col=1:2,lwd=2)

## ----eval=FALSE---------------------------------------------------------------
#  Y <- x[,2];X <- x[,1]
#  d1 <- lm(Y~X)
#  d1$coefficients

## ----eval=FALSE---------------------------------------------------------------
#  qqnorm(d1$residuals)
#  qqline(d1$residuals)

## ----eval=FALSE---------------------------------------------------------------
#  Gelman.Rubin <- function(psi) {
#    psi <- as.matrix(psi)
#    n <- ncol(psi)
#    k <- nrow(psi)
#    psi.means <- rowMeans(psi)
#    B <- n * var(psi.means)
#    psi.w <- apply(psi, 1, "var")
#    W <- mean(psi.w)
#    v.hat <- W*(n-1)/n + (B/n)
#    r.hat <- v.hat / W
#    return(r.hat)
#  }
#  
#  dR <- function(x) {
#    if (any(x < 0)) return (0)
#    return((x / 4^2) * exp(-x^2 / (2*4^2)))
#  }
#  
#  Rayleigh.chain <- function(sigma, x0, N){
#    x <- numeric(N)
#    x[1] <- x0
#    u <- runif(N)
#    k <- 0
#    for (i in 2:N) {
#      y <- rnorm(1, x[i-1], sigma)
#      if (u[i] > (dR(y)/dR(x[i-1])))
#        x[i] <- x[i-1]
#      else {
#        x[i] <- y
#      }
#    }
#    x
#  }

## ----eval=FALSE---------------------------------------------------------------
#  sigma <- 1; k <- 4; n <- 15000; b <- 1000
#  x0 <- c(5, 10, 15, 20)
#  X <- matrix(0, nrow=k, ncol=n)
#  for (i in 1:k){
#    X[i, ] <- Rayleigh.chain(sigma, x0[i], n)
#  }
#  psi <- t(apply(X, 1, cumsum))
#  for (i in 1:nrow(psi))
#    psi[i,] <- psi[i,] / (1:ncol(psi))

## ----eval=FALSE---------------------------------------------------------------
#  rhat <- rep(0, n)
#  for (j in (b+1):n)
#    rhat[j] <- Gelman.Rubin(psi[,1:j])
#  plot(rhat[(b+1):n], type="l", xlab="", ylab="R")
#  abline(h=1.2, lty=2)

## ----eval=FALSE---------------------------------------------------------------
#  y <- mcmc.list(mcmc(X[1,]), mcmc(X[2,]), mcmc(X[3,]), mcmc(X[4,]))
#  gelman.diag(y)

