## ----eval=FALSE---------------------------------------------------------------
#  library('bootstrap')
#  library('DAAG')

## ----eval=FALSE---------------------------------------------------------------
#  x <- c(3, 5, 7, 18, 43, 85, 91, 98, 100, 130, 230, 487)
#  SN <- Basic <- Percent <- BCa <- numeric(2)
#  B <- 10000
#  theta <- numeric(B)
#  for (i in 1:B) {
#    j <- sample(1:12, 12, replace = T)
#    theta[i] <- mean(x[j])
#  }
#  y <- sort(theta)
#  bootse <- sd(theta)
#  SN <- c(mean(x) - qnorm(0.975)*bootse, mean(x) + qnorm(0.975)*bootse)
#  Basic <- c(2*mean(x) - y[ceiling(B*0.975)], 2*mean(x) - y[floor(B*0.025)])
#  Percent <- c(y[floor(B*0.025)], y[ceiling(B*0.975)])
#  z <- qnorm(mean(theta < mean(x)))
#  theta.j <- numeric(12)
#  for (i in 1:12) {
#    theta.j[i] <- mean(x[-i])
#  }
#  x.j <- mean(theta.j)
#  a <- sum((x.j-theta.j)^3)/6/sum(abs(x.j-theta.j)^3)
#  alpha1 <- pnorm(z + (z + qnorm(0.025))/(1 - a*(z + qnorm(0.025))))
#  alpha2 <- pnorm(z + (z + qnorm(0.975))/(1 - a*(z + qnorm(0.975))))
#  BCa <- c(y[floor(B*alpha1)], y[ceiling(B*alpha2)])
#  d <- data.frame('Standard Normal' = SN, Basci = Basic, Percentile = Percent, BCa = BCa)
#  row.names(d) <- c('LCL', 'UCL')
#  d

## ----eval=FALSE---------------------------------------------------------------
#  n <- nrow(scor)
#  theta.j <- numeric(n)
#  for (i in 1:n) {
#    Sigma <- cov(scor[-i,])
#    e <- eigen(Sigma, symmetric = T)
#    lambda <- e$values
#    theta.j[i] <- max(lambda)/sum(lambda)
#  }
#  e1 <- eigen(cov(scor), symmetric = T)
#  lambda1 <- e1$values
#  theta.hat <- max(lambda1)/sum(lambda1)
#  bias.hat <- (n-1)*(mean(theta.j) - theta.hat)
#  se.hat <- (n-1)^2/n*var(theta.j)
#  rbind(bias.hat, se.hat)

## ----eval=FALSE---------------------------------------------------------------
#  m <- ironslag$magnetic
#  c <- ironslag$chemical
#  n <- length(m)
#  e1 <- e2 <- e3 <- e4 <- numeric(n*(n-1)/2)
#  k <- 0
#  for (i in 1:(n-1)) {
#    for (j in (i+1):n) {
#      k <- k + 1
#      y <- m[-c(i,j)]
#      x <- c[-c(i,j)]
#      J1 <- lm(y ~ x)
#      yhat1 <- J1$coef[1] + J1$coef[2] * c[c(i,j)]
#      e1[k] <- sum((m[c(i,j)] - yhat1)^2)
#      J2 <- lm(y ~ x + I(x^2))
#      yhat2 <- J2$coef[1] + J2$coef[2] * c[c(i,j)] + J2$coef[3] * c[c(i,j)]^2
#      e2[k] <- sum((m[c(i,j)] - yhat2)^2)
#      J3 <- lm(log(y) ~ x)
#      logyhat3 <- J3$coef[1] + J3$coef[2] * c[c(i,j)]
#      yhat3 <- exp(logyhat3)
#      e3[k] <- sum((m[c(i,j)] - yhat3)^2)
#      J4 <- lm(log(y) ~ log(x))
#      logyhat4 <- J4$coef[1] + J4$coef[2] * log(c[c(i,j)])
#      yhat4 <- exp(logyhat4)
#      e4[k] <- sum((m[c(i,j)] - yhat4)^2)
#    }
#  }
#  cbind(mean(e1), mean(e2), mean(e3), mean(e4))

