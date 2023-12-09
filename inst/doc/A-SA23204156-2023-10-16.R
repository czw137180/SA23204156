## ----eval=FALSE---------------------------------------------------------------
#  library(bootstrap)

## ----eval=FALSE---------------------------------------------------------------
#  m <- 1000
#  M <- 10000
#  FWER.bonf <- FDR.bonf <- TPR.bonf <- numeric(M)
#  FWER.bh <- FDR.bh <- TPR.bh <- numeric(M)
#  for (i in 1:M) {
#    p <- numeric(1000)
#    p[1:950] <- runif(950)
#    p[951:1000] <- rbeta(50, 0.1, 1)
#    p.bonf <- p.adjust(p, method = 'bonferroni')
#    p.bh <- p.adjust(p, method = 'fdr')
#    FWER.bonf[i] <- max(p.bonf[1:950] < 0.1)
#    FDR.bonf[i] <- sum(p.bonf[1:950] < 0.1)/sum(p.bonf < 0.1)
#    TPR.bonf[i] <- sum(p.bonf[951:1000] < 0.1)/50
#    FWER.bh[i] <- max(p.bh[1:950] < 0.1)
#    FDR.bh[i] <- sum(p.bh[1:950] < 0.1)/sum(p.bh < 0.1)
#    TPR.bh[i] <- sum(p.bh[951:1000] < 0.1)/50
#  }
#  d <- data.frame(FWER=c(mean(FWER.bonf),mean(FWER.bh)), FDR=c(mean(FDR.bonf),mean(FDR.bh)), TPR=c(mean(TPR.bonf),mean(TPR.bh)))
#  row.names(d) <- c('Bonf', 'B-H')
#  d

## ----eval=FALSE---------------------------------------------------------------
#  lambda <- 2
#  B <- 1000
#  m <- 1000
#  bias1 <- sd1 <- numeric(m)
#  bias2 <- sd2 <- numeric(3)
#  bias3 <- sd3 <- numeric(3)
#  k <- 1
#  for (n in c(5,10,20)) {
#    for (i in 1:m) {
#      B <- 1000
#      lambdastar <- numeric(B)
#      x <- rexp(n, lambda)
#      lambdahat <- 1/mean(x)
#      for (j in 1:B) {
#        xstar <- sample(x, replace = T)
#        lambdastar[j] <- 1/mean(xstar)
#      }
#      bias1[i] <- mean(lambdastar) - lambdahat
#      sd1[i] <- sd(lambdastar)
#    }
#    bias2[k] <- mean(bias1)
#    sd2[k] <- mean(sd1)
#    bias3[k] <- lambda/(n-1)
#    sd3[k] <- n*lambda/(n-1)/sqrt(n-2)
#    k <- k+1
#  }
#  d2 <- data.frame(bias.bootstrap = bias2, bias.theoretical = bias3, sd.bootstrap = sd2, sd.theoretical = sd3)
#  row.names(d2) <- c('n=5', 'n=10', 'n=20')
#  d2

## ----eval=FALSE---------------------------------------------------------------
#  B <- 200
#  n <- nrow(law)
#  R <- numeric(B)
#  r <- se <- t <- numeric(1000)
#  corrhat <- cor(law$LSAT, law$GPA)
#  for (j in 1:1000) {
#    for (b in 1:B) {
#      i <- sample(1:n, size = n, replace = T)
#      LSAT <- law$LSAT[i]
#      GPA <- law$GPA[i]
#      R[b] <- cor(LSAT, GPA)
#    }
#    r[j] <- mean(R)
#    se[j] <- sd(R)
#    t[j] <- (r[j] - corrhat)/se[j]
#  }
#  sehat <- mean(se)
#  QT <- quantile(t, c(0.05/2, 1-0.05/2), type = 1)
#  names(QT) <- rev(names(QT))
#  CI <- rev(corrhat - QT*sehat)
#  CI

