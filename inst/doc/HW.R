## ----eval=FALSE---------------------------------------------------------------
#  u <- runif(1000) #生成均匀分布样本
#  x1 <- 1:1000
#  x1[which(u<0.5)] <- log(2*u[which(u<0.5)])
#  x1[which(u>=0.5)] <- -log(2-2*u[which(u>=0.5)])    #利用逆变换得到Laplace分布的样本

## ----eval=FALSE---------------------------------------------------------------
#  hist(x1, prob = TRUE, main = expression(f(x)==frac(1,2)*e^(-abs(x))))
#  curve(1/2*exp(-abs(x)), col = "red", add = TRUE, lwd =2)

## ----eval=FALSE---------------------------------------------------------------
#  my.rbeta <- function(a,b,n){# Beta分布的参数和样本量
#    k <- 0                    #k用来控制生成的样本量
#    x <- numeric(n)           #用来储存生成的样本
#    while (k < n) {
#      u <- runif(1) #生成均匀分布
#      y <- runif(1) #生成g(x)的样本
#      c <- (a-1)^(a-1)*(b-1)^(b-1)/((a+b-2)^(a+b-2))
#      rho_y <- y^(a-1)*(1-y)^(b-1) #计算rho(y)
#      if(u <= rho_y){  #接受拒绝的过程
#        x[k+1] <- y
#        k <- k+1
#      }
#    }
#    x                #返回生成的样本
#  }

## ----eval=FALSE---------------------------------------------------------------
#  x2 <- my.rbeta(3,2,1000)
#  hist(x2, prob = T, main = expression(f(x)==frac(1,B(3,2))*x^2*(1-x)))
#  curve(1/beta(3,2)*x^2*(1-x), col = "red", add = T, lwd =2)

## ----eval=FALSE---------------------------------------------------------------
#  my.repane <- function(n){  #指定样本量
#    x <- numeric(n)
#    k <- 0
#    while (k < n) {    #按照算法得到样本
#      k <- k+1
#      u <- runif(3, -1, 1)
#      if(which.max(abs(u))==3){
#        x[k] <- u[2]
#      } else
#      {
#        x[k] <- u[3]
#      }
#    }
#    x
#  }

## ----eval=FALSE---------------------------------------------------------------
#  x3 <- my.repane(1000)
#  hist(x3, prob = T, main = expression(f(x)==frac(3,4)*(1-x^2)))
#  curve(3/4*(1-x^2), col = "red", add = T, lwd =2)

## ----eval=FALSE---------------------------------------------------------------
#  my.sample <- function(x, size, prob = rep(1,length(x))){
#    if(length(prob)!=length(x)){   # 判断变量输入是否合理
#      print("The length of x and prob is not equal!")
#    }else
#    {
#      p <- prob/sum(prob)
#      cp <- cumsum(p)
#      u <- runif(size)      # 生成均匀分布
#      r <- x[findInterval(u,cp)+1]    #逆变换
#      return(r)         #放回所生成的样本
#    }
#  }

## ----eval=FALSE---------------------------------------------------------------
#  x4 <- my.sample(c(1,2,3,4), 1e5, c(1))

## ----eval=FALSE---------------------------------------------------------------
#  prob <- runif(4, 0, 10)  #随机生成权重
#  p <- prob/sum(prob)      #将权重转换为概率
#  x5 <- my.sample(c(1,2,3,4), 1e5, prob = prob)   #生成样本
#  ct <- as.vector(table(x5))          #统计频率
#  ct/sum(ct)/p             #用比值进行比较每个取值的观测的频率和概率

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

