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

