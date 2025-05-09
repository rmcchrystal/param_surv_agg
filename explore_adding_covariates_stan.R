library(tidyverse)
library(cmdstanr)

## Used copilot to translate BUGS to Stan. Appears to recover values

## generate some aggregate data based on Gompertz ----
myvar = sample(0:1, 10, replace = TRUE)
myrate <- 0.2
mybeta <- 2
myrates <- if_else(myvar ==1, myrate*mybeta, myrate) 
fu <- rexp(10, 1)
mya <- rlnorm(10, log(0.5), 0.1)
ps <- flexsurv::pgompertz(fu, shape = mya, rate = myrates)
ps_exp <- pexp(fu, rate = myrate)
ps
ps_exp
Ns <- runif(10, 100, 1000) %>% round()
r <- rbinom(10, size = Ns, prob = ps)

mdl <- cmdstanr::cmdstan_model("gomp_stan.stan")
fit <- mdl$sample(data = list(r = r, 
                                 dt = fu,
                                 n = Ns,
                                 N = length(Ns),
                                 var1 = myvar))
fit
smry <- fit$summary()

## examine some shape, rate parameter recovery and r recovery
## shape
xmnb <- smry %>% filter(str_detect(variable, "b_out"))
xmnb %>% mutate(true = myrates) %>% select(variable, true, everything())

## rate
xmna <- smry %>% filter(str_detect(variable, "a_out"))
xmna %>% mutate(true = mya) %>% select(variable, true, everything())

# counts
xmnr <- smry %>% filter(str_detect(variable, "r_new"))
xmnr %>% mutate(true = r) %>% select(variable, true, everything())


