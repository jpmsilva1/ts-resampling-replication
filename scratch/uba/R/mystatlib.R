
## ======================================================================
## Standard Regression Errors
## ----------------------------------------------------------------------
## Coded by : Rita P. Ribeiro 
## ======================================================================

mse <- function(ypred,y,...) mean((y - ypred)^2)

mad <- function(ypred,y,...) mean(abs(y - ypred))

rmse <- function(ypred,y,...) sqrt(mean((y - ypred)^2))


nmse <- function(ypred,y,naive.y)
  pmin(mse(ypred,y)/mse(naive.y,y),1.000000)

  
nmad <- function(ypred,y,naive.y)
  pmin(mad(ypred,y)/mad(naive.y,y),1.000000)



nrmse <- function(ypred,y,naive.y)
  pmin(rmse(ypred,y)/rmse(naive.y,y),1.000000)
 

##=========================================================================
## Auxiliary Function
## getndp
##-------------------------------------------------------------------------
## check the number of digits which fully represent
## the value
## > getndp(0.04)
## 2
## the minimum integer!
## 0.04 * 10^(getndp(0.04))
## format(x,nsmall=getndp(x))
##
## Available on R help
##=========================================================================

getndp <- function(x, tol=2*.Machine$double.eps)
{
  ndp <- 0
  while(!isTRUE(all.equal(x, round(x, ndp), tol=tol))) ndp <- ndp+1 
  if(ndp > -log10(tol))
    warning("Tolerance reached, ndp possibly underestimated.")
  ndp 
}


format.value <- function(p,dsig=7)   
  as.double(formatC(p,digits=dsig,format="f",mode="double"))


is.number <- function(value)
  gregexpr("^[-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?$",value) > -1



##=========================================================================
## 
##-------------------------------------------------------------------------
## This set of functions enables the generation of a stratified sample
## based on quantiles (it depends on Hmisc).
## 
## Coded by: Rita P. Ribeiro
##=========================================================================
get.strat.idx <- function(y,g=10) as.numeric(Hmisc:::cut2(y,g=g))

  
ensemble.idx <- function(idx.sample,prs,strat=F,prop.size=1,
                         repl=T,seed=1) {

  set.seed(seed)
  
  if(!strat)
    sample(idx.sample,size=prop.size*length(idx.sample),
           prob=prs,rep=repl) ## bootstrap
  else
    sample.strat(idx.sample,prs,prop.size=prop.size,rep=repl)
}


sample.strat <- function(idx.sample,prs,prop.size=1,repl=T) {

  new.idx <- NULL

  all.qs <- unique(idx.sample)

  for(i in all.qs) {
    q <- which(idx.sample == i)
    prs.q <- prs[q]

    n.q <- as.integer(length(q))

    if(prop.size != 1)
      n.q <- round(ifelse(!repl,pmin(prop.size*n.q,n.q),prop.size*n.q)) 

    if(n.q == 1)
      new.idx <- c(new.idx,q)
    else if(n.q > 1)
      new.idx <- c(new.idx,sample(q,size=n.q,prob=prs.q,rep=repl))
  }
  new.idx
}
  

##=========================================================================
## cross.val
##-------------------------------------------------------------------------
## This function implements the cross-validation method.
## Coded by: Rita Ribeiro
##========================================================================
cross.val <- function(data,n.rep=1,n.folds=10,
                      seed=sample(100,1),
                      strat=F,n.strats=10) {

  if(n.folds < 1)
    stop("Invalid number of folds\n")

  n <- length(data)

  if(n.folds * n.strats > length(data)) {
    n.strats <- as.integer(length(data) / n.folds)
    cat("Changing the number of strats to",n.strats,"\n")
  }
    
  l <- alist()
 
  l$iteration <- vector(mode="list",length=n.rep)
  
  if(strat)
    idx.sample <- get.strat.idx(data,n.strats)
  else
    idx.sample <- 1:n
  
  prs <- rep(1L/n,n)

  fold.size <- as.integer(n/n.folds)
    
  for(i in 1:n.rep) {
    
    idx.sample.fold <- idx.sample
    prop.size <- 1/n.folds
   
    l$iteration[[i]]$fold <- vector(mode="list",length=n.folds)
            
    set.seed(seed*i)

    fold.idx <- alist()

   
    for(f in 1:n.folds) {      
      
      
      l$iteration[[i]]$fold[[f]] <-
        ensemble.idx(idx.sample.fold,prs,strat=T,
                     prop.size=prop.size,repl=F)

      idx.sample.fold <- idx.sample.fold[-l$iteration[[i]]$fold[[f]]]
      
      if(f != n.folds ) prop.size <- 1/(n.folds - f)      
    }
            
  }

  l
  
}


