##=========================================================================
## ubar
##-------------------------------------------------------------------------
## This function 
## UbaR - Utility-based Rules derived from an ensemble of rpart trees
## data = NULL
##  hens = "boost" -> always
## Coded by: Rita P. Ribeiro
##=========================================================================

# --------------------------------------------------------------
# constructor function of ubar class
ubar <- function(formula, data, weights, subset,                                 
                 phi.parms, loss.parms, util.parms,
                 rules.parms, cp = 0, 
                 update.phi=FALSE, fn = NULL, ...) {

  
  mf0 <- match.call(expand.dots = FALSE)  
  m <- match(c("formula", "data", "weights","subset"), names(mf0), 0)
  mf <- mf0[c(1, m)]
  mf$drop.unused.levels <- TRUE
  mf[[1]] <- as.name("model.frame")
  mf <- eval.parent(mf) ## it removes observations with NA
  y <- model.response(mf)
  nobs <- length(y)
  
  ec <- eval.control(y, update.phi = update.phi,
                     phi.parms, loss.parms, util.parms, ...)  


  ## ubar object
  ## --------------------------------------------------
  object <- new("ubar",
                call=as.character(mf0), ## to change afterwards
                target.var=all.vars(attr(mf, "terms"))[1],                
                y=y,
                phi.parms=ec$phi.parms, loss.parms=ec$loss.parms,
                util.parms=ec$util.parms,                    
                rules.parms=ubar.control(rules.parms,...),
                rp.parms=rpart.control(cp=cp, ...),
                def.rule=mean(y),
                rules.cond=character(),
                rules.yval=matrix(NA,nrow=0,ncol=2,
                  dimnames=list(NULL,c("yval","wt"))),
                rules.info=matrix(NA,nrow=0,ncol=3,
                  dimnames=list(NULL,c("n","mad","util")))
                ) 
  ## --------------------------------------------------
  
  treesL <- vector(mode="list",length=object@rules.parms$ntree)
  treesW <- rep(1/object@rules.parms$ntree,object@rules.parms$ntree)
  
  if(object@rules.parms$strat)
    idx.sample <- get.strat.idx(y)
  else
    idx.sample <- 1:nobs

  prs <- rep(1L/nobs,nobs)
  
  
  bl <- learner('urpart',
                pars=list(cp=object@rp.parms$cp,                  
                  phi.parms=ec$phi.parms,loss.parms=ec$loss.parms,util.parms=ec$util.parms))  

  it.idx <- ensemble.idx(idx.sample,prs,strat=object@rules.parms$strat,seed=1)


  md <- numeric()
  for(i in 1:object@rules.parms$ntree) {
    set.seed(10*i); md[i] <- 2 + floor(rexp(1,0.5))    
  }  

 
  maxrules <- 0
  nt <- 0
  switch(object@rules.parms$ens,
         ## ----------------------------------
         "bagg" = {
           
           for(i in 1:object@rules.parms$ntree) {
                          
             cat("*");

             set.seed(i); 
                          
             it.tree <- runLearner(bl,mf,subset=it.idx,maxdepth=md[i])
             it.rules <- length(grep("<leaf>",it.tree$frame[,1]))

             if(it.rules > 1) {  ## discard root trees             
               nt <- nt + 1
               maxrules <- maxrules + it.rules
               treesL[[nt]] <- it.tree               
             }

             it.idx <- ensemble.idx(idx.sample,prs,strat=object@rules.parms$strat,seed=i+1)
           } ## end iterations
         }, ## end bagging
         ## ----------------------------------
         "boost" = {

           lboost <- list(weight.it=NULL,prs=prs)
           
           for(i in 1:object@rules.parms$ntree) {

             cat("*");
                          
             set.seed(i)
             
             it.tree <- runLearner(bl,mf,subset=it.idx,maxdepth=md[i])
             it.rules <- length(grep("<leaf>",it.tree$frame[,1]))

             if(it.rules > 1) {  ## discard root trees             
               nt <- nt + 1
               maxrules <- maxrules + it.rules
               treesL[[nt]] <- it.tree               
               
               lboost <- ubar.boost.update(object, treesL[1:nt], treesW[1:nt],
                                           mf[it.idx,], prs)

               treesW[nt] <- lboost$weight.it
               
             }
             ## proceed to the next iteration with (previous / new) prs                                                           
             set.seed(i+1); it.idx <- sample(nobs,prob=lboost$prs,replace=FALSE)
             
           } ## end iterations
         } ## end boosting
         )
           
  cat("\n")

  if(maxrules) {
    
    object@rules.cond <- vector(mode="character",length=maxrules)
    object@rules.yval <- matrix(NA,nrow=maxrules,ncol=2,
                                dimnames=list(NULL,c("yval","wt")))  
    last.n <- 0

    for(j in 1:nt) {
      rp.rules <- rules.rpart(treesL[[j]])
      n <- NROW(rp.rules)
      idx <- last.n+(1:n)    
      object@rules.cond[idx] <- rp.rules[,2]
      object@rules.yval[idx,1] <- as.numeric(rp.rules[,1])
      object@rules.yval[idx,2] <- treesW[j]
      last.n <- last.n + n
    }          
    
    cat('Total Nr of rules (with duplicates): ',last.n,'\n')

    object <- ubar.rm.duplicated(object)
    
    cat('Effective Nr of rules: ',nr.rules(object),'\n')
  
    object <- ubar.prune(object, mf, object@rules.parms$sel)

  }
  
  ## change to save afterwards
  if(!is.null(fn)) save(object,file=fn)

  object
  
}  

##=========================================================================
## ubar.rules.update
##=========================================================================
ubar.rules.update <- function(object,idx) {

  n.idx <- length(idx)

  if(n.idx) {    
    object@rules.cond <- object@rules.cond[idx]
    
    if(n.idx == 1)
      object@rules.yval <-
        matrix(object@rules.yval[idx,],nrow=1,ncol=2,
               dimnames=list(NULL,c("yval","wt")))
    else
      object@rules.yval <- object@rules.yval[idx,1:2]

   
  }

  object
}

##=========================================================================
## ubar.boost.update
## To update the probabilities of sampling for the next iteration, as
## well as to assign weight to the rules.
##
## beta.conf = inverse measure of confidence in the predictor;
## low beta means high confidence in the prediction
## we can use: linear, square or cubic law.
## Here, we chose to use the quadratic law for beta (k = 2).
##
##
## If all the preds are considered to incur into a cost, and thus,
## are incorrect then beta makes the current iteration have almost no
## effect in the final result; there are no probailities to be updated;
## we have to re-learn it all over again.
##
## If all the preds are considered to bring a positive benefit, and thus,
## are correct then the current iteration will have a big effect in the
## final result; all the probabilities have to be reduced;
## in practice, all will remain the same.
##
## Otherwise the probabilities have to be updated, that is reduced, in the
## for the benefit preds.
## The smaller the loss, the more the weight is reduced,
## making the probability smaller that this pattern will be picked
## as a member of the next training set.
##
## Z is the normalization factor chosen such that prs is a distribution
## again, ie, such that its integral sums up to one.
##-------------------------------------------------------------------------
## According to AdaBoost.RT (Solomatine, D. and Shrestha, D. (2004))
## Coded by: Rita P. Ribeiro
##=========================================================================
ubar.boost.update <- function(object, 
                         treesL, treesW,
                         data, prs, k=2) {

  ## use rpart prediction functionality -> eficiency reasons
  ## ------
  m <- sapply(treesL,predict,data)
  w <- treesW/sum(treesW)
  preds <- apply(w * t(m),2,sum)
  ## ------
  
  uv <- util(preds,data[,object@target.var],
             phi.parms = object@phi.parms,
             loss.parms = object@loss.parms,                    
             util.parms = object@util.parms,
             return.uv=TRUE)
  
  n <- length(preds)

  if(object@util.parms$umetric == "MU") 
    uv <- (uv + 1) / 2 ## to uniformize the analysis

  benef.preds <- which(uv > 0.5)
  
  n.benef <- length(benef.preds)
  
  if(!n.benef) {

    beta.conf <- 1 - min(prs)^(k+1) ## to not to be the same data set

  } else if(n.benef == n) {

    beta.conf <- min(prs)^(k+1)
       
  } else {
    
    cost.preds <- (1:n)[-benef.preds]

    ## where did I get this idea?
    ##err.rate <- sum(exp(log(prs[cost.preds]) + log(1-ur[cost.preds])))

    err.rate <- sum(prs[cost.preds])
    
    beta.conf <- err.rate^k
  
    
    if(!beta.conf) stop("beta.conf can never be zero\n")


    prs[benef.preds] <- prs[benef.preds] * beta.conf

    Z <- sum(prs)

    prs <- prs/Z
    
  }
  
  list(weight.it=log(1/beta.conf),prs=prs)

}



##=========================================================================
## ubaRules.prune
##-------------------------------------------------------------------------
## This function gather all rules in one theory;
## select the best rules of the theory.
## AT THE END ONLY ONE DEFAULT RULE SHOULD EXIST
## 
## Coded by: Rita P. Ribeiro
##=========================================================================
ubar.prune <- function(object, data, sel) {
    
  nr <- nr.rules(object)
  
  trigger.matrix <- trigger.rules(nr, object@rules.cond, data)  
  
  eval.matrix <-
    eval.rules(nr, object@rules.yval,
               object@util.parms$umetric,
               object@y, trigger.matrix,
               object@phi.parms, object@loss.parms, object@util.parms)
    
  ntm <- NCOL(trigger.matrix)
  nem <- NCOL(eval.matrix)
  
  if(nr > 1) { 
    dec.list <- topsel(sel, nr, eval.matrix)
    object <- ubar.rules.update(object,dec.list)
  } else {
    dec.list <- 1
  }

  
  object <- ubar.info(object, ##dec.list,
                      matrix(eval.matrix[dec.list,],ncol=nem),
                      matrix(trigger.matrix[dec.list,],ncol=ntm))
  
  object
}

##=========================================================================
## topsel
##-------------------------------------------------------------------------
## This function 
##
## Coded by: Rita P. Ribeiro
##=========================================================================
topsel <- function(sel=0.5, nr, eval.matrix) {
    
  unsel.idx <- (1:nr)
  
  rank.idx <- sort(eval.matrix[unsel.idx,ncol(eval.matrix)],decreasing=T,
                   index.return=T)$ix       

  if(sel <= 1) sel <- round(nr*sel)
 
  unsel.idx[rank.idx[1:sel]]
}

##=========================================================================
## ubar.rm.duplicated
##=========================================================================
ubar.rm.duplicated <- function(object) {
  
  
  nr <- nr.rules(object)
  idx <- which(duplicated(object@rules.cond))

 
  
  if(length(idx) > 0) {

    dup.cond <- unique(object@rules.cond[idx])

    for(cond in dup.cond) { ## by rule      
      j <- which(object@rules.cond %in% cond)        
      object@rules.yval[j,1] <- sum(object@rules.yval[j,1] * object@rules.yval[j,2])/sum(object@rules.yval[j,2])
      object@rules.yval[j,2] <- mean(object@rules.yval[j,2])
    }

    to.keep <- (1:nr)[-idx]
  
    object <- ubar.rules.update(object,to.keep)
    
  }

  object
}
##=========================================================================
## ubar.info
##=========================================================================
ubar.info <- function(object, 
                      eval.matrix, trigger.matrix) {
  
  nr <- nr.rules(object)

  cat('Nr of rules:',nr,'\nr')
  
  object@rules.info <- matrix(NA,nrow=nr,ncol=3)
  colnames(object@rules.info) <-  c("n","mad","util")
  
  for(i in 1:nr) {
    exs <- which(trigger.matrix[i,])
    n <- length(exs)
    object@rules.info[i,"n"] <- n
    object@rules.info[i,"mad"] <- (sum(abs(object@y[exs] - object@rules.yval[i,1]))/n)        
    object@rules.info[i,"util"] <- eval.matrix[i,1] ##?
  }

  object
}

nr.rules <- function(object) NROW(object@rules.yval)

