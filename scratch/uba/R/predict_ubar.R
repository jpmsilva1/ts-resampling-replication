##=========================================================================
## predict.ubar
##-------------------------------------------------------------------------
## This function 
## Coded by: Rita P. Ribeiro
##=========================================================================

predict.ubar <- function(object, newdata,           
                         return.preds=T, return.trigger=F,...) {

  if(!inherits(object, "ubar")) stop("Not legitimate rules model")
  
  nr <- nr.rules(object)
      
  if(nr == 0) {
    
    if(return.preds)
      preds <- rep(object@def.rule,NROW(newdata))

    trigger.matrix <- NULL
    
  } else {

    trigger.matrix <- trigger.rules(nr, object@rules.cond, newdata)
                
    if(return.preds) {
                  
      preds <-
        as.vector(apply(trigger.matrix,2,trigger.value,
                        object@rules.yval,                        
                        object@def.rule,object@rules.parms$trig),
                  mode="numeric")
     }
    
  }
  
  if(return.trigger) attr(preds,'trigger.matrix') <- trigger.matrix
  
  names(preds) <- rownames(newdata)
  preds
}

##=========================================================================
## trigger.value
##-------------------------------------------------------------------------
## This function returns the trigger matrix, that is, the set of
## examples covered by each rule of the rule set.
## Coded by: Rita P. Ribeiro
##=========================================================================
trigger.value <- function(trigger.list, rules.yval,
                          def.yval, htrig) {
   
  r.trigg <- which(trigger.list)    
  
  nr.trigg <- length(r.trigg)
  
  if(nr.trigg == 0) {

    def.yval

  } else if(nr.trigg == 1) {

    rules.yval[r.trigg,1]

  } else {

    switch(htrig,
           "avg" = mean(rules.yval[r.trigg,1]),
           "best" =                
           rules.yval[r.trigg[which.max(rules.yval[r.trigg,2])],1],
           "avgw" = {               
             w <- sapply(rules.yval[r.trigg,2],
                         function(v,t) v/t, sum(rules.yval[r.trigg,2]))
             
             sum(w*rules.yval[r.trigg,1])
           })
  }    
}


##=========================================================================
## trigger.rules
##-------------------------------------------------------------------------
## This function returns the trigger matrix, that is, the set of
## examples covered by each rule of the rule set.
## Coded by: Rita P. Ribeiro
##=========================================================================
trigger.rules <- function(nr, rules.cond, data) {

  nex <- NROW(data)
  trigger.matrix <- matrix(F,nr,nex)
  colnames(trigger.matrix) <- rownames(data)

  
  if(nr == 0) return(trigger.matrix)      


  for(i in 1:nr) 
    trigger.matrix[i,] <- with(data,eval(parse(text = rules.cond[i])))
  
  
  trigger.matrix
  
}

##=========================================================================
## specificity
##-------------------------------------------------------------------------
## This function calculates the specificity of a rule, that is, the
## number of examples uniquely covered by that rule.

## If each call to FUN returns a vector of length n, then apply
## returns an array
## If the calls to FUN return vectors of different lengths,
## apply returns a list of length 
## Coded by: Rita P. Ribeiro
##=========================================================================
spec.rules <- function(trigger.matrix) {  


  exs.spec <- (apply(trigger.matrix,2,sum) == 1)

  if(any(exs.spec))
    unique(apply(trigger.matrix[,exs.spec],2,which))
  else
    NULL
}


##=========================================================================
## remove.identical
##-------------------------------------------------------------------------
## This function removes identical rules from a set of rules, that is
## the ones which have the same former conditions and chooses the one
## that is more valuable.
## Coded by: Rita P. Ribeiro
##=========================================================================

remove.identical <- function(nr, rules, trigger, eval) {

  remove.idx <- NULL
  overlap <- vector(mode="logical",length=length(eval))
 
  ## correlation between rules and not between examples
  ttrigger <- t(trigger)
    
  for(r in 1:nr) {

    if(!overlap[r]) {

      identical.rules <- which(apply(ttrigger,2,function(x) identical(x,ttrigger[,r])))
                                  
      if(length(identical.rules) > 1) {
                       
        overlap[identical.rules] <- T
        
        remove.idx <-
          c(remove.idx,
            identical.rules[identical.rules !=
                            identical.rules[which.max(eval[identical.rules])]])

      }
    }
  }
  
  remove.idx
}


##=========================================================================
## eval.rules
##-------------------------------------------------------------------------
## This function evaluates a set of rules according to a set of statistics.
## I can make it a end-user function, i.e., making possible to redesign
## phi, benef and utility from scratch ... 
## Coded by: Rita P. Ribeiro
##=========================================================================
eval.rules <- function(nr, rules.yval,
                       eval.s, y,                        
                       trigger.matrix,
                       phi.parms,
                       loss.parms,
                       util.parms) {
  
  ns <- length(eval.s)
  
  eval.matrix <- matrix(NA,nr,ns)
  colnames(eval.matrix) <- eval.s
      
  for(rule.idx in 1:nr)    
    eval.matrix[rule.idx,1:ns] <- eval.rule(eval.s, y, 
                                            rules.yval[rule.idx,1],
                                            trigger.matrix[rule.idx,],
                                            phi.parms, loss.parms,
                                            util.parms)
  eval.matrix
}

##=========================================================================
## eval.rule
##-------------------------------------------------------------------------
## This function evaluates a given rule according to a specified list of
## statistics.
## The evaluation is done per rule, for efficiency reasons: it is better
## if we can evaluate precision, recall, F-measure, etc... all at one time.
## Coded by: Rita P. Ribeiro
##=========================================================================

eval.rule <- function(eval.s, y, resp.value,
                      trigger.list,
                      phi.parms, loss.parms, util.parms) {
  
  exs <- which(trigger.list)
  
  n <- length(exs)

  ns <- length(eval.s)
  
  if(n == 0) return(rep(0,ns))

  naive.y <- mean(y)
  yexs <- y[exs]
  yexs.pred <- rep(resp.value,n)

  

  s.val <-  vector(mode="list",length=ns)
 

  for(s in eval.s) {
        
    if(s %in% utilMetrics) {

      s.val[[s]] <- util(yexs.pred,yexs,
                         phi.parms = phi.parms,loss.parms = loss.parms,
                         util.parms = util.parms)
    } else {

      if(length(grep(s,"mad",ignore.case=T)))
        naive.y <- median(y)
      else
        naive.y <- mean(y)
      
      ## the value should reflect the quality of the rule
      ## standard regression estimates reflect errors
      s.val[[s]] <- 1 - do.call(paste(tolower(s),"regr",sep="."),
                                list(yexs.pred,yexs,naive.y))
      
    }
  }

  unlist(s.val) 
}

