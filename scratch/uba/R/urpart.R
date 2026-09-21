

##=========================================================================
## urpart
##-------------------------------------------------------------------------
## This function returns the best cv rpart tree according to an utility
## measure.
## At this stage, we do not allow any other method then anova, thus
## method parameter is not an option to the user.
## Coded by: Rita P. Ribeiro
##=========================================================================
urpart <- function(formula, data, weights, subset,
                   cp = 0, minsplit = 3, maxsurrogate = 0,
                   maxdepth=30,
                   phi.parms, loss.parms, util.parms,
                   update.phi = FALSE, verbose = F,...) {
  
   mf <- match.call(expand.dots = FALSE)
   m <- match(c("formula", "data", "weights","subset"), names(mf), 0)
   mf <- mf[c(1, m)]
   mf$drop.unused.levels <- TRUE
   mf[[1]] <- as.name("model.frame")
   mf <- eval.parent(mf)

   y <- model.response(mf)
 
   tree <- rpart(mf, cp=cp, minsplit = minsplit,
                 maxdepth=maxdepth,
                 maxsurrogate = maxsurrogate, ...)
  
   ymat <- xpred.rpart(tree)

   ec <- eval.control(y, update.phi = update.phi,
                      phi.parms, loss.parms, util.parms, ...)

  
   xutil <- apply(ymat,2,function(ypred)
                  util(ypred,tree$y,
                       ec$phi.parms, ec$loss.parms, ec$util.parms, return.uv = FALSE)
                  )
   i <- which.max(xutil)

   ## not sure yet if it is a bug
   my.prune.rpart(tree,cp=tree$cptable[i,1])

}

my.prune.rpart <- function (tree, cp, ...) 
{
    ff <- tree$frame
    id <- as.integer(row.names(ff))
    toss <- id[ff$complexity <= cp & ff$var != "<leaf>"]
    if (length(toss) == 0) 
        return(tree)
    newx <- snip.rpart(tree, toss)
    temp <- pmax(tree$cptable[, 1L], cp)
    
    keep <- match(unique(temp), temp)
    
    newx$cptable <- tree$cptable[keep, , drop = FALSE]

    ###newx$cptable[max(keep), 1L] <- cp

    newx$cptable[as.character(max(keep)), 1L] <- cp
    newx
}



