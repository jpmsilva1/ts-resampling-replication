##=========================================================================
## pn.space.R
## adapted from ROCR  
##-------------------------------------------------------------------------
## Coded by: Rita Ribeiro (based on Fawcett06 and DG06) 
##=========================================================================


## ======================================================================
## Utility-based binormal pn values 
## ----------------------------------------------------------------------
## Coded by : Rita P. Ribeiro 
## ======================================================================
## ----------------------------------------------------------------------
## pn.curve
## ----------------------------------------------------------------------
pn.prediction <- function(ypred, y,
                          phi.parms, loss.parms, util.parms,
                          update.phi = FALSE, pooling=FALSE,                  
                          calibrate = FALSE, ...) {

  exp <- valid.input(ypred,y)
      
  if(pooling) 
    exp <- pool.predictions(exp$ypred,exp$y)

  n <- length(exp$ypred)

  ec <- eval.control(exp$y[[1]], update.phi = update.phi,
                     phi.parms, loss.parms, util.parms, utype="rank",...)

  
  predictions <- list()
  labels <- list()
  cutoffs <- list()
  fp <- list()
  tp <- list()
  fn <- list()
  tn <- list()
  n.pos <- list()
  n.neg <- list()
  n.pos.pred <- list()
  n.neg.pred <- list()

  cat('Number of runs:',n,'\n')

  for(i in 1:n) {
           
    cl <- phi2pn(exp$ypred[[i]], exp$y[[i]],
                 ec$util.parms$event.thr, ec$phi.parms)
    
    predictions <- c(predictions, list(cl$ypred))
    labels <- c(labels, list(cl$y))
    
    ny <- length(cl$ypred)
    
    if(ec$util.parms$binorm.est)
      npts <- ec$util.parms$ipts
    else
      npts <- ny

    if(ec$util.parms$ustep > 0)
      npts <- npts / ec$util.parms$ustep
    
    res <- .C("r2util_pn",
              n = ny,
              y = as.double(exp$y[[i]]),
              ypred = as.double(exp$ypred[[i]]),
              phi.parms = phi2double(ec$phi.parms),
              loss.parms = loss2double(ec$loss.parms),
              util.parms = util2double(ec$util.parms),
              cutoffs = double(npts),
              tp = double(npts),
              fp = double(npts),
              util.pos = double(1),
              util.neg = double(1),
              pn.size=integer(1))[c('cutoffs','tp','fp',
                'util.pos','util.neg','pn.size')]
       
    n.pos <- c(n.pos, res$util.pos)
    n.neg <- c(n.neg, res$util.neg)
    
    
    cutoffs <- c( cutoffs, list( res$cutoffs[1:res$pn.size] ))
    tp <- c( tp, list( res$tp[1:res$pn.size] ))
    fp <- c( fp, list( res$fp[1:res$pn.size] ))
    
    fn <- c( fn, list(format.value(n.pos[[i]] - tp[[i]])))
    tn <- c( tn, list(format.value(n.neg[[i]] - fp[[i]])))
      
    n.pos.pred <- c(n.pos.pred, list(format.value(tp[[i]] + fp[[i]])))
    n.neg.pred <- c(n.neg.pred, list(format.value(tn[[i]] + fn[[i]])))
    
  }
  
  
  return( new("prediction",
              predictions=predictions,
              labels=labels,
              cutoffs=cutoffs,
              fp=fp,
              tp=tp,
              fn=fn,
              tn=tn,
              n.pos=n.pos,
              n.neg=n.neg,
              n.pos.pred=n.pos.pred,
              n.neg.pred=n.neg.pred))

}

## ----------------------------------------------------------------------
## pn.curve
## ----------------------------------------------------------------------
pn.curve <- function(ypred, y,
                     phi.parms, loss.parms, util.parms,
                     update.phi = FALSE, pooling=FALSE, 
                     curve.type="roc",
                     calibrate = FALSE,...) {
  
 
  exp <- valid.input(ypred,y)
      
  if(pooling) 
    exp <- pool.predictions(exp$ypred,exp$y)

  n <- length(exp$ypred)
  
  ec <- eval.control(exp$y[[1]], update.phi = update.phi,
                     phi.parms, loss.parms, util.parms, utype="rank",...)

    
  get.roc <- ifelse(curve.type == "roc",TRUE,FALSE)
  get.hull <- FALSE
  
  x.values <- list()
  y.values <- list()
  alpha.values <- list()

  for(i in 1:n) {
           
    ny <- length(exp$ypred[[i]])

    if(ec$util.parms$ustep > 0)
      npts <- pmax(1000,ny / ec$util.parms$ustep) ## 1000 is to be settable by ipts
    else
      npts <- ny
    
    res <- .C("r2util_curve",
              get.roc = get.roc,
              get.hull = get.hull,
              n = ny,
              y = as.double(exp$y[[i]]),
              ypred = as.double(exp$ypred[[i]]),
              phi.parms = phi2double(ec$phi.parms),
              loss.parms = loss2double(ec$loss.parms),
              util.parms = util2double(ec$util.parms),
              x.values = double(npts),
              y.values = double(npts),
              alpha.values = double(npts),
              xy.size = integer(1))[c('x.values','y.values',
                'alpha.values','xy.size')]
    
    x.values <- c( x.values, list( res$x.values[1:res$xy.size] ))
    y.values <- c( y.values, list( res$y.values[1:res$xy.size] ))
    alpha.values <- c( alpha.values, list( res$alpha.values[1:res$xy.size] ))
    
  }
    
  return(new("performance",
             x.name       = ifelse(curve.type=="roc","False Positive Rate","Recall"),
             y.name       = ifelse(curve.type=="roc","True Positive Rate","Precision"),
             alpha.name   = "cutoffs",
             x.values     = x.values,
             y.values     = y.values,
             alpha.values = alpha.values))
}

## ----------------------------------------------------------------------
## pn.curve
## ----------------------------------------------------------------------
rank.scores <- function(ypred, y,
                        phi.parms, loss.parms, util.parms,
                        update.phi = FALSE,
                        pooling = FALSE, 
                        calibrate = FALSE,...) {
  
  exp <- valid.input(ypred,y)
      
  if(pooling) 
    exp <- pool.predictions(exp$ypred,exp$y)

  n <- length(exp$ypred)
  
  ec <- eval.control(exp$y[[1]], update.phi = update.phi,
                     phi.parms, loss.parms, util.parms, utype="rank",...)

  rank.sc <- calib.sc <- idx.sc <- list()
  cat('Number of runs:',n,'\n')

  for(i in 1:n) {

    res <- util.rank(exp$ypred[[i]],exp$y[[i]],
                     ec$phi.parms,ec$loss.parms,ec$util.parms)
                    
    
    rank.sc <- c( rank.sc, list( res$rank.scores))
    calib.sc <- c( calib.sc, list( res$calib.scores))
    idx.sc <- c( idx.sc, list( res$idx.scores))
    
    
  }
    
  list(rank.scores=rank.sc,calib.scores=calib.sc,idx.scores=idx.sc)
}

##=========================================================================
## Pool all the predictions together.
## 
##-------------------------------------------------------------------------
## Coded by: Rita Ribeiro 
##=========================================================================
pool.predictions <- function(ypred, y) 
  list(ypred=list(unlist(ypred)),y=list(unlist(y)))

##=========================================================================
## phi2pn
## From relevance information to pn predictions
##-------------------------------------------------------------------------
## Coded by: Rita Ribeiro (based on Fawcett06 and DG06) 
##=========================================================================
phi2pn <- function(ypred, y, event.thr, phi.parms) {

  y.phi <- phi(y,phi.parms=phi.parms)
  y.event <- as.factor(ifelse(y.phi >= event.thr, 1, 0))

  ypred.phi <- phi(ypred,phi.parms=phi.parms)

  list(ypred=ypred.phi, y=y.event)
}




valid.input <- function(ypred, y) {


  ## check that ypred and y are lists

  if(is.null(ypred <- checklist(ypred)))
    stop("Format of predictions is invalid")
  
  if(is.null(y <- checklist(y)))
    stop("Format of true values is invalid")
  
  if(length(ypred) != length(y))
    stop(paste("Number of cross-validation runs must be equal",
               "for predictions and true values."))
  if(! all(sapply(ypred, length) == sapply(y, length)))
    stop(paste("Number of predictions in each run must be equal",
               "to the number of true values for each run."))

  
  ## only keep prediction/label pairs that are finite numbers
  for (i in 1:length(ypred)) {    
    finite.bool <- is.finite(ypred[[i]] )
    ypred[[i]] <- ypred[[i]][finite.bool]
    y[[i]] <- y[[i]][finite.bool]
  }

  list(ypred=ypred,y=y)
  
}

checklist <- function(y) {

  if (is.data.frame(y)) {
    names(y) <- c()
    y <- as.list(y)
  } else if (is.matrix(y)) {
    y <- as.list(data.frame( y))
    names(y) <- c()
  } else if (is.vector(y) && !is.list(y)) {
    y <- list(y)
  } else if (is.factor(y) || is.ordered(y) ||
             !is.list(y)) {
    y <- NULL
  }
  y
}

