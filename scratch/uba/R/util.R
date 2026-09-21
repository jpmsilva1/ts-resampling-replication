#==========================================================================#
# Utility.R                                                                #
#                                                                          #
# PhD.                                                                     #
# Rita P. Ribeiro                                                          #
# Last Modified: August 2008                                               #
#--------------------------------------------------------------------------#
#==========================================================================#

## keep up to date (in R and C)

### temporary solution

## -----------------------------------------------------------------------
## R side
baseMetrics <- "MU"

normMetrics <- "NMU"

empMetrics <- c("AUCROC.emp","AUCPR.emp","H.emp")

rankMetrics <- c("AUCROC","AUCPR","MAP11","BFM",
                 "P","R","Fm",
                 empMetrics)

utilMetrics <- c(baseMetrics, normMetrics, rankMetrics)

utilTypes <- vector()
utilTypes <- c(rep("base",length(baseMetrics)),
               rep("norm",length(normMetrics)),
               rep("rank",length(rankMetrics)))
names(utilTypes) <- utilMetrics

## -----------------------------------------------------------------------

## # ======================================================================
## ## Utility
## ## This function estimates the utility, given by some pre-established
## ## utility metric, on a set of predictions. Optionally, it can return
## ## the utility values associated to each prediction.
## # ======================================================================


# ======================================================================
# Utility
## redundant?
# This function calculates the utility associated to each prediction.
# ======================================================================

util <- function(ypred, y,
                 phi.parms, loss.parms, util.parms,
                 return.uv = FALSE) {

  util.parms <- util.control(util.parms)

  n <- length(ypred)
    
  res <- .C("r2util",
            n = n,
            y = as.double(y),
            ypred = as.double(ypred),
            phi.parms = phi2double(phi.parms),
            loss.parms = loss2double(loss.parms),
            util.parms = util2double(util.parms),
            return.uv = as.integer(return.uv),
            u = double(n))$u
  
  if(!return.uv)
    res[1]
  else
    res
}


### temporary
util.rank <-  function(ypred, y,
                       phi.parms, loss.parms, util.parms) {

  util.parms <- util.control(util.parms)

  n <- length(ypred)
  
  res <- .C("r2util_rankscores",
            n = n,
            y = as.double(y),
            ypred = as.double(ypred),
            phi.parms = phi2double(phi.parms),
            loss.parms = loss2double(loss.parms),
            util.parms = util2double(util.parms),
            rank.scores = double(n),
            calib.scores = double(n),
            idx.scores = integer(n))[c('rank.scores','calib.scores','idx.scores')]

    
 
  res
}



# ======================================================================
# util.setup
# This function does the control of bump parameters
## this setup has to be improved ....
# ======================================================================

## ustep = 0.01
util.setup <- function(umetric = NULL, utype = NULL, use.util = TRUE,
                       p = .5, Bmax = 1,                        
                       event.thr = 1, score.thr = 0.5,
                       binorm.est = FALSE, ipts = 1000,
                       beta = 1, min.tpr = 0, max.fpr = 1,
                       maxprec = FALSE, calibrate = FALSE,
                       ustep = 0.01, ...) {

  if(!is.null(umetric) ||
     (is.null(umetric) && is.null(utype))) {
    umetric <- match.arg(umetric,utilMetrics)
    utype <- as.character(utilTypes[umetric])
  } else if(!is.null(utype)) {    
    utype <- match.arg(utype,unique(as.character(utilTypes)))
    umetric <- names(utilTypes[utilTypes == utype])[1]
  }



  if(p < 0 || p > 1) {
    warning("The value of 'p' supplied was out of range." ,
         "The default value is used instead.")
    p <- .5
  }

  if(utype == "rank" && ipts < 10) {
    warning("'ipts' is very small and it may result in highly inaccurate estimates.")
  }

  if(utype == "rank" && event.thr == 0) {
    warning("The value of 'event.thr' should be > 0, to enable the ranking",
            "A value 0.01 is assigned.")
    event.thr <- .01
  }

  if(event.thr < 0 || event.thr > 1) {
    warning("The value of 'event.thr' supplied was out of range." ,
            "The default value is used instead.")
    event.thr <- 1
  }

  
  if(score.thr < 0 || score.thr > 1) {
    warning("The value of 'score.thr' supplied was out of range." ,
            "The default value is used instead.")
    score.thr <- .5
  }

  if(min.tpr < 0 || min.tpr > 1) {
    warning("The value of 'min.tpr' supplied was out of range." ,
            "The default value is used instead.")
    min.tpr <- 0
  }

  if(max.fpr < 0 || max.fpr > 1) {
    warning("The value of 'max.fpr' supplied was out of range." ,
            "The default value is used instead.")
    max.fpr <- 1
  }


  if(ustep < 0 || ustep > 1) {
    warning("The value of 'ustep' supplied was out of range." ,
            "The default value is used instead.")
    ustep <- 0
  }
  
  list(umetric = umetric, utype=utype, use.util = use.util,        
       p = p, Bmax = Bmax, 
       event.thr = event.thr, score.thr = score.thr,
       binorm.est = binorm.est, ipts = ipts,
       beta = beta, min.tpr = min.tpr, max.fpr = max.fpr,
       maxprec = maxprec, calibrate = calibrate, ustep = ustep)
}


util.control <- function(util.parms,...) {

  call <- match.call()

  utilP <- util.setup(...)

  if(!missing(util.parms)) {
    utilP[names(util.parms)] <- util.parms

    utilP <- do.call(util.setup,utilP)
  }
  
  utilP
  
}

util2double <- function(util.parms) {
  
  util.parms$umetric <- match(util.parms$umetric,utilMetrics) - 1
  
  ## to improve
  util.parms$utype <- match(util.parms$utype,
                            unique(as.character(utilTypes))) - 1

  as.double(unlist(util.parms))
}

## ======================================================================
## util.isometrics
## Plot the surface U: Y x Y -> [-Bmax,Bmax]
## add all the parameters related to data or utility parameters
## ======================================================================
util.isometrics <- function(y, z,
                            phi.parms, loss.parms, util.parms, update.phi =FALSE,
                            ## ----------------------------------
                            add.lines = FALSE, plot.title = NULL, z.lim = NULL,                            
                            ...) {
  
  if(update.phi) { ## is.missing one of the parms
    phi.parms <- phi.control(y,phi.parms,...)
    loss.parms <- loss.control(y,loss.parms,...)
  }
  
  util.parms <- util.control(util.parms,...)
  
  if(missing(z)) 
    z <- util3D.values(y,phi.parms,loss.parms,util.parms)

  if(is.null(z.lim))
    z.lim <- c(-util.parms$Bmax,util.parms$Bmax)

  if(add.lines)
    add.lines <- phi.parms$control.pts[3*(1:phi.parms$npts-1) + 1]
  else
    add.lines <- NULL
  
  if(is.null(plot.title))
    plot.title <- substitute(U[phi]^p~" Utility Isometrics",list(p=util.parms$p))
  
  isometrics.plot(z$input,z$input,z$output,
                  z.lim = z.lim,
                  plot.title = plot.title,
                  add.lines = add.lines, ...)
  
}

## ======================================================================
## util.surface
## Plot the surface U: Y x Y -> [-Bmax,Bmax]
## ======================================================================
util.surface <- function(y, z,
                         phi.parms, loss.parms, util.parms, update.phi =FALSE,
                         plot.title = NULL, z.lim = NULL,
                         gran = 100, ...) {

  if(update.phi)  {## is.missing one of the parms
    phi.parms <- phi.control(y, phi.parms,...) 
    loss.parms <- loss.control(y, loss.parms,...)
  }
  
  util.parms <- util.control(util.parms,...)

  
  if(missing(z)) 
    z <- util3D.values(y,phi.parms,loss.parms,util.parms,gran)

  if(is.null(z.lim))
    z.lim <- c(-util.parms$Bmax,util.parms$Bmax)
  
  if(is.null(plot.title))
    plot.title <- substitute(U[phi]^p~" Utility Surface",list(p=util.parms$p))
  
  surface.plot(z$input,z$input,z$output,
               z.lim = z.lim,
               plot.title = plot.title,
               zlab = expression(U[phi]^p~(hat(Y)~","~Y)), ...)  
}

## ======================================================================
## util3D.values
## Obtains the utility values to plot in the surface
## ======================================================================
util3D.values <- function(y, phi.parms, loss.parms, util.parms,
                          gran = 500) {
 
  x <- seq(min(y),max(y),len=gran)
     
  z <- outer(x, x, util,
             phi.parms, loss.parms, util.parms, return.uv=TRUE)

  list(input=x,output=z)
}



## ---------------------------------------------------------------

