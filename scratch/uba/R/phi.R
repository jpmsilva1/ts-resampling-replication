#==========================================================================#
# phi.R                                                                    #
#                                                                          #
# PhD                                                                      #
# Rita P. Ribeiro                                                          #

#--------------------------------------------------------------------------#
## Objective: relevance definition
## 
##  
#==========================================================================#

# ======================================================================
# The phi function specifies the regions of interest in the target
# variable. It does so by performing a Monotone Cubic Spline
# Interpolation over a set of maximum and minimum relevance points. 
# The notion of RELEVANCE can be associated with rarity.
# Nonetheless, this notion may depend on the domain experts knowledge.
# ======================================================================

## keep up to date (in R and C)
phiMethods <- c("extremes","range")

# ======================================================================
# new.phi
## phi -> interface with C
# This function escapes the control parameters and does all at once
# on the C side.
# Maybe is redundant
# ======================================================================
phi <- function(y, phi.parms, only.phi=TRUE) {
  
  n <- length(y)

  res <- .C("r2phi",
            n = as.integer(n),
            y = as.double(y),
            phi.parms = phi2double(phi.parms),
            y.phi = double(n),
            yd.phi = double(n),
            ydd.phi = double(n)
            )[c('y.phi','yd.phi','ydd.phi')]

  if(only.phi)
    res$y.phi
  else
    res
}

phi.bumps.info <- function(y, phi.parms, loss.parms) {
  
  n <- length(y)
  
  res <- .C("r2bumps_info",
            n = as.integer(n),
            y = as.double(y),
            phi.parms = phi2double(phi.parms),
            loss.parms = loss2double(loss.parms))

  
}

# ======================================================================
# phi.setup
# This function does the control of loss parameters
# ======================================================================
phi.setup <- function(y, method = phiMethods,
                      extr.type = NULL, control.pts = NULL, ...) {

  method <- match.arg(method, phiMethods)

   
  control.pts <- do.call(paste("phi",method,sep="."),
                         c(list(y=y), extr.type = extr.type,
                           list(control.pts=control.pts),...))
  
  
  list(method = method, 
       npts = control.pts$npts, control.pts = control.pts$control.pts)
}

phi.control <- function(y, phi.parms, ...) {

  call <- match.call()
  
  phiP <- phi.setup(y,...)

  if(!missing(phi.parms)) {
    phiP[names(phi.parms)] <- phi.parms
      
    phiP <- do.call(phi.setup,c(list(y=y),phiP))    
  }
  
  phiP
  
}

phi2double <- function(phi.parms) {
  phi.parms$method <- match(phi.parms$method,phiMethods) - 1

  as.double(unlist(phi.parms))
}

## ======================================================================
## phi.extremes
## EXTREMES: 
## As we know that S1 and S2 have the same sign, the slope
## of the adjacent values will be the weighted harmonic mean between
## the median and the more extreme outlier.
## adjH = min{ y | y >= Q3 + coef * IQR}
## adjL = max{ y | y <= Q1 - coef * IQR}
## For adjL and adjH, coef = 1.5
## but for extreme outliers we can assign coef = 3
## stats =  lower whisker, the first quartile, the median,
##          the third quartile and the extreme of the upper whisker
## ======================================================================
phi.extremes <- function(y, extr.type = c("both","high","low"),
                         coef=1.5,...) {

  extr.type <- match.arg(extr.type)
  
  control.pts <- NULL
  
  extr <- boxplot.stats(y,coef=coef)

  r <- range(y)
  
  if(extr.type %in% c("both","low") &&
     any(extr$out < extr$stats[1])) {

    ## adjL
    control.pts <- rbind(control.pts,c(extr$stats[1],1,0))
  
  } else {

    ## min
    control.pts <- rbind(control.pts,c(r[1],0,0))
  }

  ## median 
  control.pts <- rbind(control.pts,c(extr$stats[3],0,0))
           
  if(extr.type %in% c("both","high") &&
     any(extr$out > extr$stats[5])) {
       
    ## adjH
    control.pts <- rbind(control.pts,c(extr$stats[5],1,0))
  
  } else {

    ## max
    control.pts <- rbind(control.pts,c(r[2],0,0))

  }

  npts <- NROW(control.pts)

 
  list(npts = npts,
       control.pts = as.numeric(t(control.pts)))##,
           
}


## ======================================================================
## phi.range
## ======================================================================
phi.range <- function(y, control.pts, ...) {
  
  ## if it comes from pre-set env
  if(!is.null(names(control.pts))) 
    control.pts <- matrix(control.pts$control.pts,nrow=control.pts$npts,byrow=T)
  
  if(missing(control.pts) || !is.matrix(control.pts) ||
     (NCOL(control.pts) > 3 || NCOL(control.pts) < 2))
    stop('The control.pts must be given as a matrix in the form: \n',
         '< x, y, m > or, alternatively, < x, y >')

  npts <- NROW(control.pts)
  dx <- control.pts[-1L,1L] - control.pts[-npts,1L]
  
  if(any(is.na(dx)) || any(dx == 0))
    stop("'x' must be *strictly* increasing (non - NA)")
  
  if(any(control.pts[,2L] > 1 | control.pts[,2L] < 0))
    stop("phi relevance function maps values only in [0,1]")
  
  control.pts <- control.pts[order(control.pts[,1L]),]
  
  if(NCOL(control.pts) == 2) {
    
    ## based on "monoH.FC" method
    dx <- control.pts[-1L,1L] - control.pts[-npts,1L]
    dy <- control.pts[-1L,2L] - control.pts[-npts,2L]
    Sx <- dy / dx
    
    ## constant extrapolation
    m <- c(0, (Sx[-1L] + Sx[-(npts-1)]) / 2, 0)
    
    control.pts <- cbind(control.pts,m)
    
  }
  
 
  r <- range(y)
  npts <- NROW(matrix(control.pts,ncol=3))

 
  list(npts = npts,
       control.pts = as.numeric(t(control.pts)))##,

}

## ======================================================================
## phi.plot
## Plots the phi function phi: Y -> [0,1] 
## ======================================================================
phi.plot <- function(y,
                     phi.parms, update.phi=TRUE,
                     plot.title = "The Relevance Function",
                     xlab = expression(y),
                     ylab = expression(phi(y)),
                     add.boxplot = F,
                     anonymize = F,
                     pp = T,
                     ...) {

  

  y.plot <- seq(min(y),max(y),len=length(y))

  if(update.phi)
    phi.parms <- phi.control(y,phi.parms, ...)
  
  
  y.phi <- phi(y.plot,phi.parms)
  
  plot.title <- parse(text=paste('phi',"~-~",deparse(plot.title),"~-~",
                        phi.parms$method,sep=""))
  
  
  def.par <- par("mar")
  
  
  if(add.boxplot) {
    
    ## --------- boxplot ------------------
    
    l <- layout(matrix(c(2,1)),c(1,1),c(3.5,1.5))
                                            
    par(mar=c(2,4,1.5,2)+0.1)
    boxplot(y,horizontal=T,xaxt='n',cex.axis=1,cex.lab=1)

    mtext("box-plot",2,line=2,cex=1)
    
    rug(jitter(y),side=3,ticksize=0.1)

  }

  par(mar=c(2,4,4,2)+0.1)
  
  plot(y.plot,y.phi,type="l",
       main=plot.title,              
       xlab="",ylab="",cex.main=1.2,cex.lab=1,cex.axis=1,
       xaxt='n', yaxt='n',...)
      
  control.xx <- phi.parms$control.pts[3*(1:phi.parms$npts-1) + 1]
  control.yy <- phi.parms$control.pts[3*(1:phi.parms$npts-1) + 2]
  
  if(pp) {
    points(control.xx,control.yy,pch=19,cex=0.7)
    abline(v=control.xx,h=control.yy,lty=3,col="lightgrey")
  }
      
  if(!anonymize) {
    
    axis(1,at=control.xx,labels=round(control.xx,2))
    axis(2,at=c(0,control.yy,1),labels=c(0,round(control.yy,2),1))

  } else {

    if(phi.parms$method == "extremes") {
      
      ## to improve -- not efficient
      extrs <- boxplot.stats(y)$stats
      extrs.phi <- phi(extrs,phi.parms)
      
      axis(1,at=extrs,labels=c(expression(adj[L]),expression(Q[1]),expression(tilde(Y)),expression(Q[3]),expression(adj[H])),cex.axis=1)
      axis(2,at=extrs.phi,labels=round(extrs.phi,2))

    } else {

      axis(1,at=control.xx,labels=parse(text=paste("x[",1:phi.parms$npts,"]",sep="")),cex.axis=1)
      axis(2,at=control.yy,labels=parse(text=paste("y[",1:phi.parms$npts,"]",sep="")),cex.axis=1)
      
    }
  }
        
  mtext(ylab,side=2,line=2.5,cex=1)
  mtext(xlab,side=1,line=2,cex=1)
  
  par(def.par)
}


## ======================================================================
## util.isometrics
## Plot the surface Phi: Y x Y -> [0,1]
## add all the parameters related to data or utility parameters
## ======================================================================
phi.isometrics <- function(y, z,
                           phi.parms, update.phi = FALSE,
                           p = 0.5, 
                            ## ----------------------------------
                           add.lines = NULL,
                           gran = 100,
                           ...) {


  if(update.phi)
    phi.parms <- phi.control(y,phi.parms,...)
  
  if(missing(z)) 
    z <- phi3D.values(y, phi.parms, p, gran)
  
  add.lines <- phi.parms$control.pts[3*(1:phi.parms$npts-1) + 1]

  isometrics.plot(z$input,z$input,z$output,
                  zcol.lim=c(-1,1),
                  plot.title = expression(Phi^p~" Relevance Isometrics"),                   
                  add.lines = add.lines, ...)
  
}


## ======================================================================
## phi.surface
## Plots the joint phi surface Phi: YxY -> [0,1] 
## ======================================================================
phi.surface <- function(y, z, phi.parms, update.phi = FALSE,
                        p = 0.5, 
                        gran = 100, ...) {

  if(update.phi)
    phi.parms <- phi.control(y,phi.parms,...)
  
  if(missing(z))
    z <- phi3D.values(y, phi.parms, p, gran)

    
  surface.plot(z$input,z$input,z$output,
               z.lim = c(-1,1), zcol.lim = c(-1,1),
               plot.title = expression(Phi^p~" Relevance Surface"),
               zlab = expression(Phi^p~(hat(Y)~","~Y)), ...)
  
}



## ======================================================================
## util3D.values
## Obtains the utility values to plot in the surface
## ======================================================================
phi3D.values <- function(y, phi.parms, p = 0.5, gran = 100) {
  
  x <- seq(min(y),max(y),len=gran)

  x.phi <- phi(x,phi.parms)
  
  z <- outer(x.phi, x.phi, eval.jphi, p=p)
  
  list(input=x,output=z)
}

eval.jphi <- function(y.phi, ypred.phi, p=0.5) {

  n <- length(y.phi)
  
  res <- .C("r2jphi_eval",
            n = as.integer(n),
            y.phi = as.double(y.phi),
            ypred.phi = as.double(ypred.phi),
            p=as.double(p),
            jphi = double(n))$jphi
  res
}



### ------------------------------------------------------------------




