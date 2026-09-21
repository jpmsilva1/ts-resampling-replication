#==========================================================================#
# Uba plots                                                             #
#                                                                          #
# PhD.                                                                     #
# Rita P. Ribeiro                                                          #
# Last Modified: February 2009                                              #
#--------------------------------------------------------------------------#
## Objective: Plot a 3D function.
#==========================================================================#


# ======================================================================
# plot3D
# This function 
## use independent range for color scale and persp plot
## and at the same time mantaining the z axes equal to the
## color scale
## > plot3D("soma",1:200,1:1000)
## z.lab=substitute(italic(C[m]), list(m = 1)),
# =====================================================================

surface.plot <- function(x,y,z,
                         plot.title = "Function Surface",
                         xlab = expression(hat(Y)),
                         ylab = expression(Y),                         
                         zlab = expression(f~(hat(Y)~","~Y)),
                         z.lim = NULL, zcol.lim = NULL,                  
                         plot.persp = c("x0y0","x1y1","x1y0","x0y1","x01","y01"),
                         colored = T, add.leg = T, add.contour = F) {
  
  plot.persp <- match.arg(plot.persp)
  
  old.par <- par(c("mgp","mar","oma","cex"))
  
  par("mgp"=c(1,2,1))
  
  if(add.leg) 
    par("mar"= c(5,3,2,1)+.1)
  else
    par("mar"= c(2,3,2,1)+.1)
  
  surface.colors <- canvas.cols(colored=T,surf=T)
  
  shade <- NA

  if(plot.persp == "x01") {

    xaxt <- "s"; yaxt <- "n"; ylab <- NULL
    
  } else if(plot.persp == "y01") {

    yaxt <- "s"; xaxt <- "n"; xlab <- NULL

  } else {

    shade <- 0.5; yaxt <- xaxt <- "s"
   
  }
  
  phi <- 30
  ltheta <- canvas.theta(plot.persp) - 135
  lphi <- phi - 15

  ##ltheta <- lphi <- 0
  
  zorig <- z.lim  
  z.lim <- range(z,na.rm=TRUE)
  z.lim <- range(c(zorig,z.lim)+c(-1e-7,+1e-7))
    
  zcol <- drape.color(z,col=surface.colors,zlim = z.lim)$color.index
   
  persp(x,y,z,
        zlim = zorig,
        col = zcol,
        ##col = drapecol(z,col=surface.colors),
        theta = canvas.theta(plot.persp),
        phi = phi,
        shade = shade,
        ltheta = ltheta, lphi = lphi,
        expand = 0.8,
        d = 1.5,
        ticktype = "detailed",
        border = NA,
        las = 3,
        xlab = "", ylab = "", zlab = "",
        cex.axis = 0.55,tck = 0,tcl = 0,
        cex.main = 1.2, main = plot.title) -> res
  
  
  mtext(ylab,1,line=-2.5,adj=0.2,las=1,cex=0.7)
  mtext(xlab,1,line=-2,adj=0.8,las=1,cex=0.7)
  mtext(zlab,2,line=1,adj=0.65,padj=0,las=1,cex=0.7)
  
  
  if(add.leg) {    
    par(oma=c(0,0,0,0),cex.axis=0.75)
    image.plot(zlim=zorig, col=surface.colors,legend.only =TRUE,
               horizontal=TRUE,legend.width=0.5,legend.mar=3)
  }

  
  par(old.par)

  
}

# ======================================================================
## 
# This function 
#
# =====================================================================
isometrics.plot <- function(x,y,z,
                            plot.title = "Function Isometrics",
                            xlab = expression(hat(Y)),
                            ylab = expression(Y),
                            z.lim = NULL, zcol.lim = NULL,
                            add.lines = NULL, axis.lines = NULL,
                            colored = T, add.contour = T) {  

  surface.colors <- canvas.cols(colored=T,surf=T)

  
  if(is.null(z.lim)) z.lim <- range(z,na.rm=TRUE)
  
  zcol.lim <- c(z.lim[1]-1e-7,z.lim[2]+1e-7)
  
  image.plot(x,y,z,
             zlim = zcol.lim,  ## for now!!           
             xaxt = 'n',yaxt = 'n',
             col = surface.colors,
             legend.shrink = 1,
             legend.width = 0.8,
             legend.args = list(text="",side=4, line=1),
             cex.axis = 0.75,
             xlab = xlab,ylab = ylab,main = plot.title,cex.main = 1.2)    
  
   
  if(add.contour)
    contour(x,y,z,lwd=0.6,
            drawlabels=T,labcex=0.7,add=T,cex=0.3)
  
  if(is.null(axis.lines)) {
    axis(1)
    axis(2)
  } else {
    axis(1, at=axis.lines$x$at, labels=axis.lines$x$labels, las=2)
    axis(1, at=axis.lines$y$at, labels=axis.lines$y$labels, las=2)
  }

  if(!is.null(add.lines))
    abline(v=add.lines,h=add.lines,lty=3,lwd=1.1)
  box()
  
}

## ======================================================================
##  Here, the objective  is to plot several models averaged curves
##  Each list entry representing one model's curve.
## (We can also make another plot for showing the average, further on)
## ======================================================================

pn.plot <- function(perf.obj, avg = "none", add = FALSE,
                    plot.title = NULL, col=TRUE, colorize = FALSE,
                    add.leg = FALSE, add.grid =FALSE, perf.curve = FALSE,
                    add.pts = TRUE,
                    h = NULL, v = NULL, leg.loc = "topright",
                    lwd=1.2,ltys=NULL,type=NULL,...) {


  ## for each model, plot one or more curves (according to avg)
  
  n.models <- length(perf.obj)

  if(add.leg) {
    if(is.null(names(perf.obj))) names(perf.obj) <- paste("uba curve",1:n.models,sep="")
  }

  
  cc <- curve.canvas(n.models,
                     plot.title = "",xlab="",ylab="",col=col,
                     add.grid = add.grid, add.leg = add.leg, add.pts = add.pts,
                     leg.loc = leg.loc,leg.txt = names(perf.obj),ltys=ltys,...)
  
  
    ## if(is.null(ltys))
    ##   ltys <- 1:n.models
    ## else
    ##   ltys <- rep(ltys,n.models)
  
  for(i in 1:n.models) {
    
    .plot.performance(perf.obj[[i]], avg = avg,add = TRUE, 
                      xlim=c(0,1),ylim=c(0,1),colorize = colorize,
                      col=cc$cols[i],                    
                      lwd=lwd,
                      lty=cc$ltys[i],...)

    if(add.pts) {
      xs <- seq(0,1,by=0.1)
      idx <- sapply(xs,function(v) which.min(abs(perf.obj[[i]]@x.values[[1]] - v)))
      points(xs,perf.obj[[i]]@y.values[[1]][idx],col = cc$cols[i],pch=cc$pchs[i],cex=0.7)
      
    }      
  }


  if(!is.null(v) || !is.null(h)) abline(v=v,h=h,col="grey")
  
  mtext(perf.obj[[1]]@x.name,line=2,side=1)
  mtext(perf.obj[[1]]@y.name,line=2,side=2)

  if(is.null(plot.title))  plot.title <- paste(ifelse(perf.obj[[1]]@x.name == "Recall","PR-U","ROC-U")," graph")
  mtext(plot.title,line=2,side=3,cex=1.2)
 

}

## ======================================================================
##  
## 
## ======================================================================
curve.canvas <- function(n.curves,
                         plot.title, xlab, ylab,
                         col = TRUE, idx = NULL,
                         add.leg = TRUE, add.grid = TRUE,add.pts=TRUE,
                         leg.loc = "topright", leg.txt, ltys = NULL,...) {

  ### patch to solve later
  if(!col) ## grayscale
    cc <- canvas.setup(n.curves, col = TRUE,  colored = FALSE, ...)
  else
    cc <- canvas.setup(n.curves, col = TRUE, colored = TRUE, ...)
    
  ##cc <- canvas.setup(n.curves, col = TRUE, idx = idx, colored = TRUE, ...)
  
  if(is.null(ltys))
    cc$ltys <- 1:length(leg.txt)
  else
    cc$ltys <- rep(ltys,length(leg.txt))

  if(add.pts)
    cc$pchs <- (1:length(leg.txt))-1
  else
    cc$pchs <- NA
    
  ## NOT WORKING PROPERLY
  if(add.leg && leg.loc == "out")  {
    df.l <- layout(matrix(c(2,1),ncol=2),width=c(3,1))
    par(mar=c(0,0,0,0)+.5)      
    plot.new()

    
    legend(x=0.1,y=1,legend=leg.txt,
           lwd=0.5,
           col=cc$cols,pt.bg=cc$cols,           
           pch=cc$pchs,
           lty=cc$ltys,pt.lwd=1.2,
           merge=T,text.width=0.9,cex=0.9)         
  }

  
  
  def.par <- par(no.readonly = TRUE,
                 mar=c(5,4,3,0.5)+.1
                 ) # save default, for resetting...
  on.exit(par(def.par))  
    
  plot(c(0, 1), c(0, 1),
       xlim=c(0,1),ylim=c(0,1),
       type = "n", 
       main = plot.title,
       xlab = xlab,ylab = ylab)

  if(add.grid)
    abline(h = seq(0, 1, by = 0.1), v = seq(0, 1, by = 0.1), 
           lty = 3, lwd = 0.5, col = "grey")
  
 
  
  if(add.leg && leg.loc != "out") {
    switch(leg.loc,
           "topright" = {x <- 0.8; y <- 1},
           "bottomright" = {x <- 0.8; y <- 0.2}          
           )    
    
    legend(x=x,y=y,legend=leg.txt,
           lwd=0.8,
           col=cc$cols,pt.bg=cc$cols,
           lty=cc$ltys,pt.lwd=0.5,
           pch=cc$pchs,
           merge=T,text.width=0.9,
           bg=c("lightgrey"),
           bty="n",
           cex=0.8)
  } 

  
  cc
}

## ======================================================================
##  
## 
## ======================================================================
## It THESE thrs that are suppost to be plotted
curve.thrs <- function(curve.pts, thrs = NULL, n.thrs = 10,
                       show.thrs = F, pos = 4, offset = 2,
                       cc.pch, cc.col) {

  if(is.null(thrs))
    thrs <- format.value(seq(0,1,by=1/n.thrs))

  ## DON'T LIKE THIS ...
  p.idx <-
    unlist(sapply(thrs,
                  function(xv,xx)
                  which(format.value(abs(xx - xv)) < 1e-7)[1],
                  unlist(curve.pts[,1])))
  
  points(curve.pts[p.idx,1], curve.pts[p.idx,2],               
         type="p", pch=cc.pch,col="black",lwd=0.5,bg=cc.col,
         cex=0.75)

  if(show.thrs)
    text(curve.pts[p.idx,1],curve.pts[p.idx,2],thrs,
         pos = pos, offset = offset,cex=0.6)

}

