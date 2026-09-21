
## ======================================================================
##  All the canvas setting for plots
## 
## ======================================================================
canvas.setup <- function(n, col = T, pch = c(21:25,4:6,8:7), colored = T, ...) {
      
  if(col)
    cols <- canvas.cols(n, colored = colored,...)
  else
    cols <- rep(1,n)

  n.pchs <- length(pch)
  if(!n.pchs) pch <- 21
  
  if(n.pchs > 1) 
    pchs <- pch[(0:(n-1) %% n.pchs) + 1]
  else
    pchs <- rep(pch,n)

  lty <- 1:n
  
  list(cols = cols, pchs = pchs, lty = lty)
}


## ======================================================================
##  
## 
## ======================================================================
canvas.leg <- function(leg.text,cols,pchs) {


  plot.new()
  par(mar=c(5,0,2,1)+0.1)
  plot.window(c(0,1), c(0,1))
    
  legend(x=0,y=1,leg.text,
         lwd=0.5,y.intersp=2,
         col=cols,pt.bg=cols,
         lty=rep(1,length(cols)),pt.lwd=0.5,
         pch=pchs,
         merge=T,text.width=0.9,
         bg=c("lightgrey"),
         bty="n",
         cex=0.6)

}


## ======================================================================
##  
## 
## ======================================================================
canvas.cols <- function(n = 64, colored=T, surf=F, ...) {

  if(colored)
    if(surf)
      cols <- color.scheme('orskblu7',n)
    else
      cols <- color.scheme('RdOrBlu',n)
  else
    cols <- color.scheme('mygrayscale',n)
  

}

## RColorBrewer
color.scheme <- function(name, n = 64) {
  switch(name,
         'rdyblu7' = cs <- colorRampPalette(c('#d73027','#fc8d59','#fee090',
           '#ffffbf', '#e0f3f8','#91bfdb','#4575b4'))(n),
         'orskblu7' = cs <- colorRampPalette(c('#FF6600','#FFCC00','#FFFF66',
           '#CCFFFF','#99FFFF','#33FFFF','#0099FF'))(n),
         'RdYlBlu' = cs <- colorRampPalette(c('#d73027', '#CCFFFF','#0099FF'))(n),
         'RdOrBlu' = cs <- colorRampPalette(c('#C11A1A', '#F8B230','#4D2ADE'))(n),
         'mygrayscale' =  cs <- colorRampPalette(c('#CACACA','#8C8C8C','#373737'))(n)
         )
  cs
}

canvas.theta <- function(theta.persp,theta.init=-45) {

  theta <- switch(theta.persp,
                  "x0y0"=theta.init,
                  "x1y0"=theta.init+90,
                  "x1y1"=theta.init+180,                  
                  "x0y1"=theta.init+270,
                  "x01"=0,
                  "y01"=90)
  theta

}
