## ======================================================================
## CDdiag.R
## 
## Performance analysis through Critical Difference diagrams.
##
## List of functions: cdDiag.test, cdDiag.input, 
## Rita P. Ribeiro
## ======================================================================


# --------------------------------------------------------------
# class def

setClass("cdTest",
         representation(data.name="character", ## input info 
                        evalStat.name="character",
                        model.names="character",
                        stat.max="logical",
                        target.p.value = "numeric",                        
                        statistic="numeric", ## friedman test info
                        parameter="numeric",
                        p.value="numeric",
                        method="character",
                        posthoc.test="character", ## post-hoc test info
                        ##crit.dist = crit.dist,
                        ranks="matrix",
                        cd.pvalues="data.frame",
                        avg.ranks="numeric"
                        ))         

# --------------------------------------------------------------
# constructor function
cdTest <- function (dn, sn, msn, sm, tpv, ## input info
                    s, p, pv, m, ## friedman test info
                    pht, rs, ps, avg.rs ## post-hoc test info
                    ) 
{
  o <- new("cdTest")

  o@data.name <- dn
  o@evalStat.name <- sn
  o@model.names <- msn
  o@stat.max <- sm
  o@target.p.value <- tpv

  o@statistic <- s
  o@parameter <- p
  o@p.value <- pv
  o@method <- m
  
  o@posthoc.test <- pht
  o@ranks <- rs
  o@cd.pvalues <- ps
  o@avg.ranks <- avg.rs

  o
}


##=========================================================================
## CDdiag.test
##-------------------------------------------------------------------------
## This function does a statistical comparison of a set of prediction
## methods over multiple datasets.
## It takes the result of the evaluation of an experiment and ranks
## the models based on the results obtained by a specific statistic over all the
## datasets and runs/folds.
##
## INPUT
##
##          <m1>  <m2>      ...    <mi> 
## <ds1>      --    --      ...      --
## <ds2>      --    --      ...      --
## ...
## <dsn>      --    --      ...      --
##
## OUTPUT:
## The result of the Friedman /Posthoc Nemenyi Test
## Coded by: Rita P. Ribeiro
## -----------------------------------------------------------------------
## Example
## x <- matrix(abs(rnorm(20)),5,4)
## rownames(x) <- paste("ds",1:NROW(x),sep="")
## colnames(x) <- paste("model",1:NCOL(x),sep="")
## x[,2] <- 2*x[,2] ## introduce a "significant" difference

## cdt <- cd.test(x)
## cdt <- cd.test(x,stat.max=T)
##=========================================================================

cd.test <- function(x, stat.max = F,
                    p.value = 0.05,                        
                    plot=F,
                    model.names = NULL,
                    stat.name = NULL, ## only for plot convenience
                        ...) {

  ft <- friedman.test(x)

  print(ft)

  
  
  if(is.null(model.names))
    model.names <- colnames(x)
  
  
  ## Nemenyi test
  nyt <- cd.posthoc(x, stat.max, 
                        p.value,
                        model.names=model.names,
                        stat.name=stat.name
                        )
   
  cdt <- cdTest(ft$data.name, ## input info 
                as.character(stat.name), ## as it might be NULL
                model.names,
                stat.max,
                p.value,                
                ft$statistic, ## friedman test info                
                ft$parameter,
                ft$p.value,
                ft$method,
                ##crit.dist,                
                nyt$posthoc.test, ## post-hoc test info
                nyt$ranks,
                nyt$cd.pvalues,
                nyt$avg.ranks
      )

  if(plot) plot(cdt, ...)       

 
  cdt
}



##=========================================================================
## PostHocTest
##-------------------------------------------------------------------------
## This function perfoms a post-hoc test. Only Nemenyi test is available.
### Nemenyi-Damico-Wolfe-Dunn test (joint ranking)
### Hollander & Wolfe (1999), page 244 
### (where Steel-Dwass results are given)
## INPUT
##
##          <m1>  <m2>      ...    <mi> 
## <ds1>      --    --      ...      --
## <ds2>      --    --      ...      --
## ...
## <dsn>      --    --      ...      --
##
## Coded by: Rita P. Ribeiro
##=========================================================================
cd.posthoc <- function(x, stat.max = F,
                       p.value=0.05, test.name="Posthoc Bonferroni Test",
                       model.names = NULL,
                       stat.name = NULL
                       ) {  
  n.models <- NCOL(x)
  n.ds <- NROW(x)
     
  new.x <- data.frame(stat=as.numeric(t(x)),model=factor(rep(colnames(x),n.ds)),ds=factor(gl(n.ds,n.models)))
  
  ## -x because we want to rank first the maximum utility
  if(stat.max)
    new.x$stat <- -new.x$stat
  
  out.FR <- uba:::my.FR.multi.comp(Y=new.x$stat,X=new.x$model,blocks=new.x$ds,nblocks=n.ds,conf=1-p.value)

  rownames(out.FR$ranks) <- rownames(x)


  
  list(posthoc.test = test.name,       
       ##crit.dist = crit.dist,
       ranks = out.FR$ranks,
       cd.pvalues = out.FR$cd.pvalues,
       avg.ranks = out.FR$avg.ranks)


}


##=========================================================================
## subset
##-------------------------------------------------------------------------
## This method selects a set of models from the cd test object.
##
## Coded by: Rita P. Ribeiro
## -----------------------------------------------------------------------
## Example
## > subset(x, models=c("model2","model4"))
##=========================================================================
setMethod("subset",
           signature(x='cdTest'),
          function(x, models="all") {
  
  
  if(models[1] != "all") {

  
    former.models <- x@model.names
    models.idx <- selModelsIdx(models,former.models)
    
    ms <- x@model.names[models.idx]
  
    x@avg.ranks <- x@avg.ranks[ms]  

    pairwise.ms <- sapply(rownames(x@cd.pvalues),strsplit,"-")
    
    selmodel <- sapply(pairwise.ms,function(y) {     
      y <- unlist(strsplit(y,"-")) %-~% "\\s"
      t <- all(y %in% models)
    })
    
    x@cd.pvalues <- subset(x@cd.pvalues,selmodel)      
    x@ranks <- x@ranks[,ms]

    x@model.names <- names(x@avg.ranks)
  }
   
  x
}
)
##=========================================================================
## CDdiag.plot
##-------------------------------------------------------------------------
## This method plots the Critical Difference diagram for a cd test.
## 
## Coded by: Rita P. Ribeiro
## > x <- cd.test(res)
## > plot(x)
## > plot(subset(x, models=c("model2","model4")))
##=========================================================================
setMethod("plot",
           signature(x="cdTest",y='missing'),
          function(x,y,
                   models.caption=NULL,
                   add.leg = TRUE, colored = TRUE,
                   models.pchs = NULL,
                   plot.xaxis = TRUE, x.las = 2, x.cex = 0.8,
                   codification.char='-',
                   fn = NULL, ...) {
  
  init.n <- length(x@avg.ranks) 

  if(!init.n) {
    print(x)
    cat('Nothing to plot.\n')    
  } else {
   
    if(is.null(models.caption))
      models.caption <- x@model.names
    

    
    n.models <- length(x@avg.ranks)
    r.idx <- sort(x@avg.ranks,decreasing=T,index.return=T)$ix

    x@avg.ranks <- x@avg.ranks[r.idx]
    modnames <- models.caption[r.idx]
    
   
    cc <- canvas.setup(n.models,col=colored, pchs=models.pchs)

    cc$cols <- cc$cols[r.idx]
    cc$pchs <- cc$pchs[r.idx]
        
    marked <- rep(F,n.models)
    
    if(add.leg)
      layout(matrix(c(1,2)),height=c(3,7),width=c(1,1,1))
    else 
      layout(matrix(c(1,2)),height=c(2,8),width=c(1,1,1))
    
    op <- par(mar=c(1,1,3,1)+0.1)
    plot.new()
    title("Critical Difference Diagram")

    mtext(paste(x@evalStat.name," Pairwise Comparisons at ",(1-x@target.p.value)*100,
                "% conf. level"," (- - dashed line means a significant difference)",sep=""),
          cex=x.cex*1.1,side=3,line=0)
       
    if(add.leg) {      
      legend("center",             
             legend=modnames,
             pch=cc$pchs,col=cc$cols,pt.bg=cc$cols,
             cex=x.cex,ncol=round(n.models / 3))
    }

    n.pv <- NROW(x@cd.pvalues)
    
    if(plot.xaxis)
      par(mar=c(5,1,1,1)+0.1)
    else
      par(mar=c(1,1,1,1)+0.1)
    
    xx <- range(x@avg.ranks); 
    xx <- rev(range(c(floor(xx),ceiling(xx))))
    
    plot(1,type='n',xaxt='n',yaxt='n',ylim=c(n.pv,0),
         xlim=xx, ylab="",xlab="",xpd=TRUE)
 
    axis(3,at=seq(round(xx[1]),round(xx[2])),cex=0.6)
    abline(v=x@avg.ranks,col=cc$cols,lwd=1,lty=1)

    if(plot.xaxis)
      axis(1,at=unique(x@avg.ranks),cex.axis=x.cex,
           cdTest.xaxis(modnames,x@avg.ranks),las=x.las)
           
    
    l <- n.pv
    for(i in 1:(n.models-1))
      for(j in (i+1):n.models) {

        pv <- which(row.names(x@cd.pvalues) %in% 
                    c(paste(names(x@avg.ranks)[i],"-",names(x@avg.ranks)[j],sep=""),
                      paste(names(x@avg.ranks)[j],"-",names(x@avg.ranks)[i],sep="")))      
        
        avail.line <- vector(mode="numeric",length=n.models)
        
        if(x@avg.ranks[i] != x@avg.ranks[j]) {
          
          marked[c(i,j)] <- T

          if(length(grep("Reject",as.character(x@cd.pvalues[pv,4]))))
             segments(x@avg.ranks[i],l,x@avg.ranks[j],l,lty=3,lwd=1,col=gray(0.7))             
          else     
             segments(x@avg.ranks[i],l,x@avg.ranks[j],l,lwd=0.75)
          
          points(x@avg.ranks[c(i,j)],rep(l,2),
                 col=cc$cols[c(i,j)],pch=cc$pchs[c(i,j)],cex=0.85,bg=cc$cols[c(i,j)])

          
        } else {
          
          if(!avail.line[i]) avail.line[i] <- l
          
        }
        l <- l - 1
      }
    
    
    if(!all(marked)) {
      
      models.idx <- which(!marked)
      
      points(x@avg.ranks[models.idx],avail.line[models.idx],
             col=cc$cols[models.idx],bg=cc$cols[models.idx],
             pch=cc$pchs[models.idx],cex=0.85)
    }    
      
    par(op)
  } 
}
)

##=========================================================================
## show method
##-------------------------------------------------------------------------
## This method shows the information regarding Critical Difference Test.
##
##
## Coded by: Rita P. Ribeiro
##=========================================================================
setMethod("show","cdTest",
          function(object) { 
                                    
            ## initial setup
            spaces <- 2
            digits=getOption("digits") - 2
            indent <- paste(rep(" ", spaces * 2), collapse = "")
                        
            cat("\n")
            cat(indent,"Critical Difference Test with p-value = ",
                object@target.p.value," stat.max:",object@stat.max,"\n\n")
            
            cat(indent,object@method,"\n\n")
            
            cat(names(object@statistic),": ",object@statistic," ")
            cat(names(object@parameter),": ",object@parameter," ")
            cat("p-value: ",object@p.value,"\n\n")
            
            if(is.na(object@p.value) ||
               object@p.value >= object@target.p.value) {
              cat("No significant differences among the models",
                  " were reported at ",(1 - object@target.p.value)*100, "% conf. level.\n\n")
            } else {
              
              cat(indent,"PostHoc Nemenyi Test \n\n")
              n.models <- length(object@avg.ranks)
              r.idx <- sort(object@avg.ranks,decreasing=T,index.return=T)$ix
              object@avg.ranks <- object@avg.ranks[r.idx]
              
              model.names <- names(object@avg.ranks)
              
              cat(paste("\nCritical Difference Values obtained by ",object@posthoc.test,
                        ".\n",sep=""))
              ##colnames(object@cd.pvalues) <- "p-value"
              Hmisc:::print.char.matrix(object@cd.pvalues,col.names=T)   
              
              cat("\nRanking of the models over the set of datasets.\n")
              rank.df <- rbind(object@ranks[,r.idx],round(object@avg.ranks,2))
              colnames(rank.df) <- model.names
              Hmisc:::print.char.matrix(rank.df,col.names=T)    
              
              
              cat("\n\nSignificant differences obtained at",(1-object@target.p.value)*100,
                  "% conf.level.\n")
              ##cat("\n\nStatistic:",object@evalStat.name,"\n")
              cat("---------------------------------------------------------\n")

              all.pv <- as.numeric(levels(cdt@cd.pvalues[,5])[as.numeric(cdt@cd.pvalues[,5])])

              
              for(i in 1:(n.models-1))
                for(j in (i+1):n.models) {
                  
                  pv <- which(row.names(object@cd.pvalues) %in% 
                              c(paste(names(object@avg.ranks)[i],"-",names(object@avg.ranks)[j],sep=""),
                                paste(names(object@avg.ranks)[j],"-",names(object@avg.ranks)[i],sep="")))
                                  
                  if(object@avg.ranks[i] != object@avg.ranks[j]) {
                    
                    if(all.pv[pv] < object@target.p.value) { ## significant (95% conf. level)        
                      cat('-',model.names[i],'is significantly',
                          ifelse(object@avg.ranks[i] <  object@avg.ranks[j],"better","worse"),'than',
                          model.names[j],'\n')
                    }
                  }
                }
            }
            cat('\n')
          }
          )




## adapted from asbio
#####
## removed "bug" in asbio, which uses levels for the average ranks naming
#####    
##mr <- as.matrix(mean.ranks)
##row.names(mr) <- levels(X)
##res$mean.rank.in.blocks <- mr

my.FR.multi.comp <- function (Y, X, blocks, nblocks, conf = 0.95) 
{
    block.ranks <- matrix(ncol = nlevels(X), nrow = nblocks)
    for (i in 1:nblocks) {
        block.ranks[i, ] <- rank(Y[blocks == i])
    }

    
    mod.names <- unique(X) ## the first nblocks to appear
    colnames(block.ranks) <- mod.names
    
    mean.ranks <- apply(block.ranks, 2, mean)


    
    R1 <- tapply(Y, X, length)
    r <- length(R1)
    dif.mat <- outer(mean.ranks, mean.ranks, "-")
    diffs <- dif.mat[upper.tri(dif.mat)]
    SE.diff <- sqrt((r * (r + 1))/(6 * nblocks))
    B <- qnorm(1 - ((1 - conf)/(2 * (r^2 - r)/2)))
    p.val <- 2 * pnorm(abs(diffs)/SE.diff, lower = FALSE)
    p.adj <- ifelse(p.val * ((r^2 - r)/2) >= 1, 1, round(p.val * 
        ((r^2 - r)/2), 6))

    
    
    hwidths <- B * SE.diff
    val <- round(cbind(diffs, diffs - hwidths, diffs + hwidths), 
        5)
    Decision <- ifelse((val[, 2] > 0 & val[, 3] > 0) | val[, 
        2] < 0 & val[, 3] < 0, "Reject H0", "FTR H0")


    res <- list()

    names(mean.ranks) <- mod.names

    res$avg.ranks <- mean.ranks
    
    res$ranks <- block.ranks

    val <- as.data.frame(cbind(val, Decision, p.adj))
    lvl <- outer(mod.names, mod.names, function(x1, x2) {
        paste(x1,x2, sep = "-")
    })
    dimnames(val) <- list(lvl[upper.tri(lvl)], c("Diff", "Lower", 
        "Upper", "Decision", "Adj. p-value"))
    res$cd.pvalues <- val

    res
}



##=========================================================================
## my.rank
##-------------------------------------------------------------------------
## This function ranks the a set of values in a decreasing way,
## handling eventual ties. It doesn't handles NAs.
## Coded by: Rita Ribeiro
##=========================================================================
my.rank <- function(x,decr=T) {

  n <- length(x)
  
  r.idx <- 1:n
  s <- sort(x,decreasing=decr,index.return=T)$ix

  r.idx[s] <- 1:n

  for(i in 1:(n-1)) {
    ties <-  which(!is.na(match(x[(i+1):n],x[i]))) + i
    r.idx[c(i,ties)] <- mean(r.idx[c(i,ties)])
  }

  r.idx
}




##=========================================================================
## CDdiag.xaxis
##-------------------------------------------------------------------------
## This function arranges the model names so as they can be plotted in
## the x-axis of the CD diagram.
##=========================================================================
cdTest.xaxis <- function(model.names,avg.ranks) {
                                        

  n <- length(model.names)
  x.legend <- NULL
  flagged <- rep(F,n)
    
  for(m in 1:n) {

    if(!flagged[m]) {

      flagged[m] <- T
      same.avg <- which(avg.ranks == avg.ranks[m])
      same.avg <- same.avg[same.avg != m]
      
      if(length(same.avg) > 0) {
        
        x.legend <-
          c(x.legend,
            paste(model.names[c(m,same.avg)][sort(avg.ranks[c(m,same.avg)],
                                                  decreasing=T,
                                                  index.return=T)$ix],
                  collapse="/\n"))
        flagged[same.avg] <- T
        
      } else {
        x.legend <- c(x.legend,model.names[m])
      }
    }
  }
  x.legend
  
}




##=========================================================================
## cdDiag.input
##-------------------------------------------------------------------------
## This function summarizes, by a given stat, the results obtained
## by each model over all the runs for each pid included in the experiment.
## Optionally, a sub set of models can be selected.
## This structure is useful for CD.diagram.
##
## INPUT:
##
## $<pid1> problem id
##   [[1]]
##            <stat1>     ...  <stati>   
## <model1>        _      ...      _    
## <model2>        _      ...      _    
## ...
## <modelk>        _      ...      _    
##
##   [[n]]
##            <stat1>     ...  <stati>   
## <model1>        _      ...      _    
## <model2>        _      ...      _    
## ...
## <modelk>        _      ...      _
##
## $<pid2> ...
##
## OUTPUT: 
## 
##         <model1>  <model2>      ...    <modeli> 
## <pid1>        --        --      ...          --
## <pid2>        --        --      ...          --
##
## Coded by: Rita P. Ribeiro
##=========================================================================
cdDiag.input <- function(x, models = "all",                          
                         stat = "NMU", base.comp = "mean",
                         codification.char) {
  
  all.stats <- colnames(x[[1]][[1]])

### TEMPORARY ascii code 175
  all.stats <- strsplit(all.stats,codification.char)[[1]][1]
  
  if(!stat %in% all.stats)
    stop("\nNo information about ",stat,".\n",
         "(",paste(all.stats,collapse=","),")\n")
  
  
  former.models <- rownames(x[[1]][[1]])        
  models.idx <- selModelsIdx(models,former.models)    
  models <- former.models[models.idx]

  n.models <- length(models)
  
  ds.names <- names(x)
  n.ds <- length(ds.names)
  
  all.x <- matrix(0,nrow=n.ds,ncol=length(models))
  rownames(all.x) <- ds.names
  colnames(all.x) <- models

  
  if(stat == "nrules")
    stat.idx <- 1
  else
    stat.idx <- grep(stat,colnames(x[[1]][[1]]))##[1]
   
  for(ds in ds.names) 
    for(m in models)       
      all.x[ds,m] <-
        do.call(base.comp,list(sapply(x[[ds]],
                                      function(run) run[m,stat.idx],
                                      simplify=T)))
      
  list(exp = all.x, model.names = colnames(all.x), stat.name = stat)
   
}
  
##=========================================================================
## selModelsIdx
##-------------------------------------------------------------------------
## Coded by: Rita Ribeiro 
##=========================================================================
selModelsIdx <- function(sel.models="all",all.models) {
  
  models.idx <- 1:length(all.models)

  print(sel.models)
  print(all.models)
  
  if(sel.models[1] != "all") 
    models.idx <- matchRegExpr(sel.models,all.models)
  
  if(!length(models.idx))
    stop("No valid models were selected.")
  
  models.idx  
}

## ======================================================================
## matchRegExpr
## This function
## ======================================================================
matchRegExpr <- function(m.expr,m.vector) {

  pos <- rep(F,length(m.vector))
  
  for(e.i in 1:length(m.expr))
    pos <- pos | (m.vector %~% m.expr[e.i])
  which(pos)

}
          
