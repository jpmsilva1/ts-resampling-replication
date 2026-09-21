
##=========================================================================
## extract.rules
##-------------------------------------------------------------------------
## This function takes a regression tree model and extracts from it the
## simplified set of rules.
##
## > data(Boston)
## > ft <- rpart(medv ~ .,Boston)
## > rules.b <- extract.rules(ft,Boston,'medv',fn="BostonURrules")
## Based on code by Jaqueline B. Pugliesi
## Coded by: Rita P. Ribeiro
##=========================================================================
rules.rpart <- function(x, compact = FALSE, 
                        simplify=TRUE, digits=getOption("digits"))
{
  
  if (!inherits(x, "rpart")) stop(cat("Not a legitimate rpart tree\n"))  
  frm <- x$frame

  leafs <- row.names(frm[grep("<leaf>",frm[,1]),])
    
  ##
  ## Print each leaf node as a rule.
  ##

  nr <- length(leafs) 
  rules.frame <- matrix("TRUE",nrow=nr,ncol=2)  
  colnames(rules.frame) <- c("yval","cond")

  for (i in 1:nr) {
   
    rules.frame[i,1] <- frm[leafs[i],]$yval

    pth <- path.rpart(x, nodes=as.numeric(leafs[i]), print.it=FALSE)    
    pth <- pth[[1]][-1]
    n.pth <- length(pth)
    if(!n.pth) {
      rules.frame[i,2] <- "TRUE"
    } else {

      ## not the most efficient way to deal with but
      ## just for evaluation purposes within the data set.
      cat.splits <- which(sapply(pth,function(lt) !grepl(">",lt) && !grepl("<",lt)))

      for(cs in cat.splits) {
        aux <- strsplit(pth[cs],"=")
        rhs <- aux[[1]][2]        
        pth[cs] <- paste(aux[[1]][1]," %in% ","c(",paste(paste("'",strsplit(rhs,",")[[1]],"'",sep=""),collapse=","),")",sep="")
      }
      rules.frame[i,2] <- paste(pth,collapse = " & ")
    }

  }

  rules.frame
}

print.rules.rpart <- function(x, compact = FALSE,
                        simplify=TRUE, digits=getOption("digits"))
{
  
   if (!inherits(x, "rpart")) stop(cat("Not a legitimate rpart tree"))
  
  #
  # Get some information.
  #
  rtree <- length(attr(x, "ylevels")) == 0
  target <- as.character(attr(x$terms, "variables")[2])
  frm <- x$frame
  names <- row.names(frm)
  ylevels <- attr(x, "ylevels")
  ds.size <-  x$frame[1,]$n
  #
  # Print each leaf node as a rule.
  #
  if (rtree)
    # Sort rules by coverage
    ordered <- rev(sort(frm$n, index=TRUE)$ix)
  else
    # Sort rules by probabilty of second class (usually the last in binary class)
    ordered <- rev(sort(frm$yval2[,5], index=TRUE)$ix)
  for (i in ordered)
  {
    if (frm[i,1] == "<leaf>")
    {
      # The following [,5] is hardwired and works on one example....
      if (rtree)
        yval <- frm[i,]$yval
      else
        yval <- ylevels[frm[i,]$yval]
      cover <- frm[i,]$n
      pcover <- round(100*cover/ds.size)
      if (! rtree) prob <- frm[i,]$yval2[,5]
      cat("\n")
      pth <- path.rpart(x, nodes=as.numeric(names[i]), print.it=FALSE)
      pth <- unlist(pth)[-1]
      if (! length(pth)) pth <- "True"
      if (compact)
      {
        cat(sprintf("R%03s ", names[i]))
        if (rtree)
          cat(sprintf("[%2.0f%%,%0.2f]", pcover, pcover))
        else
          cat(sprintf("[%2.0f%%,%0.2f]", pcover, prob))
        cat(sprintf(" %s", pth), sep="")
      }
      else
      {
        cat(sprintf(" Rule number: %s ", names[i]))
        if (rtree)
          cat(sprintf("[%s=%s cover=%d (%.0f%%)]\n",
                      target, yval, cover, pcover))
        else
          cat(sprintf("[%s=%s cover=%d (%.0f%%) prob=%0.2f]\n",
                      target, yval, cover, pcover, prob))
        cat(sprintf("   %s\n", pth), sep="")
      }
    }
  }
  cat("\n")
  invisible(ordered)
 
}




  






