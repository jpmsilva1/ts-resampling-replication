##=========================================================================
## uba_classes
##-------------------------------------------------------------------------
## Definition of classes and methods in uba package
## Coded by: Rita P. Ribeiro
##=========================================================================


# --------------------------------------------------------------
# class def
## UBA PARAMETERIZATION (for a given dataset)

setClass("uba.par",
         representation(phi.parms="list",
                        loss.parms="list",
                        util.parms="list"))         

# --------------------------------------------------------------
# constructor function

uba.par <- function (pP, lP, uP) 
{
    o <- new("uba.par")
    o@phi.parms <- pP
    o@loss.parms <- lP
    o@util.parms <- uP
    o
}


# --------------------------------------------------------------
# show function

setMethod("show","uba.par",
          function(object) {
            cat("\n\t\t UBA Parameterization \n\n")
            cat("\nPhi Parameters ::\n\n")
            cat(sprintf("\t Method = %s\n",object@phi.parms$method))
            ##cat(sprintf("\t Nr. Points = %f\n",object@phi.parms$npts))
            cat(sprintf("\t Control Points:\t \t y \t phi(y) \t phi'(y)\n"))
            for(i in 1:object@phi.parms$npts) 
              cat(sprintf("\t \t %.5f \t %.5f \t %.5f\n",
                          object@phi.parms$control.pts[(i-1)*object@phi.parms$npts + 1],
                          object@phi.parms$control.pts[(i-1)*object@phi.parms$npts + 2],
                          object@phi.parms$control.pts[(i-1)*object@phi.parms$npts + 3]
                          ))
            
            cat("\nLoss Parameters ::\n\n")
            cat(sprintf("\t ymin = %.5f ymax = %.5f\n",object@loss.parms$ymin,object@loss.parms$ymax))
            cat(sprintf("\t tloss = %.5f\n",object@loss.parms$tloss))
            cat(sprintf("\t epsilon = %.5f\n",object@loss.parms$epsilon))

            cat("\nUtil Parameters ::\n\n")
            cat(sprintf("\t utility metric = %s type = %s use.util = %d\n",
                        object@util.parms$umetric,object@util.parms$utype,object@util.parms$use.util))
            cat(sprintf("\t p = %.2f Bmax = %3d\n",object@util.parms$p,object@util.parms$Bmax))
            cat(sprintf("\t event.thr = %.2f score.thr = %.2f\n",object@util.parms$event.thr,object@util.parms$score.thr))
            cat(sprintf("\t binorm.est = %d ipts = %3d \n",object@util.parms$binorm.est,object@util.parms$ipts))
            cat(sprintf("\t beta = %.2f min.tpr = %.2f max.fpr = %.2f maxprec = %d \n",
                        object@util.parms$beta,object@util.parms$min.tpr,object@util.parms$max.fpr,object@util.parms$maxprec))
            cat(sprintf("\t calibrate = %d ustep = %.3f \n",object@util.parms$calibrate,object@util.parms$ustep))           
            cat('\n')
          }
          )

# --------------------------------------------------------------
# class def
setClass("uba.learner",
         contains = 'learner')
##         representation(uba.pars="uba.par"))


# --------------------------------------------------------------
# constructor function
uba.learner <- function(func,pars=list(
                               uba.pars=list(phi.parms=list(),loss.parms=list(),util.parms=list())
                               ##,...
                               )) {
  ## should test for the validity of uba.pars
  if (missing(func)) stop("\nYou need to provide a function name.\n")
  ##new("uba.learner",learner(func=func,pars=pars),uba.pars=uba.pars)
  new("uba.learner",learner(func=func,pars=pars))
}



# --------------------------------------------------------------
# Methods:


# show
setMethod("show","uba.learner",
          function(object) {            
            cat('\nUBA Learner:: ',deparse(object@func),'\n\nParameter values\n')
            for(n in names(object@pars))
              if(n != "uba.pars")
                cat('\t',n,' = ',deparse(object@pars[[n]]),'\n')
            print(object@pars$uba.pars) ## invokes show method
            cat('\n\n')
          }
          )

setMethod("show","learner",
          function(object) {            
            cat('\n(RR) Learner:: ',deparse(object@func),'\n\nParameter values\n')
            for(n in names(object@pars))
              if(n != "uba.pars")
                cat('\t',n,' = ',deparse(object@pars[[n]]),'\n')
            print(object@pars$uba.pars) ## invokes show method
            cat('\n\n')
          }
          )

##---------------------------------------------------------------------------------------
##### From DMwR
# summary
setMethod("summary",
          signature(object="cvRun"),
          function(object,...) {
            cat('\n== Summary of a Cross Validation Experiment ==\n')
            print(object@settings)
            cat('\n* Data set :: ',object@dataset@name)
            print(object@learner) ## invokes show method
            ##cat('\n* Learner  :: ',object@learner@func,' with parameters ')
            ##for(x in names(object@learner@pars))
            ##  cat(x,' = ',
            ##      object@learner@pars[[x]],' ')
            cat('\n\n* Summary of Experiment Results:\n\n')
            apply(object@foldResults,2,function(x)
                  c(avg=mean(x,na.rm=T),std=sd(x,na.rm=T),
                    min=min(x,na.rm=T),max=max(x,na.rm=T),
                    invalid=sum(is.na(x)))
        )
          }
          )




# --------------------------------------------------------------
# class def (assuming regression problems only)
setClass("ubar",
         representation(call="character", target.var="character",
                        y="numeric",
                        ##uba.pars="uba.par",
                        phi.parms="list",
                        loss.parms="list",
                        util.parms="list",
                        rules.parms="list",
                        rp.parms="list",
                        def.rule = "numeric",
                        rules.cond="character",
                        rules.yval="matrix",
                        rules.info="matrix"
                        ))         

## setValidity("ubar",
##             function(object) {
##               if(length(object@rules.cond) != NROW(rules.yval))
##                 return("The number of rules specified in conditions and in predictions differ!")
##               TRUE
##             }
##           )

## setGeneric("updateRules<-",
##            function(x, idx) standardGeneric("updateRules<-")
##            )

## setReplaceMethod("updateRules", "ubar",
##                  function(x, idx) {
##                    x@rules.cond <- x@rules.cond[idx]
##                    x@rules.yval <- x@rules.yval[idx,]
##                    validObject(x)
##                    x
##                  }
##                  )

## ensure consistency
## setMethod("updateIdx","ubar",
##           function(object,new.idx) {
##             object@rules.cond <- object@rules.cond[new.idx]
##             object@rules.yval <- object@rules.yval[new.idx,]
##             if(NROW(object@rules.info))
##               object@rules.info <- object@rules.info[new.idx,]

##           }
##           )

# --------------------------------------------------------------
# show function

setMethod("show","ubar",
          function(object) {
            for(i in 1:NROW(object@rules.yval)) {
              cat(sprintf("\n\n Rule nr %s ", i))

              cat(sprintf("[wt=%.3f, n=%.0f, u=%.3f, mad=%.3f]\n ",                         
                          object@rules.yval[i,2],
                          object@rules.info[i,1],
                          object@rules.info[i,3],
                          object@rules.info[i,2]))
                          

              cat(sprintf("\t%s=%f\n",
                          object@target.var,
                          object@rules.yval[i,1]))                     
                                                    
              cat("\t\t",gsub("&","and \\\n\t\t",object@rules.cond[i]))              
            }          
            cat('\n')
          }
          )


