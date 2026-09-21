#==========================================================================#
# loss.R                                                                   #
#                                                                          #
# PhD                                                                      #
# Rita P. Ribeiro                                                          #
# Last Modified:                                             #
#--------------------------------------------------------------------------#
## Objective: Error Calibration
## 
##  
#==========================================================================#

##lossFuns <- c("AD","SE")
##normFuns <- c("BRAYCURTIS","LINEAR","SIG")


# ======================================================================
# loss.setup
# This function setups all the parameters associated with the loss fun.
# In particular, it has to estimate the normalized loss.
## unifies all the introduced params but 
## it gives more precedence to loss.parms
# ======================================================================
loss.setup <- function(y, ymin = NULL, ymax = NULL,
                       tloss = NULL, epsilon = 0.1, ...) {

  r <- range(y)
    
  if(is.null(ymin)) ymin <- r[1]

  if(is.null(ymax)) ymax <- r[2]
  
  if(!(ymax - ymin > 0)) {
    warning("The 'ymin' and 'ymax' value are invalid." ,
            "The range of the target var Y are used instead.")
    ymin <- r[1]; ymax <- r[2]
  }

  
  if(is.null(tloss)) {

    ## Cherkassky and Ma, 2002
    n <- length(y)
    tau <- 3
    ## noise level
    meanY <- mean(y)
    tL <- abs(meanY - y)
    tloss <- tau * sd(tL) * sqrt(log(n)/n)
  }

  if(!is.numeric(tloss))
    stop("Invalid value for 'tloss'")
    
  list(ymin = ymin, ymax = ymax, tloss = tloss,
       epsilon = epsilon) 

}

loss.control <- function(y, loss.parms, ...) {

  call <- match.call()

  lossP <- loss.setup(y, ...)
  
   if(!missing(loss.parms)) {
    lossP[names(loss.parms)] <- loss.parms

    lossP <- do.call(loss.setup,c(list(y=y),lossP))
  }
  
  lossP
  
}

loss2double <- function(loss.parms) {

  as.double(unlist(loss.parms))
}

# ======================================================================
# 
# This function 
# =====================================================================
AD <- function(x,y) abs(x-y)
SE <- function(x,y) (x-y)^2





