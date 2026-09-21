#SCCS @(#)hurreyT.control.s	1.10 07/05/01
    # We want to pass any ... args to hurrey.control, but not pass things
    #  like "dats=mydata" where someone just made a typo.  The use of ...
    #  is just to allow things like "cp=.05" with easier typing

## ======================================================================
## Here is what goes to system
## ======================================================================
eval.control <-
  function(y, update.phi = FALSE, phi.parms, loss.parms, util.parms,
           ...) {

    if(update.phi || missing(phi.parms)) { 
      phi.parms <- phi.control(y, phi.parms, ...)              
      loss.parms <- loss.control(y, loss.parms, ...)
    } else if(missing(loss.parms)) {
      loss.parms <- loss.control(y, loss.parms, ...)
    }
    util.parms <- util.control(util.parms,...)

    
    list(phi.parms=phi.parms, loss.parms=loss.parms, util.parms=util.parms)
  }

## ======================================================================

## ======================================================================
ubar.control <- function(rules.parms, ...) { 

  rc <- ubar.setup(...)

  if(!missing(rules.parms)) {
    
    rc[names(rules.parms)] <- rules.parms
      
    rc <- do.call(ubar.setup,rc)    
  }

  rc
}
  
ubar.setup <- function(ens = c("boost","bagg"), 
                       ntree = 10, sel = 1,
                       trig = c("avgw","avg","best"),
                       strat = T, ...) {

  ens <- match.arg(ens)
  
  if(!is.logical(strat)) {
    warning("'strat' parameter should be logical. Default value TRUE was assumed.")
    strat <- T
  }
  
  if(sel <= 0) {
    warning("The selection parameter 'sel' should be > 0;",
            "The default value of 1 was used instead.")
    sel <- 1
  }

  trig <- match.arg(trig)
  
  list(ens = ens, ntree = ntree, 
       sel = sel, trig = trig, strat = strat)
}


