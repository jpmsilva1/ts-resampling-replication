library(uba)
data(NO2Emissions)
y <- NO2Emissions$LNO2
ypred <- sample(y)
pP <- phi.control(y,method="extremes",type="high") 
yphi <- phi(y,phi.parms=pP) 
lP <- loss.control(y) 
uP <- util.control(umetric="MU",p=0.95) 
us <- util(ypred,y,phi.parms=pP,loss.parms=lP,util.parms = uP,return.uv=TRUE)
util.surface(y,phi.parms=pP,loss.parms=lP,util.parms = uP)
util.isometrics(y,phi.parms=pP,loss.parms=lP,util.parms = uP)

mu <- util(ypred, y, phi.parms=pP, loss.parms=lP, util.parms = uP)
uP <- util.control(umetric="P",event.thr=1) 
prec <- util(ypred, y, phi.parms=pP, loss.parms=lP, util.parms = uP)
print(prec)
uP <- util.control(umetric="R",event.thr=1) 
rec <- util(ypred, y, phi.parms=pP, loss.parms=lP, util.parms = uP)
print(rec)
uP <- util.control(umetric="F",event.thr=1,beta=1) 
fm <- util(ypred, y, phi.parms=pP, loss.parms=lP, util.parms = uP)
print(fm)
uP <- util.control(umetric="AUCPR",event.thr=1) 
aucpr <- util(ypred, y, phi.parms=pP, loss.parms=lP, util.parms = uP)
print(aucpr)
uP <- util.control(umetric="AUCROC",event.thr=1) 
aucroc <- util(ypred, y, phi.parms=pP, loss.parms=lP, util.parms = uP)
print(aucroc)
uP <- util.control(event.thr=0.75)

pr <- pn.curve(ypred, y,
           curve.type="pr",
           phi.parms = pP,
           loss.parms = lP,
           util.parms = uP)

pn.plot(list(M=pr),add.leg=TRUE)          

uP <- util.control(event.thr=0.5)
roc <- pn.curve(ypred, y,
           curve.type="roc",
           phi.parms = pP,
           loss.parms = lP,
           util.parms = uP)
pn.plot(list(M=roc),add.leg=TRUE, leg.loc="bottomright")    
