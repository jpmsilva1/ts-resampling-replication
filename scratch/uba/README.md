# UBA

The uba package is developed in R and comprehends a set of tools designated for regression algorithms in non-uniform costs domains. Namely, it includes utility-based performance metrics, utility-based curves and an utility-based regression rules ensemble system.

A more complete description of these functionalities is available on my [PhD Thesis: Utility-based Regression](http://www.dcc.fc.up.pt/~rpribeiro/publ/rpribeiroPhD11.pdf).

## Installation

## Help

### Example

library(uba)
library(rpart)
data(NO2Emissions)
y <- NO2Emissions$LNO2
m <- rpart(LNO2 ~.,NO2Emissions)
ypred <-predict(m)
pP <- phi.control(y,method="extremes",extr.type="high") 
lP <- loss.control(y) 




