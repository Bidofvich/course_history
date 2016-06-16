#week 1 class 1
getwd() - get working directory
read.csv("mydata.csv")
dir() - ls

myfunc <- function(x)
{
	y <- rnorm(100)
	mean(y)
}
#save as myfunc.R
source("myfunc.R")

second <- function(x)
{
	x + rnorm(length(x))
}

#week1 class 2
<- = assignment operator
x <- 1
msg <- "hello"
x <- 1:20 #means a vector(list) from 1 to 20 including 1 and 20.
numbers in r = numeric objects(double precision real no.)
L to represent integer
1 gives numeric object
1L give integer
Inf = infinity or 1/0
NaN = not a number or 0/0
#to create vector use vector()
#c() func use to create vectors of objects
x <- c(0.5, 0.6)
##x is 0.5 0.6 vector
#using vector
x <- vector("numeric", length=10)
#x is 0 0 0 0 0 0 0 0 0 0
#if mixing objects vector type is least common denominator in objects.
> x <- c(1.7, "a") # becomes character vector
> x
[1] "1.7" "a"  
class(x) # to tell which objects type eg integer.
#explicit coercion
as.numeric(x) # to change into numeric
as.logical(x) #to change into logical
as.character(x) #to change into character
#to create list
x <- list(1, "a", TRUE, 1+4i)
[1] # to show vector
[[1]] # to show list index

#Matrix
m <- matrix(nrow=2, ncol= 3)
dim(m) #dimension
> 2 3
attributes(m)
# matrix is filled column wise
> m <- matrix(1:6, nrow=2, ncol=3)
> m
     [,1] [,2] [,3]
[1,]    1    3    5
[2,]    2    4    6
> m <- 1:10
> dim(m)
NULL
> dim(m) <- c(2, 5)
> m
     [,1] [,2] [,3] [,4] [,5]
[1,]    1    3    5    7    9
[2,]    2    4    6    8   10

#column binding and row binding
> x <- 1:3
> y <- 10:12
> cbind(x, y)
     x  y
[1,] 1 10
[2,] 2 11
[3,] 3 12
> rbind(x, y)
  [,1] [,2] [,3]
x    1    2    3
y   10   11   12

#factor represent categorical data
> x <- factor(c("yes", "yes", "no", "yes", "no"))
> x
[1] yes yes no  yes no 
Levels: no yes #Level is alphabetical order
> table(x)
x
 no yes 
  2   3 
> unclass(x)
[1] 2 2 1 2 1
attr(,"levels")
[1] "no"  "yes"

> x <- factor(c("yes", "yes", "no", "yes", "no"), levels = c("yes", "no"))
> x
[1] yes yes no  yes no 
Levels: yes no #level is accordingly in factor

#missing values = NA or NaN
#NaN for mathematical , NA for everything else

is.na()
is.nan()
#NA have class, integer NA, character NA.
#NAN is NA but not converse.
is.na(x)

#Data frames - tables
> x <- data.frame(foo = 1:4, bar = c(T, T, F, F))
> x
  foo   bar
1   1  TRUE
2   2  TRUE
3   3 FALSE
4   4 FALSE
> nrow(x)
[1] 4
> ncol(x)
[1] 2

#names for r object
> x <- 1:3
> names(x)
NULL
> names(x) <- c("foo", "bar", "norf")
> x
 foo  bar norf 
   1    2    3 
> names(x)
[1] "foo"  "bar"  "norf"
#names for list
> x <- list(a=1, b=2, c=3)
> x
$a
[1] 1

$b
[1] 2

$c
[1] 3
#names for matrix
> x <- matrix(1:4 , nrow=2, ncol=2)
> dimnames(x) <- list(c('a', 'b'), c('c', 'd'))
> x
  c d
a 1 3
b 2 4

#Reading Data
read.table()
args:
file=name of file
header= logical indicating file has header line
sep= string indicating how col are separated(, , ; ,spaces,tab)
colClasses = character vector indicating class of each col
nrows= no. of rows
comment.char = indicating comment character (ex # )
skip=skip no. of lines from beginning
stringsAsFactors = character var be coded as factors?

#Read help page for read.table
dput and dump stores extra meta data
#Connections
#subset
[ = returns object of same class
[[ = extracts element of list
$ = extracts element of list by name 
> x <- c('a', 'b', 'c', 'c', 'd', 'a')
> x[1]
[1] "a"
> x[1:4]
[1] "a" "b" "c" "c"
> x[x > "a"]
[1] "b" "c" "c" "d"
> u <- x > 'a'
> u
[1] FALSE  TRUE  TRUE  TRUE  TRUE FALSE
> x[u]
[1] "b" "c" "c" "d"

> x <- list(foo = 1:4, bar = 0.6)
> x
$foo
[1] 1 2 3 4

$bar
[1] 0.6

> x[1]
$foo
[1] 1 2 3 4

> x[[1]]
[1] 1 2 3 4
> x$bar
[1] 0.6
> x[['bar']]
[1] 0.6
> x['bar']
$bar
[1] 0.6
> x <- list(foo = 1:4, bar = 0.6, baz = 'hello')
> x[c(1, 3)] #multiple value possible in single brackett
$foo
[1] 1 2 3 4

$baz
[1] "hello"

> x <- list(foo = 1:4, bar = 0.6, baz = 'hello')
> name <- 'foo'
> x[[name]] # [[ can be used with computed indices
[1] 1 2 3 4
> x$name # $ with literal names only
NULL
> x$foo
[1] 1 2 3 4
#Subsetting matrix
> x <- matrix(1:6, 2, 3)
> x
     [,1] [,2] [,3]
[1,]    1    3    5
[2,]    2    4    6
> x[1, 2]
[1] 3
> x[1, ]
[1] 1 3 5
> x[, 2]
[1] 3 4
> x[1, 2]
[1] 3
> x[1, 2, drop = FALSE]
     [,1]
[1,]    3
> x[1, , drop = FALSE]
     [,1] [,2] [,3]
[1,]    1    3    5
#Partial Matching works with [[ and $
> x <- list(aardvark = 1:5)
> x$a
[1] 1 2 3 4 5
> x[['a']]
NULL
> x[['a', exact = FALSE]]
[1] 1 2 3 4 5
#removing na
> x <- c(1, 2, NA, 4, NA, 5)
> bad <- is.na(x)
> x[!bad]
[1] 1 2 4 5
> x <- c(1, 2, NA, 4, NA, 5)
> y <- c('a', 'b', NA, 'd', NA, 'f')
> good <- complete.cases(x, y)
> good
[1]  TRUE  TRUE FALSE  TRUE FALSE  TRUE
> x[good]
[1] 1 2 4 5
> y[good]
[1] "a" "b" "d" "f"
#Vectorized operations
> x <- matrix(1:4, 2, 2); y <- matrix(rep(10, 4), 2, 2)
# rep reapeats 10 four times
> x
     [,1] [,2]
[1,]    1    3
[2,]    2    4
> y
     [,1] [,2]
[1,]   10   10
[2,]   10   10
> x * y 	# element wise multiplication
     [,1] [,2]
[1,]   10   30
[2,]   20   40
> x / y
     [,1] [,2]
[1,]  0.1  0.3
[2,]  0.2  0.4
> x %*% y 	# true matrix multiplication
     [,1] [,2]
[1,]   40   40
[2,]   60   60

#Week 2
#control structures = control flow of execution of program
common structures are:
1. if, else:
  if(condition){

  }
  else if(condition){

  }
  else{

  }

  if (x > 3){
  y <- 10
  } else {
  y <- 0
  }

  or

  y <- if(x>3){
  10
  } else {
  0
  }
2. for:
for(i in 1:10){
  print(i)
}

x<- c("a", "b", "c", "d")
for(i in 1:4) print(x[i])

for(i in seq_along(x)){ #creates integer sequence from 1 to length
  print(x[i])
}

for(i in seq_len(nrow(x))){ #creates seq for integers

}
for(letter in x) print(letter)

3. while:
while(count < 10) {
  count <- count + 1
}
conditions are evaluated from left to right
4. repeat: #infinite loop
break is used to exit loop
repeat{
  if(){
    break
  }
  else{

  }
}

5. break
6. next #continue in python
7. return

#Functions
add2 <- function(x, y) {
  x + y
}

above <- function (x, n = 10) { #n default value is 10
  use <- x > n
  x[use]
}
> x <- 1:20
> above(x, 12)
[1] 13 14 15 16 17 18 19 20


columnmean <- function(y, removeNA = TRUE) {
      nc <- ncol(y)
      means <- numeric(nc) # vector of 0's of length nc
      for(i in 1:nc){
        means[i] <- mean(y[, i], na.rm = removeNA)
      }
      means
}

#Arguments are processed only when it is used in function/ This is laziness of R.

# "..." argument used when extending another func and dont want to copy whole args list of original func. read about ...
myplot <- function (x, y, type="l", ...) {
    plot(x, y, type = type, ...)
}
> paste('a', 'b', sep = ':')
[1] "a:b"

#Lexical scoping
make.power <- function(n) {
  pow <- function(x) {
    x^n
  }
  pow
}

> cube <- make.power(3)
> square <- make.power(2)
> cube(3)
[1] 27
> square(3)
[1] 9


> ls(environment(cube))
[1] "n"   "pow"
> get("n", environment(cube))
[1] 3
> ls(environment(square))
[1] "n"   "pow"
> get("n", environment(square))
[1] 2
#lexical scoping: variables are used from parent environment than global. this is diff bw dynamic and lexical scoping

#Date time functions

#Week 3
#Loop Functions
lapply: loop over a list and evaluate func on each element
lapply(list, func, ...) # always return a list
> x <- list(a = 1:5, b= rnorm(10))
> x
$a
[1] 1 2 3 4 5

$b
 [1] -1.4078903 -1.2967673  0.3882797  0.9474098  1.1219709  1.0511261
 [7] -1.2687816  0.1132720  0.9099762  1.5828267

> lapply(x, mean)
$a
[1] 3

$b
[1] 0.2141422

> x <- 1:4 #runif(x) generates x random no.s
> lapply(x, runif, min = 0, max = 10)
[[1]]
[1] 2.163159

[[2]]
[1] 2.617133 2.001741

[[3]]
[1] 5.388037 6.877513 2.301749

[[4]]
[1] 0.6648614 7.3627358 5.4646219 7.1335149

> x <- list(a = matrix(1:4, 2, 2), b = matrix(1:6, 3, 2))
> x
$a
     [,1] [,2]
[1,]    1    3
[2,]    2    4

$b
     [,1] [,2]
[1,]    1    4
[2,]    2    5
[3,]    3    6

> lapply(x, function(elt) elt[, 1])
$a
[1] 1 2

$b
[1] 1 2 3


sapply: same as lapply but simplify result
> x <- list(a=1:4, b= rnorm(10), c=rnorm(20, 1), d= rnorm(100, 5))
> lapply(x, mean)
$a
[1] 2.5

$b
[1] 0.2477846

$c
[1] 1.096821

$d
[1] 4.922971

> sapply(x, mean)
        a         b         c         d 
2.5000000 0.2477846 1.0968211 4.9229708 
if result is list with length = 1, then vector is returned
if result is list with length same and > 1, then matrix is returned
else list is returned

apply: apply a func over margins of array
apply( x, margin, fun, ...)
x is array
margin is integer vector which margins should be retained
fun is func applied
... is for other args to passed to fun
> x <- matrix(rnorm(200), 20, 10)
> apply(x, 2, mean) # 2 means all col
 [1]  0.150103495  0.007596779 -0.033488145  0.255615419 -0.044776892
 [6]  0.026579299 -0.037776536  0.094427053 -0.325085772  0.125249558
> apply(x, 1, mean) # 1 means all rows
 [1]  0.47256620  0.15599813 -0.18707849  0.12059917  0.40265923 -0.07233030
 [7]  0.42992597 -0.29018023 -0.48022047  0.60602027  0.25458548 -0.03885957
[13]  0.22370498 -0.28886992  0.43493417 -0.27257590 -0.65404988  0.20981672
[19] -0.18982006 -0.39993699

rowSums equivalent = apply(x, 1, sum)
rowMeans equivalent = apply(x, 1, mean)
colSums equivalent = apply(x, 2, sum)
colMeans equivalent = apply(x, 2, mean)

> x <- matrix(rnorm(200), 20, 10)
> apply(x, 1, quantile, probs = c(.25, .75))
          [,1]       [,2]       [,3]       [,4]       [,5]      [,6]      [,7]
25% -0.3476663 -0.3587228 -0.9417290 -0.2537246 -0.2661843 0.2293373 -1.015777
75%  0.7901921  0.5378400  0.8702174  0.7814421  0.6225367 0.6841100  1.067515
          [,8]      [,9]      [,10]      [,11]      [,12]      [,13]      [,14]
25% -0.8219746 -0.313048 -0.2152329 -0.8611793 -0.8536629 -0.3383362 -0.7507792
75%  0.4461323  0.938519  0.9392170  0.4506565  0.8596230  0.8463268  0.6029322
         [,15]      [,16]      [,17]      [,18]      [,19]     [,20]
25% -1.1004048 -0.4342453 -0.5252540 -1.2204091 -0.3257892 0.1352914
75%  0.2091723  0.1012309  0.9739697  0.1187215  0.7728642 0.5646373

tapply: apply a func over subsets of vector
function (X, INDEX, FUN = NULL, ..., simplify = TRUE)  
x is vector
INDEX is factor or list of factors
fun is func
... other args to passed to FUN
simplify simplify result

mapply: multivariate version of lapply
apply func parallely over set of args
function (FUN, ..., MoreArgs = NULL, SIMPLIFY = TRUE, USE.NAMES = TRUE)  
... contains args to apply over
MoreArgs list of other args to fun
SIMPLIFY result should be simplified
list(rep(1, 4), rep(2, 3), rep(3, 2), rep(4, 1)) is similar to mapply below:
> mapply(rep, 1:4, 4:1)
[[1]]
[1] 1 1 1 1

[[2]]
[1] 2 2 2

[[3]]
[1] 3 3

[[4]]
[1] 4

> noise <- function(n, mean, sd) rnorm(n, mean, sd)
> noise(5,1 ,2)
[1]  1.953397  1.031996  1.889350 -4.532291  2.232731
> noise(1:5, 1:5, 2) #working for one only
[1] 4.883696 1.390834 2.570663 3.266493 6.230559
> mapply(noise, 1:5, 1:5, 2)
[[1]]
[1] 0.9239916

[[2]]
[1] -2.146581  5.534295

[[3]]
[1] 1.495853 4.469184 2.476475

[[4]]
[1] 1.584920 7.476185 2.540689 3.799988

[[5]]
[1] 3.3708845 3.4403847 5.8268199 7.2260864 0.7257307

#Split
function (x, f, drop = FALSE, ...) 
x is vector or list
f is factor or list of factors
drop empty factor level drop or not

#invisible func stops autoo prinnting  of return type of func

Debugging tools:
traceback
debug
browser
trace
recover

#Week 4

str :compactly display the internal structure of r object.
str: a brief summary
str(str)
str(ls)
x <- rnorm(100, 2, 4)
summary(x)
str(x)

f <- gl(40, 10)
str(f)

generating random no.s
rnorm: generate randorm normal variates with a given mean and standard deviation.
dnorm: evaluate the normal prob density (with a given mean/sd) at a point(vector of points)
pnorm: evaluate the cumulative distribution function for a normal distribution
rpois: generate random poisson variates with a given rate.

setting the random no. seed with set.seed ensures reproducibility.
set.seed(1)

generating random number from linear model
set.seed(20)
x <- rnorm(100)
e <- rnorm(100, 0, 2)
y <- 0.5 + 2*x + e
summary(y)
plot(x, y)


set.seed(10)
x <- rbinom(100, 1, 0.5)
e <- rnorm(100, 0, 2)
y <- 0.5 + 2 * x + e
summary(y)
plot(x, y)


set.seed(1)
x <- rnorm(100)
log.mu <- 0.5 + 0.3 * x
y <- rpois(100, exp(log.mu))
summary(y)
plot(x, y)

Random sampling

set.seed(1)
sample(1:10, 4)
sample(1:10, 4)
sample(letters, 4)
sample(1:10) ## permutation
sample(1:10)
sample(1:10, replace = TRUE)

simulation summary
drawing samples from specific probability distribution can be done with r* func.
standard distribution are built in: Normal, poisson, binomial, exponential, gamma etc
the sample func can be used to draw random samples from arbitrary vectors.
setting the random no. generator seed via set.seed is critical for reproducibility.


R profiler: code slow? opyimize it.
performance analysis or r profiler for where code is using most of time
system.time() #returns amount oftime to evaluate the expression

Rprof() 
summaryRprof(): summarizes the output from Rprof
dont use system.time() and Rprof() together