Week 1
getwd()
setwd('./data-cleaning') # set working dir
file.exists('directoryName')
dir.create('directoryName')

if(!file.exists('data')) {
	dir.create('data')
}

download.file(url='', destfile='', method='') #useful for csv tsv etc

fileUrl <- "https://data.baltimorecity.gov/api/views/dz54-2aru/rows.csv?accessType=DOWNLOAD"
download.file(fileUrl, destfile = "./data/cameras.csv", method = "curl") #use method = curl for https

## [1] "cameras.csv"

dateDownloaded <- date()
dateDownloaded

read.table()
cameraData <- read.table('./data/cameras.csv', sep = ',', header = TRUE)
cameraData <- read.csv('./data/cameras.csv')
head(cameraData)

more important parameters:
quote: there are any quoted value or not, quote="" means no quotes
na.strings : set the character that represent missing value
nrows: how many rows to read of file , eg nrows=10
skip: number of lines to skip before read

fileUrl <- "https://data.baltimorecity.gov/api/views/dz54-2aru/rows.xlsx?accessType=DOWNLOAD"
download.file(fileUrl, destfile="./data/cameras.xlsx", method="curl")
dateDownloaded <- date()
install.packages('xlsx')
library(xlsx)
cameraData <- read.xlsx("./data/cameras.xlsx", sheetIndex=1, header = TRUE)
head(cameraData)

colIndex <- 2:3
rowIndex <- 1:4
cameraData <- read.xlsx("./data/cameras.xlsx", sheetIndex=1, colIndex=colIndex, rowIndex=rowIndex)
cameraData
write.xlsx()
read.xlsx2()
read.xlsx()

library(XML)
fileUrl <- "http://www.w3schools.com/xml/simple.xml"
doc <- xmlTreeParse(fileUrl, useInternal=TRUE)
rootNode <- xmlRoot(doc)
xmlName(rootNode)
names(rootNode)
rootNode[[1]]
rootNode[[1]][[2]]
xmlSApply(rootNode, xmlValue)

XPath
/node top level node
//node node at any level
node[@attr-name] node with attr name
node[@attr-name='bob'] node with attr name = 'bob'

xpathSApply(rootNode, "//name", xmlValue)
xpathSApply(rootNode, "//price", xmlValue)

#fileUrl <- "http://espn.go.com/nfl/team/.../name/bal/baltimore-ravens" wrong link
doc <- htmlTreeParse(fileUrl, useInternal=TRUE)
scores <- xpathSApply(doc, "//li[@class='score'", xmlValue)
teams <- xpathSApply(doc, "//li[@class='team-name'", xmlValue)
scores
teams

library(jsonlite)
jsonData <- fromJSON("https://api.github.com/users/jtleek/repos")
names(jsonData)
names(jsonData$owner)
jsonData$owner$login
myjson <- toJSON(iris, pretty=TRUE)
cat(myjson)
iris2 <- fromJSON(myjson)
head(iris2)

data.table
library(data.table)
DF = data.frame(x=rnorm(9), y=rep(c('a','b','c'), each=3), z=rnorm(9))
head(DF, 3)

DT =  data.table(x=rnorm(9), y=rep(c('a','b','c'), each=3), z=rnorm(9))
head(DT, 3)
tables()
DT[2, ]
DT[DT$y=="a",]
DT[c(2, 3)]
DT[, c(2, 3)]

{
	x = 1
	y = 2
}
k = {print(10); 5}

DT[, list(mean(x), sum(z))]
DT[, table(y)]
DT[, w:=z^2] #used to add new column w to table having z square

DT2 <- DT
DT[, y:=2] #changing Dt also changes DT2
DT[, m:= {tmp <- (x+z); log2(tmp+5)}] #multiple operations
DT[, a:=x>0] #plyr like operations
DT[, b:= mean(x+w), by=a] # group by a ie where a is true take mean of true no.s and apply mean to all those rows having true similarly false.
set.seed(123);
DT <- data.table(x=sample(letters[1:3], 1E5, TRUE))
DT[, .N, by=x] #count the no. of times grouped by N. 1E5 means 100,000

#keys
DT <- data.table(x=rep(c("a", "b", "c"), each=100), y=rnorm(300))
setkey(DT, x)
DT['a']
DT1 <- data.table(x=c('a', 'a', 'b', 'dt1'), y=1:4)
DT2 <- data.table(x=c('a', 'b', 'dt2'), z=5:7)
setkey(DT1, x); setkey(DT2, x)
merge(DT1, DT2)

#fast reading
big_df <- data.frame(x=rnorm(1E6), y=rnorm(1E6))
file <- tempfile()
write.table(big_df, file=file, row.names=FALSE, col.names=TRUE, sep="\t", quote=FALSE)
system.time(fread(file))

system.time(read.table(file, header=TRUE, sep="\t"))


#Week 2
library(RMySQL)
ucscDb <- dbConnect(MySQL(), user="genome", host="genome-mysql.cse.ucsc.edu")
result <- dbGetQuery(ucscDb, "show databases;"); dbDisconnect(ucscDb);

hg19 <- dbConnect(MySQL(), user="genome", db="hg19", host="genome-mysql.cse.ucsc.edu")
allTables <- dbListTables(hg19)
length(allTables)
allTables[1:5]
dbListTables(hg19, "affyU133Plus2") #no. of columns
dbGetQuery(hg19, "select count(*) from affyU133Plus2") # no. of rows

affyData <- dbReadTable(hg19, "affyU133Plus2")
head(affyData)
query <- dbSendQuery(hg19, "select * from affyU133Plus2 where misMatches between 1 and 3")
affMis <- fetch(query); quantile(affMis$misMatches)
affMisSmall <- fetch(query, n=10); dbClearResult(query);
dim(affMisSmall) # dimensions
dbDisconnect(hg19)