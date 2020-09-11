library(sparklyr)
library(dplyr)
library(roll)
library(corrplot)
Sys.setenv(TZ='Asia/Hong Kong')

path = '/mnt/c/Users/cyril/Desktop/BDT/IndependentProject2/'
setwd(path)
spark_dir = "/mnt/c/Users/cyril/Desktop/sparktmp/"

config = spark_config()
config$`sparklyr.shell.driver-java-options` <-  paste0("-Djava.io.tmpdir=", spark_dir)
config$`sparklyr.shell.driver-memory` <- "4G"
config$`sparklyr.shell.executor-memory` <- "4G"
config$`spark.yarn.executor.memoryOverhead` <- "512"
config$`sparklyr.shell.driver-class-path` <- "/usr/share/java/mysql-connector-java-8.0.15.jar"
config$spark.executor.heartbeatInterval='10000000s'
config$spark.network.timeout= '100000000s'

sc <- spark_connect(master = "local", config = config)

db_tbl <- spark_read_jdbc(sc,
                  name    = "symbols",  memory=TRUE,
                  options = list(url      = "jdbc:mysql://localhost:3306/symbols?serverTimezone=UTC",
                                 user     = "root",
                                 password = "2124",
                                 dbtable  = "(select time,symbol,ret from symbols.all where time between '2018-06-01' and '2018-07-01'  ) as my_query"))

  
all = db_tbl %>% sdf_pivot(time ~ symbol,fun.aggregate = list(ret = "mean"))  
all = sdf_sort(a, 'time')
all = collect(all)
spark_write_csv(all,paste0(path,'returns.csv'))

db_tbl <- spark_read_jdbc(sc,
                          name    = "symbols",  memory=TRUE,
                          options = list(url      = "jdbc:mysql://localhost:3306/symbols?serverTimezone=UTC",
                                         user     = "root",
                                         password = "2124",
                                         dbtable  = "(select time,symbol,Close from symbols.all where time between '2018-06-01' and '2018-07-01'  ) as my_query"))


all = db_tbl %>% sdf_pivot(time ~ symbol,fun.aggregate = list(Close = "mean"))  
all = sdf_sort(all, 'time')
spark_write_csv(all,paste0(path,'close3.csv'))


# Plot correlation matrix
df = as.data.frame(collect(all))
df = as.data.frame(df)
remove = NULL
for(i in 1:ncol(df)) remove = c(remove,ifelse(length(which(is.na(df[,i])))== length(df[,i]),i,0))
df = df[,-remove]
df = df[complete.cases(df),]
corrplot(cor(df[,c(2:16,which(colnames(df) == 'BTCUSDT'))]), method="color")
