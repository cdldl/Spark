import numpy as np
from numpy.linalg import eigh
import pandas as pd
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.mllib.linalg import DenseVector
import pyspark.sql.functions as fns
from pyspark.sql.types import DoubleType, Row
from pyspark.sql.window import Window
import matplotlib.pyplot as plt
from pyspark.ml.stat import Correlation
import pyspark.sql.functions as F

sc = SparkContext()
sqlContext = SQLContext(sc)

path = '/mnt/c/Users/cyril/Desktop/BDT/IndependentProject2/retts.csv/'
filename = 'part-00000-edd80d2a-e8cf-4b20-8678-ca9a51ee7325-c000.csv'


def get_input_df(path, filename):
    data2 = pd.read_csv(path + filename, index_col='time', parse_dates=True)
    cols = data2.columns
    data = data2.resample('60T').agg({'sum'})
    data.columns = cols
    data = data.loc[:, (data != 0).any(axis=0)]
    return sqlContext.createDataFrame(data)


def update_columns(df):
    typeCol = [x[1] for x in df.dtypes]
    notDouble = [i for i, x in enumerate(typeCol) if x != 'double']
    drops = [df.columns[i] for i in notDouble]
    df = df.select([c for c in df.columns if c not in drops])
    columns = df.columns
    for column in columns:
        df = df.withColumn(column, F.when(
            F.isnan(F.col(column)), 0).otherwise(F.col(column)))
    df = df.na.fill(0).cache()
    return df


def transform_to_dense_vector(df):
    # Transform to dense vector
    rdd = df.rdd.map(lambda x: Row(features=DenseVector(x)))
    df3 = sqlContext.createDataFrame(rdd)
    df3 = df3.cache()
    return df3


def estimateCovariance(df):
    m = df.select(df['features']).rdd.map(lambda x: x[0]).mean()
    dfZeroMean = df.select(df['features']).rdd.map(
        lambda x: x[0]).map(lambda x: x-m)  # subtract the mean
    return dfZeroMean.map(lambda x: np.outer(x, x)).sum() / df.count()


def pca(df, k=2):
    cov = estimateCovariance(df)
    col = cov.shape[1]
    eigVals, eigVecs = eigh(cov)
    inds = np.argsort(eigVals)
    eigVecs = eigVecs.T[inds[-1:-(col+1):-1]]
    components = eigVecs[0:k]
    f = df.select(df['features']).rdd.map(
        lambda x: x[0]).map(lambda x: np.dot(x, components.T))
    # Return the `k` principal components, `k` scores, and all eigenvalues
    return f


def convert_factors(factors, k):
    # Convert Factors to the right format
    f = factors.collect()
    f = np.array(f)
    rdd1 = sc.parallelize(f)
    rdd2 = rdd1.map(lambda x: [float(i) for i in x])
    fac = rdd2.toDF([str(x) for x in list(range(1, k+1))])
    return fac


def get_weighted_asset_return(df, col, fac, w):
    df_tmp = df.select(col)
    df_tmp = df_tmp.selectExpr(str(col) + " as label")
    df2 = df_tmp.withColumn("row_num", fns.monotonically_increasing_id())
    df2 = df2.withColumn("index", fns.row_number().over(w))
    df2 = df2.cache()
    fac = fac.withColumn("row_num", fns.monotonically_increasing_id())
    fac = fac.withColumn("index", fns.row_number().over(w))
    df_all = fac.join(df2, df2.index == fac.index)
    df_all = df_all.drop('row_num').drop('index')
    return fac, df_all


def get_model(vec):
    # , regParam=0.3, elasticNetParam=0.8)
    lr = LinearRegression(featuresCol='features',
                          labelCol='label', maxIter=1000)
    model = lr.fit(vec)
    return model


def get_coefficients(model):
    return model.coefficients


def get_residuals(model):
    return model.summary.residuals


def get_standardized_residuals(res):
    resstd = res.select(fns.stddev(
        'residuals').alias('std')).collect()[0]['std']
    resstd = res.withColumn('std', fns.lit(resstd))
    resstd = resstd.withColumn('resstd', fns.col('residuals') / fns.col('std'))
    resstd = resstd.drop('residuals')
    return resstd


def pcaVar(df, k=2):
    cov = estimateCovariance(df)
    col = cov.shape[1]
    eigVals, eigVecs = eigh(cov)
    inds = np.argsort(eigVals)
    eigVecs = eigVecs.T[inds[-1:-(col+1):-1]]
    components = eigVecs[0:k]
    eigVals = eigVals[inds[-1:-(col+1):-1]]  # sort eigenvals
    score = df.select(df['features']).rdd.map(
        lambda x: x[0]).map(lambda x: np.dot(x, components.T))
    # Return the `k` principal components, `k` scores, and all eigenvalues
    return eigVals


def varianceExplained(df, k=1):
    eigenvalues = pcaVar(df, k)
    vars = []
    for i in range(k):
        vars.append(sum(eigenvalues[0:i])/sum(eigenvalues))
    return vars


def heatmap2d(arr):
    plt.imshow(arr, cmap='viridis')
    plt.colorbar()
    plt.savefig('corrMatrix.pdf')


def CorrelationMatrix(df):
    vector_col = "corr_features"
    assembler = VectorAssembler(inputCols=df.columns, outputCol=vector_col)
    df_vector = assembler.transform(df).select(vector_col)
    matrix = Correlation.corr(df_vector, vector_col)
    cor = matrix.collect()[0]["pearson({})".format(vector_col)].values
    cor2 = pd.DataFrame(np.array(cor).reshape(
        len(df.columns), len(df.columns))).iloc[:15, :15]
    heatmap2d(cor2)


def main():
    # Create dataframe
    df = get_input_df(path, filename)
    df = update_columns(df)

    # Print Scree plot
    # var = varianceExplained(transform_to_dense_vector(df), 20)
    # plt.plot(np.abs(np.diff(var)))
    # plt.savefig('screePlot.pdf')

    # Correlation matrix
    # CorrelationMatrix(transform_to_dense_vector(df))

    # Compute PCA
    k = 2
    factors = pca(transform_to_dense_vector(df), k)
    fac = convert_factors(factors, k)

    correct_negative_diff = fns.udf(lambda diff: max(diff, 0.0), DoubleType())
    w = Window.orderBy("row_num")

    # For each asset, compute its VaR
    def f(col, df, k, fac, w):
        # get the weighted asset return and join with pca factors
        fac, df_all = get_weighted_asset_return(df, col, fac, w)

        # Regress each asset return against pca factors
        vectorAssembler = VectorAssembler(
            inputCols=[str(i) for i in range(1, k+1)], outputCol='features')
        vec = vectorAssembler.transform(df_all)
        model = get_model(vec)
        beta = get_coefficients(model)
        res = get_residuals(model)
        resstd = get_standardized_residuals(res)

        # join coefs and standardized residuals to get proportion of VaR relying on factors vs residuals
        beta_star = np.append(beta, resstd.select('std').first())
        resstd = resstd.drop('std')
        data = vec.select('label')
        Var_fm = vec.approxQuantile('label', [0.05], 0.001)

        # epsilon is apprx. using Silverman's rule of thumb (bandwidth selection)
        # the constant 2.575 corresponds to a triangular kernel
        data = data.withColumn('count', fns.lit(float(df.count())))
        power = data.count()**(-0.2)
        eps = data.select((2.575*fns.stddev('label') *
                           power).alias('eps')).collect()[0]['eps']
        data = data.withColumn('eps', fns.lit(eps))

        # compute marginal VaR as expected value of factor returns, such that the
        # asset return was incident in the triangular kernel region peaked at the
        # VaR value and bandwidth = epsilon.
        data = data.withColumn('var', fns.lit(Var_fm[0]))
        data = data.withColumn(
            'k_weight', 1 - (fns.abs(fns.col('label') - fns.col('var')) / fns.col('eps')))
        data = data.withColumn(
            'k_weight', correct_negative_diff(fns.col('k_weight')))
        resstd = resstd.withColumn(
            "row_num", fns.monotonically_increasing_id())
        resstd = resstd.withColumn("index", fns.row_number().over(w))
        data = data.withColumn("row_num", fns.monotonically_increasing_id())
        data = data.withColumn("index", fns.row_number().over(w))

        # join factors, residuals and k_weights
        data = fac.join(resstd, ['index']).join(data, ['index'])
        data = data.drop('row_num').drop('index')
        data = data.cache()

        col_names = data.columns
        for x in range(k+2):
            new_column_name = col_names[x]
            data = data.withColumn(new_column_name, (getattr(
                data, col_names[x]) * getattr(data, 'k_weight')))

        mVaR = data.select(*[fns.mean(c).alias(c)
                             for c in data.columns[:(k+1)]]).collect()

        # correction factor to ensure that sum(cVaR) = asset VaR
        cf = Var_fm[0] / np.sum(np.array(mVaR)*beta_star)
        mVaR = cf * np.array(mVaR)
        cVaR = mVaR * beta_star
        pcVaR = 100 * cVaR / Var_fm[0]
        return pcVaR, Var_fm

    with open('output1.txt', 'w') as out_file:
        for col in df.columns[:2]:
            pcVaR, Var_fm = f(col, df, k, fac, w)
            out_file.write('pcVaR: {}, Var_fm: {}\n'.format(pcVaR, Var_fm))


if __name__ == '__main__':
    main()
