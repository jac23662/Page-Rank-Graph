import pyspark
from pyspark.sql import *
# Build SparkSession
spark = pyspark.sql.SparkSession.builder.appName("MyApp").config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.2-s_2.12").getOrCreate()
spark.sparkContext.setCheckpointDir("/users/thomas1b/checkpoints")
import graphframes
from graphframes.examples import Graphs
from graphframes import *

import time
import sys
from pyspark.sql.types import StructField
from pyspark.sql.types import StructType
from pyspark.sql.types import StringType
from pyspark.sql.functions import lit, col, coalesce, sum, when
from graphframes.lib import Pregel

start_time = time.time()
#Grab input and output files from parameters
infile = (sys.argv[1])
outfile = (sys.argv[2])

#schema for the edge dataframe
schema = StructType([StructField("src", StringType(), True), StructField("dst", StringType(), True)])

#edges creation
# Reads in a tab seperated list file and removes comments, then stores the file as a dataframe
# this would be the edges dataframe with the source and destination as the header
edges = spark.read.format("csv").option("delimiter", "\t").option("comment", "#").option("header","false").schema(schema).load(infile)


#vertices creation
# based off of the edges src column, we get all the distinct values and create a vertices dataframe
# we renamed the column header as the id, for later use in graph creation
temp = edges.select(['src']).distinct()
vertices = temp.withColumnRenamed('src','id')

# save the number of vertices in the graph
numVertices = vertices.count()

#add the outDegrees of each vertices to the vertices dataframe for later use in the iterations
vertices = GraphFrame(vertices, edges).outDegrees

#Graph creation based on the vertices and the edges
g = GraphFrame(vertices, edges)

# page rank iteration
alpha = 0.15

#number of iterations setMaxIter is set to 10, as specified
ranks = g.pregel.setMaxIter(10).withVertexColumn("rank", lit(1.0 / numVertices), coalesce(Pregel.msg(), lit(0.0)) * lit(1.0 - alpha) + lit(alpha)).sendMsgToDst(Pregel.src("rank") / Pregel.src("outDegree")).aggMsgs(sum(Pregel.msg())).run()
#ranks.show()

# save the resulting dataframe with updated page ranks to output
ranks.write.mode("overwrite").format("csv").save(outfile)

print("----- %s seconds -----" % (time.time() - start_time))
print("\n\nsuccess\n\n")

