import sys

from pyspark import SparkContext, SparkConf
 
if __name__ == "__main__":
	 
	# create Spark context with Spark configuration
	conf = SparkConf().setAppName("Read Text to RDD - Python")
	sc = SparkContext(conf=conf)
	 

	text_file = sc.textFile("divina_commedia.txt")
	counts = text_file.flatMap(lambda line: line.split(" ")) \
	      			.map(lambda word: (word, 1)) \
	        		.reduceByKey(lambda a, b: a + b) \
	        		.coalesce(1)
	#counts.map(toCSVLine)
	counts.saveAsTextFile("counted_words")

	print("\n ciaone! \n")