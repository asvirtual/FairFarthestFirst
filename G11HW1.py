from pyspark import SparkContext, SparkConf
import sys
import os
import math
import random as rand


def fairFFT(P, kA, kB):
    if len(P) == 0: return []
    
    S = [rand.choice(P)]
    k = kA + kB
    if (S[0][1] == "A"): kA -=1
    else: kB -= 1
    
    for _ in range(1, k):
        center = None
        maxDist = 0
        #print(k ,kA, kB)
        
        for x in [x for x in P if x not in S]:
            if (
                (x[1] == "A" and kA <= 0) or
                (x[1] == "B" and kB <= 0)
            ):
                continue
            
            minDist = math.inf
            for c in S:
                d = 0
                for dimIdx, xi in enumerate(x[0]):
                    d += (float(xi) - float(c[0][dimIdx])) ** 2
                
                d = math.sqrt(d)
                if (d < minDist): 
                    minDist = d
            
            if minDist > maxDist:
                center = x
                maxDist = minDist
        
        # print("Center:", center, "MaxDist:", maxDist)
        if center[1] == "A": kA -= 1
        else: kB -= 1
        
        if center is not None: # To handle the case where we run out of points of a certain label
            S.append(center)
    
    return S


def MRFairFFT(data, kA, kB):
    data = (data
        .mapPartitions(lambda it: fairFFT(list(it), kA, kB)) # R1 Reduce phase
        .collect()                                           # R2 Shuffle+Grouping
    )
    
    return fairFFT(data, kA, kB) # R2 Reduce phase


def formatPointset(dataPoint):
    components = dataPoint.split(",")
    floatComp = []
    for comp in components[:-1]:
        floatComp.append(float(comp))
        
    return ((floatComp, components[-1]))


def main():    
    assert len(sys.argv) == 5, "Wrong input params"

    # SPARK SETUP
    conf = SparkConf().setAppName('HW1')
    sc = SparkContext(conf=conf)
    
    # 1. Read number of partitions
    kA, kB, L = sys.argv[2], sys.argv[3], sys.argv[4]
    assert kA.isdigit() and kB.isdigit(), "K must be an integer"
    kA, kB, L = int(kA), int(kB), int(L)

    # 2. Read input file and subdivide it into K random partitions
    data_path = sys.argv[1]
    assert os.path.isfile(data_path), "File or folder not found"
    
    data = sc.textFile(data_path).repartition(numPartitions=L).cache().map(formatPointset)
    print(f"N = {data.count()}")
    
    labels_count = (data.map(lambda p: (p[1], 1))
        .reduceByKey(lambda a, b: a + b)
        .collectAsMap())
    
    numA = labels_count.get("A", 0)
    numB = labels_count.get("B", 0)

    print("A =", numA, "B =", numB)
    print(MRFairFFT(data, kA, kB))


if __name__ == "__main__":
    main()
