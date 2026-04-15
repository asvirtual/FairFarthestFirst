from pyspark import SparkContext, SparkConf
import sys
import os
import math
import random as rand
import time


'''
    Compute dist(x1, x2) = sqrt( (x1[0] - x2[0])^2 + ... + (x1[d] - x2[d]) ^ 2 )
'''
def dist(x1, x2):
    assert len(x1[0]) == len(x2[0]), f"Incompatible dimensions between x1 = {x1[0]} and x2 = {x2[0]}"
    return math.sqrt(
        sum(
            [(x1[0][i] - x2[0][i]) ** 2 for i in range(len(x1[0]))]
        )
    ) 


def fairFFT(P, kA, kB):
    if len(P) == 0: return [] 
    
    totA = sum([1 for x in P if x[1] == "A"])
    totB = len(P) - totA
    
    sol = [rand.choice(P)]
    
    k = kA + kB 
    countA, countB = kA, kB # Keep track of "budgets" for the two labels
    
    # Edge case: the budget kL for label L is bigger than the number of points with label L in P
    #   In this case - (totL < kL) - we pick kL - totL points of the opposite label L'

    if totB < kB: 
        countB = totB
        countA += kB - totB
    
    if totA < kA:
        countA = totA
        countB += kA - totA

    if (sol[0][1] == "A"): countA -=1
    else: countB -= 1

    for _ in range(1, k):
        center = None
        maxDist = 0
        
        for x in [x for x in P if x not in sol]:
            # Current point has a label whose budget already ran out, it's not a legal center candidate
            if (
                (x[1] == "A" and countA <= 0) or 
                (x[1] == "B" and countB <= 0)
            ):
                continue
            
            # Compute dist(x, S) = min{ dist(x, c) for every c in S }
            d = math.inf
            for c in sol: d = min(dist(x, c), d)
            
            # Keep track of the farthest point (i.e. center candidate) from S accordingly to dist(x, S)
            if d > maxDist:
                center = x
                maxDist = d
        
        if center is not None: # To handle the case where we run out of points of a certain label
            if center[1] == "A": countA -= 1
            else: countB -= 1

            sol.append(center)
    
    return sol


def MRFairFFT(inputPoints, kA, kB):
    inputPoints = (inputPoints
        .mapPartitions(lambda it: fairFFT(list(it), kA, kB)) # R1 Reduce phase
        .collect()                                           # R2 Shuffle+Grouping
    )
    
    return fairFFT(inputPoints, kA, kB)                      # R2 Reduce phase


def computeRadius(inputPoints, sol):    
    return (inputPoints
        .map(lambda point: min([ dist(point, center) for center in sol ]))
        .mapPartitions(lambda x: [max(x, default=0)]) # max(x, default=0) to handle the case where a partition is empty
        .max())

def pointsetToFloat(inputPoint):
    components = inputPoint.split(",")        
    return ( [float(comp) for comp in components[:-1]], components[-1] )


def main():    
    assert len(sys.argv) == 5, "Wrong input params"

    # SPARK SETUP
    conf = SparkConf().setAppName('G11HW1')
    sc = SparkContext(conf=conf)
    
    # 1. Read number of partitions
    kA, kB, L = sys.argv[2], sys.argv[3], sys.argv[4]
    assert kA.isdigit() and kB.isdigit(), "K must be an integer"
    kA, kB, L = int(kA), int(kB), int(L)

    # 2. Read input file and subdivide it into K random partitions
    data_path = sys.argv[1]
    s = f"File path = {data_path}, KA = {kA}, KB = {kB}, L = {L}"
    #assert os.path.isfile(data_path), "File or folder not found"
    
    inputPoints = sc.textFile(data_path).repartition(numPartitions=L).cache().map(pointsetToFloat)

    N = inputPoints.count()

    labels_count = (inputPoints
        .map(lambda p: (p[1], 1))
        .reduceByKey(lambda a, b: a + b)
        .collectAsMap())
    
    numA = labels_count.get("A", 0)
    numB = labels_count.get("B", 0)

    s += f"\nN = {N}, NA = {numA}, NB = {numB}"
    
    
    start = time.perf_counter()
    sol = MRFairFFT(inputPoints, kA, kB)
    end = time.perf_counter()

    for center in sol:
        s += f"\nCenter = {center[0]} Label = {center[1]}"
    
    radius = computeRadius(inputPoints, sol)
    s += f"\nObjective function = {radius}"
    s += f"\nRunning time of MRFairFFT = {int((end - start)*1000)} ms"

    print(s)


if __name__ == "__main__":
    main()