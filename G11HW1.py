from pyspark import SparkContext, SparkConf
import sys
import os
import math
import random as rand


def fairFFT(P, k, kA, kB):
    assert k == kA + kB, "kA + kB != k"
    # [1, 2, "A"]
    S = [rand.choice(P)]
    if (S[0].split(",")[-1] == "A"): kA -=1
    else: kB -= 1
    # S[0] = S[0].split(",")[:-1]
    
    for _ in range(1, k):
        center = None
        maxDist = 0
        print(k ,kA, kB)
        
        for x in [x for x in P if x not in S]:
            if (
                (x.split(",")[-1] == "A" and kA <= 0) or
                (x.split(",")[-1] == "B" and kB <= 0)
            ):
                continue
            
            d = 0
            minDist = math.inf
            for c in S:
                for dimIdx, xi in enumerate(x.split(",")[:-1]):
                    d += (float(xi) - float(c.split(",")[dimIdx])) ** 2
                
                d = math.sqrt(d)
                if (d < minDist): 
                    minDist = d
            
            if minDist > maxDist:
                center = x
                maxDist = minDist
                
        if center.split(",")[-1] == "A": kA -= 1
        else: kB -= 1
        print(center, kA, kB)
        S.append(center)
    
    return S


def MRFairFFT():
    pass


def main():
    with open("testinputN32D2 copy.csv", "r") as f:
        P = f.read().strip().split("\n")
        print(P)
        print(fairFFT(P, 4, 2, 2))
        
         


if __name__ == "__main__":
    main()
