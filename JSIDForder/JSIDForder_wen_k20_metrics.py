# -*- coding:utf-8 -*-
#由词的类和jsidf计算文本向量
from __future__ import division
from numpy import *
from compiler.ast import flatten
from sklearn import metrics
from sklearn.metrics import pairwise_distances
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def loadDocument(infile, outfile):
    fr = open(infile)
    stopkey = [line.strip() for line in open('stopword.txt','r').readlines()]
    doc = []
    # event = []
    for line in fr.readlines():
        words = line.strip().split()
        # print words[2]
        # event.append(words[2])
        wordlist = []
        for word in words:
            if not (word in stopkey):
                wordlist.append(word)
        doc.append(wordlist)
    fout = open(outfile, 'w')
    for line in doc:
        for word in line:
            fout.write(word + ' ')
        fout.write('\n')
    fr.close()
    fout.close()
    print 'finished loading document!'
    return doc


def JaccardSimilarityCoefficient(key, clustAssment, doc, k):
    dimWord = 0
    intersectWord = 0
    jaccard = mat(zeros((len(doc), k)))
    for i in range(len(doc)):
        for j in range(k):
            for m in range(clustAssment.shape[0]):
                if clustAssment[m, 0] == j:
                    dimWord += 1
                    if key[m] in doc[i]:
                        intersectWord += 1
            unionWord = len(doc[i]) + dimWord - intersectWord
            # print i, intersectWord, unionWord
            jaccard[i, j] = intersectWord/unionWord
            dimWord = 0
            intersectWord = 0
    print 'finished jaccard computing!'
    # print jaccard
    return jaccard

def idf(key, clustAssment, doc, k, Tgap = 2):
    result = mat(zeros((k, 1)))
    for i in range(k):
        wordlist = []
        for m in range(clustAssment.shape[0]):
            if clustAssment[m, 0] == i:
                wordlist.append(key[m])
        xlist = []
        for j in range(len(doc)):
            for word in wordlist:
                if word in doc[j]:
                    xlist.append(j)
                    break
        # print xlist
        sum = 0
        b = 0
        sig = 1
        if len(xlist) >= 2:
            for index in range(1, len(xlist)):
                sum += math.log(xlist[index] - xlist[index - 1] + 1)
            if xlist[index] - xlist[index - 1] > Tgap:
                b += 1
            sig = math.log((len(xlist) - b) ** 2 / (xlist[len(xlist) - 1] - xlist[0] + 1) + 2)
        # print 'b:', b, 'significance:', sig
        result[i, 0] = (math.log(len(doc) / (1 + sum))) * sig

    print 'finished idf computing!'
    return result

def jsidf(jaccard, idfvalue):
    result = mat(zeros(jaccard.shape))
    for j in range(jaccard.shape[1]):
        for i in range(jaccard.shape[0]):
            result[i, j] = jaccard[i, j] * idfvalue[j, 0]
    # print result
    print 'finished jsidf computing'
    return result

def loadClass(infile):
    fr = open(infile, 'r')
    # fout = open(outfile, 'w')
    classList = []
    key = []
    for line in fr.readlines():
        curLine = line.strip().split()
        key.append(curLine[0])
        classList.append(curLine[1])
    fr.close()
    dataMat = mat(zeros((len(classList), 2)))
    for i in range(len(classList)):
        dataMat[i, 0] = classList[i]
    # for i in range(len(classList)):
    #     fout.write(key[i] +' '+str(dataMat[i, 0])+'\n')
    print 'finished load class!'
    return key, dataMat

def docV():
    k = 100
    doc = loadDocument('wenzhou_out.txt', 'wenzhou_doc.txt')
    key, dataMat = loadClass('wenzhou_classes_sorted.txt')
    jaccard = JaccardSimilarityCoefficient(key, dataMat, doc, k)
    idfvalue = idf(key, dataMat, doc, k)
    docVec = jsidf(jaccard, idfvalue)

    savetxt('wenzhou_docVec.txt', docVec, fmt='%s', newline='\n')

    return docVec


def distCosine(vector1, vector2):
    return sum(multiply(vector1, vector2)) / sqrt(sum(power(vector1, 2))*sum(power(vector2, 2)))

# 随机生成初始的质心
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(array(dataSet)[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
    return centroids

def kMeans(dataSet, k, timelist, distMeas=distCosine, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))  # create mat to assign data points
    # to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  # for each data point assign it to the closest centroid
            minDist = -1
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI > minDist:
                    minDist = distJI;
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist
        # print centroids
        for cent in range(k):  # recalculate centroids
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]  # get all the point in this cluster
            centroids[cent, :] = mean(ptsInClust, axis=0)  # assign centroid to mean
    return centroids, clusterAssment

def biKmeans(dataSet, k, distMeas=distCosine):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList =[centroid0] #create a list with one centroid
    for j in range(m):#calc initial Error
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j, :])**2
    while (len(centList) < k):
        lowestSSE = -1
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]#get the data points currently in cluster i
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:,1])#compare the SSE to the currrent minimum
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            print "sseSplit, and notSplit: ",sseSplit,sseNotSplit
            if (sseSplit + sseNotSplit) > lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #change 1 to 3,4, or whatever
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print 'the bestCentToSplit is: ',bestCentToSplit
        print 'the len of bestClustAss is: ', len(bestClustAss)
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#replace a centroid with two best centroids
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#reassign new clusters, and SSE
    return mat(centList), clusterAssment

def RootMeanSquare(clusterAssing, k):
    P = 100
    sumsquare = clusterAssing.sum(axis=0)
    return sqrt(sumsquare[0,1]/(P*(clusterAssing.shape[0]-k)))

def Rsquared(clusterAssing, docVector):
    c = mean(docVector, axis=0)
    linesum=0
    for i in range(docVector.shape[0]):
        linesum += distCosine(docVector[i, :], c)
    sumsquare = clusterAssing.sum(axis=0)
    return (linesum - sumsquare[0,1])/linesum

def ModifiedHubertStatistic(docVector,clusterAssing,centroid):
    n = docVector.shape[0]
    factor1 = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                factor1 += distCosine(docVector[i, :], docVector[j, :]) * distCosine(centroid[int(clusterAssing[i, 0]), :], centroid[int(clusterAssing[j, 0]), :])
    return 2/((n-1)*n) * factor1

def getLabels(clusterAssing):
    labelist = []
    for i in range(clusterAssing.shape[0]):
        labelist.append(int(clusterAssing[i, 0]))
    labels = array(labelist)
    return labels

def CalinskiHarabazIndex(docVector, labels):
    return metrics.calinski_harabaz_score(array(docVector), labels)


def SilhouetteIndex(docVector, labels):
    return metrics.silhouette_score(array(docVector), labels, metric='cosine')

def findEventCenter():
    k = 20
    docVector = docV()
    myCentroids, clustAssing = biKmeans(docVector, k)

    labels = getLabels(clustAssing)

    root_mean_square = RootMeanSquare(clustAssing, k)
    R_squared = Rsquared(clustAssing, docVector)
    Modified_Hubert = ModifiedHubertStatistic(docVector, clustAssing, myCentroids)
    calinski = CalinskiHarabazIndex(docVector, labels)
    silhouette = SilhouetteIndex(docVector, labels)

    fmetrics = open("metrics.txt",'w')
    fmetrics.write('root_mean_square:'+str(root_mean_square)+'\n')
    fmetrics.write('R_squared:' + str(R_squared) + '\n')
    fmetrics.write('Modified_Hubert:' + str(Modified_Hubert) + '\n')
    fmetrics.write('calinski:' + str(calinski) + '\n')
    fmetrics.write('silhouette:' + str(silhouette))
    fmetrics.close()


    # docVector = mat([[0,1],[1,2],[2,3],[3,4]])
    # myCentroids = mat([[0,1],[3,3]])
    # clustAssing = mat([[0,0.1],[0,0.2], [1,0.1],[1,0.2]])

    centerlist = mat(zeros((shape(myCentroids)[0], 1)))
    minError = mat(zeros((shape(myCentroids)[0], 1)))
    for i in range(shape(myCentroids)[0]):
        minError[i, 0] = -1

    for j in range(shape(clustAssing)[0]):
        curError = clustAssing[j, 1]
        if curError > minError[clustAssing[j, 0], 0]:
            minError[int(clustAssing[j, 0]), 0] = curError
            centerlist[int(clustAssing[j, 0]), 0] = j

    clusterIndex = []
    for i in range(k):
        lineIndex = []
        for j in range(shape(clustAssing)[0]):
            if clustAssing[j, 0] == i:
                lineIndex.append(j)
        clusterIndex.append(lineIndex)

    clusterIndex2 = []
    cents = argsort(centerlist, axis=0)
    for i in range(cents.shape[0]):
        clusterIndex[cents[i, 0]].sort()
        clusterIndex2.append(clusterIndex[cents[i, 0]])

    event = []
    date = []
    fr = open('wenzhou.txt', 'r')
    for line in fr.readlines():
        words = line.strip().split()
        event.append(words[2])
        date.append(words[0])
    fr.close()

    # timelist = []
    # for i in range(len(date)):
    #     dt = date[i]
    #     timeArray = time.strptime(dt, "%Y年%m月%d日")
    #     timestamp = time.mktime(timeArray)
    #     timelist.append(timestamp)

    fout = open("wenzhou_docClass.txt", 'w')
    for i in range(k):
        fout.write('cluster: '+str(i)+' center:')
        fout.write(date[int(centerlist[cents[i, 0], 0])])
        fout.write(event[int(centerlist[cents[i, 0], 0])])
        fout.write(' '+str(len(clusterIndex2[i])))
        for j in range(len(clusterIndex2[i])):
            fout.write(' ' + date[clusterIndex2[i][j]])
            fout.write(event[clusterIndex2[i][j]])
        fout.write('\n')
    fout.close()

    f = open("clusterIndex.txt", 'w')
    for i in range(len(clusterIndex)):
        f.write(str(centerlist[i]) + ':')
        for j in range(len(clusterIndex[i])):
            f.write(str(clusterIndex[i][j]) + ' ')
        f.write('\n')

    print "The centual events are as follows:"
    print centerlist
    print clusterIndex

def main():
    findEventCenter()

if __name__ == '__main__':
    # docVector = mat([[1.1,0.3],[10,5],[1,0],[10,6]])
    # cluster = mat([[0, 0.2], [1, 0.01], [0, 0.9], [1, 0.99]])
    # centroid = mat([[1, 0], [10, 5.5]])
    # # a = ModifiedHubertStatistic(docVector, cluster, centroid)
    # # print a
    # labels =  getLabels(cluster)
    # b=CalinskiHarabazIndex(docVector,labels)
    # print b
    main()
