#-*-coding:utf-8-*-
import numpy
import pandas as pd
import xlrd
import math

def loadExcel(dataPath):
    #返回的数据类型为dataFrame,导入的数据是存在标题行的
    ori_data = pd.read_excel(dataPath)
    #转化为numpy的展示方式，转化之后第一行消失
    ori_data = ori_data.values
    #增加标签栏
    label = ["编号","色泽","根蒂","敲声","纹理","脐部","触感","密度","含糖率","好瓜"]
    label_exist = [0,0,0,0,0,0,0,0,0,0]
    temp_data = []
    for i in range(len(ori_data)):
        temp_data.append([])
        for j in range(len(label)):
            temp_data[i].append(ori_data[i][j])
    #将数据转化到列表中
    ori_data = temp_data
    return ori_data, label,label_exist

def mostClass(result):
    #创建一个字典
    className = {}
    for name in result:
        #字典中的键值不存在，增加一个新的
        if name not in className.keys():
            className[name] = 0
        className[name] += 1
    #依据键值对进行排序
    sortedClass = sorted(className.iteritems(), reverse = True)
    #返回最多的类别
    return sortedClass[0][0]

def calEnt(ori_data):
    result = {}
    #得带对应类别的数目
    for data in ori_data:
        if data[-1] not in result.keys():
            result[data[-1]] = 0
        result[data[-1]]  += 1
    Ent = 0.0
    #依据公式进行熵的计算
    for key in result:
        Pk = float(result[key]/len(ori_data))
        Ent -= Pk* math.log(Pk,2)
    return Ent

#若使用属性划分，则划分出来的子集整理为小数据集
def spiltSet(value, i, ori_data, seq):
    sub_set = []
    if i==7 or i==8:
        if seq == 0:
            for j in ori_data:
                if j[i] < value:
                    temp = j[:i]
                    temp.extend(j[i + 1:])
                    sub_set.append(temp)
        elif seq == 1:
            for j in ori_data:
                if j[i] >= value:
                    temp = j[:i]
                    temp.extend(j[i + 1:])
                    sub_set.append(temp)
    else:
        # 对应属性下的数据集
        for j in ori_data:
            if j[i] == value:
                temp = j[:i]
                temp.extend(j[i + 1:])
                sub_set.append(temp)

    return sub_set

def continue_value(sum_en,index,ori_data):
    conGain = 0.0
    classEnt = 0.0
    bestGain =0.0
    conFlag = 0
    feature = [example[index] for example in ori_data]
    sorted(feature)
    for i in range(len(feature)-1):
        midVal = float((feature[i]+feature[i+1])/2.0)
        for j in range(2):
            subSet = spiltSet(midVal,index,ori_data,j)
            # 总增益
            classEnt -= float(calEnt(subSet) * len(subSet) / len(ori_data))
        conGain = sum_en + classEnt
        if bestGain < conGain:
            bestGain = conGain
            conFlag = midVal
    return conGain, conFlag

def gain_choose(ori_data):
    #计算熵
    sum_enropy = calEnt(ori_data)
    temp_enropy = 0.0
    flag2 = -1
    #跳过编号，从第1个开始。因为编号的增益是最大的，使用编号之后算法停止，然而类别太多
    #不考虑最后一列的好瓜列表
    for i in range(1,len(ori_data[0])-1):
        feature = [example[i] for example in ori_data]
        feature = set(feature)
        classEnt = 0.0
        if i==7 or i==8:
            Gain_data, conFlag = continue_value(sum_enropy,i,ori_data)
        else:
            for value in feature:
                 #将属性的每一个值进行计算对应的增益
                 subSet = spiltSet(value, i, ori_data,-1)
                 #总增益
                 classEnt -= float(calEnt(subSet)*len(subSet)/len(ori_data))
            Gain_data = sum_enropy + classEnt
        #选出最大的增益
        if Gain_data>temp_enropy:
            flag = i;
            temp_enropy = Gain_data
            print("2flag")
            print (flag)
            print(flag2)
        if flag ==7 or flag == 8:
            flag2 = conFlag
    #返回最大增益属性
    return flag,flag2

#构建决策树
def createTree(ori_data,label,lEx):
    subLabels = []
    print(label)
    #每一组数据的结果保存在temp_result中
    temp_result = [result[-1] for result in ori_data]
    #如果只有一种结果，那么直接返回这一种结果，无需继续构建决策树
    if len(set(temp_result)) == 1:
        return temp_result [0]
    #若每组数据中的属性值都相同，则无须继续构建，直接返回所占比重最大的结果;或者没有属性值是无须构建
    #取除结果以外的数据进行对比
    temp_result2 = [result[:len(label)-1] for result in ori_data]
    flag = 1
    for i in range(len(temp_result2)):
        for j in range(len(temp_result2[0])):
            if temp_result2[i][j] != temp_result2[i+1][j]:
                flag=0
                break;
        if flag == 0:
            break;
    if flag == 1 or len(ori_data[0]) == 1:
        return mostClass(temp_result)
    #计算增益，返回增益最大的对应的属性
    best_Feature,flag2 = gain_choose(ori_data)
    bestLabel = label[best_Feature]
    decision_tree = {bestLabel:{}}
    lEx[best_Feature] = 1
    #del(label[best_Feature])
    print(bestLabel,flag2)
    if best_Feature == 7 or best_Feature==8:
        for j in range(2):
            for p in range(len(lEx)):
                if lEx[p] == 0:
                    subLabels.append(label[p])
            #递归进行划分
            decision_tree[bestLabel][flag2] = createTree(spiltSet(flag2, best_Feature, ori_data, j), subLabels, lEx)
    else:
        featValues = [example[best_Feature] for example in ori_data]
        # 得到属性下的各个值
        uniqueVals = set(featValues)
        for value in uniqueVals:
            for p in range(len(lEx)):
                if lEx[p] == 0:
                    subLabels.append(label[p])
            #递归进行划分
            decision_tree[bestLabel][value] = createTree(spiltSet(value, best_Feature, ori_data,-1), subLabels, lEx)
    return decision_tree


if __name__ == "__main__":
    train_path = "/home/stardream/DDML/MachineLearning/dataSet/watermelon3.0.xlsx"
    #加载表格中的数据
    dataSet, labels,l_exist = loadExcel(train_path)
    #创建树模型
    decisionTree = createTree(dataSet,labels,l_exist)
    print(decisionTree)


