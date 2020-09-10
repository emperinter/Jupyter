
```python
```

# anaconda

# kaggle网址
> 一个Kaggle是一个数据建模和数据分析竞赛平台。企业和研究者可在其上发布数据，统计学者和数据挖掘专家可在其上进行竞赛以产生最好的模型。这一众包模式依赖于这一事实，即有众多策略可以用于解决几乎所有预测建模的问题，而研究者不可能在一开始就了解什么方法对于特定问题是最为有效的。

# numpy
矩阵等计算

- 查找xxxx的帮助文档
```python
print(help(numpy.xxxx))
```
- 构造多维数组
```python
numpy.array()
```

注意里面的值必须为同一类型，否则有类型转换；

eg:

```python
# 构造数组
# 一维数组
vector = numpy.array([5,10,15,20])
# 二维数组，注意有两个括号
matrix = numpy.array([[5,10,15],[20,25,30],[35,40,45]])
print(vector)
print(matrix)
```

> # 如何构造三维以上的数组是一个难点所在
> mumpy/0.ipynb/13*14

- 查看xxx的结构/用于debug
```python
print(xxxx.shape)
``` 

- 查询 
```python
print(xxxx[x,y])
print(xxxxx[,1]) # 取第一列
print(xxxx[:,0:2]) #第一和第二列
```

- 切片

```python
print(xxxx[n:m])
```

- 判断值是否存在
```python
xxx == m # xxx 中是否有m/会判断每一个值
```

- **整体值类型改变**



```python
v = numpy.array(["1","2","3"])
print(v.dtype)
print(v)
v = v.astype(float)
print(v.dtype)
print(v)
```

- 求极值

- 指定维度求值
```python
# 对行求值
m = numpy.array([
    [5,10,15],
    [20,25,30],
    [35,40,45]
])
m.sum(axis=1)
# answer:
# array([ 30,  75, 120])
```

```python
#对列求值
m = numpy.array([
    [5,10,15],
    [20,25,30],
    [35,40,45]
])
m.sum(axis=0)#answer:
#array([60, 75, 90])
```

- 变换为矩阵

```python
import numpy as np
print(np.arange(15))
a = np.arange(15).reshape(3,5)
a
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14]
# array([[ 0,  1,  2,  3,  4],
#        [ 5,  6,  7,  8,  9],
#        [10, 11, 12, 13, 14]])
```

- 求出array维度
```python
print(xxx.ndim)
```

- 求出array元素个数
```python
print(xxx.size)
```

- 初始化矩阵为0
```python
import numpy as np
np.zeros((3,4))
# array([[0., 0., 0., 0.],
#        [0., 0., 0., 0.],
#        [0., 0., 0., 0.]])
```

- 指定类型

```python
np.ones((2,3,4),dtype=np.int32)

# array([[[1, 1, 1, 1],
#         [1, 1, 1, 1],
#         [1, 1, 1, 1]],

#        [[1, 1, 1, 1],
#         [1, 1, 1, 1],
#         [1, 1, 1, 1]]])
```

- 得出一个序列
```python
np.arange(10,30,5)#该数>10 且< 30 从10开始每次加5
```

```python
np.arange(10,30,5).reshape(4,3)# 注意元素个数是否够用
# array([[10, 15],
#        [20, 25]])
```

- 随机模块
```python
np.random.random((2,3)) # 第一个random是调用模块，第二个是调用函数，(2,3)是构造一个2*3的矩阵
# array([[0.05134094, 0.63073588, 0.14218974],
#        [0.86727903, 0.95890848, 0.39738407]])
```

- 在一个区间上[x,y]平均间隔去取n个数
> np.linspace[x,y,m]

```python
np.linspace(2,3,5)
# array([2.  , 2.25, 2.5 , 2.75, 3.  ])
```

- 数学运算
```python
import numpy as np
a = np.array([20,30,40,50])
b = np.arange(4)
print(a)
print(b)
print("a - b " , a - b) # 对应位置相减
print("a - b - 1 :" , a - b - 1) 
print("b**2" , b**2)
print("a < 35" , a < 35)
# [20 30 40 50]
# [0 1 2 3]
# a - b  [20 29 38 47]
# a - b - 1 : [19 28 37 46]
# b**2 [0 1 4 9]
# a < 35 [ True  True False False]
```

- 矩阵乘法
```python
A = np.array([
    [1,1],
    [0,1]
])
B = np.array([
    [2,0],
    [3,4]
])
print('------A-------')
print(A)
print('------B-------')
print(B)
print('------A*B-------')
print(A*B) #对应位置相乘
print('------A.dot(B)-------')
print(A.dot(B)) # 矩阵乘法
print('------np.dot(A,B)-------')
print(np.dot(A,B)) # 也为矩阵乘法
# ------A-------
# [[1 1]
#  [0 1]]
# ------B-------
# [[2 0]
#  [3 4]]
# ------A*B-------
# [[2 0]
#  [0 4]]
# ------A.dot(B)-------
# [[5 4]
#  [3 4]]
# ------np.dot(A,B)-------
# [[5 4]
#  [3 4]]
```

- 数学公式
e/平法等等
```python
import numpy as np
B = np.arange(3)
print(B)
print(np.exp(B)) # e**B
print(np.sqrt(B)) # _/`B``
# [0 1 2]
# [1.         2.71828183 7.3890561 ]
# [0.         1.         1.41421356]
```

```python
import numpy as np
a = np.floor(10*np.random.random((3,4))) # np.floor() //向下取整
print(a)
print('-------------')
print(a.ravel()) # 把矩阵拉成向量
print('-------------')
a.shape = (3,4) # 把向量拉成矩阵
#
# a.shape = (3,-1)
# -1帮你自动计算后一个维度的个数
#
print(a)
print('-------------')
print(a.T) # 矩阵转置
# [[3. 5. 8. 6.]
#  [5. 6. 6. 7.]
#  [1. 6. 2. 5.]]
# -------------
# [3. 5. 8. 6. 5. 6. 6. 7. 1. 6. 2. 5.]
# -------------
# [[3. 5. 8. 6.]
#  [5. 6. 6. 7.]
#  [1. 6. 2. 5.]]
# -------------
# [[3. 5. 1.]
#  [5. 6. 6.]
#  [8. 6. 2.]
#  [6. 7. 5.]]
```

- 矩阵拼接
```python
# 矩阵拼接
import numpy as np
a = np.floor(10*np.random.random((2,2)))
b = np.floor(10*np.random.random((2,2)))
print('----------a-----------')
print(a)
print('----------b-----------')
print(b)
print('----------------------')
print(np.hstack((a,b))) # 按行拼接
print('----------------------')
print(np.vstack((a,b))) # 按列拼接
# ----------a-----------
# [[2. 0.]
#  [9. 7.]]
# ----------b-----------
# [[2. 0.]
#  [6. 9.]]
# ----------------------
# [[2. 0. 2. 0.]
#  [9. 7. 6. 9.]]
# ----------------------
# [[2. 0.]
#  [9. 7.]
#  [2. 0.]
#  [6. 9.]]
```

- 矩阵数据切割
```python
#数据分割
a = np.floor(10*np.random.random((2,12)))
print(a)
print('------------')
print(np.hsplit(a,3))  # 按行切分，3切分成3份，得到三个array值
print('------------')
print(np.hsplit(a,(3,4))) 
# split a after  the third and the fourth cloumn
# 在第三行和第四行后进行切割
print('------------')
a = np.floor(10*np.random.random((12,2)))
print(a)
print('-------------')
np.vsplit(a,3)  # 按列切分
# [[4. 3. 3. 3. 7. 5. 7. 4. 6. 4. 6. 8.]
#  [9. 9. 4. 8. 0. 4. 3. 5. 1. 9. 4. 4.]]
# ------------
# [array([[4., 3., 3., 3.],
#        [9., 9., 4., 8.]]), array([[7., 5., 7., 4.],
#        [0., 4., 3., 5.]]), array([[6., 4., 6., 8.],
#        [1., 9., 4., 4.]])]
# ------------
# [array([[4., 3., 3.],
#        [9., 9., 4.]]), array([[3.],
#        [8.]]), array([[7., 5., 7., 4., 6., 4., 6., 8.],
#        [0., 4., 3., 5., 1., 9., 4., 4.]])]
# ------------
# [[8. 2.]
#  [3. 9.]
#  [3. 5.]
#  [5. 0.]
#  [4. 3.]
#  [2. 3.]
#  [0. 2.]
#  [5. 7.]
#  [5. 5.]
#  [7. 9.]
#  [3. 8.]
#  [0. 0.]]
# -------------
# [array([[8., 2.],
#         [3., 9.],
#         [3., 5.],
#         [5., 0.]]), array([[4., 3.],
#         [2., 3.],
#         [0., 2.],
#         [5., 7.]]), array([[5., 5.],
#         [7., 9.],
#         [3., 8.],
#         [0., 0.]])]
```

- 复制

```python
# 复制/有俩种方法
# 浅复制
c = a.view() # 浅复制，共用一套值
print(c is a)
c.shape = (2,6)
print('a.shape: ' ,a.shape)
print('c.shape: ' ,c.shape)
c[0,4] = 1234    # a 的值也变量，a和c共用了一套值
print(a)
print(id(a))
print(id(c))
# False
# a.shape:  (3, 4)
# c.shape:  (2, 6)
# [[   0    1    2    3]
#  [1234    5    6    7]
#  [   8    9   10   11]]
# 2540538182992
# 2540538442256
#
#
#
# 深复制
d = a.copy()
print(d is a)
d[0,0] = 9999
print('------d-------')
print(d)
print('------a-------')
print(a)
# False
# ------d-------
# [[9999    1    2    3]
#  [1234    5    6    7]
#  [   8    9   10   11]]
# ------a-------
# [[   0    1    2    3]
#  [1234    5    6    7]
#  [   8    9   10   11]]
```

- 索引

```python
#索引
import numpy as np
data = np.sin(np.arange(20).reshape(5,4))
print(data)
ind = data.argmax(axis = 0) # 按列来进行计算
print(ind) # 输出每一列的最大值所在的行（以0开始），索引
data_max = data[ind,range(data.shape[1])] 
print(data_max)
# [[ 0.          0.84147098  0.90929743  0.14112001]
#  [-0.7568025  -0.95892427 -0.2794155   0.6569866 ]
#  [ 0.98935825  0.41211849 -0.54402111 -0.99999021]
#  [-0.53657292  0.42016704  0.99060736  0.65028784]
#  [-0.28790332 -0.96139749 -0.75098725  0.14987721]]
# [2 0 3 1]
# [0.98935825 0.84147098 0.99060736 0.6569866 ]
```

- 在行和列进行扩展
```python
# 扩展
import numpy as np
a = np.arange(0,40,10)
print(a)
b = np.tile(a,(3,5)) #构造一个三行五列的二维数组，每一个元素都是a
print(b)
# [ 0 10 20 30]
# [[ 0 10 20 30  0 10 20 30  0 10 20 30  0 10 20 30  0 10 20 30]
#  [ 0 10 20 30  0 10 20 30  0 10 20 30  0 10 20 30  0 10 20 30]
#  [ 0 10 20 30  0 10 20 30  0 10 20 30  0 10 20 30  0 10 20 30]]
```

- 排序

```python
#排序
import numpy as np
a = np.array([[4,3,5],
             [1,6,1],
             [0,2,3]])
print(a)
print('------按列排序-------')
b = np.sort(a,axis = 0) #对二维数组排序,0为按列排序，1为按行排序
print(b)
#b
a.sort(axis = 1)
print('--------按行排序-----') #对二维数组排序,0为按列排序，1为按行排序
print(a)

print('################')
a = np.array([5,3,1,2])
j = np.argsort(a)   # 索引，求最小值索引（编号）

print('-------最小值索引------')
print(j)
print('-------排序结果------')
print(a[j])   # 排序完之后的结果
# [[4 3 5]
#  [1 6 1]
#  [0 2 3]]
# ------按列排序-------
# [[0 2 1]
#  [1 3 3]
#  [4 6 5]]
# --------按行排序-----
# [[3 4 5]
#  [1 1 6]
#  [0 2 3]]
# ################
# -------最小值索引------
# [2 3 1 0]
# -------排序结果------
# [1 2 3 5]
```
































# pandas 
数据处理库

## 读取文件
> pandas.read_csv('xxxx.csv')

```python
import pandas as pd
m = pd.read_csv('m0.csv')
print(type(m))
print(m.dtypes)
# 字符值为object

# print(help(pd.read_csv))
# <class 'pandas.core.frame.DataFrame'>
# noteid        int64
# notebook     object
# username     object
# date         object
# dtype: object

```

## 显示数据
```python
m.head()  #把数据显示出来，默认显示前五条
# m.head(3) #显示前3条
# m.tail(4) # 显示后四行
# print(m.columns) # 显示列的指标
# print(m.shape) # 查看数据维度（m,n）=》表示总共有m个样本,每个样本有n个指标
```

## 取数据
```python
# location
### 取第一个数据
print(m.loc[0])
print('----------')
print(m.loc[3])

### 取从3开始到6结束
m.loc[3:6]

### 取1，3，5
id = {1,3,5}
m.loc[id]

### 一列一列取
col = m['noteid'] # 列名来定义，如无则默认为第一行为列名

print(col)

### 取多列
col = ['noteid','username']
data = m[col]
print(data)

# 取以什么来结尾的列
# 比如找以(kg)结尾的

import pandas as pd
m = pd.read_csv('m0.csv')

col_names = m.columns.tolist()  #把当前列名变成list
print(col_names)

gram_columns = []

for c in col_names:
    if c.endswith("(kg)"):    # 以(kg)结尾
        gram_columns.append(c)  #追加

gram_def = m[gram_columns]

print(gram_def.head(3))


# noteid                0
#  school               A
# username          adads
# height(cm)          160
# weight(kg)           40
# date          2019/8/27
# Name: 0, dtype: object
# ----------
# noteid                3
#  school               D
# username      zcvxvzxcv
# height(cm)          163
# weight(kg)           43
# date          2019/8/30
# Name: 3, dtype: object
```

## 运算

```python
# 四则运算
div_m = m['height(cm)'] / 100
print(div_m)


# 0     1.60
# 1     1.61
# 2     1.62
# 3     1.63
# 4     1.64
# 5     1.65
# 6     1.66
# 7     1.67
# 8     1.68
# 9     1.69
# 10    1.70
# Name: height(cm), dtype: float64

```


> 新加一列

```python
# 相同维度的运算
import pandas as pd
m = pd.read_csv('m0.csv')
data = m['height(cm)'] * m['weight(kg)']
print(data)
print(data.shape)
print('======新加一列======')
height_m = m['height(cm)'] / 100
m['height(m)'] = height_m  # 是添加了，但并未写入文件
print(m['height(m)'])   
print(m)
# 0     6400
# 1     6601
# 2     6804
# 3     7009
# 4     7216
# 5     7425
# 6     7636
# 7     7849
# 8     8064
# 9     8281
# 10    8500
# dtype: int64
# (11,)
# ======新加一列======
# 0     1.60
# 1     1.61
# 2     1.62
# 3     1.63
# 4     1.64
# 5     1.65
# 6     1.66
# 7     1.67
# 8     1.68
# 9     1.69
# 10    1.70
# Name: height(m), dtype: float64
#     noteid     school   username  height(cm)  weight(kg)       date  height(m)
# 0        0          A      adads         160          40  2019/8/27       1.60
# 1        1          B    sdfsadf         161          41  2019/8/28       1.61
# 2        2          C    sdfasdf         162          42  2019/8/29       1.62
# 3        3          D  zcvxvzxcv         163          43  2019/8/30       1.63
# 4        4          E     sdfasf         164          44  2019/8/31       1.64
# 5        5     支持v在v在     必胜德国法国         165          45   2019/9/1       1.65
# 6        6      在v秩序册     在v出租车v         166          46   2019/9/2       1.66
# 7        7     支持v在v从    在v从中选出v         167          47   2019/9/3       1.67
# 8        8  v自行车v自行车v     在v出租车v         168          48   2019/9/4       1.68
# 9        9      在v现在v    自行车v在v从         169          49   2019/9/5       1.69
# 10      10  豆腐干豆腐干大锅饭     在v自行车v         170          50   2019/9/6       1.70

```

## 排序
```python
import pandas as pd
m = pd.read_csv('m0.csv')

print('-------升序------')
m.sort_values("weight(kg)",inplace=True) # 默认ascending为Ture，用来升序
print(m["weight(kg)"])

print('-------降序------')

m.sort_values("weight(kg)",inplace=True,ascending=False) # 默认ascending为Ture，用来升序
print(m['weight(kg)'])

# -------升序------
# 0      40
# 1      41
# 3      43
# 4      44
# 6      46
# 7      47
# 8      48
# 9      49
# 10     50
# 5      66
# 2     888
# Name: weight(kg), dtype: int64
# -------降序------
# 2     888
# 5      66
# 10     50
# 9      49
# 8      48
# 7      47
# 6      46
# 4      44
# 3      43
# 1      41
# 0      40
# Name: weight(kg), dtype: int64
```                 


## NAN
pandas认为NAN为缺失值，或打印不出来的值。一般把缺失值放到最后


## 例子
```python
import pandas as pd
import numpy as np
data = pd.read_csv('titanic/train.csv') # https://www.kaggle.com/c/titanic/data
data.head()

#
# 数据内容
#
# pclass 仓位等级
# SibSp 兄弟姐妹数量
# Parch 父母/子女
# Fare 船票价格
# Cabin 床仓编号/NaN无该值
# Embarked 登船地点/码头

print('-------------')

age = data["Age"]
print(age.loc[0:10]) # 读取前十个值
print('@@@@@@@@@@@@@@@@@')
age_is_null = pd.isnull(age) # 判断缺失值，false则不是，ture则是缺失值
print(age_is_null)
print('@@@@@@@@@@@@@@@@')
age_null_true = age[age_is_null] # 筛选，这里传入的为（true/false）,把true的留下来
print(age_null_true)
print('@@@@@@@@@@@@@@@@')
age_null_count = len(age_null_true) # 当前长度
print(age_null_count)

# 未处理缺失值的情况
print('************未处理的情况**************')
mean_age = sum(data['Age']) / len(data['Age'])  # 有缺失值则结果为NaN,
print(mean_age)

# 处理后的情况
print('&&&&&&&&&&&&&&处理后的情况&&&&&&&&&&')
good_ages = data['Age'][age_is_null == False]
correct_mean_age = sum(good_ages) / len(good_ages)
print(correct_mean_age)

print('&&&&&&&&&&&&&&Pandas函数mean()来实现&&&&&&&&&&')
# pandas默认的函数来实现以上功能
correct_mean_age = data['Age'].mean()
print(correct_mean_age)

# 每个仓位等级的平均价格
print('每个仓位等级的平均价格')
passanger_classes = [1,2,3]
fares_by_class = {}
for this_class in passanger_classes:
    pclass_rows = data[data["Pclass"] == this_class ]
    pclass_fares = pclass_rows["Fare"]
    fare_for_class = pclass_fares.mean()
    fares_by_class[this_class] = fare_for_class
print(fares_by_class)

# pandas快速来实现
## 函数pivot_table
print('依靠函数pivot_table来实现上述功能')
passanger_survival = data.pivot_table(index="Pclass",values="Fare",aggfunc=np.mean)  # index:以谁为基准，values：index和什么的关系,aggfunc:指什么关系
print(passanger_survival)

print('================================')
# 
# 默认求平均值
#
passanger_age = data.pivot_table(index="Pclass",values="Age") # 求平均年龄，少写一个aggfun。按照默认求均值来操作
print(passanger_age)

print('++++++++++++++++++++++++++++++++')
port_stats = data.pivot_table(index="Embarked",values=["Fare","Survived"],aggfunc=np.sum) #一个量和其它两个量之间的关系
print(port_stats)

## 函数 dropna
print('==============dropna/把缺失值丢掉==================')
#specifying axis = 1 or axis="columns" will drop any columns that have null values
drop_na_columns = data.dropna(axis=1)  
print(drop_na_columns)
print('++==++')
new_passanger_survival = data.dropna(axis = 0,subset=['Age','Sex'])  # 如果这俩列有缺失值，则把有缺失值当前对应的行给去掉
print(new_passanger_survival)

print('^^^^^^^^^^^查找/定位到一个具体值^^^^^^^^^^^^^')
row_index_83_age = data.loc[83,"Age"]  # 83表示行，"Age"表示这一行"Age"这一列；一下同理
row_index_1000_pclass = data.loc[766,"Pclass"]
print(row_index_83_age)
print(row_index_1000_pclass)


# -------------
# 0     22.0
# 1     38.0
# 2     26.0
# 3     35.0
# 4     35.0
# 5      NaN
# 6     54.0
# 7      2.0
# 8     27.0
# 9     14.0
# 10     4.0
# Name: Age, dtype: float64
# @@@@@@@@@@@@@@@@@
# 0      False
# 1      False
# 2      False
# 3      False
# 4      False
#        ...  
# 886    False
# 887    False
# 888     True
# 889    False
# 890    False
# Name: Age, Length: 891, dtype: bool
# @@@@@@@@@@@@@@@@
# 5     NaN
# 17    NaN
# 19    NaN
# 26    NaN
# 28    NaN
#        ..
# 859   NaN
# 863   NaN
# 868   NaN
# 878   NaN
# 888   NaN
# Name: Age, Length: 177, dtype: float64
# @@@@@@@@@@@@@@@@
# 177
# ************未处理的情况**************
# nan
# &&&&&&&&&&&&&&处理后的情况&&&&&&&&&&
# 29.69911764705882
# &&&&&&&&&&&&&&Pandas函数mean()来实现&&&&&&&&&&
# 29.69911764705882
# 每个仓位等级的平均价格
# {1: 84.1546875, 2: 20.662183152173913, 3: 13.675550101832993}
# 依靠函数pivot_table来实现上述功能
#              Fare
# Pclass           
# 1       84.154687
# 2       20.662183
# 3       13.675550
# ================================
#               Age
# Pclass           
# 1       38.233441
# 2       29.877630
# 3       25.140620
# ++++++++++++++++++++++++++++++++
#                 Fare  Survived
# Embarked                      
# C         10072.2962        93
# Q          1022.2543        30
# S         17439.3988       217
# ==============dropna/把缺失值丢掉==================
#      PassengerId  Survived  Pclass  \
# 0              1         0       3   
# 1              2         1       1   
# 2              3         1       3   
# 3              4         1       1   
# 4              5         0       3   
# ..           ...       ...     ...   
# 886          887         0       2   
# 887          888         1       1   
# 888          889         0       3   
# 889          890         1       1   
# 890          891         0       3   

#                                                   Name     Sex  SibSp  Parch  \
# 0                              Braund, Mr. Owen Harris    male      1      0   
# 1    Cumings, Mrs. John Bradley (Florence Briggs Th...  female      1      0   
# 2                               Heikkinen, Miss. Laina  female      0      0   
# 3         Futrelle, Mrs. Jacques Heath (Lily May Peel)  female      1      0   
# 4                             Allen, Mr. William Henry    male      0      0   
# ..                                                 ...     ...    ...    ...   
# 886                              Montvila, Rev. Juozas    male      0      0   
# 887                       Graham, Miss. Margaret Edith  female      0      0   
# 888           Johnston, Miss. Catherine Helen "Carrie"  female      1      2   
# 889                              Behr, Mr. Karl Howell    male      0      0   
# 890                                Dooley, Mr. Patrick    male      0      0   

#                Ticket     Fare  
# 0           A/5 21171   7.2500  
# 1            PC 17599  71.2833  
# 2    STON/O2. 3101282   7.9250  
# 3              113803  53.1000  
# 4              373450   8.0500  
# ..                ...      ...  
# 886            211536  13.0000  
# 887            112053  30.0000  
# 888        W./C. 6607  23.4500  
# 889            111369  30.0000  
# 890            370376   7.7500  

# [891 rows x 9 columns]
# ++==++
#      PassengerId  Survived  Pclass  \
# 0              1         0       3   
# 1              2         1       1   
# 2              3         1       3   
# 3              4         1       1   
# 4              5         0       3   
# ..           ...       ...     ...   
# 885          886         0       3   
# 886          887         0       2   
# 887          888         1       1   
# 889          890         1       1   
# 890          891         0       3   

#                                                   Name     Sex   Age  SibSp  \
# 0                              Braund, Mr. Owen Harris    male  22.0      1   
# 1    Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   
# 2                               Heikkinen, Miss. Laina  female  26.0      0   
# 3         Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   
# 4                             Allen, Mr. William Henry    male  35.0      0   
# ..                                                 ...     ...   ...    ...   
# 885               Rice, Mrs. William (Margaret Norton)  female  39.0      0   
# 886                              Montvila, Rev. Juozas    male  27.0      0   
# 887                       Graham, Miss. Margaret Edith  female  19.0      0   
# 889                              Behr, Mr. Karl Howell    male  26.0      0   
# 890                                Dooley, Mr. Patrick    male  32.0      0   

#      Parch            Ticket     Fare Cabin Embarked  
# 0        0         A/5 21171   7.2500   NaN        S  
# 1        0          PC 17599  71.2833   C85        C  
# 2        0  STON/O2. 3101282   7.9250   NaN        S  
# 3        0            113803  53.1000  C123        S  
# 4        0            373450   8.0500   NaN        S  
# ..     ...               ...      ...   ...      ...  
# 885      5            382652  29.1250   NaN        Q  
# 886      0            211536  13.0000   NaN        S  
# 887      0            112053  30.0000   B42        S  
# 889      0            111369  30.0000  C148        C  
# 890      0            370376   7.7500   NaN        Q  

# [714 rows x 12 columns]
# ^^^^^^^^^^^查找^^^^^^^^^^^^^
# 28.0
# 1
```

### 排序
```python
## 排序
import pandas as pd
passanger_data = pd.read_csv('titanic/train.csv')
new_passanger_survival = passanger_data.sort_values("Age",ascending=False) # 按年龄来降序排序
print(new_passanger_survival)
passanger_reindex = new_passanger_survival.reset_index(drop=True) # 把index值（索引值）按降序规则来来重新排序
print('--------------------------')
print(passanger_reindex.loc[0:10])  # \ 表示换行显示


#      PassengerId  Survived  Pclass                                      Name  \
# 630          631         1       1      Barkworth, Mr. Algernon Henry Wilson   
# 851          852         0       3                       Svensson, Mr. Johan   
# 493          494         0       1                   Artagaveytia, Mr. Ramon   
# 96            97         0       1                 Goldschmidt, Mr. George B   
# 116          117         0       3                      Connors, Mr. Patrick   
# ..           ...       ...     ...                                       ...   
# 859          860         0       3                          Razi, Mr. Raihed   
# 863          864         0       3         Sage, Miss. Dorothy Edith "Dolly"   
# 868          869         0       3               van Melkebeke, Mr. Philemon   
# 878          879         0       3                        Laleff, Mr. Kristo   
# 888          889         0       3  Johnston, Miss. Catherine Helen "Carrie"   

#         Sex   Age  SibSp  Parch      Ticket     Fare Cabin Embarked  
# 630    male  80.0      0      0       27042  30.0000   A23        S  
# 851    male  74.0      0      0      347060   7.7750   NaN        S  
# 493    male  71.0      0      0    PC 17609  49.5042   NaN        C  
# 96     male  71.0      0      0    PC 17754  34.6542    A5        C  
# 116    male  70.5      0      0      370369   7.7500   NaN        Q  
# ..      ...   ...    ...    ...         ...      ...   ...      ...  
# 859    male   NaN      0      0        2629   7.2292   NaN        C  
# 863  female   NaN      8      2    CA. 2343  69.5500   NaN        S  
# 868    male   NaN      0      0      345777   9.5000   NaN        S  
# 878    male   NaN      0      0      349217   7.8958   NaN        S  
# 888  female   NaN      1      2  W./C. 6607  23.4500   NaN        S  

# [891 rows x 12 columns]
# --------------------------
#     PassengerId  Survived  Pclass                                  Name   Sex  \
# 0           631         1       1  Barkworth, Mr. Algernon Henry Wilson  male   
# 1           852         0       3                   Svensson, Mr. Johan  male   
# 2           494         0       1               Artagaveytia, Mr. Ramon  male   
# 3            97         0       1             Goldschmidt, Mr. George B  male   
# 4           117         0       3                  Connors, Mr. Patrick  male   
# 5           673         0       2           Mitchell, Mr. Henry Michael  male   
# 6           746         0       1          Crosby, Capt. Edward Gifford  male   
# 7            34         0       2                 Wheadon, Mr. Edward H  male   
# 8            55         0       1        Ostby, Mr. Engelhart Cornelius  male   
# 9           281         0       3                      Duane, Mr. Frank  male   
# 10          457         0       1             Millet, Mr. Francis Davis  male   

#      Age  SibSp  Parch      Ticket     Fare Cabin Embarked  
# 0   80.0      0      0       27042  30.0000   A23        S  
# 1   74.0      0      0      347060   7.7750   NaN        S  
# 2   71.0      0      0    PC 17609  49.5042   NaN        C  
# 3   71.0      0      0    PC 17754  34.6542    A5        C  
# 4   70.5      0      0      370369   7.7500   NaN        Q  
# 5   70.0      0      0  C.A. 24580  10.5000   NaN        S  
# 6   70.0      1      1   WE/P 5735  71.0000   B22        S  
# 7   66.0      0      0  C.A. 24579  10.5000   NaN        S  
# 8   65.0      0      1      113509  61.9792   B30        C  
# 9   65.0      0      0      336439   7.7500   NaN        Q  
# 10  65.0      0      0       13509  26.5500   E38        S  

```

## 自定义函数
```python
import pandas as pd
passanger_data = pd.read_csv('titanic/train.csv')

# this function returns the bundbreath(第一百行) item from a series
def hundredth_row(column):
    # Extract the hundredth item
    hundredth_items  = column.loc[99] # start form 0
    return hundredth_items

# Return the hundredth item from each column
hundredth_row = passanger_data.apply(hundredth_row)  # apply 用来申请执行
print(hundredth_row)

print()

# 返回每一个缺失值的个数
def not_null_count(column):
    column_null = pd.isnull(column)
    null = column[column_null]
    return len(null)

column_null_count = passanger_data.apply(not_null_count)
print(column_null_count)

# 
def which_class(row):
    pclass = row['Pclass']
    if pd.isnull(pclass):
        return 'Unknown'
    elif pclass == 1:
        return 'First Class'
    elif pclass == 2:
        return 'Second Class'
    elif pclass == 3:
        return 'Third Class'

classes = passanger_data.apply(which_class,axis = 1)  # axis ?? means what??
print(classes)

print()

def is_minor(row):
    if row['Age'] < 18:
        return True
    else:
        return False
minors = passanger_data.apply(is_minor,axis = 1)

# print minors

def generate_age_label(row):
    age = row['Age']
    if pd.isnull(age):
        return 'Unknown'
    elif age < 18:
        return 'minor'
    else:
        return 'adult'

age_labels = passanger_data.apply(generate_age_label,axis = 1)
print(age_labels)


print('--------------')
passanger_data['age_labels'] = age_labels
age_group_survival = passanger_data.pivot_table(index="age_labels",values="Survived")  #默认求各年龄获救的平局值
print(age_group_survival)

# PassengerId                  100
# Survived                       0
# Pclass                         2
# Name           Kantor, Mr. Sinai
# Sex                         male
# Age                           34
# SibSp                          1
# Parch                          0
# Ticket                    244367
# Fare                          26
# Cabin                        NaN
# Embarked                       S
# dtype: object

# PassengerId      0
# Survived         0
# Pclass           0
# Name             0
# Sex              0
# Age            177
# SibSp            0
# Parch            0
# Ticket           0
# Fare             0
# Cabin          687
# Embarked         2
# dtype: int64
# 0       Third Class
# 1       First Class
# 2       Third Class
# 3       First Class
# 4       Third Class
#            ...     
# 886    Second Class
# 887     First Class
# 888     Third Class
# 889     First Class
# 890     Third Class
# Length: 891, dtype: object

# 0        adult
# 1        adult
# 2        adult
# 3        adult
# 4        adult
#         ...   
# 886      adult
# 887      adult
# 888    Unknown
# 889      adult
# 890      adult
# Length: 891, dtype: object
# --------------
#             Survived
# age_labels          
# Unknown     0.293785
# adult       0.381032
# minor       0.539823
```

## Series
```python
## Import the Series object from pandas
## ???
from pandas import Series  
passanger_data = pd.read_csv('titanic/train.csv') 
series_files = passanger_data['Name']  # 其中的一列
Passanger_name = series_files.values # series该列里面的值

# print(type(Passanger_name))
# print(Passanger_name)
print('---------------')
series_rt = passanger_data['Ticket']
rt_ticket = series_rt.values
# print(rt_scores)
series_custom = Series(rt_ticket,index = Passanger_name) # 用名字当索引
# series_custom[['Odahl, Mr. Nils Martin','Jonkoff, Mr. Lalio']] # 打印  这和下面大打印只能存在一个，如都存在，这一个不会显示
print('################')
fiveten = series_custom[10:20]
print(fiveten)

## 排序
print('\n\n--------排序-------------\n')
series_files = passanger_data['PassengerId']  # 其中的一列
Passanger_id = series_files.values # series该列里面的值

series_age = passanger_data['Age']
Passanger_age = series_age.values

series_custom1 = Series(Passanger_age,index = Passanger_id)  # 把id当索引
original_index = series_custom1.index.tolist()
#print original_index
sorted_index = sorted(original_index)
sorted_by_index = series_custom1.reindex(sorted_index)
print(sorted_by_index)

print('----按index（键）排序---------')
sc2 = series_custom1.sort_index()  # 按键排序 
print(sc2)
print('\n----按values(值)排序---------') # 按值排序
sc3 = series_custom1.sort_values()
print(sc3)

print('===========================')
## 数学运算
# The value in a Series object are treated as an ndayyat,the core data type in Numpy
import numpy as np
# Add each value with each other
print(np.add(series_custom1,series_custom1))  # 值一样这之间相加，值不一样则对应相加????

# Apply sin function to each other
#
# np.sine(series_custom)
#
# module 'numpy' has no attribute 'sine'

# Return the hightest value (vill return a single value not a Series)
np.max(series_custom1)

# will actually return a Series object with a boolean value for each ticket
# > 50 返回一些true/false值
series_custom > 50
series_greater_than_50 = series_custom[series_custom > 50] # 拿true/false来返回值

series_one = series_custom > 50
series_two = series_custom < 75
both_criteria = series_custom[series_one & series_two]

print(both_criteria)


print('-=-=-=-=-=-=-=-=不同票价求平均值=-=-=-=-=-n\n\n\n\n\n')

# data alignment same index
# 不同票价求平均值
tt_critics = Series(passanger_data['ticket'].values,index=passanger_data['PassengerId'])
tt_users = Series(passanger_data['ticket2'].values,index=passanger_data['PassengerId'])      
tt_mean= (tt_critics + tt_users) /2
print(tt_mean)

# -------1--------
# #######2########
# Sandstrom, Miss. Marguerite Rut                            10
# Bonnell, Miss. Elizabeth                                   11
# Saundercock, Mr. William Henry                             12
# Andersson, Mr. Anders Johan                                13
# Vestrom, Miss. Hulda Amanda Adolfina                       14
# Hewlett, Mrs. (Mary D Kingcome)                            15
# Rice, Master. Eugene                                       16
# Williams, Mr. Charles Eugene                               17
# Vander Planke, Mrs. Julius (Emelia Maria Vandemoortele)    18
# Masselmani, Mrs. Fatima                                    19
# dtype: int64


# --------排序-------------

# 1      22.0
# 2      38.0
# 3      26.0
# 4      35.0
# 5      35.0
#        ... 
# 887    27.0
# 888    19.0
# 889     NaN
# 890    26.0
# 891    32.0
# Length: 891, dtype: float64
# ----按index（键）排序---------
# 1      22.0
# 2      38.0
# 3      26.0
# 4      35.0
# 5      35.0
#        ... 
# 887    27.0
# 888    19.0
# 889     NaN
# 890    26.0
# 891    32.0
# Length: 891, dtype: float64

# ----按values(值)排序---------
# 804    0.42
# 756    0.67
# 645    0.75
# 470    0.75
# 79     0.83
#        ... 
# 860     NaN
# 864     NaN
# 869     NaN
# 879     NaN
# 889     NaN
# Length: 891, dtype: float64
# ===========================
# 1      44.0
# 2      76.0
# 3      52.0
# 4      70.0
# 5      70.0
#        ... 
# 887    54.0
# 888    38.0
# 889     NaN
# 890    52.0
# 891    64.0
# Length: 891, dtype: float64
# -=-=-=-=-=-=-=-=-=-=-=-=-=-n





# Nosworthy, Mr. Richard Cater                          51
# Harper, Mrs. Henry Sleeper (Myna Haxtun)              52
# Faunthorpe, Mrs. Lizzie (Elizabeth Anne Wilkinson)    53
# Ostby, Mr. Engelhart Cornelius                        54
# Woolner, Mr. Hugh                                     55
# Rugg, Miss. Emily                                     56
# Novel, Mr. Mansouer                                   57
# West, Miss. Constance Mirium                          58
# Goodwin, Master. William Frederick                    59
# Sirayanian, Mr. Orsen                                 60
# Icard, Miss. Amelie                                   61
# Harris, Mr. Henry Birkhardt                           62
# Skoog, Master. Harald                                 63
# Stewart, Mr. Albert A                                 64
# Moubarek, Master. Gerios                              65
# Nye, Mrs. (Elizabeth Ramell)                          66
# Crease, Mr. Ernest James                              67
# Andersson, Miss. Erna Alexandra                       68
# Kink, Mr. Vincenz                                     69
# Jenkin, Mr. Stephen Curnow                            70
# Goodwin, Miss. Lillian Amy                            71
# Hood, Mr. Ambrose Jr                                  72
# Chronopoulos, Mr. Apostolos                           73
# Bing, Mr. Lee                                         74
# dtype: int64
# -=-=-=-=-=-=-=-=-=-=-=-=-=-n





# PassengerId
# 1        5.0
# 2        5.5
# 3        6.0
# 4        6.5
# 5        7.0
#        ...  
# 887    448.0
# 888    448.5
# 889    449.0
# 890    449.5
# 891    450.0
# Length: 891, dtype: float64
```

```python
import pandas as pd
# will return a new DataFrame that is indexed by the values in the specified column
# and will drop that cloumn from the DataFrame
# without the PannengerId dropped

# DataFrame来指定一个索引值

passenger_data = pd.read_csv('titanic/train.csv')
print(type(passanger_data) )
passenger_ticket = passanger_data.set_index('Name',drop=False) # 把ticket当成一个索引
print(passenger_ticket.index) # 打印index 值


#
# 目前怀疑是数据的问题，一下索引都失败了
# 具体问题详

print('\n\n\n\n=========================')
#Slice using either bracket notation or loc[]
passenger_data["Moran,Mr.James":"Sandstrom,Miss.Marguerite Rut"]

# Specific ticiket
passanger_data.loc["Moran,Mr.James":"Sandstrom,Miss.Marguerite Rut"]

# Select list of movies
tickets  = ["Sandstrom,Miss.Marguerite Rut","Moran,Mr.James","Rice,Master.Eugene"]
passenger_data.loc[tickets]
# <class 'pandas.core.frame.DataFrame'>
# Index(['Braund, Mr. Owen Harris',
#        'Cumings, Mrs. John Bradley (Florence Briggs Thayer)',
#        'Heikkinen, Miss. Laina',
#        'Futrelle, Mrs. Jacques Heath (Lily May Peel)',
#        'Allen, Mr. William Henry', 'Moran, Mr. James',
#        'McCarthy, Mr. Timothy J', 'Palsson, Master. Gosta Leonard',
#        'Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)',
#        'Nasser, Mrs. Nicholas (Adele Achem)',
#        ...
#        'Markun, Mr. Johann', 'Dahlberg, Miss. Gerda Ulrika',
#        'Banfield, Mr. Frederick James', 'Sutehall, Mr. Henry Jr',
#        'Rice, Mrs. William (Margaret Norton)', 'Montvila, Rev. Juozas',
#        'Graham, Miss. Margaret Edith',
#        'Johnston, Miss. Catherine Helen "Carrie"', 'Behr, Mr. Karl Howell',
#        'Dooley, Mr. Patrick'],
#       dtype='object', name='Name', length=891)




# =========================
# ---------------------------------------------------------------------------
# TypeError                                 Traceback (most recent call last)
# <ipython-input-31-6c844267ac99> in <module>
#      18 print('\n\n\n\n=========================')
#      19 #Slice using either bracket notation or loc[]
# ---> 20 passenger_data["PassengerId":"ticket"]
#      21 
#      22 # Specific ticiket

# F:\Software\PYTHON\lib\site-packages\pandas\core\frame.py in __getitem__(self, key)
#    2959 
#    2960         # Do we have a slicer (on rows)?
# -> 2961         indexer = convert_to_index_sliceable(self, key)
#    2962         if indexer is not None:
#    2963             return self._slice(indexer, axis=0)

# F:\Software\PYTHON\lib\site-packages\pandas\core\indexing.py in convert_to_index_sliceable(obj, key)
#    2356     idx = obj.index
#    2357     if isinstance(key, slice):
# -> 2358         return idx._convert_slice_indexer(key, kind="getitem")
#    2359 
#    2360     elif isinstance(key, str):

# F:\Software\PYTHON\lib\site-packages\pandas\core\indexes\base.py in _convert_slice_indexer(self, key, kind)
#    3188             if self.is_integer() or is_index_slice:
#    3189                 return slice(
# -> 3190                     self._validate_indexer("slice", key.start, kind),
#    3191                     self._validate_indexer("slice", key.stop, kind),
#    3192                     self._validate_indexer("slice", key.step, kind),

# F:\Software\PYTHON\lib\site-packages\pandas\core\indexes\base.py in _validate_indexer(self, form, key, kind)
#    5069             pass
#    5070         elif kind in ["iloc", "getitem"]:
# -> 5071             self._invalid_indexer(form, key)
#    5072         return key
#    5073 

# F:\Software\PYTHON\lib\site-packages\pandas\core\indexes\base.py in _invalid_indexer(self, form, key)
#    3338             "cannot do {form} indexing on {klass} with these "
#    3339             "indexers [{key}] of {kind}".format(
# -> 3340                 form=form, klass=type(self), key=key, kind=type(key)
#    3341             )
#    3342         )

# TypeError: cannot do slice indexing on <class 'pandas.core.indexes.range.RangeIndex'> with these indexers [PassengerId] of <class 'str'>
```

## 类型转换

```python
## 类型转换

# The apply() method in Pandas allows us to specify Python logic
# The apply() method requires you to pass in a vectorized operation
# that can be applied over each Series object.
import numpy as np
import pandas as pd

passenger_data = pd.read_csv('titanic/train.csv')

#returns the data types as a Series
types = passenger_data.dtypes
print(types)

# filter data types to just floats,index attributes returns just column names
float_columns = types[types.values == 'int64'].index
#use bracket notation to filter columns to just float columns
float_df = passenger_data[float_columns]
print(float_df)
# 'x' is a Series object representing a column
deviations =float_df.apply(lambda x: np.std(x))

print('--------------------------')

print(deviations)

print('\n\n命名函数lambda')
rt_mt_user = float_df[['ticket','ticket2']]
rt_mt_user.apply(lambda x:np.std(x),axis=1)  #对每一个指标算标准差
# PassengerId      int64
# Survived         int64
# Pclass           int64
# Name            object
# Sex             object
# Age            float64
# SibSp            int64
# Parch            int64
# Ticket          object
# Fare           float64
# Cabin           object
# Embarked        object
# ticket           int64
# ticket2          int64
# dtype: object
#      PassengerId  Survived  Pclass  SibSp  Parch  ticket  ticket2
# 0              1         0       3      1      0       0       10
# 1              2         1       1      1      0       1       10
# 2              3         1       3      0      0       2       10
# 3              4         1       1      1      0       3       10
# 4              5         0       3      0      0       4       10
# ..           ...       ...     ...    ...    ...     ...      ...
# 886          887         0       2      0      0     886       10
# 887          888         1       1      0      0     887       10
# 888          889         0       3      1      2     888       10
# 889          890         1       1      0      0     889       10
# 890          891         0       3      0      0     890       10

# [891 rows x 7 columns]
# --------------------------
# PassengerId    257.209383
# Survived         0.486319
# Pclass           0.835602
# SibSp            1.102124
# Parch            0.805605
# ticket         257.209383
# ticket2          0.000000
# dtype: float64


# 命名函数lambda
# 0        5.0
# 1        4.5
# 2        4.0
# 3        3.5
# 4        3.0
#        ...  
# 886    438.0
# 887    438.5
# 888    439.0
# 889    439.5
# 890    440.0
# Length: 891, dtype: float64
```