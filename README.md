# numpy-快速入门教程, 翻译自[numpy官网教程](https://docs.scipy.org/doc/numpy-dev/user/quickstart.html)

# 提前准备

在阅读本教程之前你应该基本了解Python,如果你想回顾,可以参看[Python教程](https://docs.python.org/3/tutorial/).

如果你想本地运行教程中的示例,你还必须安装包含numpy的python环境,请参阅[环境安装](http://scipy.org/install.html).

# 基础概念
NumPy的最重要的对象是齐次多维数组,它通常是一个数字类型的元素表，所有元素类型相同，由整数元组组成。在Numpy中, 数组的维度(dimensions)也被称为轴(axes),数组的轴的个数叫做数组的秩(rank)

例如: 在三维空间的一个坐标点`[1,2,1]`是一个秩(rank)为1的数组,因为它只有一个轴,而这个轴的长度是3

例如: 在下面的例子`[[1.,0.,0.],[0.,1.,2.]]`中,数组的秩为2(它有两个维度).第一个维度长度为2,第二个维度长度为3 (类似2*3的矩阵)
    
NumPy数组类叫`ndarray`,与`numpy.array`是同一个类。不过请注意, `numpy.array`和标准Python库类`array.array`不一样,`array.array`只处理一维数组,而且提供的功能也比较少。

`ndarray`类具有更多的属性:
### ndarray.ndim : 
    数组轴的个数，在python中，轴的个数被称作秩
### ndarray.shape : 
    指示数组在每个维度上大小的整数元组。例如一个n排m列的矩阵，它的shape属性将是(n,m),这个元组的长度显然是秩，即维度或者ndim属性
### ndarray.size : 
    数组的元素的总个数,也等于元素shape(n,m)的乘积(n*m). 
### ndarray.dtype : 
    描述数组中元素类型的对象，可以使用标准的Python类型或是指定的dtype类型。此外NumPy也提供自己的类型: numpy.int32、numpy.int16 numpy.float64
python交互式界面测试:
```python
>>> import numpy as np
>>> a = np.arange(15).reshape(3, 5)
>>> a
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14]])
>>> a.shape
(3, 5)
>>> a.ndim
2
>>> a.dtype.name
'int64'
>>> a.itemsize
8
>>> a.size
15
>>> type(a)
<type 'numpy.ndarray'>
>>> b = np.array([6, 7, 8])
>>> b
array([6, 7, 8])
>>> type(b)
<type 'numpy.ndarray'>
```

编写代码整体测试:
```python
#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np

array = [[1, 2, 3], [4, 5, 6]]
mydarray = np.array(array)

print('数组维度(轴)的个数:' + str(mydarray.ndim))
print('数组shape值:' + str(mydarray.shape))
print('数组元素的总个数:' + str(mydarray.size))
print('数组中元素类型:' + str(mydarray.dtype))
```
输出结果:
```python
数组维度(轴)的个数:2
数组shape值:(2, 3)
数组元素的总个数:6
数组中元素类型:int32
```

# 创建数组
有几种方法可以创建数组。
例如,可以从常规的Python列表创建一个数组, 或是array函数通过元组创建, 所创建的数组类型由原序列中的元素类型推导而来.
```python
>>> import numpy as np
>>> a = np.array([2,3,4])
>>> a
array([2, 3, 4])
>>> a.dtype
dtype('int64')
>>> b = np.array([1.2, 3.5, 5.1])
>>> b.dtype
dtype('float64')
```
 一个常见的错误是:传递了多个数值作为参数给`array`, 而不是提供一个由多个数值组成的列表。
 ```python
 >>> a = np.array(1,2,3,4)    # 错误
>>> a = np.array([1,2,3,4])  # 正确
 ```
`array`可以将 数组的数组 转为为二维数组, 将 数组的数组的数组 转为为三维数组, 一次类推.

```python
>>> b = np.array([(1.5,2,3), (4,5,6)])
>>> b
array([[ 1.5,  2. ,  3. ],
       [ 4. ,  5. ,  6. ]])
``` 
数组中值的类型可以在创建时指定
```python
>>> c = np.array( [ [1,2], [3,4] ], dtype=complex )
>>> c
array([[ 1.+0.j,  2.+0.j],
       [ 3.+0.j,  4.+0.j]])
```
通常，数组的元素开始都是未知的，但是它的大小已知。因此，NumPy提供了一些使用占位符创建数组的函数。这降低了扩展数组的所需消耗.
通过函数`zeros`可以创建一个全是0的数组; `ones`创建一个全1的数组; 函数`empty`创建一个内容随机并且依赖与内存状态的数组;默认创建的数组类型(dtype)都是`float64`.

```python
>>> np.zeros( (3,4) )
array([[ 0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.]])
>>> np.ones( (2,3,4), dtype=np.int16 )                # dtype 也可以指定
array([[[ 1, 1, 1, 1],
        [ 1, 1, 1, 1],
        [ 1, 1, 1, 1]],
       [[ 1, 1, 1, 1],
        [ 1, 1, 1, 1],
        [ 1, 1, 1, 1]]], dtype=int16)
>>> np.empty( (2,3) )                                 # uninitialized, output may vary
array([[  3.73603959e-262,   6.02658058e-154,   6.55490914e-260],
       [  5.30498948e-313,   3.14673309e-307,   1.00000000e+000]])
```
为了创建一个数列，NumPy提供一个类似arange的函数返回数组而不是列表:
```python
>>> np.arange( 10, 30, 5 )
array([10, 15, 20, 25])
>>> np.arange( 0, 2, 0.3 )                 # it accepts float arguments
array([ 0. ,  0.3,  0.6,  0.9,  1.2,  1.5,  1.8])
```
当`arange`接受浮点型参数时，由于浮点数精度有限，通常无法预测获得的元素的值。因此，最好使用函数`linspace`去接收我们想要的元素值,代替用range来指定步长。
```python
>>> from numpy import pi
>>> np.linspace( 0, 2, 9 )                 # 9 numbers from 0 to 2
array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ,  1.25,  1.5 ,  1.75,  2.  ])
>>> x = np.linspace( 0, 2*pi, 100 )        # useful to evaluate function at lots of points
>>> f = np.sin(x)
```

参见更多示例: array, zeros, zeros_like, ones, ones_like, empty, empty_like, arange, linspace, rand, randn, fromfunction, fromfile参考： [NumPy示例](https://docs.scipy.org/doc/numpy/reference/generated)  