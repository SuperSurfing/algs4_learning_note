

## 摘要

&emsp;&emsp;本文是 Robert Sedgewick 的[《Algorithms 4th》](https://algs4.cs.princeton.edu/home/)的学习笔记。




## 网络资源

[CodeCogs - Latex](https://www.codecogs.com/latex/eqneditor.php?lang=zh-cn)

[Source Code Download Link](https://algs4.cs.princeton.edu/code/)

> Eclipse (manual). Download algs4.jar to a folder and add algs4.jar to the classpath variable to the build path of a project via Project -> Properties -> Java Build Path -> Libaries -> Add External JARs.

~~~
import edu.princeton.cs.algs4.*;

public class Hello {
    public static void main(String[] args) {
        System.out.println("Hello world");
        StdOut.print("Please intput a string: ");
        String s= StdIn.readString();//读入一个字符串
        StdOut.println("Hello World! 2018-08-20 "+s);
    }
}
~~~

[Algorithms 4th - B站](https://www.bilibili.com/video/av66547337/)

[算法导论 - B站](https://www.bilibili.com/video/av66468649?p=1)

[Test Driven Development](https://www.jetbrains.com/help/idea/tdd-with-intellij-idea.html)

[Junit 单元测试在 intelliJ IDEA 中的应用](https://blog.csdn.net/antony9118/article/details/51736135)

[IntelliJ IDEA 常用插件一览](https://juejin.im/entry/5c4f9f9d6fb9a049a5714c8c)

[Unit Testing with JUnit - Tutorial](https://www.vogella.com/tutorials/JUnit/article.html)

> A JUnit _test_ is a method contained in a class which is only used for testing. This is called a _Test class_. To define that a certain method is a test method, annotate it with the `@Test` annotation.

[Junit4 Getting Started](https://github.com/junit-team/junit4/wiki/Getting-started)


 [参考Coursera课程，Algorithm 4th 学习算法](https://blog.csdn.net/buzaidiannianguoqu/article/details/81870960)

[Algorithms, 4th Edition 算法4精华笔记](https://blog.csdn.net/garfielder007/article/details/96288978)

[algorithms-learn-note - github](https://github.com/iDube/algorithms-learn-note)




## Chapter1_Fundamentals

**Steps to developing a usable algorithm**:

- Model a problem
- Find an algorithm to solve it
- Fast enough ? Fits in memory ?
- If not, figure out why
- Find a way to address the problem
- Iterable until satisfied



### 1.1_Programming_Model


这一节主要介绍基本的 Java 语法，需要特别注意 Array 和二维数组。

~~~
double[][] a = new double[M][N]
~~~

注：**Java 中一切皆对象，包括 int, double, double[] 等都是对象**。

Euclid's Aglorithm for the Greatest Common Divisor

~~~
public static int gcd(int p, int q) {
    if (q == 0) return p;
    int r = p % q;
    return gcd(q, r);
}
~~~


Typical static methods:

~~~
#
~~~


[牛顿法和二分法平方根](https://www.jianshu.com/p/dcd73888ac3a)

```java
/**
 * 牛顿迭代法求平方根
 * @param  number   求值的数
 * @param  accuracy 精度
 * @return          Double
 */
public static double NewtonSqrt(double number, double accuracy) {
         //第一个猜测值
        double guess = number / 2;
        int count = 0;
        if (number < 0) {
            return Double.NaN;
        }
        //当两个猜测的差值大于精度即return
        while (Math.abs(guess - (number / guess)) > accuracy) {
            //迭代公式推导而成
            guess = (guess + (number / guess)) / 2;
            count++;
            System.out.printf("try count = %d, guess = %f\n", count, guess);
        }
        System.out.printf("final result = %f\n", guess);
        return guess;
    }
```

```java
    public static double DichotomySqrt(double number, double accuracy) {
        double higher = number;
        double lower = 0.0;
        double middle = (lower + higher) / 2;
        double last_middle = 0.00;
        int count = 0;
        if (number < 0) {
            return Double.NaN;
        }
        while (Math.abs(middle - last_middle) > accuracy) {
            if (middle * middle > number) {
                higher = middle;
            } else {
                lower = middle;
            }
            last_middle = middle;
            middle = (lower + higher) / 2;
            count++;
            System.out.printf("Dichotomy try count = %d, guess = %f\n", count, last_middle);
        }
        System.out.printf("Dichotomy final result = %f\n", last_middle);
        return last_middle;
    }
```

[Computing Square Roots with Newton's Method](https://pages.mtu.edu/~shene/COURSES/cs201/NOTES/chap06/sqrt-1.html)

[wiki - Newton sqrt](https://en.wikipedia.org/wiki/Newton%27s_method)

牛顿法的推导：

$$ x^{2}= a \Rightarrow x^{2}-a=0 $$

$$f(x) = x^{2}-a$$

![enter image description here](https://upload-images.jianshu.io/upload_images/1634490-52782329108a5301?imageMogr2/auto-orient/strip%7CimageView2/2/w/510/format/webp)


$$x_{n+1} = x_{n} - \frac{f(x_{n})}{f'(x_{n})} \Rightarrow x_{n+1} = \frac{1}{2}(x_{n}+\frac{a}{x})$$

上面这个就是递推公式。


练习题中的数学概念：

[二项式分布](https://zhuanlan.zhihu.com/p/24692791)

> 在单次试验中，结果A出现的概率为p，结果B出现的概率为q，p+q=1。那么在n=10，即10次试验中，结果A出现0次、1次、……、10次的概率各是多少呢？这样的概率分布呈现出什么特征呢？这就是二项分布所研究的内容。

$$b(x,n,p)=C_{n}^{x}p^{x}q^{n-x}$$

> 在大样本的情况下，二项分布的计算会很麻烦，这时可以采用正态分别来近似，其条件是np和n(1-p)都大于5。采用正态分布的参数为：

$$\mu = np, \rho =\sqrt{np(1-p)}$$

[正态分布 - wiki](https://zh.wikipedia.org/wiki/%E6%AD%A3%E6%80%81%E5%88%86%E5%B8%83)

[调和级数 - wiki](https://zh.wikipedia.org/wiki/%E8%B0%83%E5%92%8C%E7%BA%A7%E6%95%B0)

$$\sum_{k=1}^{\infty}\frac{1}{k}=1 + \frac{1}{2} + \frac{1}{3} + \frac{1}{4} + ...$$


#### Binomial

这个练习题演示了两种计算“二项式分布”的算法：一是递归法，一个是动态规划。

其中，递归法的递推公式可以从杨辉三角到二项式，再展开合并即可得到。

$$C_{N}^{k} = C_{N-1}^{k} + C_{N-1}^{k-1}$$

$$b(x,n,p)=C_{n}^{x}p^{x}q^{n-x}$$

DP 算法的思想如下：

1. 当 k = 0 时，公式可以简化
2. 将原先的递归计算倒过来，从杨辉三角型的顶部往底部计算，减少递归实例。
3. 两个算法的**递推公式相同**。


#### Sattolo's Algorithm

这是一个洗牌算法。

#### RightTriangle

需要注意，这个三角形的边长 3，4，5 。



### 1.2_Data_Abstraction

了解即可


### 1.3_Bags_Queues_Stacks

![enter image description here](https://algs4.cs.princeton.edu/13stacks/images/collection-apis.png)


注意：

这一章引入的三个基本 ADTs 都具有两个特点：

- Generics 泛型
- Iterable 可迭代


此外，这一节还引入了 Linked-list，并用它来实现上述的 ADTs

~~~
private class Node {
   Item item;
   Node next;
}
~~~


### 1.4_Analysis_of_Algorithms

这一章的内容非常重要，请结合 [Algs4 - Analysis of Algorithms](https://algs4.cs.princeton.edu/14analysis/) 认真学习。

一般对算法的分析包括两个方面：

- 时间复杂度（how long）
- 空间复杂度（memory）

科学的步骤如下：

- Observe some feature of the natural world, generally with precise measurements.（观察特征）
- Hypothesize a model that is consistent with the observations.（假设模型）
- Predict events using the hypothesis.（预测）
- Verify the predictions by making further observations.（验证）
- Validate by repeating until the hypothesis and observations agree.（反复校验和修正）



算法开发的步骤：

- Model the problem
- Find an algorithm to solve it
- Fast enough? Fits in the memory ?
- If not, figure out why
- Find a way to address the problem
- Iterate until satified


这一节同 TreeSum（三个数之和）和 DoublingTest （problem size 倍数增长）来观察：Time-size 的关系。

![enter image description here](https://algs4.cs.princeton.edu/14analysis/images/loglog.png)


很明显，这个算法的性能指数级的，取对数后，slope 为3，换句话说，它的时间复杂度为 O(n^3) 。


> Mathematical models. The total running time of a program is determined by two primary factors: **the cost of executing each statement** and **the frequency of execution of each statement**.

如书上所说，可以通过数学模型来分析算法的性能。简化数学模型，下面是

**Order-of-growth classifications**（N-阶导数）：

![enter image description here](https://algs4.cs.princeton.edu/14analysis/images/classifications.png)

> Separating the algorithm from the implementation on a particular
computer is a powerful concept because it allows us to develop knowledge about the
performance of algorithms and then apply that knowledge to any computer. 

算法分析的步骤：

- Develop an input model, including a definition of the problem size.
-  Identify the inner loop.
- Define a cost model that includes operations in the inner loop.
- Determine the frequency of execution of those operations for the given input.

其中，input model 可以简单地理解为输入参数。而 cost model 可以理解为“执行次数最多操作”。以 BinarySearch 算法为例，input model 是"输入数组的长度"，而 cost model 是 "compare the values of two array entries"。

此外，IO 操作也会影响算法的性能，因此需要将它考虑在内。“Coping with dependence on inputs.”

> Whitelist.  The input model is the N numbers in the whitelist and the M numbers
on standard input where we assume M >> N; the inner loop is the statements in
the single  while loop; the cost model is the compare operation (inherited from
binary search); and the analysis is immediate given the analysis of binary search—
the number of compares is at most M (lg N + 1).

上的算法估算，考虑到 M 次 IO 操作。

Proposition C - Doubling Ratio 实验（P207）表明 ，当输入规模翻倍时，程序时间 T 将趋近于 2^b 倍。比如，TreeSum 的时间复杂度是 N 的3次方。因此，当输入规模翻番时，T 将趋近于 8 倍。


练习题 6 的答案：

外层步长乘以 2 ，因此它可以表示为  $log_{2}N$，内存则是简单的 N，合起来就是

$$Nlog_{2}N$$

Creative_Problems：

第 4 题，关于 ThreeSum 和 FourSum 的思考：

1. TwoSum 可以先排序（Math.sort(), NlogN），然后迭代（N）并用 Binary Search 查找负值（logN），合计 NlogN 。
2. ThreeSum 同样先排序（NlogN）,然后双重循环迭代（N*N），循环内部用 Binary Search 查找负值（logN），合计 N*N*logN 。
3. FourSum 先双重循序计算两两之和（N*N），对 sum array 排序（2Nlog2N = 2N*(logN + 1) ），然后迭代（2N）并用 Binary Search 查找负值，合计 2N*(logN + 1) 。


第 20 题，BitonicMax：

1. 这个是 Binary Search 的变形，每次迭代，判断 mi 处在 increasing sequence 还是 decreasing sequence 。减而治之！

~~~
    // find the index of the maximum in a bitonic subarray a[lo..hi]
    public static int max(int[] a, int lo, int hi) {
        if (hi == lo) return hi;
        int mid = lo + (hi - lo) / 2;
        if (a[mid] < a[mid + 1]) return max(a, mid+1, hi);
        if (a[mid] > a[mid + 1]) return max(a, lo, mid);
        else return mid;
    }
~~~



### 递归变迭代

一般来说，尾递归（recurse call 是函数的最后一条语句）都可以很容易改为迭代。以 Binary Search 为例：

~~~
    public static int rank(int key, int[] a) {
        return rank(key, a, 0, a.length - 1);
    }

    public static int rank(int key, int[] a, int lo, int hi) {
        if (lo > hi) return -1;
        int mi = lo + (hi - lo) / 2;
        if (key < a[mi]) return rank(key, a, lo, mi);
        else if (key > a[mi]) return rank(key, a, mi, hi);
        else return mi;
    }
~~~

Binary Search 递归算法实际上是：（从中间）分割搜索空间，减而治之！

将这个递归改为迭代，核心思想是：（从两端）压缩搜索空间（lo <= hi），快速跳跃！

~~~
    public static int indexOf(int[] a, int key) {
        int lo = 0;
        int hi = a.length - 1;
        while (lo <= hi) {
            // Key is in a[lo..hi] or not present.
            int mid = lo + (hi - lo) / 2;
            if      (key < a[mid]) hi = mid - 1;
            else if (key > a[mid]) lo = mid + 1;
            else return mid;
        }
        return -1;
    }
~~~


### 线性回归

这一节讲解 LinearRegression.java 和 Polynomial.java 的大致思路。

[梯度向量与梯度下降法](https://blog.csdn.net/Sagittarius_Warrior/article/details/78365581)





### 1.5_Union_Find


