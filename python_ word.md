1.改良递归：

```
import sys

sys.setrecursionlimit(20000)
```

这个用来改善递归深度，正常递归深度只有1000

```
from functools import lru_cache

@lru_cache(maxsize=None)  # 使用 LRU 缓存，maxsize=None 表示缓存大小不受限
def fibonacci(n):
```

这个是对递归实行记忆化。

2.直接接受全部信息：

```
import sys

# 重新定义 input 为 sys.stdin.read
input = sys.stdin.read

# 读取所有输入
data = input()

# 解析输入
lines = data.strip().split('\n')  # 将输入按行分割
n = int(lines[0])  # 第一行是整数 n
matrix = [list(map(int, line.split())) for line in lines[1:]]  # 剩下的行是矩阵数据

# 打印解析结果
print(f"n: {n}")
print("Matrix:")
for row in matrix:
    print(row)
```

3.dfs在oj中第一行加上一句：

```
# pylint: skip-file
```

4.保留小数：

```
number = 123.456789
formatted_number = f"{number:.2f}"  # 保留两位小数
print(formatted_number)  # 输出: 123.46
```

5.输入的if-else简化：

```
new_step = step + (0 if ma[nx][ny] == 1 else 1)
```

6.关于堆

```
以下是heapq的一些常用方法：

heapq.heappush(heap, item)：向堆中插入元素item。堆会自动调整以保持最小堆属性。

heapq.heappop(heap)：从堆中弹出并返回最小的元素。如果堆为空，则抛出异常。剩余的堆仍然保持最小堆属性。

heapq.heapify(x)：将列表x转换成堆，原地操作，时间复杂度为O(n)。

heapq.heappushpop(heap, item)：先执行heappush，然后立即执行heappop。这个组合操作比分别调用heappush和heappop要快。

heapq.heapreplace(heap, item)：先执行heappop，然后立即执行heappush。这个组合操作与heappushpop不同，因为它首先弹出了最小值，然后再添加新值。

heapq.nlargest(n, iterable[, key])：返回iterable中最大的n个元素组成的列表。key参数指定一个单参数排序函数，类似于sorted()的key参数。

heapq.nsmallest(n, iterable[, key])：返回iterable中最小的n个元素组成的列表。key参数同上。
```

7.关于字典的键

```
1.数字类型：
整数 (int)
浮点数 (float)
复数 (complex) （虽然不常见）
2.字符串 (str)：
字符串是不可变序列，因此非常适合用作字典的键。
3.元组 (tuple)：
如果元组包含的所有元素都是不可变且可哈希的，那么这个元组本身也是可哈希的，可以作为字典的键。
例如：(1, 2) 或 ('a', 'b') 是有效的键，但 (1, []) 不是，因为列表是可变的。
```

8.简化：

1.对于结果要取模的题，可以在运算过程中一直取，对最终结果不影响。

2.学会简化，比如，看到数值普遍是某个数的倍数，直接除以它；要求加和是某个数的倍数，直接给所有数减去他，等等。



9.不知道几个输入：

```
while True:
    try:
    except EOFError:
```

10.同时取元素和序号

```
enumerate(s)
```

11.bisect

```
以下是 bisect 模块中包含的主要函数：

bisect.bisect_left(a, x, lo=0, hi=len(a)):
查找在已排序列表 a 中元素 x 应该插入的索引位置，以保持列表有序。如果有相同元素，插入到左边。

bisect.bisect_right(a, x, lo=0, hi=len(a)) 或者 bisect.bisect(a, x, lo=0, hi=len(a)):
类似于 bisect_left，但是如果有相同元素，会将元素插入到右边。

bisect.insort_left(a, x, lo=0, hi=len(a)):
将元素 x 插入到已排序列表 a 中，保持列表有序。如果有相同元素，x 会被插入到左边。

bisect.insort_right(a, x, lo=0, hi=len(a)) 或者 bisect.insort(a, x, lo=0, hi=len(a)):
类似于 insort_left，但如果有相同元素，x 会被插入到右边。
参数 lo 和 hi 可选，用来指定搜索范围，缺省情况下是整个列表。
```

12.copy

```
import copy
#浅拷贝：
original_list = [1, 2, 3]
copied_list = copy.copy(original_list)
#深拷贝：
deep_copied_list = copy.deepcopy(original_list)
```

13.生成全排列：

```
import itertools

# 生成一个列表的全排列
elements = [1, 2, 3]
permutations = list(itertools.permutations(elements))
#结果：[(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)]

```

14.生成素数的欧拉筛：

```
def sieve_of_euler(limit):

    # 初始化布尔列表表示是否为素数，以及用于保存素数的列表

​    is_prime = [True] * (limit + 1)
​    primes = []
​    
​    for n in range(2, limit + 1):
​        if is_prime[n]:
​            primes.append(n)
​        
​        for p in primes:
​            if n * p > limit:
​                break
​            is_prime[n * p] = False
​            

            # 确保每个合数只被它的最小质因数筛去一次

​            if not (n % p):
​                break
​    
​    return primes
```



# 示例用法
print(sieve_of_euler(30))

15.不要忘了子数组和

16.变二进制：bin（x），但得到的是字符串并且前面还有ob

二转十：binary_str = "1010"
decimal_num = int(binary_str, 2) 

17.

```
calendar.monthrange(year, month)：返回一个包含两个整数的元组，第一个是该月第一天是一周中的哪一天（0-6），第二个是该月有多少天。
calendar.weekday(year, month, day)：返回给定日期对应的星期几，以数字表示（0-6）。
calendar.isleap(year)：判断给定年份是否为闰年。
calendar.leapdays(y1, y2)：计算从 y1 到但不包括 y2 年之间的所有闰年的总天数。
```

18.math

```
print(math.pow(2,3)) # 8.0
print(math.pow(2,2.5))
print(math.comb(5,3)) # 组合数，C53
print(math.factorial(5))5的阶乘
```

