```


开酒店：

from bisect import bisect_left

def max_profit(n, k, m, p):

    # dp[i] 表示前 i 个地点能获得的最大利润

​    dp = [0] * (n + 1)

​    for i in range(1, n + 1):

        # 使用二分查找找到位置与 m[i-1] 至少差 k 的最大 j 值

​        j = bisect_left(m, m[i-1] - k)  # 找到第一个小于 m[i-1] - k 的位置

        # 在 i 位置开餐馆的最大利润 (开和不开的最大值)

​        dp[i] = max(dp[i-1], p[i-1] + dp[j])

​    return dp[n]

# 主程序处理输入

t = int(input())
for _ in range(t):
    n, k = map(int, input().split())
    m = list(map(int, input().split()))
    p = list(map(int, input().split()))
    print(max_profit(n, k, m, p)
```

开酒店：二分做法，这里的dp【i】指的是前i个能达到的最大的盈利。这是很合理的，因为每次更新都在和前一个比较，如果比他小了，就会被直接取代。而下面的：

```
def max_profit(n, k, loc, prof):
    dp = prof[:]  # Initialize dp with profits at each location

    for i in range(n):
        for j in range(i):
            if loc[i] - loc[j] > k:  # Check if the distance is greater than k
                dp[i] = max(dp[i], dp[j] + prof[i])

    return max(dp)

for _ in range(int(input())):
    n, k = map(int, input().strip().split())
    locations = list(map(int, input().strip().split()))
    profits = list(map(int, input().strip().split()))

    print(max_profit(n, k, locations, profits))
```

这里的dp【i】指的是‘如果在i处开店铺，他的盈利最大值’，所以在最后还要取max

```
from bisect import bisect_right
from collections import deque

def find_first_smaller(li1,a):  #输入一个列表和一个数
    index=bisect_right(li1,a)
    if index==0:
        return -1
    else:
        return index-1

n=int(input())
height=list(map(int,input().split()))
ganggan=deque()
ganggan.append(height[0])
for i in range(1,n):
    keyonggangan=find_first_smaller(ganggan,height[i])
    if keyonggangan==-1:
        ganggan.appendleft(height[i])
    else:
        ganggan[keyonggangan]=height[i]
print(len(ganggan))
```

这个代码用到了deque



```


n = int(input())
矩阵的保护圈：
s = [[401]*(n+2)]
mx = s + [[401] + [0]*n + [401] for _ in range(n)] + s 

dirL = [[0,1], [1,0], [0,-1], [-1,0]]

row = 1
col = 1
N = 0
drow, dcol = dirL[0]

for j in range(1, n*n+1):
    mx[row][col] = j
    if mx[row+drow][col+dcol]: #如果数值是0，为false；如果不是0，为true
        N += 1
        drow, dcol = dirL[N%4]
    
    row += drow
    col += dcol

for i in range(1, n+1):
    print(' '.join(map(str, mx[i][1:-1])))
```

boredom:

方法1：直接拉大数递归

```
input()
c = [0] * 100001
for m in map(int, input().split()):
    c[m] += 1

dp = [0] * 100001
for i in range(1, 100001):
    dp[i] = max(dp[i - 1], dp[i - 2] + i * c[i])

print(max(dp))
```

方法2：正统递归，对每个点设两个值，一个表示选了，一个表示没选，最后两个里选一个大的。

```
n=int(input())

num=list(map(int,input().split()))
num.sort()
t=num[-1]
chuxiancishu=[0]*(t+1)
for i in num:
    chuxiancishu[i]+=1
dp=[[0,0]for j in range(t+1)]  #分别表示选了i和不选i
dp[1][1]=chuxiancishu[1]
for i in range(2,t+1):
    dp[i][0]=max(dp[i-2][0]+chuxiancishu[i]*i,dp[i-2][1]+chuxiancishu[i]*i)
    dp[i][1]=max(dp[i-1][0],dp[i-1][1])
print(max(dp[t][0],dp[t][1]))
```

方法3：方法1的极致简化版

```
n = input()
s=[0]*100002
for i in map(int, input().split()):
    s[i] += i

a = b = 0
for d in s:
    a,b = max(a,b),a+d

print(max(a,b))
```



小偷背包类问题：

1.经典写法，二维矩阵

```
#第⼀步建⽴⽹格(横坐标表示[0,c]整数背包承重):(n+1)*(c+1)

def knapsack(n, c, w, p):

    cell = [[0 for j in range(c+1)]for i in range(n+1)]（表示取到第i个物品时，容量为j的背包能取的最大价值）

    for j in range(c+1):

        #第0⾏全部赋值为0，物品编号从1开始.为了下⾯赋值⽅便

        cell[0][j] = 0

    for i in range(1, n+1):

        for j in range(1, c+1):

            #⽣成了n*c有效矩阵，以下公式w[i-1],p[i-1]代表从第⼀个元素w[0],p[0]开始取。

            if j >= w[i-1]:

                cell[i][j] = max(cell[i-1][j], p[i-1] + cell[i-1][j - w[i-1]])

            else:

                cell[i][j] = cell[i-1][j]

    return cell

goodsnum, bagsize = map(int, input().split())

#goodsnum, bagsize = 3, 4

*value, = map(int, input().split())

*weight, = map(int, input().split())

#value, weight = [1500, 3000, 2000], [1, 4, 3] # guitar, stereo, laptop

cell = knapsack(goodsnum, bagsize, weight, value)

print(cell[goodsnum][bagsize])
```

2.简化版，一维数组：

```
def knapsack(bagsize: int, values: list, weights: list, dp: list):
    """
    动态规划解决0-1背包问题。
    
    :param bagsize: 背包的最大容量
    :param values: 每个物品的价值列表
    :param weights: 每个物品的重量列表
    :param dp: 用于存储最大价值的状态转移表，初始化为全0
    :return: 返回更新后的dp数组，其中dp[j]表示容量为j时的最大价值
    """
    
    # 遍历每个物品
    for i in range(len(values)):
        # 倒序遍历背包容量，从bagsize到当前物品的重量（包括）
        # 这样可以确保每个物品只被选择一次
        for j in range(bagsize, weights[i] - 1, -1):
            # 更新dp[j]，考虑两种情况：
            # 1. 不选当前物品，价值保持不变：dp[j]
            # 2. 选当前物品，价值增加：dp[j-weight[i]] + values[i]
            # 取两者中的最大值
            dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
    
    return dp

# 主程序开始
if __name__ == '__main__':
    # 读取输入数据
    goodsnum, bagsize = map(int, input().split())  # 物品数量和背包容量
    
    # 读取每个物品的价值和重量
    values = list(map(int, input().split()))
    weights = list(map(int, input().split()))
    
    # 初始化dp数组，长度为bagsize+1，默认值为0
    dp = [0] * (bagsize + 1)
    
    # 调用knapsack函数计算最大价值
    result_dp = knapsack(bagsize, values, weights, dp)
    
    # 输出结果，即dp数组中最后一个元素，代表最大容量下的最大价值
    print(max(result_dp))  # 注意这里使用max是为了处理bagsize为0的情况
```



可能的错误：

```
for i in range(1, n + 1):
    for l in range(0, W - w[i] + 1):
        f[l + w[i]] = max(f[l] + v[i], f[l + w[i]])
# 由 f[i][l + w[i]] = max(max(f[i - 1][l + w[i]], f[i - 1][l] + w[i]),
# f[i][l + w[i]]) 简化⽽来
```

问题：可能会一个数据加了好几次，所以我们才会反过来加。



对于完全背包以及可能存在的‘背包不能装满’问题：

```
INF=float("inf")

for _ in range(int(input())):
    a,b=map(int,input().split())
    all_weight=b-a
    dp=[0]+[INF]*all_weight
    coins=[]
    a=int(input())
    for _ in range(a):
        value,weight=map(int,input().split())
        coins.append((value,weight))

    for i in range(a):
        value,weight=coins[i]
        for j in range(weight,all_weight+1):
            if dp[j-weight]!=INF:
                dp[j]=min(dp[j],dp[j-weight]+value)

    if dp[-1]==INF:
        print('This is impossible.')
    else:
        print(f'The minimum amount of money in the piggy-bank is {dp[-1]}.')
```

可以使用无穷大来解决。就是只对不是无穷大的数来进行计算。

0—1背包，同样考虑“背包不能装满”问题：

```
def knapsack(bagsize_:int,weights_:list,values_:list):
    dp_=[-1]*(bagsize_+1)
    dp_[0]=0
    global n
    for j in range(len(weights_)):
        for t in range(bagsize_,weights_[j]-1,-1):
            if dp_[t-weights_[j]]!=-1:
                dp_[t]=max(dp_[t-weights_[j]]+values_[j],dp_[t])
    return dp_[bagsize_]


bagsize,n=map(int,input().split())
weights=[]
values=[]
for i in range(n):
    weight,value=map(int,input().split())
    weights.append(weight)
    values.append(value)


print(knapsack(bagsize,weights,values))
```

实际上和之前的是一样的，只不过加入了对“-1”的检验

打怪兽（heapq与堆）：

    from collections import defaultdict
    import heapq
    
    # 处理多个测试用例
    for _ in range(int(input())):
        # 读取输入数据
        n, m, b = map(int, input().split())
        d = defaultdict(list)
        
        # 收集每个时间点的所有伤害值，并维护一个大小为m的堆
        for _ in range(n):
            t, x = map(int, input().split())
            if len(d[t]) < m:
                heapq.heappush(d[t], x)  # 如果当前时间点的伤害列表长度小于m，则直接加入堆
            else:
                heapq.heappushpop(d[t], x)  # 否则，将新伤害值与堆顶元素比较，只保留最大的m个伤害值
        
        # 计算每个时间点的总伤害
        for t in d:
            d[t] = sum(d[t])
        
        # 按照时间点的发生顺序排序
        dp = sorted(d.items())
        
        # 应用伤害并检查生命值是否耗尽
        for t, damage in dp:
            b -= damage
            if b <= 0:
                print(t)
                break
        else:
            print('alive')



dfs水坑问题：

1.常规解法：

```
import sys
sys.setrecursionlimit(20000)

directions=[(1,1),(1,-1),(-1,1),(-1,-1),(0,1),(1,0),(0,-1),(-1,0)]

def dfs(x,y):
    field[x][y]='.'
    for dx,dy in directions:
        nx=x+dx
        ny=y+dy
        if 0<=nx<=n-1 and 0<=ny<=m-1 and field[nx][ny]=='W':
            dfs(nx,ny)

n,m=map(int,input().split())  #n为行数，m为列数

field=[list(input()) for _ in range(n)]

pools=0
for i in range(n):
    for j in range(m):
        if field[i][j]=='W':
            dfs(i,j)
            pools+=1
print(pools)
```

2.加快速度，可使用栈来代替dp：

```
directions=[(1,1),(1,-1),(-1,1),(-1,-1),(0,1),(1,0),(0,-1),(-1,0)]

def dfs(x,y):
    stack=[(x,y)]
    while stack:
        d,t=stack.pop()
        field[d][t]='.'
        for dx,dy in directions:
            nx=d+dx
            ny=t+dy
            if 0<=nx<=n-1 and 0<=ny<=m-1 and field[nx][ny]=='W':
                stack.append((nx,ny))

n,m=map(int,input().split())  #n为行数，m为列数

field=[list(input()) for _ in range(n)]

pools=0
for i in range(n):
    for j in range(m):
        if field[i][j]=='W':
            dfs(i,j)
            pools+=1
print(pools)
```

水坑问题的另一个变种：

```
#pylint: skip-file
directions=[(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,-1),(-1,1)]

def dfs(x,y):
    area=1
    maze[x][y]='.'
    for dx,dy in directions:
        nx,ny=x+dx,y+dy
        if 0<=nx<n and 0<=ny<m and maze[nx][ny]=='W':
            area+=dfs(nx,ny)
    return area

for i in range(int(input())):
    n,m=map(int,input().split())
    maze=[]
    for j in range(n):
        maze.append(list(input()))
    max_area=0
    for q in range(n):
        for w in range(m):
            if maze[q][w]=='W':
                max_area=max(max_area,dfs(q,w))
    print(max_area)
```

这里的重点是不要学迷宫的硬套，选择对每一个dfs都加入一个area来记数，否则会相当难思考





2.迷宫问题:

关键点：1.检查起始点是不是有障碍

​               2.走过的点要做标记以及注意回溯（使用栈的时候不用回溯，因为栈会把每一个可能的路线都加进栈里）

方法1：使用栈，并且直接在原矩阵里标记

```
# 读取迷宫的大小
n = int(input())

# 读取迷宫的每一行，并将其转换为二维列表
maze = [list(map(int, input().split())) for i in range(n)]

# 定义移动方向，这里只允许向右和向下移动
directions = [(0, 1), (1, 0)]

# 检查起点和终点是否为0，如果不是，直接输出"No"
if maze[0][0] != 0 or maze[-1][-1] != 0:
    print('No')
else:
    # 初始化栈，从起点 (0, 0) 开始
    stack = [(0, 0)]
    
    # 使用栈进行深度优先搜索
    while stack:
        # 弹出栈顶元素
        d, t = stack.pop()
        
        # 将当前格子标记为已访问
        maze[d][t] = 1
        
        # 如果到达终点，输出"Yes"并退出循环
        if d == t == n - 1:
            print('Yes')
            break
        else:
            # 遍历所有可能的移动方向
            for dx, dy in directions:
                nx, ny = d + dx, t + dy
                
                # 检查新位置是否在边界内且是可通行的（值为0）
                if 0 <= nx < n and 0 <= ny < n and maze[nx][ny] == 0:
                    stack.append((nx, ny))
    else:
        # 如果栈为空且未找到路径，输出"No"
        print('No')

```

这是方法2：加入一个检验矩阵

```
# 读取迷宫的大小
n = int(input())

# 读取迷宫的每一行，并将其转换为二维列表
maze = [list(map(int, input().split())) for _ in range(n)]

# 定义移动方向，这里只允许向右和向下移动
directions = [(0, 1), (1, 0)]

# 检查起点和终点是否为0，如果不是，直接输出"No"
if maze[0][0] != 0 or maze[-1][-1] != 0:
    print('No')
else:
    # 初始化检查矩阵，用于记录哪些位置已经被访问过
    visited = [[False] * n for _ in range(n)]
    
    # 初始化栈，从起点 (0, 0) 开始
    stack = [(0, 0)]
    
    # 使用栈进行深度优先搜索
    while stack:
        # 弹出栈顶元素
        current_x, current_y = stack.pop()
        
        # 将当前格子标记为已访问
        visited[current_x][current_y] = True
        
        # 如果到达终点，输出"Yes"并退出循环
        if current_x == n - 1 and current_y == n - 1:
            print('Yes')
            break
        else:
            # 遍历所有可能的移动方向
            for dx, dy in directions:
                next_x, next_y = current_x + dx, current_y + dy
                
                # 检查新位置是否在边界内且是可通行的（值为0），并且未被访问过
                if 0 <= next_x < n and 0 <= next_y < n and maze[next_x][next_y] == 0 and not visited[next_x][next_y]:
                    stack.append((next_x, next_y))
    else:
        # 如果栈为空且未找到路径，输出"No"
        print('No')
```

方法三：递归

```
def dfs(mx, visited, x, y):
    # 如果到达右下角，返回True
    if x == n - 1 and y == n - 1:
        return True

    # 定义向右和向下的移动方向
    directions = [(0, 1), (1, 0)]

    for dx, dy in directions:
        nx = x + dx
        ny = y + dy

        # 检查新坐标是否在矩阵范围内，是否已经访问过，以及是否可以通过
        if 0 <= nx < n and 0 <= ny < n and not visited[nx][ny] and mx[nx][ny] == 0:
            visited[nx][ny] = True
            if dfs(mx, visited, nx, ny):
                return True
            

    return False

# 读取输入
n = int(input())
mx = [list(map(int, input().split())) for _ in range(n)]

# 初始化访问标记数组
visited = [[False] * n for _ in range(n)]

# 检查起始点是否可以通过
if mx[0][0] == 1:
    print('No')
else:
    visited[0][0] = True
    if dfs(mx, visited, 0, 0):
        print('Yes')
    else:
        print('No')
```

走迷宫的路径数：

```
# 定义四个方向：右、下、左、上
directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]

# 全局计数器，用于记录从起点到终点的路径数量
path_count = 0

def dfs(current_x, current_y):
    global path_count

    # 如果当前格子是终点，计数器加一
    if maze[current_x][current_y] == 'e':
        path_count += 1
    else:
        # 遍历四个方向
        for dx, dy in directions:
            next_x, next_y = current_x + dx, current_y + dy
            # 检查新位置是否在迷宫范围内且不是障碍物
            if 0 <= next_x < rows and 0 <= next_y < cols and maze[next_x][next_y] != 1:
                # 标记当前格子为已访问
                maze[current_x][current_y] = 1
                # 递归调用DFS
                dfs(next_x, next_y)
                # 回溯，恢复当前格子的状态
                maze[current_x][current_y] = 0

# 读取迷宫的行数和列数
rows, cols = map(int, input().split())

# 读取迷宫的具体布局
maze = [list(map(int, input().split())) for _ in range(rows)]

# 将终点设置为 'e'
maze[rows - 1][cols - 1] = 'e'

# 从起点 (0, 0) 开始进行DFS
dfs(0, 0)

# 输出从起点到终点的路径数量
print(path_count)
```

这个就需要回溯了，属于dfs，因为在这里，我们既需要把上一个点标记，以防止走回去，又需要解除标记，否则这个点之后就会被挡住，其他的路径也走不了。

递归：波兰表达式：

递归做法:

```
def exp(poland):
    # 从列表中弹出第一个元素
    l = poland.pop(0)
    
    # 如果当前元素是运算符
    if l == '+':
        # 递归计算左操作数和右操作数，并返回它们的和
        return exp(poland) + exp(poland)
    elif l == '-':
        # 递归计算左操作数和右操作数，并返回它们的差
        return exp(poland) - exp(poland)
    elif l == '*':
        # 递归计算左操作数和右操作数，并返回它们的积
        return exp(poland) * exp(poland)
    elif l == '/':
        # 递归计算左操作数和右操作数，并返回它们的商
        return exp(poland) / exp(poland)
    else:
        # 如果当前元素是操作数，直接将其转换为浮点数并返回
        return float(l)

# 读取输入并分割成列表
poland_expression = input().split()

# 计算表达式的值
result = exp(poland_expression)

# 输出结果，保留六位小数
print(f'{result:.6f}')
```

使用栈：

```
from collections import deque

# 读取输入并分割成列表
input_expression = input().split()

# 初始化一个双端队列（栈）来存储操作数和中间结果
evaluation_stack = deque()

# 定义运算符列表
operators = ['+', '-', '*', '/']

# 从右到左遍历输入的波兰表达式
for i in range(len(input_expression) - 1, -1, -1):
    # 如果当前元素是运算符
    if input_expression[i] in operators:
        # 从栈中弹出两个操作数
        operand1 = evaluation_stack.pop()
        operand2 = evaluation_stack.pop()
        
        # 使用 eval 函数计算表达式的值，并将结果压入栈中
        result = eval(f"{operand1} {input_expression[i]} {operand2}")
        evaluation_stack.append(result)
    else:
        # 如果当前元素是操作数，直接将其转换为浮点数并压入栈中
        evaluation_stack.append(float(input_expression[i]))

# 输出最终结果，保留六位小数
for result in evaluation_stack:
    print(f'{result:.6f}')
```

k—tree

这道题的关键是，已经达成目标的值，我们要避免它之后的继续运算。最开始使用dp，没法阻止后续运算，所以最好改用递归+记忆化，可以节省时间

这是dp

```
n,k,d=map(int,input().split())  #要求的值总数，数的分叉数，最小的值
dp=[[[0,0]for j in range(k**i)] for i in range(n+1)]
num=0
for i in range(1,n+1):
    for t in range(len(dp[i])):
        if (t+1)%k!=0:
            dp[i][t][-1] = dp[i-1][t//k][-1]+(t+1)%k
            dp[i][t][0]=max(dp[i-1][t//k][0],(t+1)%k)
        else:
            dp[i][t][-1] = dp[i - 1][t // k][-1] +  k
            dp[i][t][0] = max(dp[i - 1][t // k][0], k)
        if dp[i][t][-1]==n and dp[i][t][0]>=d:
            num+=1
print((num%1000000007))
```

这是递归：

```
from functools import lru_cache
@lru_cache(maxsize=None)
def digui(c:int,max_d:int):

    if c==0 and max_d>=d:

        return 1
    elif c<0 or (c==0 and max_d<d):
        return 0
    else:
        a=0
        for i in range(1,k+1):
            t=max(max_d,i)
            a+=digui(c-i,t)
        return a





n,k,d=map(int,input().split())  #要求的值总数，数的分叉数，最小的值
num=0
num+=digui(n,0)
print((num%1000000007))
```

回文序列：（这个是力扣的，学输入格式）：

```
class Solution(object):
    def expandaroundcenter(self, s, left, right):
        while left>=0 and right<len(s) and s[left]==s[right]:
            left -= 1
            right += 1
        return left+1,right-1
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        start=0
        end=0
        for i in range(len(s)):
            s1,e1=self.expandaroundcenter(s,i,i)
            s2,e2=self.expandaroundcenter(s,i,i+1)
            if e1-s1>start-end :
                start=s1
                end=e1
            if e2-s2>end-start :
                start=s2
                end=e2
        return s[start:end+1]
solution = Solution()
print(solution.longestPalindrome("babad"))
```

最大子矩阵问题：

这里需要关注的点有二：kadena算法和输入方式

```

def max_submatrix(matrix1:list):
    def kadane(li1: list):
        max_now = li1[0]
        max_so_far = li1[0]
        for x in li1[1:]:
            max_now = max(x, max_now + x)
            max_so_far = max(max_so_far, max_now)
        return max_so_far
    len1=len(matrix1)
    max_matrix=float('-inf')
    for left in range(len1):
        temp = [0] * len1
        for right in range(left,len1):
            for k in range(len1):
                temp[k]+= matrix1[k][right]
            max_matrix=max(max_matrix,kadane(temp))
    return max_matrix

n=int(input())
nums=[]
while len(nums)<n*n:
    nums.extend(input().split())
ma=[list(map(int,nums[i*n:(i+1)*n])) for i in range(n)]
print(max_submatrix(ma))
```

核电站：

此题是比较重要的模板题

```
n,m=map(int,input().split())   #n个坑，m个连着就会炸
dp=[0]*(n+1)
dp[0]=1
for i in range(1,n+1):
    if i<m:
        dp[i]=dp[i-1]*2
    elif i==m:
        dp[i]=dp[i-1]*2-1
    else:
        dp[i]=dp[i-1]*2-dp[i-m-1]
print(dp[n])
```

土豪：

两个dp序列

```
values=list(map(int,input().split(',')))
dp1=[0]*len(values)
dp2=[0]*len(values)
dp1[0]=values[0]
dp2[0]=values[0]
for i in range(1,len(values)):
    dp1[i]=max(dp1[i-1]+values[i],values[i])
    dp2[i]=max(dp2[i-1]+values[i],values[i],dp1[i-1])
print(max(dp2))
```

孤岛之间距离：用于寻找两个代码块之间的距离，同时还有在找到一个目标之后直接停止双层循环的方法

```
import heapq

# 定义四个可能的移动方向：右、左、下、上
directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

# 获取矩阵的大小n，并初始化一个n*n的空列表maze用于存储输入的矩阵。
n = int(input())
maze = [[] for _ in range(n)]
for i in range(n):
    # 读取每一行的输入，并将其转换为整数列表后添加到maze中。
    a = input().strip()
    for j in range(len(a)):
        maze[i].append(int(a[j]))
m = len(maze[0])  # 获取矩阵的列数m

# 定义一个函数来查找值为1的第一个位置作为起点。
def find_start_position(maze):
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            if maze[i][j] == 1:
                return i, j  # 返回找到的第一个1的位置
    return None  # 如果没有找到任何1，则返回None

# 查找并设置起始点坐标(x0, y0)
x0, y0 = find_start_position(maze)

# 定义Dijkstra算法实现的函数。
def dijkstra(step: int, x: int, y: int) -> int:
    q = []  # 初始化优先队列（最小堆）
    visited = set()  # 创建一个集合记录已访问的节点
    
    # 将起始点及其步数(0)加入优先队列。
    heapq.heappush(q, (step, x, y))
    
    while q:
        step_now, x_now, y_now = heapq.heappop(q)  # 弹出并处理当前步数最少的节点
        
        if (x_now, y_now) in visited:
            continue  # 如果该节点已经被访问过，则跳过
        
        visited.add((x_now, y_now))  # 标记当前节点为已访问
        
        # 如果当前节点值为1且不是起点，则返回当前步数（即最短路径长度）。
        if maze[x_now][y_now] == 1 and step_now > 0:
            return step_now
        
        # 对于每个可能的移动方向，计算新位置的新坐标。
        for dx, dy in directions:
            nx, ny = x_now + dx, y_now + dy
            
            # 如果新位置在矩阵范围内且未被访问过，则将新位置及所需步数加入优先队列。
            if 0 <= nx < n and 0 <= ny < m and (nx, ny) not in visited:
                # 计算新位置的步数：如果新位置是0，则步数+1；如果是1，则步数不变。
                step_new = step_now + (1 if maze[nx][ny] == 0 else 0)
                
                # 将新位置及其步数加入优先队列。
                heapq.heappush(q, (step_new, nx, ny))
    
    return -1  # 如果没有找到其他的1，则返回-1表示不可达

# 调用dijkstra函数，并打印结果。
print(dijkstra(0, x0, y0))
```

哈希表：

这个实际上就是对字典的间的灵活运用

```
class Solution(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        from collections import defaultdict
        a=defaultdict(list)
        for s in strs:
            sorted_s = ''.join(sorted(s))
            a[sorted_s].append(s)
        return list(a.values())
if __name__ == '__main__':
    s = Solution()
    print(s.groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))
```

第二道：

while循环处非常巧妙

```
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        longest_streak = 0
        num_set = set(nums)

        for num in num_set:
            if num - 1 not in num_set:
                current_num = num
                current_streak = 1

                while current_num + 1 in num_set:
                    current_num += 1
                    current_streak += 1

                longest_streak = max(longest_streak, current_streak)

        return longest_streak
```

 Dijkstra

1.特殊应用，不是矩阵

```
mport heapq

def dijkstra(n, edges, s, t):
    # 构建图的邻接表表示法
    graph = [[] for _ in range(n)]  # 初始化一个空的邻接表
    for u, v, w in edges:
        graph[u].append((v, w))  # 无向图，因此两边都要添加
        graph[v].append((u, w))
    
    # 使用优先队列（最小堆）存储待处理节点及其当前距离
    pq = [(0, s)]  # (distance, node)，初始时将起点的距离设为0
    
    # 记录访问过的节点，避免重复处理
    visited = set()
    
    # 存储每个节点到起点的最短距离，初始化为无穷大
    distances = [float('inf')] * n
    distances[s] = 0  # 起点到自身的距离是0
    
    while pq:
        dist, node = heapq.heappop(pq)  # 取出距离最小的节点
        
        # 如果当前节点就是目标节点，则返回最短距离
        if node == t:
            return dist
        
        # 如果该节点已经被访问过，则跳过
        if node in visited:
            continue
        
        # 标记当前节点为已访问
        visited.add(node)
        
        # 遍历当前节点的所有邻居
        for neighbor, weight in graph[node]:
            # 如果邻居节点未被访问过
            if neighbor not in visited:
                new_dist = dist + weight
                
                # 如果找到了更短的路径到达邻居节点
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist  # 更新最短距离
                    heapq.heappush(pq, (new_dist, neighbor))  # 将更新后的邻居加入优先队列
    
    # 如果遍历结束没有找到目标节点，返回-1表示不可达
    return -1

# 示例：读取输入数据并调用dijkstra函数求解最短路径问题
if __name__ == "__main__":
    # 假设输入已经准备好，实际应用中应通过标准输入获取这些值
    n, m, s, t = map(int, input().split())  # 节点数n、边数m、起始节点s、目标节点t
    edges = [list(map(int, input().split())) for _ in range(m)]  # 每条边的信息（u, v, w）
    
    # 解决问题并打印结果
    result = dijkstra(n, edges, s, t)
    print(result)
```

并查集：

1.班级数量问题

```
# 并查集查找函数，带路径压缩优化
def find(x):
    # 如果当前节点不是根节点，则递归查找其父节点，并进行路径压缩
    if parent[x] != x:  
        parent[x] = find(parent[x])  # 路径压缩：将当前节点直接连接到根节点
    return parent[x]  # 返回根节点

# 并查集合并函数
def union(x, y):
    # 找到x和y各自的根节点，并将其中一个根节点设置为另一个根节点的父节点
    rootX = find(x)
    rootY = find(y)
    parent[rootX] = rootY  # 合并两个集合

# 读取输入数据
n, m = map(int, input().split())  # n是节点数，m是边数

# 初始化每个节点的父节点为自己，表示每个节点都是独立的集合
parent = list(range(n + 1))  # parent[i] == i 表示元素i是该集合的根结点

# 处理每一条边，将相连的节点所在的集合合并
for _ in range(m):
    a, b = map(int, input().split())  # 读取一条边连接的两个节点
    union(a, b)  # 合并这两个节点所在的集合

# 计算有多少个不同的连通分量
# 通过find函数找到每个节点所属的根节点，并加入到一个集合中，最终集合的大小就是连通分量的数量
classes = set(find(x) for x in range(1, n + 1))  # 对所有节点执行find操作，得到它们各自所属的根节点

# 输出结果：连通分量的数量
print(len(classes))
```

2.分别说出不同班级的人数

```
# 并查集查找函数，带路径压缩优化
def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])  # 路径压缩：将当前节点直接连接到根节点
    return parent[x]

# 并查集合并函数
def union(x, y):
    root_x = find(x)
    root_y = find(y)
    if root_x != root_y:
        parent[root_x] = root_y  # 合并两个集合
        size[root_y] += size[root_x]  # 更新合并后集合的大小

# 主程序开始
n, m = map(int, input().split())  # n是学生数量，m是关系数量

# 初始化每个学生的父节点为自己，表示每个学生都是独立的班级
parent = list(range(n + 1))  # parent[i] == i 表示元素i是该集合的根结点
size = [1] * (n + 1)  # 初始化每个学生的班级大小为1

# 处理每一条关系，将相连的学生所在的班级合并
for _ in range(m):
    a, b = map(int, input().split())  # 读取一对有关系的学生
    union(a, b)  # 合并这两个学生所在的班级

# 处理查询部分
k = int(input())  # 查询次数
queries = [tuple(map(int, input().split())) for _ in range(k)]  # 读取所有查询

# 输出查询结果
for a, b in queries:
    print("Yes" if find(a) == find(b) else "No")  # 如果两个学生的根节点相同，则它们在同一个班级

# 计算并打印班级数量和班级大小
classes = [size[x] for x in range(1, n + 1) if x == parent[x]]  # 统计每个根节点对应的班级大小
print(len(classes))  # 输出班级数量
print(' '.join(map(str, sorted(classes, reverse=True))))  # 输出按降序排列的班级大小
```

单调栈：

```
class Solution:
    def trap(self, height: List[int]) -> int:
        stack = []  # 用于存储柱子的索引，帮助我们找到左右边界
        water = 0  # 记录可以捕获的雨水总量

        for i in range(len(height)):
            # 当当前高度大于栈顶元素所指的高度时，意味着找到了一个可能的右边界
            while stack and height[i] > height[stack[-1]]:
                top = stack.pop()  # 弹出栈顶元素，它代表了一个局部低点
                
                if not stack:
                    break  # 如果栈为空，则没有左边界，无法形成容器盛水
                
                # 计算当前柱子与上一个未被弹出的栈顶元素之间的距离
                distance = i - stack[-1] - 1
                
                # 计算两个边界之间能够容纳的最大水高，即两边较低的高度减去局部低点的高度
                bounded_height = min(height[i], height[stack[-1]]) - height[top]
                
                # 将这一部分的水量加到总水量中
                water += distance * bounded_height
            
            # 将当前柱子的索引压入栈中，作为潜在的左边界或中间柱子
            stack.append(i)
        
        return water  # 返回总的接水量
```

滑动

```
rom collections import deque
from typing import List

class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        # 如果输入数组为空或k为0，直接返回空列表
        if not nums or k == 0:
            return []
        
        n = len(nums)
        
        # 如果窗口大小为1，则每个元素都是其自身的窗口最大值
        if k == 1:
            return nums
        
        # 初始化双端队列，用于存储索引，并确保队列中的值递减
        deque_index = deque()  
        res = []  # 存储结果的最大值
        
        for i in range(n):
            # 移除滑出窗口的元素（队首元素），即当队首元素的索引超出当前窗口范围时移除它
            if deque_index and deque_index[0] < i - k + 1:
                deque_index.popleft()
            
            # 移除所有小于当前元素的队尾元素，以保持队列中元素的递减顺序
            while deque_index and nums[deque_index[-1]] < nums[i]:
                deque_index.pop()
            
            # 将当前元素的索引加入队列
            deque_index.append(i)
            
            # 从第 k 个元素开始记录结果，因为只有这时窗口才完整形成
            if i >= k - 1:
                # 队首始终是窗口的最大值，因此将其添加到结果列表中
                res.append(nums[deque_index[0]])
        
        return res

if __name__ == "__main__":
    sol = Solution()
    print(sol.maxSlidingWindow([1,3,-1,-3,5,3,6,7], 3))  # 输出 [3,3,5,5,6,7]
```

题目：划分字母区间

```
class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        # 初始化一个长度为26的列表 'last'，用于记录每个字母最后出现的位置。
        # 假设输入字符串只包含小写字母，因此使用26个位置来表示'a'到'z'。
        last = [0] * 26
        
        # 遍历字符串 's'，记录每个字符最后出现的位置。
        for i, ch in enumerate(s):
            # 将字符转换为其在字母表中的索引（例如，'a' 对应 0，'b' 对应 1），并更新 'last' 列表中相应位置的值为当前索引 'i'。
            last[ord(ch) - ord("a")] = i

        # 初始化一个空列表 'partition'，用于保存每一段的长度。
        partition = list()
        
        # 初始化两个变量 'start' 和 'end'，都设置为0。
        # 'start' 表示当前段的起始位置，'end' 表示当前段的结束位置。
        start = end = 0
        
        # 再次遍历字符串 's'，这次是为了确定每一段的边界。
        for i, ch in enumerate(s):
            # 更新 'end' 为当前字符最后出现的位置和现有 'end' 的较大值。
            # 这确保了当前段包含所有与当前字符相关的部分。
            end = max(end, last[ord(ch) - ord("a")])
            
            # 如果当前索引 'i' 等于 'end'，说明已经找到了一个完整的段：
            if i == end:
                # 计算这一段的长度（'end - start + 1'），并将其添加到 'partition' 列表中。
                partition.append(end - start + 1)
                
                # 更新 'start' 为 'end + 1'，准备开始新的段。
                start = end + 1
        
        # 返回最终的 'partition' 列表，其中包含了所有段的长度。
        return partition
```

旋转矩阵：

```
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        将给定 n x n 二维矩阵原地旋转 90 度。

        参数:
        matrix (List[List[int]]): n x n 的二维矩阵，需要原地旋转。
        
        返回:
        None: 修改的是输入的矩阵本身，不返回任何值。
        """
        n = len(matrix)  # 获取矩阵的大小 n
        
        # 创建一个新的 n x n 矩阵用于存储旋转后的结果
        # 注意：不能使用 matrix_new = matrix 或 matrix_new = matrix[:]
        # 因为这两种方式都是浅拷贝，会使得 matrix_new 和 matrix 指向同一块内存，
        # 当修改 matrix_new 时，matrix 也会被修改，这不符合我们想要的效果。
        matrix_new = [[0] * n for _ in range(n)]
        
        # 遍历原始矩阵中的每一个元素，并根据旋转规则将其放置到新矩阵中对应的位置
        # 旋转规则是将原始矩阵中的第 i 行变为新矩阵中的第 n-i-1 列
        for i in range(n):
            for j in range(n):
                matrix_new[j][n - i - 1] = matrix[i][j]
        
        # 将新矩阵的内容复制回原始矩阵
        # 注意：这里使用 matrix[:] 而不是 matrix = matrix_new
        # 因为后者只是改变了局部变量 matrix 的指向，并不会影响到调用者传入的矩阵对象
        matrix[:] = matrix_new
```

建筑建设：

特殊的区间问题，对于可移动但必须包含一个变量的区间，可以把它们全部加入用来排序的集合中，因为在它们之间只可能选一个，其余的都会有重复。

```
def generate_intervals(x, width, m):
    temp = []
    for start in range(max(0, x-width+1), min(m, x+1)):
        end = start+width
        if end <= m:
            temp.append((start, end))
    return temp


n, m = map(int, input().split())
plans = [tuple(map(int, input().split())) for _ in range(n)]
intervals = []
for x, width in plans:
    intervals.extend(generate_intervals(x, width, m))
intervals.sort(key=lambda x: (x[1], x[0]))
排序，先按x【1】排，如果x【1】一样再按下【0】排。
cnt = 0
last_end = 0
for start, end in intervals:
    if start >= last_end:
        last_end = end
        cnt += 1
print(cnt)
```

这个是取了右端点来排序，目的是放下更多的区间。

这一道的目的是排满整个序列：

他就是从左侧开始排的，方法是选每个左端小于目前已经排满的右边的序列，取他们右端点的最大值。

```
import heapq
n=int(input())
li=list(map(int,input().split()))
monitor=[]
for i in range(n):
    heapq.heappush(monitor,(i-li[i],i+li[i]))
start=-1
end=n-1
cnt=0
while start<end and monitor:
    max_end=start
    while monitor and monitor[0][0]<=start+1:
        a,b=heapq.heappop(monitor)
        max_end=max(max_end,b)
    start=max_end
    cnt+=1
print(cnt)

```

### 区间问题

#### 1 区间合并

给出一堆区间，要求**合并**所有**有交集的区间** （端点处相交也算有交集）。最后问合并之后的**区间**。

![img](https://pic4.zhimg.com/80/v2-6e3bb59ed6c14eacfa1331c645d4afdf_1440w.jpg)

<center>区间合并问题示例：合并结果包含3个区间</center>

【**步骤一**】：按照区间**左端点**从小到大排序。

【**步骤二**】：维护前面区间中最右边的端点为ed。从前往后枚举每一个区间，判断是否应该将当前区间视为新区间。

假设当前遍历到的区间为第i个区间 [l_i,r_i]，有以下两种情况：

- l_i <=ed：说明当前区间与前面区间**有交集**。因此**不需要**增加区间个数，但需要设置 ed = max(ed, r_i)。

- l_i > ed: 说明当前区间与前面**没有交集**。因此**需要**增加区间个数，并设置 ed = max(ed, r_i)。

  ```python
  list.sort(key=lambda x:x[0])
  st=list[0][0]
  ed=list[0][1]
  ans=[]
  for i in range(1,n):
  	if list[i][0]<=ed:
          ed=max(ed,list[i][1])
      else:
          ans.append((st,ed))
          st=list[i][0]
          ed=list[i][1]
  ans.append((st,ed))
  ```

  

#### 2 选择不相交区间

给出一堆区间，要求选择**尽量多**的区间，使得这些区间**互不相交**，求可选取的区间的**最大数量**。这里端点相同也算有重复。

![img](https://pic1.zhimg.com/80/v2-690f7e53fd34c39802f45f48b59d5c5a_1440w.webp)

<center>选择不相交区间问题示例：结果包含3个区间</center>

【**步骤一**】：按照区间**右端点**从小到大排序。

【**步骤二**】：从前往后依次枚举每个区间。

假设当前遍历到的区间为第i个区间 [l_i,r_i]，有以下两种情况：

- l_i <= ed：说明当前区间与前面区间有交集。因此直接跳过。

- l_i > ed: 说明当前区间与前面没有交集。因此选中当前区间，并设置 ed = r_i。

  ```python
  list.sort(key=lambda x:x[1])
  ed=list[0][1]
  ans=[list[0]]
  for i in range(1,n):
  	if list[i][0]<=ed:
          continue
      else:
          ans.append(list[i])
          ed=list[i][1]
  ```

  

#### 3 区间选点问题

给出一堆区间，取**尽量少**的点，使得每个区间内**至少有一个点**（不同区间内含的点可以是同一个，位于区间端点上的点也算作区间内）。

![img](https://pica.zhimg.com/80/v2-a7ef021e1191ec53f20609ba870b44ba_1440w.webp)

<center>区间选点问题示例，最终至少选择3个点</center>



这个题可以转化为上一题的**求最大不相交区间**的数量。

【**步骤一**】：按照区间右端点从小到大排序。

【**步骤二**】：从前往后依次枚举每个区间。

假设当前遍历到的区间为第i个区间 [l_i,r_i]，有以下两种情况：

- l_i <=ed：说明当前区间与前面区间有交集，前面已经选点了。因此直接跳过。

- l_i > ed: 说明当前区间与前面没有交集。因此选中当前区间，并设置 ed = r_i。

  ```python
  list.sort(key=lambda x:x[1])
  ed=list[0][1]
  ans=[list[0][1]]
  for i in range(1,n):
  	if list[i][0]<=ed:
          continue
      else:
          ans.append(list[i][1])
          ed=list[i][1]
  ```

  

#### 4 区间覆盖问题

给出一堆区间和一个目标区间，问最少选择多少区间可以**覆盖**掉题中给出的这段目标区间。

如下图所示： 

![img](https://pic3.zhimg.com/80/v2-66041d9941667482fc51adeb4a616f64_1440w.webp)

<center>区间覆盖问题示例，最终至少选择2个区间才能覆盖目标区间</center>

【**步骤一**】：按照区间左端点从小到大排序。

**步骤二**】：**从前往后**依次枚举每个区间，在所有能覆盖当前目标区间起始位置start的区间之中，选择**右端点**最大的区间。

假设右端点最大的区间是第i个区间，右端点为 r_i。

最后将目标区间的start更新成r_i

```python
q.sort(key=lambda x:x[0])
#start,end 给定
ans=0
ed=q[0][1]
for i in range(n):
    if q[i][0]<=start<=q[i][1]:
        ed=max(ed,q[i][1])
        if ed>=end:
            ans+=1
			break
    else:
        ans+=1
        start=0
        start+=ed
```

#### 5 区间分组问题

给出一堆区间，问最少可以将这些区间分成多少组使得每个组内的区间互不相交。 

![img](https://pic2.zhimg.com/80/v2-6c6a045d481ddc44c66b046ef3e7d4cd_1440w.webp)

<center>区间分组问题示例，最少分成3个组</center>

【**步骤一**】：按照区间左端点从小到大排序。

【**步骤二**】：从**前往后**依次枚举每个区间，判断当前区间能否被放到某个现有组里面。

（即判断是否存在某个组的右端点在当前区间之中。如果可以，则不能放到这一组）

假设现在已经分了 m 组了，第 k 组最右边的一个点是 r_k，当前区间的范围是 [L_i,R_i] 。则：

如果L_i <r_k 则表示第 i 个区间无法放到第 k 组里面。反之，如果 L_i > r_k， 则表示可以放到第 k 组。

- 如果所有 m 个组里面没有组可以接收当前区间，则当前区间新开一个组，并把自己放进去。
- 如果存在可以接收当前区间的组 k，则将当前区间放进去，并更新当前组的 r_k = R_i。

**注意：**

为了能快速的找到能够接收当前区间的组，我们可以使用**优先队列 （小顶堆）**。

优先队列里面记录每个组的右端点值，每次可以在 O(1) 的时间拿到右端点中的的最小值。

```python
import heapq
list.sort(key=lambda x: x[0])
min_heap = [list[0][1]]    
for i in range(1, n):
    if list[i][0] >= min_heap[0]:
        heapq.heappop(min_heap)
    heapq.heappush(min_heap, list[i][1])
num=len(min_heap)
```

