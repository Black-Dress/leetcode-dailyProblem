import heapq
from collections import defaultdict
import sys


# 并查集
class ufset:
    def __init__(self, n: int):
        self.parent = [i for i in range(n)]
        # 按秩合并所需要的数组
        # rank = [0 for i in range(n)]

    def find(self, x) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # 路径压缩,将x所在的那一条树枝上所有的节点都连到根
        while x != root:
            origin = self.parent[x]
            self.parent[x] = root
            x = origin
        return root

    def merge(self, x, y):
        root_x, root_y = self.find(x), self.find(y)
        if root_x != root_y:
            self.parent[root_x] = root_y


class ufset_721:
    def __init__(self, accounts: [[str]]):
        self.parent = dict()
        for i in accounts:
            for j in range(1, len(i)):
                self.parent[i[j]] = i[j]

    def find(self, account: str) -> str:
        root = account
        while root != self.parent[root]:
            root = self.parent[root]
        # 路径压缩
        while account != root:
            origin = self.parent[account]
            self.parent[account] = root
            account = origin
        return root

    def merge(self, x: str, y: str):
        root_x, root_y = self.find(x), self.find(y)
        if root_x != root_y:
            self.parent[root_x] = root_y


class Solution:
    # 509
    def fib(self, n: int) -> int:
        if n < 2:
            return n
        res = [0 for i in range(n+1)]
        res[0], res[1] = 0, 1
        for i in range(2, n+1):
            res[i] = res[i-1]+res[i-2]
        return res[-1]

    # 830
    def largeGroupPositions(self, s: str) -> [[int]]:
        res = list()
        index = 1
        for i in range(1, len(s)):
            if s[i] == s[i-1]:
                index += 1
            else:
                index = 1
            if index == 3:
                res.append([i-2, i])
            if index > 3:
                res[-1][1] = i
        return res

    # 239
    def maxSlidingWindow(self, nums: [int], k: int) -> [int]:
        # 利用优先队列
        n = len(nums)
        q = [(-nums[i], i) for i in range(k)]
        heapq.heapify(q)
        res = [-q[0][0]]
        for i in range(k, n):
            heapq.heappush(q, (-nums[i], i))
            while q[0][1] <= i-k:
                heapq.heappop(q)
            res.append(-q[0][0])
        return res

    # 399
    def calcEquation(self, equations: [[str]], values: [float], queries: [[str]]) -> [float]:
        # 初始化equations
        graph = defaultdict(int)
        arrary = set()
        for i in range(len(equations)):
            a, b = equations[i]
            graph[(a, b)] = values[i]
            graph[(b, a)] = 1/values[i]
            arrary.add(a)
            arrary.add(b)
        for i in arrary:
            for j in arrary:
                for k in arrary:
                    if graph[(j, i)] and graph[(i, k)]:
                        graph[(j, k)] = graph[(j, i)] * graph[(i, k)]
        res = list()
        for x, y in queries:
            if graph[(x, y)]:
                res.append(graph[(x, y)])
            else:
                res.append(-1)
        return res

    # 547
    def findCircleNum(self, isConnected: [[int]]) -> int:
        # 利用floyd算法将间接到达的节点变为直接到达
        arrary = [i for i in range(len(isConnected))]
        for i in arrary:
            for j in arrary:
                for k in arrary:
                    if isConnected[j][i] and isConnected[i][k]:
                        isConnected[j][k], isConnected[k][j] = 1, 1
        visit = [0 for i in range(len(isConnected))]
        res = 0
        for i in range(len(isConnected)):
            if visit[i]:
                continue
            for j in range(len(isConnected[i])):
                visit[j] = 1 if isConnected[i][j] == 1 else visit[j]
            res += 1
        return res

    # 189 旋转数组
    def rotate(self, nums: [int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        k = k % n
        if n == 0 or k == 0:
            return
        temp, count = [nums[i] for i in range(k)], 0
        for i in range(k, n):
            temp[count % k], nums[i] = nums[i], temp[count % k]
            count += 1
        for i in range(k):
            temp[count % k], nums[i] = nums[i], temp[count % k]
            count += 1
        print(nums)

    # 123 买卖股票的最佳时机三
    def maxProfit(self, prices: [int]) -> int:
        if not prices:
            return 0
        n = len(prices)
        # dp[i][j] 第i天，第j状态所持有的金钱数量
        # 每一天总共有5个状态
        # 不操作，第一次买入，第一次卖出，第二次买入，第二次卖出
        dp = [[0]*5 for i in range(n)]
        # 初始化dp数组
        # 第0天的所有买入的操作都会让钱变少
        for i in range(1, 5, 2):
            dp[0][i] = -prices[0]
        # 填表
        for i in range(1, n):
            # 两次交易
            dp[i][0] = dp[i-1][0]
            for j in range(1, 5, 2):
                # 第j次买入 = max(不买，昨天最后一次买出的剩余金额-今天卖入的开销)
                dp[i][j] = max(dp[i-1][j], dp[i-1][j-1]-prices[i])
                # 第j次卖出 = max(不卖，昨天最后一次买入的股票的负债+今天卖出的价格)
                dp[i][j+1] = max(dp[i-1][j+1], dp[i-1][j]+prices[i])
        return max(dp[-1])

    # 228. 汇总区间
    def summaryRanges(self, nums: [int]) -> [str]:
        n = 0
        res = []
        while n < len(nums):
            if n+1 < len(nums) and nums[n]+1 == nums[n+1]:
                m = n
                while n+1 < len(nums) and nums[n]+1 == nums[n+1]:
                    n += 1
                res.append("{}->{}".format(nums[m], nums[n]))
            else:
                res.append(str(nums[n]))
            n += 1
        return res

    # 1202. 交换字符串中的元素
    def smallestStringWithSwaps(self, s: str, pairs: [[int]]) -> str:
        uf = ufset(len(s))
        for x, y in pairs:
            uf.merge(x, y)
        res = defaultdict(list)
        for i in range(len(s)):
            res[uf.find(i)].append(i)
        s = list(s)
        for i in res.values():
            string = sorted([s[j] for j in i])
            for j in range(len(i)):
                s[i[j]] = string[j]
        return "".join(s)

    # 1232.缀点成线
    def checkStraightLine(self, coordinates: [[int]]) -> bool:
        angle = cur = temp = 0
        for i in range(1, len(coordinates)):
            if coordinates[i][0] == coordinates[i-1][0]:
                # tan(180) = 0
                temp = 0
            elif coordinates[i][1] == coordinates[i-1][1]:
                # tan(90) = 无穷
                temp = -1
            else:
                temp = (coordinates[i][1]-coordinates[i-1][1])/(coordinates[i][0]-coordinates[i-1][0])
            if i < 2:
                angle = temp
            else:
                cur = temp
                if cur != angle:
                    return False
        return True

    # 721. 账户合并
    def accountsMerge(self, accounts: [[str]]) -> [[str]]:
        # 合并账户，通过并查集合并
        uf = ufset_721(accounts)
        index = defaultdict()
        # 记录没有邮箱的名字
        name = list()
        res = []
        temp = defaultdict(list)

        for account in accounts:
            if len(account) > 1:
                index[account[1]] = account[0]
            else:
                name.append([account[0]])
            for i in range(2, len(account)):
                index[account[i]] = account[0]
                uf.merge(account[1], account[i])

        for k in uf.parent.keys():
            temp[uf.find(k)].append(k)
        for k, v in temp.items():
            res.append([])
            if index.get(k):
                res[-1].append(index[k])
            res[-1].extend(sorted(v))
        res.extend(name)
        return res

    # 1584. 连接所有点的最小费用
    def minCostConnectPoints(self, points: [[int]]) -> int:
        n = len(points)
        path = [0]
        dis = [sys.maxsize for i in range(n)]
        dis[0] = -1
        res = 0

        def updateDistance(i: int):
            for j in range(n):
                if dis[j] > 0:
                    dis[j] = min(dis[j], abs(points[i][0]-points[j][0])+abs(points[i][1]-points[j][1]))

        updateDistance(0)
        while len(path) < n:
            minindex = [sys.maxsize, 0]
            for i in range(n):
                if dis[i] > 0:
                    minindex = minindex if minindex[0] < dis[i]else [dis[i], i]
            res += minindex[0]
            dis[minindex[1]] = -1
            updateDistance(minindex[1])
            path.append(minindex[1])
        return res

    # 628. 三个数的最大乘积
    def maximumProduct(self, nums: [int]) -> int:
        nums = sorted(nums, reverse=True)
        res = 0
        a, b = nums[0]*nums[1], nums[-1]*nums[-2]
        if nums[0] <= 0:
            return a*nums[2]
        if nums[2] >= 0:
            res = max(a*nums[2], b*nums[0])
        else:
            res = b*nums[0]
        return res

    # 1489. 找到最小生成树里的关键边和伪关键边
    def findCriticalAndPseudoCriticalEdges(self, n: int, edges: [[int]]) -> [[int]]:
        pass

    # 1319. 连通网络的操作次数
    def makeConnected(self, n: int, connections: [[int]]) -> int:
        # 利用并查集获得联通图
        uf = ufset(n)
        backSelect, parentNode = 0, dict()
        for connection in connections:
            if uf.find(connection[0]) != uf.find(connection[1]):
                # 若两个节点不在同一个区间，合并
                uf.merge(connection[0], connection[1])
            else:
                backSelect += 1
        for i in range(n):
            parent = uf.find(i)
            if parentNode.get(parent) is None:
                parentNode[parent] = 1
        return len(parentNode)-1 if len(parentNode)-1 <= backSelect else -1

    # 674. 最长连续递增序列
    def findLengthOfLCIS(self, nums: [int]) -> int:
        if len(nums) == 0:
            return 0
        res = [1]
        for i in range(1, len(nums)):
            if nums[i] > nums[i-1]:
                res[-1] += 1
            else:
                res.append(1)
        return max(res)

    # 959. 由斜杠划分区域
    def regionsBySlashes(self, grid: [str]) -> int:
        grid = [list(i) for i in grid]
        n = len(grid)
        uf = ufset(n*n*4)
        # 每一格的编号
        #       i+0
        # i+1            i+3
        #
        #       i+2
        for i in range(n):
            for j in range(n):
                index = (i*n+j)*4
                # 和右边的进行合并
                if j < n-1:
                    right = index+4
                    uf.merge(index+3, right+1)
                # 和下边合并
                if i < n-1:
                    bottom = index+n*4
                    uf.merge(index+2, bottom+0)
                if grid[i][j] == '/':
                    uf.merge(index+0, index+1)
                    uf.merge(index+3, index+2)
                elif grid[i][j] == '\\':
                    uf.merge(index+0, index+3)
                    uf.merge(index+1, index+2)
                else:
                    uf.merge(index+0, index+3)
                    uf.merge(index+3, index+2)
                    uf.merge(index+2, index+1)
        res = set()
        for i in range(n*n*4):
            res.add(uf.find(i))
        return len(res)

    # 1579. 保证图可完全遍历
    def maxNumEdgesToRemove(self, n: int, edges: [[int]]) -> int:
        edges = sorted(edges, key=lambda x: x[0], reverse=True)
        ufA, ufB = ufset(n+1), ufset(n+1)
        resA, resB = [], []
        temp1, temp2 = set(), set()
        # 得到刪除的邊，以及創建聯通圖
        for i in range(len(edges)):
            edge = edges[i]
            if edge[0] == 1:
                if ufA.find(edge[1]) != ufA.find(edge[2]):
                    ufA.merge(edge[1], edge[2])
                else:
                    resA.append(i)
            elif edge[0] == 2:
                if ufB.find(edge[1]) != ufB.find(edge[2]):
                    ufB.merge(edge[1], edge[2])
                else:
                    resB.append(i)
            else:
                if ufA.find(edge[1]) != ufA.find(edge[2]) and ufB.find(edge[1]) != ufB.find(edge[2]):
                    ufA.merge(edge[1], edge[2])
                    ufB.merge(edge[1], edge[2])
                else:
                    resA.append(i)
                    resB.append(i)
        # 判断是否是联通图
        for i in range(1, n+1):
            temp1.add(ufA.find(i))
            temp2.add(ufB.find(i))
        # 得出结果
        if len(temp1) > 1 or len(temp2) > 1:
            res = -1
        else:
            if len(resB) != 0:
                resA.extend(resB)
            res = len(set(resA))
        return res

    # 724. 寻找数组的中心索引
    def pivotIndex(self, nums: [int]) -> int:
        n = len(nums)
        if n == 0:
            return -1
        sum_ = sum(nums)
        left, res = 0, -1
        for i in range(n):
            left += nums[i-1] if i >= 1 else 0
            if left == (sum_-nums[i])/2:
                res = i
                break
        return res


s = Solution()
print(s.pivotIndex([-1, -1, -1, 0, 1, 1]
                   ))
