import heapq
from collections import defaultdict


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


s = Solution()
