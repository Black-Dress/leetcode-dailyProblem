import collections
from sys import modules
from typing import Collection, Counter, List


class Solution:
    # BFS
    def LCP07_BFS(self, relation: List[List[int]], k: int, n: int) -> int:
        edges = collections.defaultdict(list)
        for i in relation:
            edges[i[0]].append(i[1])
        steps, queue = 0, collections.deque([0])
        while queue.__len__() and steps < k:
            steps += 1
            for i in range(queue.__len__()):
                to = edges[queue.popleft()]
                queue.extend(to)
        res = 0
        if steps == k:
            while queue.__len__():
                res += 1 if queue.popleft() == n - 1 else 0
        return res

    # DFS
    def LCP07_DFS(self, edges: dict, level: int, cur: int, k: int, n: int, res: List):
        if level == k:
            res[0] += 1 if cur == n - 1 else 0
            return
        for i in edges[cur]:
            self.LCP07_DFS(edges, level + 1, i, k, n, res)

    # LCP 07
    def numWays(self, n: int, relation: List[List[int]], k: int) -> int:
        # return self.LCP07_BFS(relation, k, n)
        edges, res = collections.defaultdict(list), [0]
        for i in relation:
            edges[i[0]].append(i[1])
        self.LCP07_DFS(edges, 0, 0, k, n, res)
        return res[0]

    # 1833. 雪糕的最大数量
    def maxIceCream(self, costs: List[int], coins: int) -> int:
        costs.sort()
        res, sum = 0, 0
        for i in range(costs.__len__()):
            sum += costs[i]
            res += 1
            if sum > coins:
                res -= 1
        return res

    # 451. 根据字符出现频率排序
    def frequencySort(self, s: str) -> str:
        c = Counter(s)
        sortedList = sorted(c.items(), key=lambda item: item[1], reverse=True)
        res = [k for k, v in sortedList for i in range(v)]
        return "".join(res)

    # 645. 错误的集合
    def findErrorNums(self, nums: List[int]) -> List[int]:
        bucket = [-1 for i in range(nums.__len__() + 1)]
        a, b = 0, 0
        for i in nums:
            bucket[i] += 1
        for i in range(1, bucket.__len__()):
            if bucket[i] == 1:
                a = i
            if bucket[i] == -1:
                b = i
        return [a, b]

    # 1711. 大餐计数
    def countPairs(self, deliciousness: List[int]) -> int:
        mod = 1000000007
        table = collections.defaultdict(int)
        maxNum, res = max(deliciousness), 0
        for i in deliciousness:
            j = 1
            while j <= maxNum * 2:
                res += table.get(j) if table.get(j) is not None else 0
                res %= mod
                j <<= 1
            table[i] += 1
        return res


s = Solution()
print(s.countPairs([149, 107, 1, 63, 0, 1, 6867, 1325, 5611, 2581, 39, 89, 46, 18, 12, 20, 22, 234]))
