import collections
from typing import Collection, Counter, List
from sys import maxsize
import bisect


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
    #

    def getSkyline(self, buildings: List[List[int]]) -> List[List[int]]:
        n = len(buildings)
        res, right = [], []
        preRightMax = 0
        # 先判断左边界,还需要判断是否前一个与自己断开了
        for building in buildings:
            if res.__len__() == 0:
                res.append([building[0], building[2]])
            else:
                if building[2] > res[-1][1] or building[0] > preRightMax:
                    if building[0] == res[-1][0]:
                        res[-1][1] = building[2]
                    else:
                        res.append([building[0], building[2]])
            preRightMax = max(preRightMax, building[1])
        # 右边界倒序
        buildings.reverse()
        preHeight, preLeftMin = 0, maxsize
        for building in buildings:
            if right.__len__() == 0:
                right.append([building[1], preHeight])
            else:
                if building[2] > preHeight or building[1] < preLeftMin:
                    data = [building[1], 0] if building[1] < preLeftMin else [building[1], preHeight]
                    if building[1] == right[-1][0]:
                        right[-1][1] = building[2]
                    else:
                        right.append(data)
            preLeftMin = min(preLeftMin, building[0])
            preHeight = building[2]
        res.extend(right)
        return sorted(res, key=lambda x: x[0])

    # 1818. 绝对差值和
    def minAbsoluteSumDiff(self, nums1: List[int], nums2: List[int]) -> int:
        nums_sorted = sorted(nums1)
        n = nums1.__len__()
        maxgap, res = 0, 0
        for i in range(n):
            index = bisect.bisect_left(nums_sorted, nums2[i])
            a = abs(nums_sorted[index-1] - nums2[i]) if index > 0 else maxsize
            b = abs(nums_sorted[index] - nums2[i]) if -1 < index < n else maxsize
            c = abs(nums_sorted[index+1]-nums2[i]) if index < n-1 else maxsize
            cur = abs(nums1[i]-nums2[i])
            minval = min(a, b, c)
            maxgap = max(abs(cur-minval), maxgap)
            res += cur
        res -= maxgap
        return res % (10**9+7)

    # 1846. 减小和重新排列数组后的最大元素
    def maximumElementAfterDecrementingAndRearranging(self, arr: List[int]) -> int:
        arr.sort()
        arr[0] = 1
        n, maxval = arr.__len__(), 1
        for i in range(1, n):
            arr[i] = arr[i-1]+1 if abs(arr[i]-arr[i-1]) > 1 else arr[i]
            maxval = max(maxval, arr[i])
        return maxval


s = Solution()
print(s.findErrorNums([1, 2, 2, 4]))
