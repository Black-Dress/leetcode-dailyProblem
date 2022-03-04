import bisect
from heapq import *
from typing import List


class Solution:
    def medianSlidingWindow(self, nums: List[int], k: int) -> List[float]:
        # l 比 r 多
        l, r = [], []

        def mid(l: List[int], r: List[int]) -> int:
            return -l[0] if len(l) > len(r) else (-l[0] + r[0]) / 2

        def inStack(l: List[int], r: List[int], m: int):
            if not l or m <= -l[0]:
                if len(l) == len(r):
                    heappush(l, -m)
                else:
                    heappush(r, -heappushpop(l, -m))
            else:
                if len(l) == len(r):
                    heappush(l, -heappushpop(r, m))
                else:
                    heappush(r, m)

        for i in range(k):
            inStack(l, r, nums[i])

        res = [mid(l, r)]
        for i in range(k, len(nums)):
            # 弹栈
            if nums[i - k] <= -l[0]:
                l.remove(-nums[i - k])
                heapify(l)
                while len(l) - len(r) < 0:
                    heappush(l, -heappop(r))
            else:
                r.remove(nums[i - k])
                heapify(r)
                while len(l) - len(r) > 1:
                    heappush(r, -heappop(l))
            # 入栈
            inStack(l, r, nums[i])
            res.append(mid(l, r))
        return res

    def dicesProbability(self, n: int) -> List[float]:
        # dp[i][j] 表示i颗骰子时 总和为 j 的组合数
        dp = [[0] * (n * 6 + 1) for i in range(n + 1)]
        for i in range(1, 7):
            dp[1][i] = 1
        for i in range(2, n + 1):
            for j in range(i, i * 6 + 1):
                # dp[i][j] = sum([dp[i - 1][j - k] for k in range(1, min(7, j // 2 + 1))])
                dp[i][j] = sum(dp[i - 1][j - k] if j > k else 0 for k in range(1, 7))

        total = sum(dp[n][:])
        return sorted([dp[n][i] / total for i in range(n, n * 6 + 1)])

    # 剑指 Offer 14- II. 剪绳子 II
    def cuttingRope(self, n: int) -> int:
        if n <= 3:
            return n - 1
        a, b, mod, res, x = n // 3 - 1, n % 3, 1000000007, 1, 3
        # 快速幂求余
        while a:
            if a % 2:
                res = (res * x) % mod
            x = (x**2) % mod
            a //= 2
        if b == 0:
            return (res * 3) % mod
        if b == 1:
            return (res * 4) % mod
        return (res * 6) % mod

    # 540. 有序数组中的单一元素
    def singleNonDuplicate(self, nums: List[int]) -> int:
        def dfs(l: int, r: int, nums: List[int]) -> int:
            if l > r:
                return -1
            mid = (l + r) >> 1
            # 和前面相等
            a = mid > 0 and nums[mid] == nums[mid - 1]
            # 和后面相等
            b = mid < r and nums[mid] == nums[mid + 1]
            # 奇偶
            c = (mid + 1) % 2 == 0
            if not a and not b:
                return nums[mid]
            if (a and c) or (b and not c):
                return dfs(mid + 1, r, nums)
            if (b and c) or (a and not c):
                return dfs(l, mid - 1, nums)
        return dfs(0, len(nums) - 1, nums)

    # 239. 滑动窗口最大值
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        que = []
        # 初始化
        for i in range(k):
            while que and nums[que[-1]] < nums[i]:
                que.pop()
            que.append(i)
        res = [nums[que[0]]]
        for i in range(k, len(nums)):
            # 插入
            while que and nums[que[-1]] < nums[i]:
                que.pop()
            que.append(i)
            # 弹出
            while que and que[0] <= i - k:
                que.pop(0)
            res.append(None if not que else nums[que[0]])
        return res


s = Solution()
print(s.maxSlidingWindow([1, 3, -1, -3, 5, 3, 6, 7], 3))
# print(sorted([1, 2, 3, 4, 5, 6], key=lambda x: (x == 1, x - 1)))
# print(min([1, 2, 3, 4, 5, 6], key=lambda x: (x == 1, x - 1)))
