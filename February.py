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
            else:
                r.remove(nums[i - k])
                heapify(r)

            while len(l) - len(r) > 1 or len(l) - len(r) < 0:
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


s = Solution()
print(s.dicesProbability(2))
