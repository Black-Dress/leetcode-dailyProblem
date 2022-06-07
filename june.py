from cmath import pi
import heapq
from typing import List


class Solution:
    # 473. 火柴拼正方形
    def makesquare(self, matchsticks: List[int]) -> bool:
        # 将数据分成相等的四份，找到一个可行解 DFS
        n, m = len(matchsticks), sum(matchsticks) // 4
        matchsticks.sort(reverse=True)
        if sum(matchsticks) % 4 != 0:
            return False
        edge = [0] * 4

        def dfs(idx: int) -> bool:
            if idx == n:
                return True
            for i in range(4):
                edge[i] += matchsticks[idx]
                # 在判断条件放dfs可以省略一个变量
                if edge[i] <= m and dfs(idx + 1):
                    return True
                edge[i] -= matchsticks[idx]
            return False
        return dfs(0)
        # 贪心，从大往小放火柴
        # total, n, m = sum(matchsticks), len(matchsticks), sum(matchsticks) // 4
        # if total % 4 != 0:
        #     return False
        # idx = [0] * 4
        # heapq.heapify(idx)
        # matchsticks.sort(reverse=True)
        # for i in range(n):
        #     heapq.heappush(idx, heapq.heappop(idx) + matchsticks[i])
        # res = True
        # for i in idx:
        #     res = res and i == m
        # return res and sum(idx) == total

    # 875. 爱吃香蕉的珂珂
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        # 先找到那两堆香蕉 a,b b香蕉堆的个数能够满足条件，然后在 a,b之间选择一个合理的数字
        # 查找过程可以使用二分查找
        def check(piles: List[int], h: int) -> int:
            res = 0
            for i in piles:
                res += i // h
                res += 1 if i % h != 0 else 0
            return res
        piles.sort()
        l, r, res = 1, piles[-1], 0
        while l <= r:
            res = (l + r) >> 1
            num = check(piles, res)
            if num > h:
                l = res + 1
            if num <= h:
                r = res - 1
        return res if check(piles, res) <= h else res + 1


s = Solution()
print(s.minEatingSpeed([30, 11, 23, 4, 20],
                       6))
