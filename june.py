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


s = Solution()
print(s.makesquare([10, 6, 5, 5, 5, 3, 3, 3, 2, 2, 2, 2]))
