from collections import defaultdict
import collections
from typing import Collection


class Solution:
    def __init__(self):
        self.status_576 = collections.defaultdict()

    # 576
    def findPaths(self, m: int, n: int, maxMove: int, startRow: int, startColumn: int) -> int:
        # 普通dfs 超时，需要记忆状态
        mod = 10**9+7
        direction = [(0, 1), (0, -1), (-1, 0), (1, 0)]
        if startColumn < 0 or startColumn == n or startRow < 0 or startRow == m:
            return 1
        if maxMove == 0 or (m - maxMove > startRow > maxMove - 1 and n - maxMove > startColumn > maxMove - 1):
            return 0
        res = 0
        # 状态压缩
        key = startRow*2500 + startColumn*50 + maxMove
        if(self.status_576.get(key)):
            return self.status_576[key]
        for dx, dy in direction:
            res = (res + self.findPaths(m, n, maxMove-1, startRow+dx, startColumn+dy)) % mod
        self.status_576[key] = res
        return res


s = Solution()
print(s.findPaths(2, 2, 2, 0, 0))
