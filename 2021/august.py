from collections import defaultdict
import collections
from typing import Collection, List


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

    # 526. 优美的排列
    def countArrangement(self, n: int) -> int:
        def dfs_526(cur: List[int]) -> int:
            if cur.__len__() == n:
                return 1
            res = 0
            for i in range(1, n+1):
                if cur.count(i) == 0 and (i % (cur.__len__()+1) == 0 or (cur.__len__()+1) % i == 0):
                    cur.append(i)
                    res += dfs_526(cur)
                    cur.pop()
            return res
        return dfs_526([])


s = Solution()
print(s.countArrangement(2))
