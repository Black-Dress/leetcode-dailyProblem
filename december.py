from typing import List
from queue import PriorityQueue, Queue


class Solution:
    # 1446. 连续字符
    def maxPower(self, s: str) -> int:
        maxnum, cur = 1, 1
        for i in range(1, len(s)):
            if s[i] == s[i - 1]:
                cur += 1
            else:
                cur = 1
            maxnum = max(cur, maxnum)
        return maxnum

    # 1005. K 次取反后最大化的数组和
    def largestSumAfterKNegations(self, nums: List[int], k: int) -> int:
        minqueue = PriorityQueue()
        for num in nums:
            minqueue.put(num)
        for i in range(k):
            item = minqueue.get()
            minqueue.put(-item)
        return sum(minqueue.queue)

    # 1816. 截断句子
    def truncateSentence(self, s: str, k: int) -> str:
        index = 0
        while k != 0:
            k -= 1 if s[index] == ' ' or index == len(s) - 1 else 0
            index += 1
        return s[:index].strip()

    # 1034. 边界着色
    def colorBorder(self, grid: List[List[int]], row: int, col: int, color: int) -> List[List[int]]:
        que, dx, dy = set(), [0, 0, 1, -1], [1, -1, 0, 0]
        m, n, origin = len(grid), len(grid[0]), grid[row][col]
        vist = [[0 for i in range(n)] for j in range(m)]
        que.add((row, col))
        vist[row][col] = 1
        while len(que) != 0:
            item = que.pop()
            # 判断是否需要变更颜色
            for i in range(4):
                x, y = item[0] + dx[i], item[1] + dy[i]
                # 在边界
                if x >= m or x < 0 or y >= n or y < 0:
                    grid[item[0]][item[1]] = color
                else:
                    # 在连通边界
                    if grid[x][y] != origin and vist[x][y] == 0:
                        grid[item[0]][item[1]] = color
                    else:
                        # 没有访问过，但是连通
                        if vist[x][y] == 0:
                            que.add((x, y))
                        vist[x][y] = 1
        return grid


s = Solution()
print(s.colorBorder([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0]], 1, 1, 2))
