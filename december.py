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

    # 689. 三个无重叠子数组的最大和
    def maxSumOfThreeSubarrays(self, nums: List[int], k: int) -> List[int]:
        # sum[i] 指 nums[i,k] 的和
        # l[i] = k 表示 0~i 最大sum[k],0<k<i
        # r[i] = k 表示 i~n-1 最大sum[k],i<k<n-1
        # 枚举所有的 i 得到 max(sum[l[i-k]]+sum[i]+sum[r[i+k]])
        # 只有比maxnum大的时候才更新res，就能够维护最小的字典序
        # 需要利用滑动窗口计算和，不然会超时
        sum_ = [sum(nums[i] for i in range(k))]
        for i in range(1, len(nums) - k + 1):
            sum_.append(sum_[-1] - nums[i - 1] + nums[i + k - 1])
        l, r = [0 for i in range(len(sum_))], [len(nums) - k for i in range(len(sum_))]
        for i in range(1, len(sum_)):
            l[i] = i if sum_[i] > sum_[l[i - 1]] else l[i - 1]
        for i in range(len(sum_) - 2, -1, -1):
            r[i] = i if sum_[i] >= sum_[r[i + 1]] else r[i + 1]
        maxnum, res = 0, []
        for i in range(k, len(sum_) - k):
            cur = sum_[i] + sum_[l[i - k]] + sum_[r[i + k]]
            if cur > maxnum:
                maxnum = cur
                res = [l[i - k], i, r[i + k]]
        return res

    # 807. 保持城市天际线
    def maxIncreaseKeepingSkyline(self, grid: List[List[int]]) -> int:
        n, res = len(grid), 0
        row, col = [0 for i in range(n)], [0 for i in range(n)]
        for i in range(n):
            row[i] = max(grid[i])
            col[i] = max(grid[j][i] for j in range(n))
        for i in range(n):
            for j in range(n):
                res += min(row[i], col[j]) - grid[i][j]
        return res


s = Solution()
print(s.maxIncreaseKeepingSkyline([[3, 0, 8, 4], [2, 4, 5, 7], [9, 2, 6, 3], [0, 3, 1, 0]]))
