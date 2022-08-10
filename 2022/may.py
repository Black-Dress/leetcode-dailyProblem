import imp
from typing import Counter, List
from collections import defaultdict
from NodeHelper.TreeNode import TreeNode
from pkg_resources import working_set


class Solution:
    def findRightInterval(self, intervals: List[List[int]]) -> List[int]:
        # 记录所有的右边界
        max_l = max([l for l, r in intervals])
        index_r, index_l = defaultdict(list), defaultdict(int)
        res = [-1] * len(intervals)
        for i in range(len(intervals)):
            index_r[intervals[i][1]].append(i)
            index_l[intervals[i][0]] = i
        intervals.sort(key=lambda x: x[0])
        # 设置标记数组
        nums, index = defaultdict(int), 0
        for i in range(intervals[0][0], max_l + 1):
            if intervals[index][0] < i:
                index += 1
            nums[i] = intervals[index][0]
        # 找到对应结果
        for k, v in index_r.items():
            for i in v:
                res[i] = index_l[nums[k]] if k <= max_l else -1
        return res

    # 675. 为高尔夫比赛砍树
    def cutOffTree(self, forest: List[List[int]]) -> int:
        # 利用优先队列 存储下一个位置，并且计算从当前位置到下一个位置需要花费的步数
        m, n = len(forest), len(forest[0])

        def bfs(sx: int, sy: int, ex: int, ey: int) -> int:
            points, dx, dy = [(sx, sy)], [1, -1, 0, 0], [0, 0, -1, 1]
            index, d = 1, 0
            vist = {(sx, sy): 1}
            while points:
                cur = points.pop(0)
                index -= 1
                if cur[0] == ex and cur[1] == ey:
                    return d
                for i in range(4):
                    x, y = cur[0] + dx[i], cur[1] + dy[i]
                    if x >= 0 and x < m and y >= 0 and y < n and forest[x][y] >= 1 and (x, y) not in vist:
                        points.append((x, y))
                        vist[(x, y)] = 1
                if index == 0:
                    d += 1
                    index = len(points)
            return -1
        nxt = []
        pre = [0, 0]
        res = 0
        for i in range(m):
            for j in range(n):
                if forest[i][j] > 1:
                    nxt.append([i, j, forest[i][j]])
        nxt.sort(key=lambda x: x[2])
        for i in nxt:
            d = bfs(pre[0], pre[1], i[0], i[1])
            if d < 0:
                return -1
            forest[i[0]][i[1]] = 1
            res += d
            pre = i
        return res

    # 467. 环绕字符串中唯一的子字符串
    def findSubstringInWraproundString(self, p: str) -> int:
        # 找到p中按照字典序排列的子串
        def ord_(s: str) -> int:
            return ord(s) - ord('a')
        k, n = 1, len(p)
        dp = defaultdict(int)
        dp[p[0]] = 1
        for i in range(1, n):
            if ord_(p[i]) == (ord_(p[i - 1]) + 1) % 26:
                k += 1
            else:
                k = 1
            dp[p[i]] = max(k, dp[p[i]])
        return sum(dp.values())

    # 699. 掉落的方块
    def fallingSquares(self, positions: List[List[int]]) -> List[int]:
        # 用一个数组表示 i 位置的高度，时间复杂度
        # n = max([i[0] + i[1] for i in positions])
        # height = [0] * (n + 1)
        # res = [positions[0][1]]
        # height[positions[0][0]], height[positions[0][0] + positions[0][1]] = positions[0][1], -positions[0][1]

        # def check(pos: int, bound: List[int]) -> bool:
        #     return pos in range(bound[0], bound[0] + bound[1])
        # for i in range(1, len(positions)):
        #     pos, h = positions[i][0], positions[i][1]
        #     if check(pos, positions[i - 1]) or check(pos + h - 1, positions[i - 1]):
        #         r = positions[i - 1][0] + positions[i - 1][1]
        #         if r < pos + h:
        #             preh = height[r]
        #             height[r] = 0
        #             height[pos + h] = preh
        #         else:
        #             pass
        #     height[pos] += h
        #     height[pos + h] -= h
        #     res.append(max(res[-1], sum(height[:pos + h])))
        # return res
        n = len(positions)
        heights = [0] * n
        for i, (left1, side1) in enumerate(positions):
            right1 = left1 + side1 - 1
            heights[i] = side1
            for j in range(i):
                left2, right2 = positions[j][0], positions[j][0] + positions[j][1] - 1
                if right1 >= left2 and right2 >= left1:
                    heights[i] = max(heights[i], heights[j] + side1)
        for i in range(1, n):
            heights[i] = max(heights[i], heights[i - 1])
        return heights

    # 面试题 17.11. 单词距离
    def findClosest(self, words: List[str], word1: str, word2: str) -> int:
        a, b, res = 0, 0, len(words)
        for i in range(len(words)):
            if words[i] == word1:
                a = i
            if words[i] == word2:
                b = i
            if words[a] == word1 and words[b] == word2:
                res = min(res, abs(a - b))
        return res

    def threeSumClosest(self, nums: List[int], target: int) -> int:
        # O(n^2) 确定l 然后 m r进行滑动窗口移动
        res = 100000
        nums.sort()
        for l in range(len(nums) - 2):
            m, r = l + 1, len(nums) - 1
            while m < r:
                s = nums[l] + nums[m] + nums[r]
                if abs(res - target) > abs(s - target):
                    res = s
                if s > target:
                    r -= 1
                elif s < target:
                    m += 1
                else:
                    return target
        return res

    def BSTSequences(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return [[]]
        l = self.BSTSequences(root.left)
        r = self.BSTSequences(root.right)
        res = [[]]
        if l[0] and r[0]:
            res = self.mearge(l[0], r[0], [])
        if not l[0] and r[0]:
            res = r
        if not r[0] and l[0]:
            res = l
        m = [[root.val] + i for i in res]
        print(m)
        return m

    def mearge(self, a: List[int], b: List[int], cur: List[int]) -> List[List[int]]:
        if not a and not b:
            return [cur]
        if not a:
            return [cur + b]
        if not b:
            return [cur + a]
        res = []
        res.extend(self.mearge(a[1:], b, cur + [a[0]]))
        res.extend(self.mearge(a, b[1:], cur + [b[0]]))
        return res


s = Solution()
print(s.mearge([1, 0], [4, 3], []))
