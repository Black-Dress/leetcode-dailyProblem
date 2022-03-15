from ast import parse
from calendar import c
from collections import defaultdict
from ctypes.wintypes import tagRECT
import tarfile
from typing import List, Literal, Set


class Solution:
    # 2055. 蜡烛之间的盘子
    def platesBetweenCandles(self, s: str, queries: List[List[int]]) -> List[int]:
        # 利用栈可以轻松得到查询的结果
        # 有没有方法初始化初始字符串，并用cnt进行存储，表示某区间段的结果
        # def check(query:List[int])->int:
        #     stack,res = [],0
        #     for i in range(query[0],query[1]+1):
        #         if s[i] == '|':
        #             res += i-stack[-1]-1 if stack else 0
        #             stack.append(i)
        #     return res
        # return [check(i) for i in queries]

        # 存储所有蜡烛位置，利用二分查找离边界最近的蜡烛位置
        # 二分查找最近的蜡烛
        # cnt = [i for i in range(len(s)) if s[i] == '|']
        # def index(i: int, l: int, r: int, isL: bool) -> int:
        #     while l <= r:
        #         mid = (l + r) >> 1
        #         if i == cnt[mid]:
        #             return mid
        #         if i > cnt[mid]:
        #             l = mid + 1
        #         if i < cnt[mid]:
        #             r = mid - 1
        #     return r if isL else l
        # res = []
        # for i in queries:
        #     l, r = index(i[0], 0, len(cnt) - 1, False), index(i[1], 0, len(cnt) - 1, True)
        #     res.append(max(cnt[r] - cnt[l] - (r - l), 0) if l in range(0, len(cnt)) and r in range(0, len(cnt)) else 0)
        # return res

        # 前缀和，利用两个数组存储i位置左边和右边最近的蜡烛，用一个数组存储i位置之前有多少个盘子
        n = len(s)
        l, r, candles = [0] * n, [0] * n, [0] * n
        index = -1
        for i in range(n):
            if s[i] == '|':
                index = i
            l[i] = index
        index = -1
        for i in range(n - 1, -1, -1):
            if s[i] == '|':
                index = i
            r[i] = index
        index = 0
        for i in range(n):
            if s[i] == '|':
                index += 1
                candles[i] = index
        res = []
        for a, b in queries:
            x, y = r[a], l[b]
            res.append(0 if x < 0 or y < 0 else max(y - x - (candles[y] - candles[x]), 0))
        return res

    # 798. 得分最高的最小轮调
    def bestRotation(self, nums: List[int]) -> int:
        n, diff = len(nums), [0] * (len(nums) + 1)
        for i in range(n):
            if nums[i] > i:
                # [i+1,i+1+n-nums[i]-1]
                l, r = i + 1, i + 1 + n - nums[i]
                diff[l] += 1
                diff[r] -= 1
            else:
                # [0,i-nums[i]]
                diff[0] += 1
                diff[i - nums[i] + 1] -= 1
                # [i+1,n]
                diff[i + 1] += 1
                diff[n] -= 1
        for i in range(1, n):
            diff[i] += diff[i - 1]
        res, score = 0, 0
        for i in range(n):
            if diff[i] > score:
                res = i
                score = diff[i]
        return res

    # 301. 删除无效的括号
    def removeInvalidParentheses(self, s: str) -> List[str]:
        # BFS 搜索
        def check(s: str) -> bool:
            cnt = 0
            for i in s:
                if i == '(':
                    cnt += 1
                elif i == ')':
                    cnt -= 1
                    if cnt < 0:
                        return False
            return cnt == 0
        # BFS
        que, res = set([s]), []
        while que:
            for ss in que:
                if check(ss):
                    res.append(ss)
            if res:
                break
            cur = set()
            for ss in que:
                for i in range(len(ss)):
                    # 去除重复删除
                    if i > 0 and ss[i] == s[i - 1]:
                        continue
                    if ss[i] == ')' or ss[i] == '(':
                        cur.add(ss[:i] + ss[i + 1:])
            que = cur
        return res

    # 416. 分割等和子集
    def canPartition(self, nums: List[int]) -> bool:
        # dp[i] 表示值为i的时候能否通过nums的元素构成
        a = sum(nums)
        if a % 2 != 0:
            return False
        dp = [True] + [False] * (a // 2)
        for num in nums:
            for i in range(a // 2, num - 1, -1):
                dp[i] = dp[i] or dp[i - num]
        return dp[-1]

    # 309. 最佳买卖股票时机含冷冻期
    def maxProfit(self, prices: List[int]) -> int:
        # dp[i][0] 表示持股最大收益
        # dp[i][1] 不持股且在冷冻期的最大收益
        # dp[i][2] 不持股且不在冷冻期的最大收益
        n = len(prices)
        dp = [[-prices[0], 0, 0]] + [[0] * 3 for i in range(n - 1)]
        for i in range(1, n):
            # 买
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][2] - prices[i])
            # 卖
            dp[i][1] = dp[i - 1][0] + prices[i]
            # 冷冻
            dp[i][2] = max(dp[i - 1][1], dp[i - 1][2])

        return max(dp[-1][1], dp[-1][2])

    # 2049. 统计最高分的节点数目
    def countHighestScoreNodes(self, parents: List[int]) -> int:
        # 移除某一节点,res = 左子树*右子树*(整棵树-当前树)
        # cnt[i] 表示 i节点表示的树包含的节点数量,children[i]=[l,r] 表示i节点的左节点和右节点的位置
        # 通过并查集思想进行子树节点的统计
        # cnt, children, n, res, num = defaultdict(int), defaultdict(list), len(parents), 1, 0
        # for i in range(n):
        #     index = i
        #     while index != 0:
        #         cnt[parents[index]] += 1
        #         index = parents[index]
        #     children[parents[i]].append(i)
        #     cnt[i] += 1
        # for i in range(n):
        #     l = cnt[children[i][0]] if children[i] else 1
        #     r = cnt[children[i][1]] if len(children[i]) > 1 else 1
        #     parent = max(cnt[0] - cnt[i], 1)
        #     temp = parent * l * r
        #     if temp > num:
        #         num = temp
        #         res = 0
        #     if temp == num:
        #         res += 1
        # return res
        # 先建立一颗树，利用dfs遍历的时候进行score的计算
        children, n, res, num = defaultdict(list), len(parents), 0, 0
        for i, parent in enumerate(parents):
            children[parent].append(i)

        def check(t: int):
            nonlocal num, res
            if t > num:
                num = t
                res = 0
            if t == num:
                res += 1

        def dfs(index: int) -> int:
            if not children[index]:
                check(n - 1)
                return 1
            child = [dfs(i) for i in children[index]]
            a, b = child[0], n - child[0] - 1
            for i in range(1, len(child)):
                a *= child[i]
                b -= child[i]
            check(a * max(1, b))
            return sum(child) + 1
        dfs(0)
        return res

    # 312. 戳气球
    def maxCoins(self, nums: List[int]) -> int:
        # dp[i][j] 表示区间i,j的最大金币数
        n, nums = len(nums), [1] + nums + [1]
        dp = [[0] * (n + 2) for i in range(n + 2)]
        for i in range(n, -1, -1):
            for j in range(i + 2, n + 2):
                for k in range(i + 1, j):
                    dp[i][j] = max(dp[i][j], dp[i][k] + dp[k][j] + nums[i] * nums[j] * nums[k])
        return dp[0][-1]


s = Solution()
print(s.maxCoins([1, 0, 2, 8]))
# ()())()
# (()(()((
