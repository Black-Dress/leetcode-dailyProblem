from ast import parse
from calendar import c
from collections import Counter, defaultdict
from ctypes.wintypes import tagRECT
import tarfile
from typing import List, Literal, Set


class Solution:
    # 165. 比较版本号
    def compareVersion(self, version1: str, version2: str) -> int:
        a, b, i = version1.split('.'), version2.split('.'), 0
        while i < len(a) and i < len(b):
            if int(a[i]) > int(b[i]):
                return 1
            if int(a[i]) < int(b[i]):
                return -1
            i += 1
        while i < len(a):
            if int(a[i]) > 0:
                return 1
            i += 1
        while i < len(b):
            if int(b[i]) > 0:
                return -1
            i += 1
        return 0

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

    # 438. 找到字符串中所有字母异位词
    def findAnagrams(self, s: str, p: str) -> List[int]:
        # 滑动窗口,初始化窗口之后 需要判断区间是否是异位词
        # if s[j] not in p -> i=j
        m, n, target, cnt = len(s), len(p), Counter(p), 0
        res = []
        i, j = 0, 0
        # 初始化
        while j < n:
            target[s[j]] -= 1
            cnt += 1 if target[s[j]] >= 0 else -1
            j += 1
        if cnt == n:
            res.append(i)
        # 滑动窗口
        while j < m:
            target[s[j]] -= 1
            cnt += 1 if target[s[j]] >= 0 else -1
            target[s[i]] += 1
            cnt += 1 if target[s[i]] <= 0 else -1
            j += 1
            i += 1
            if cnt == n:
                res.append(i)
        return res

    # 448. 找到所有数组中消失的数字
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        for i in range(len(nums)):
            index = nums[i]
            while nums[i] != nums[index - 1]:
                nums[i], nums[index - 1] = nums[index - 1], nums[i]
                index = nums[i]
        return [i + 1 for i in range(len(nums)) if nums[i] - 1 != i]

    # 461. 汉明距离
    def hammingDistance(self, x: int, y: int) -> int:
        res = 0
        while x != 0 or y != 0:
            res += (x & 1) ^ (y & 1)
            x >>= 1
            y >>= 1
        return res

    # 2044. 统计按位或能得到最大值的子集数目
    def countMaxOrSubsets(self, nums: List[int]) -> int:
        # len(nums)<=16 -> dfs 搜索
        # def dfs(target: int, cur: int, nums: List[int]) -> int:
        #     res = 1 if cur == target else 0
        #     if not nums:
        #         return res
        #     for i in range(len(nums)):
        #         res += dfs(target, cur | nums[i], nums[i + 1:])
        #     return res

        # # 得到最大值
        # target = 0
        # for i in range(len(nums)):
        #     target |= nums[i]
        # return dfs(target, 0, nums)

        # dp 统计所有计算的中间结果 dp[i] 表示按位或的值为i的情况有 dp[i]个
        # dp[num|i] += dp[i]
        dp = Counter([0])
        for num in nums:
            # dp[k] 会因为 dp 的改变而改变，kv的v则不会改变
            for k, v in dp.copy().items():
                dp[num | k] += v
        return dp[max(dp)]

    # 581. 最短无序连续子数组
    def findUnsortedSubarray(self, nums: List[int]) -> int:
        # 双指针
        # nums = nums
        # n = len(nums)
        # if n == 1:
        #     return 0
        # l, r = 1, n - 2
        # while l < n and nums[l] >= nums[l - 1]:
        #     l += 1
        # while r > 0 and nums[r] <= nums[r + 1]:
        #     r -= 1
        # # # 整体有序
        # if l == n:
        #     return 0
        # # 存在 l==r l<r l>r
        # minnum, maxnum = min(nums[l - 1:r + 2]), max(nums[l - 1: r + 2])
        # # 更新边界
        # while l > 0 and nums[l - 1] > minnum:
        #     l -= 1
        # while r + 1 < n and nums[r + 1] < maxnum:
        #     r += 1
        # return r - l + 1
        n = len(nums)
        maxnum, minnum = nums[0], nums[n - 1]
        l, r = 0, -1
        for i in range(n):
            if nums[i] >= maxnum:
                maxnum = nums[i]
            else:
                r = i
            if nums[n - i - 1] <= minnum:
                minnum = nums[n - i - 1]
            else:
                l = n - i - 1
        return r - l + 1


s = Solution()
print(s.solve(5, 2, 15, [3, -7, 8, -5, 9]))
# ()())()
# (()(()((
