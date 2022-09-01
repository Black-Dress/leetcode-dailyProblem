from collections import Counter, defaultdict
from curses.ascii import isdigit
from email.policy import default
import graphlib
import numbers
from operator import eq, truediv
from posixpath import split
import re
from sys import stdin
from typing import List


class node:
    def __init__(self, color: str) -> None:
        self.color = color
        self.children = []


class Solution:
    def solveEquation(self, equation: str) -> str:
        factor = val = 0
        i, n, sign = 0, len(equation), 1  # 等式左边默认系数为正
        while i < n:
            if equation[i] == '=':
                sign = -1
                i += 1
                continue

            s = sign
            if equation[i] == '+':  # 去掉前面的符号
                i += 1
            elif equation[i] == '-':
                s = -s
                i += 1

            num, valid = 0, False
            while i < n and equation[i].isdigit():
                valid = True
                num = num * 10 + int(equation[i])
                i += 1

            if i < n and equation[i] == 'x':  # 变量
                factor += s * num if valid else s
                i += 1
            else:  # 数值
                val += s * num

        if factor == 0:
            return "No solution" if val else "Infinite solutions"
        return f"x={-val // factor}"

    def solution1():
        a, b = stdin.readline().split(" ")
        # 大数是小数的倍数，大数可以删除数字小数可以删除数字
        # 最少的操作次数
        # 如何删除？永远删除大数的数字？之后再删除小数的数字？

    def solution4(self):
        n = int(stdin.readline())
        nums = list(map(int, stdin.readline().split(" ")))
        cnt = defaultdict(list)
        res = 0

        for i in range(n):
            cnt[nums[i]].append(i)
        for v in cnt.values():
            if len(v) >= 2:
                index = []
                for i in range(1, len(v)):
                    index.append(0)
                    for j in range(v[i - 1], v[i]):
                        if nums[j] < nums[v[0]]:
                            index[-1] += 1
                for i in range(len(index)):
                    res += (len(index) - i) * index[i]
                    if i != 0 and i != len(index) - 1:
                        res += (len(index) - i) * index[i]
        print(res)

    def solution3(self):
        n = int(stdin.readline())
        nums = list(map(int, stdin.readline().split(" ")))
        a, b = nums[0], nums[1]
        sum_a, sum_b = 0, 0
        res = 0
        for i, val in enumerate(nums):
            if (i + 1) % 2 == 0:
                sum_a += val
                if val > a:
                    a = val

            if (i + 1) % 2 != 0:
                sum_b += val
                if val > b:
                    b = val

        if a != b:
            res = a * (n // 2) - sum_a + b * ((n + 1) // 2) - sum_b
        else:
            res = min((a + 1) * (n // 2) - sum_a + b * ((n + 1) // 2) - sum_b, a * (n // 2) - sum_a + (b + 1) * ((n + 1) // 2) - sum_b)
        print(res)

    def solution2():
        s = stdin.readline()
        max_num = (s // 5) * 2 + (s // 3)
        i, res = 0, 0
        reds = s.split("red")
        for i in reds:
            if len(i) >= 2 and (i[:-2] == "de" or i[:2] == "er"):
                pass

    # 97. 交错字符串
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        # dp[i][j] 表示 s1[0:i],s2[0:j]能否够组成s3[0:i+j]
        if len(s3) != len(s1) + len(s2):
            return False
        dp = [[False] * (len(s2) + 1)for _ in range(len(s1) + 1)]
        dp[0][0] = True
        for i in range(len(s1) + 1):
            for j in range(len(s2) + 1):
                if i > 0:
                    dp[i][j] = dp[i][j] or dp[i - 1][j] and s1[i - 1] == s3[i + j - 1]
                if j > 0:
                    dp[i][j] = dp[i][j] or dp[i][j - 1] and s2[j - 1] == s3[i + j - 1]
        return dp[len(s1)][len(s2)]

    def wangyi01(self):
        # 可以将程序化成3个部分
        t = int(stdin.readline())
        for tt in range(t):
            if tt > 0:
                stdin.readline()
            s = stdin.readline().strip("\n")
            n, m = list(map(int, s.split(" ")))
            graph, res = [], []
            for i in range(n):
                graph.append(stdin.readline().strip('\n'))
            if n == m:
                for i in graph:
                    print(i)
                return
            # 如果是偶数
            q = m // n
            # 如果边角料是奇数
            if (m % n) % 2 != 0:
                q = (m // n) - 1
            p = (m - q * n) // 2
            # 第一部分1
            for y in range(n - p, n):
                res.append("".join([graph[y][n - p:n]] + [graph[y] * q] + [graph[y][:p]]))
            # 第二部分
            for k in range(q):
                if k == 0:
                    for y in range(n):
                        res.append("".join([graph[y][n - p: n]] + [graph[y] * q] + [graph[y][:p]]))
                else:
                    for i in range(n):
                        res.append(res[-n])
            # 第三部分
            for y in range(p):
                res.append("".join([graph[y][n - p:n]] + [graph[y] * q] + [graph[y][:p]]))
            # 输出
            for i in res:
                print(i)
            print("")

    def minWindow(self, s: str, t: str) -> str:
        def check(cnt: dict()) -> bool:
            res = 0
            for i in cnt.values():
                if i > 0:
                    return False
                res += i
            return res <= 0
        cnt, res, l = Counter(t), s, 0
        for r in range(len(s)):
            if s[r] in cnt:
                cnt[s[r]] -= 1
            # 窗口内多引入了数据，需要保证窗口内所有字符都存在，并且删除多余的数据
            if check(cnt):
                while l < r:
                    if s[l] in cnt and cnt[s[l]] >= 0:
                        break
                    if s[l] in cnt and cnt[s[l]] < 0:
                        cnt[s[l]] += 1
                    l += 1
                res = min(res, s[l:r + 1], key=lambda x: len(x))
        return res if check(cnt) else ""

    def trap(self, height: List[int]) -> int:
        stack, res = [[0, height[0]]], 0
        for i in range(1, len(height)):
            val = height[i]
            while stack and val > stack[-1][1]:
                pre = stack.pop()
                if stack:
                    res += (i - stack[-1][0] - 1) * (min(val, stack[-1][1]) - pre[1])
            stack.append([i, val])
        return res

    def gcd(self, a: int, b: int) -> int:
        a, b = max(a, b), min(a, b)
        if b == 0:
            return a
        return self.gcd(b, a % b)

    def xiecheng01(self):
        n = int(stdin.readline())
        for _ in range(n):
            num = list(stdin.readline().strip("\n"))
            if num[-1] == '0' or int(num[-1]) % 2 == 0:
                print("".join(num))
                continue
            flag = False
            for i, val in enumerate(num):
                if int(val) % 2 == 0:
                    num[-1], num[i] = num[i], num[-1]
                    flag = True
                    break
            if flag:
                print("".join(num))
            else:
                print("-1")

    def xiecheng03(self):
        # 删除一条边能够得到两块区域，不含重边和环，那么代表所有的边都能够降节点划分成两块
        # 会不会存在删除一条边无法划分成两块的情况（不会，那样就意味着存在环）
        # 通过输入判断按照顺序的情况下颜色是怎么排列的
        # 可能是一颗多叉树的情况
        # 如何判断删除哪一条边
        # dfs 查询到某一个节点的时候记录到达目前为止的颜色，如果多于三个颜色，并且外部颜色也满足条件的话就删除他的子节点的边

        def check(cnt: dict, colors: dict) -> bool:
            for k, v in cnt.items():
                if v <= 0:
                    return False
                if colors[k] - v <= 0:
                    return False
            return True

        def dfs(graph: List[List[int]], index: int, visit: set, color: str, cnt: dict, colors: dict) -> int:
            res = 0
            if check(cnt, colors):
                res += 1
            cnt[color[index]] += 1
            for i in graph[index]:
                if i not in visit:
                    visit.add(i)
                    res += dfs(graph, i, visit, color, cnt, colors)
                    visit.remove(i)
            cnt[color[index]] -= 1
            return res

        n = int(stdin.readline())
        color = stdin.readline().strip("\n")
        colors = Counter(color)
        graph = [[] for i in range(n)]
        for i in range(n - 1):
            a, b = list(map(int, stdin.readline().strip("\n").split(" ")))
            graph[a - 1].append(b - 1)
            graph[b - 1].append(a - 1)
        visit = set()
        visit.add(0)
        cnt = {'r': 0, 'g': 0, 'b': 0}
        res = dfs(graph, 0, visit, color, cnt, colors)
        print(res)

    def xiecheng04(self):
        # 找到前二的平滑值
        n = int(stdin.readline().strip("\n"))
        nums = list(map(int, stdin.readline().strip("\n").split(" ")))
        res = [0, 0]
        index = 0
        for i in range(1, n):
            val = abs(nums[i] - nums[i - 1])
            if val > res[0]:
                res[0] = val
                index = i
            if res[0] > val > res[1]:
                res[1] = val
        a = res[1]
        if index + 1 < n:
            a = abs(nums[index + 1] - nums[index - 1])
        print(min(a, res[1]))

    def guanglianda01(self):
        n = int(stdin.readline())
        a = list(stdin.readline().strip("\n").split(" "))
        b = list(stdin.readline().strip("\n").split(" "))
        cnt = defaultdict(int)
        i, j = 0, 0
        while i < n and j < n:
            if a[i] in cnt:
                i += 1
                continue
            while j < n and b[j] != a[i]:
                cnt[b[j]] = 1
                j += 1
            j += 1
            i += 1

        print(len(cnt))

    def guanglianda02(self):
        n, m = list(map(int, stdin.readline().strip("\n").split(" ")))
        res, pr = 0, 0
        nums, res = [], [0] * (n + 1)
        for i in range(m):
            nums.append(list(map(int, stdin.readline().strip("\n").split(" "))))
        nums.sort()
        for i in range(m):
            l, r, x = nums[i]
            if i == 0:
                while l <= r and x > 0:
                    res[l] = 1
                    l += 1
                    x -= 1
            else:
                while pr < l:
                    res[pr] = 1
                    pr += 1
                j = l
                while j <= pr:
                    if res[j] == 1 and sum(res[j:pr + 1] > x):
                        res[l] = 0
                    l += 1
                if l < pr + 1:
                    x -= sum(res[l:pr + 1])
                while j <= r and x > 0:
                    res[j] = 1
                    j += 1
                    x -= 1
            pr = r

        print(sum(res))


s = Solution()
s.guanglianda02()
