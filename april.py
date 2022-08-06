from collections import defaultdict, deque
from itertools import combinations, permutations, product
from math import tan
from platform import node
from typing import Counter, List, Optional
from NodeHelper.TreeNode import TreeNode
from re import L
from typing import Counter, List
from NodeHelper.ListNode import ListNode
from NodeHelper.TreeNode import TreeNode


class Solution:
    # 954. 二倍数对数组
    def canReorderDoubled(self, arr: List[int]) -> bool:
        # arr[2 * i + 1] = 2 * arr[2 * i] 表示在偶数位置上能够满足这样的条件
        # 有n//2个这样的数队据能够构成这一数组，dict判断是否存在能够对应的条件
        # 如果针对于 arr[i] 存在 2*arr[i] 和 arr[i]//2 ,优先找大的解，直到所有的位置遍历完成，构成数对的数量是否为n//2来进行判断
        cnt, n, res = Counter(arr), len(arr), 0
        if cnt[0] % 2 != 0:
            return False
        for i in sorted(cnt.keys(), key=lambda x: abs(x)):
            if i != 0 and cnt[2 * i] < cnt[i]:
                return False
            cnt[2 * i] -= cnt[i]
        return True

    # 310. 最小高度树
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        if n < 3:
            return [i for i in range(n)]
        # 利用队列，每一轮都删除度为一的节点，最后留下的一个或者两个节点就是答案
        nodes, matrix = defaultdict(int), [[] for i in range(n)]
        for s, e in edges:
            nodes[s] += 1
            nodes[e] += 1
            matrix[s].append(e)
            matrix[e].append(s)
        # 每一轮都只删除度为一的点
        que = [k for k, v in nodes.items() if v == 1]
        remain = n
        while remain > 2:
            temp = []
            for i in que:
                for j in matrix[i]:
                    nodes[j] -= 1
                    if nodes[j] == 1:
                        temp.append(j)
            remain -= len(que)
            que = temp
        return que

    # 357. 统计各位数字都不同的数字个数
    def countNumbersWithUniqueDigits(self, n: int) -> int:
        num = 0
        for i in range(1, n + 1):
            num += 9 * list(permutations(range(9), i - 1)).__len__()
        return num + 1

    # 386. 字典序排数
    def lexicalOrder(self, n: int) -> List[int]:
        # res = []

        # def dfs(pre: str) -> List[int]:
        #     if int(pre) > n:
        #         return []
        #     res = [int(pre)]
        #     for i in range(10):
        #         res.extend(dfs(pre + str(i)))
        #     return res
        # for i in range(1, 10):
        #     res.extend(dfs(str(i)))
        # return res
        res, num = [0] * (n + 1), 1
        for i in range(1, n + 1):
            res[i] = num
            if num * 10 <= n:
                num *= 10
            else:
                while num % 10 == 9 or num + 1 > n:
                    num //= 10
                num += 1
        return res[1:]

    def shortestToChar(self, s: str, c: str) -> List[int]:
        def update(res: List[int], start: int):
            # 向左
            i = start
            while i >= 0 and start - i < res[i]:
                res[i] = start - i
                i -= 1
            # 向右
            i = start + 1
            while i < len(s) and s[i] != c:
                res[i] = i - start
                i += 1
        n = len(s)
        res = [n] * n
        for i in range(n):
            if s[i] == c:
                update(res, i)
        return res

    # 780. 到达终点
    def reachingPoints(self, sx: int, sy: int, tx: int, ty: int) -> bool:
        # dp
        # dp[i][j] 可以到达 dp[i+j][j] dp[i][j+i] 从前往后进行循环遍历
        # if sx > tx or sy > ty:
        #     return False
        # dp = [[0] * (ty + 1) for i in range(tx + 1)]
        # dp[sx][sy] = 1
        # for i in range(sx, tx + 1):
        #     for j in range(sy, ty + 1):
        #         if i + j <= tx:
        #             dp[i + j][j] = max(dp[i + j][j], dp[i][j])
        #         if i + j <= ty:
        #             dp[i][i + j] = max(dp[i][i + j], dp[i][j])
        # return dp[tx][ty] == 1
        # 倒序求解，若tx>ty -> 说明上一个状态是 (tx-ty,ty) 同时为了优化时间需要进行模运算
        # 若出现整除情况，那么需要判断另一个数字能不能由整除情况构成
        while tx > sx and ty > sy:
            if tx > ty:
                tx %= ty
            else:
                ty %= tx
        if tx < sx or ty < sy:
            return False
        return (ty - sy) % tx == 0 if tx == sx else (tx - sx) % ty == 0

    # 面试题 02.01. 移除重复节点
    def removeDuplicateNodes(self, head: ListNode) -> ListNode:
        # 通过移位和与运算判断元素是否重复
        index, num, pre = head.next, 1 << head.val, head
        while index:
            while index and num & (1 << index.val) != 0:
                pre.next = index.next
                index = index.next
            num |= (1 << index.val)
            pre = index
            index = index.next
        return head

    # 388. 文件的最长绝对路径
    def lengthLongestPath(self, input: str) -> int:
        res, files, i, n = [], defaultdict(str), 0, len(input)
        while i < n:
            num = 0
            while i < n and input[i] == '\t':
                num += 1
                i += 1
            c = ''
            while i < n and input[i] != '\n':
                c += input[i]
                i += 1
            files[num] = c
            if '.' in c:
                res.append(sum([len(files[i]) for i in range(num + 1)]) + num)
            i += 1
        return 0 if not res else max(res)

    # 1161. 最大层内元素和
    def maxLevelSum(self, root: Optional[TreeNode]) -> int:
        size, temp, index = 1, 0, 1
        nums, res = [root], [1, root.val]
        while len(nums):
            cur = nums.pop(0)
            temp += cur.val
            if cur.right:
                nums.append(cur.right)
            if cur.left:
                nums.append(cur.left)
            size -= 1
            if size == 0:
                if res[1] < temp:
                    res = [index, temp]
                size = len(nums)
                index += 1
                temp = 0
        return res[0]
    # 824. 山羊拉丁文

    def toGoatLatin(self, sentence: str) -> str:
        vowel = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']
        s = sentence.split(' ')
        for i in range(len(s)):
            if s[i][0] not in vowel:
                s[i] = s[i][1:] + s[i][0]
            s[i] += "ma"
            s[i] += "a" * (i + 1)
        return " ".join(s)

    # 587. 安装栅栏
    def outerTrees(self, trees: List[List[int]]) -> List[List[int]]:
        def cross(p: List[int], q: List[int], r: List[int]) -> int:
            return (q[0] - p[0]) * (r[1] - q[1]) - (q[1] - p[1]) * (r[0] - q[0])

        n = len(trees)
        if n < 4:
            return trees

        # 按照 x 从小到大排序，如果 x 相同，则按照 y 从小到大排序
        trees.sort()

        hull = [0]  # hull[0] 需要入栈两次，不标记
        used = [False] * n
        # 求凸包的下半部分
        for i in range(1, n):
            while len(hull) > 1 and cross(trees[hull[-2]], trees[hull[-1]], trees[i]) < 0:
                used[hull.pop()] = False
            used[i] = True
            hull.append(i)
        # 求凸包的上半部分
        m = len(hull)
        for i in range(n - 2, -1, -1):
            if not used[i]:
                while len(hull) > m and cross(trees[hull[-2]], trees[hull[-1]], trees[i]) < 0:
                    used[hull.pop()] = False
                used[i] = True
                hull.append(i)
        # hull[0] 同时参与凸包的上半部分检测，因此需去掉重复的 hull[0]
        hull.pop()

        return [trees[i] for i in hull]

    # 691. 贴纸拼词
    def minStickers(self, stickers: List[str], target: str) -> int:
        # 每一个贴纸相当于字典库，使用最少的字典库能够组成target
        # 采用dfs 搜索出所有的可行组合
        # n = len(stickers)

        # def intersection(a: str, b: str) -> str:
        #     b_ = Counter(b)
        #     for i in a:
        #         b_[i] -= 1
        #     return ''.join([k * v for k, v in b_.items() if v > 0])

        # def dfs(target: str, cur: int) -> List[int]:
        #     if target == '':
        #         return [cur]
        #     res = []
        #     for i in range(n):
        #         nxt = intersection(stickers[i], target)
        #         if nxt != target:
        #             res += dfs(nxt, cur + 1)
        #     return res
        # res = dfs(target, 0)
        # return -1 if not res else min(res)

        # 状态压缩

        m = len(target)

        def dp(mask: int) -> int:
            if mask == 0:
                return 0
            res = m + 1
            for sticker in stickers:
                left, cnt = mask, Counter(sticker)
                for i, c in enumerate(target):
                    if mask >> i & 1 and cnt[c]:
                        cnt[c] -= 1
                        left ^= 1 << i
                if left < mask:
                    res = min(res, dp(left) + 1)
            return res
        res = dp((1 << m) - 1)
        return res if res <= m else -1

    # 面试题 04.06. 后继者
    def inorderSuccessor(self, root: TreeNode, p: TreeNode) -> TreeNode:
        index = 1

        def dfs(root: TreeNode) -> TreeNode:
            nonlocal index
            if not root:
                return None
            # 左 根 右
            l = dfs(root.left)
            if index == 0:
                index -= 1
                return root
            if root == p:
                index -= 1
            r = dfs(root.right)
            return l if l else r
        return dfs(root)

    # 953. 验证外星语词典
    def isAlienSorted(self, words: List[str], order: str) -> bool:
        sequence = {e: i for i, e in enumerate(order)}

        def check(a: str, b: str) -> bool:
            m, n = len(a), len(b)
            for i in range(min(m, n)):
                if sequence[a[i]] > sequence[b[i]]:
                    return False
                if sequence[a[i]] < sequence[b[i]]:
                    return True
            return False if m > n else True
        for i in range(1, len(words)):
            if not check(words[i - 1], words[i]):
                return False
        return True

    # 668. 乘法表中第k小的数
    def findKthNumber(self, m: int, n: int, k: int) -> int:
        # 讨论小于x的有多少个，x可以用二分来进行查询
        def count(x: int) -> int:
            return sum(min(x // i, n) for i in range(1, m + 1))
        l, r = 1, m * n
        while l < r:
            mid, cnt = (l + r) >> 1, count((l + r) >> 1)
            if cnt >= k:
                r = mid
            else:
                l = mid + 1
        return l

    # 1408. 数组中的字符串匹配
    def stringMatching(self, words: List[str]) -> List[str]:
        words.sort(key=lambda x: len(x))
        res = set()
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                if words[j].count(words[i]) != 0:
                    res.add(words[i])
                    break
        return list(res)


s = Solution()
print(s.stringMatching(["leetcoder", "leetcode", "od", "hamlet", "am"]))
