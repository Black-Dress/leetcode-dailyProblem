from collections import defaultdict, deque
from itertools import combinations, permutations, product
from platform import node
import queue
from typing import Counter, List


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


s = Solution()
print(s.lexicalOrder(103))
# print(list(permutations(range(9), 0)).__len__())
# print(list(combinations(range(4), 2)))
