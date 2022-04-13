from re import A
from typing import Counter, List
from NodeHelper.ListNode import ListNode


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


s = Solution()
print(s.removeDuplicateNodes(ListNode.createListNode([1, 1, 1, 5])))
