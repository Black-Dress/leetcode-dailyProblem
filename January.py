import collections
from typing import Collection, Dict, List, Literal
from NodeHelper.ListNode import ListNode
from NodeHelper.TreeNode import TreeNode
import bisect


class Solution:
    # 23. 合并K个升序链表
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        def merge(a: ListNode, b: ListNode) -> ListNode:
            res = ListNode(0)
            index = res
            while a is not None and b is not None:
                if a.val < b.val:
                    index.next = a
                    a = a.next
                else:
                    index.next = b
                    b = b.next
                index = index.next
            if a is not None:
                index.next = a
            else:
                index.next = b
            return res.next

        def merge_(l: int, r: int, lists: List[ListNode]) -> ListNode:
            if l == r:
                return lists[l]
            if l > r:
                return None
            mid = (l + r) >> 1
            return merge(merge_(l, mid, lists), merge_(mid + 1, r, lists))

        return merge_(0, len(lists) - 1, lists)

    # 32. 最长有效括号
    def longestValidParentheses(self, s: str) -> int:
        stack, n = [-1], len(s)
        res = 0
        for i in range(n):
            if s[i] == '(':
                stack.append(i)
            else:
                stack.pop()
                if len(stack) == 0:
                    stack.append(i)
                else:
                    res = max(res, i - stack[-1])
        return res

    # 33. 搜索旋转排序数组
    def search(self, nums: List[int], target: int) -> int:
        # 确定 target和nums[0]的大小关系，确定target在左半区还是右半区
        l, r = 0, len(nums) - 1
        # while l <= r:
        #     mid = (l + r) >> 1
        #     if nums[mid] == target:
        #         return mid
        #     # 如果在左半区
        #     if nums[0] <= target < nums[mid]:
        #         r = mid - 1
        #     if nums[0] <= nums[mid] < target:
        #         l = mid + 1
        #     if nums[mid] < nums[0] and target >= nums[0]:
        #         r = mid - 1
        #     # 如果在右半区
        #     if nums[mid] < target < nums[0]:
        #         l = mid + 1
        #     if target < nums[mid] < nums[0]:
        #         r = mid - 1
        #     if nums[mid] >= nums[0] and target < nums[0]:
        #         l = mid + 1
        # return -1
        while l <= r:
            mid = (l + r) >> 1
            if nums[mid] == target:
                return mid
            # 左半区
            if target >= nums[0]:
                if nums[0] <= nums[mid] < target:
                    l = mid + 1
                else:
                    r = mid - 1
            # 右半区
            else:
                if target < nums[mid] < nums[0]:
                    r = mid - 1
                else:
                    l = mid + 1
        return -1

    # 39. 组合总和
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        # 没有重复数字，可以重复选取，不能有重复解
        # def DFS(candiates: List[int], target: int, index: int, cur: List[int], res: List[List[int]]):
        #     for i in range(index, len(candidates)):
        #         if cur[0] == target:
        #             res.append(cur[1:].copy())
        #             return
        #         if cur[0] < target:
        #             cur[0] += candidates[i]
        #             cur.append(candidates[i])
        #             DFS(candiates, target, i, cur, res)
        #             cur[0] -= candidates[i]
        #             cur.pop()
        #         if cur[0] > target:
        #             return
        def DFS(candidates: List[int], target: int, index: int, cur: List[int], res: List[List[int]]):
            if target == 0:
                res.append(cur.copy())
                return
            if index == candidates.__len__() or target < candidates[index]:
                return
            for i in range(index, candidates.__len__()):
                if target >= candidates[i]:
                    DFS(candidates, target - candidates[i], i, cur + [candidates[i]], res)
                else:
                    return

        candidates.sort()
        res = []
        DFS(candidates, target, 0, [], res)
        return res

    # 40. 组合总和 II
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        # 有重复数字的情况下要求没有重复解和重复数字利用
        # def DFS(candiates: List[int], target: int, index: int, cur: List[int], res: List[List[int]]):
        #     # cur 状态已经走过，return
        #     if cnt.get("".join(str(i) for i in cur[1:])) is not None:
        #         return
        #     if cur[0] == target:
        #         res.append(cur[1:].copy())
        #         return
        #     for i in range(index, len(candidates)):
        #         if cur[0] < target:
        #             cur[0] += candidates[i]
        #             cur.append(candidates[i])
        #             # 从下一位开始
        #             DFS(candiates, target, i + 1, cur, res)
        #             # 进入一次循环之后更新，状态
        #             cnt["".join(str(j) for j in cur[1:])] = 1
        #             cur[0] -= candidates[i]
        #             cur.pop()
        #         if cur[0] > target:
        #             return

        # candidates.sort()
        # res, cnt = [], collections.defaultdict(str)
        # DFS(candidates, target, 0, [0], res)
        # return res

        # def DFS(candiates: List[int], target: int, index: int, cur: List[int], res: List[List[int]]):
        #     if cur[0] == target:
        #         res.append(cur[1:].copy())
        #         return
        #     for i in range(index, candiates.__len__()):
        #         if i > index and candiates[i] == candiates[i - 1]:
        #             continue
        #         if cur[0] < target:
        #             cur[0] += candidates[i]
        #             cur.append(candidates[i])
        #             # 从下一位开始
        #             DFS(candiates, target, i + 1, cur, res)
        #             # 进入一次循环之后更新，状态
        #             cur[0] -= candidates[i]
        #             cur.pop()
        #         else:
        #             break

        def DFS(candidates: List[int], target: int, index: int, cur: List[int], res: List[List[int]]):
            if target == 0:
                res.append(cur.copy())
                return
            if target < 0 or index == candidates.__len__() or target < candidates[index]:
                return
            for i in range(index, candidates.__len__()):
                if i > index and candidates[i] == candidates[i - 1]:
                    continue
                if candidates[i] <= target:
                    DFS(candidates, target - candidates[i], i + 1, cur + [candidates[i]], res)
                else:
                    return

        candidates.sort()
        res = []
        DFS(candidates, target, 0, [], res)
        return res

    # 216. 组合总和 III
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        # 限制数量为k，没有重复数字情况下的没有重复解，和重复数字利用
        def DFS(index: int, k: int, n: int, nums: List[int], cur: List[int], res: List[List[int]]):
            if n == 0 and k == 0:
                res.append(cur.copy())
                return
            if k == 0 or n < 0 or index == nums.__len__() or n < nums[index]:
                return
            for i in range(index, nums.__len__()):
                if nums[i] <= n:
                    DFS(i + 1, k - 1, n - nums[i], nums, cur + [nums[i]], res)
                else:
                    return
        res, nums = [], [i for i in range(1, 10)]
        DFS(0, k, n, nums, [], res)
        return res

    # 377. 组合总和 Ⅳ
    def combinationSum4(self, nums: List[int], target: int) -> int:
        # 可以重复选取，可以有重复解（顺序不同即可）
        # DFS 超时
        # def DFS(nums: List[int], target: int):
        #     nonlocal cnt, res
        #     if target > 0 and cnt.get(target) is not None:
        #         res += cnt[target]

        #     if target == 0:
        #         res += 1
        #         return
        #     if target < 0:
        #         return
        #     for num in nums:
        #         if num <= target:
        #             DFS(nums, target - num)
        #         else:
        #             return
        # res, cnt = 0, collections.Counter()
        # nums.sort()
        # DFS(nums, target)
        # dp[i] 表示和为i的时候的总和数
        # dp[i] 的组合数应该等于nums中含有数有的
        dp = [1] + [0] * target
        for i in range(1, target + 1):
            for j in nums:
                if j <= i:
                    dp[i] += dp[i - j]
        return dp[target]

    # 71. 简化路径
    def simplifyPath(self, path: str) -> str:
        stack, n = [], len(path)
        i, j = 0, 0
        while i < n and j < n:
            # 移动到目录
            while j < n and path[j] == '/':
                j += 1
            i = j
            # 移动到目录后一个斜杠
            while i < n and path[i] != '/':
                i += 1
            if j != i and path[j:i] == '..' and stack.__len__() != 0:
                stack.pop()
            if j != i and path[j:i] != '..' and path[j:i] != '.':
                stack.append("".join(path[j:i]))
            j = i
        res = "".join(['/' + i for i in stack])
        return "/" if res == "" else res

    # 1614. 括号的最大嵌套深度
    def maxDepth(self, s: str) -> int:
        stack, res = [], 0
        for i in range(s.__len__()):
            if s[i] == '(':
                stack.append('(')
            if s[i] == ')' and stack.__len__() != 0:
                stack.pop()
            res = max(res, stack.__len__())
        return res

    # 480. 滑动窗口中位数
    def medianSlidingWindow(self, nums: List[int], k: int) -> List[float]:
        # 滑动窗口维持有序，利用二分进行插入和删除
        def median(s: List[int]): return (s[(s.__len__() - 1) // 2] + s[s.__len__() // 2]) / 2
        window, res = sorted(nums[:k]), []
        res.append(median(window))
        for i, j in zip(nums[:-k], nums[k:]):
            window.pop(bisect.bisect_left(window, i))
            window.insert(bisect.bisect_left(window, j), j)
            res.append(median(window))
        return res

    # 1629. 按键持续时间最长的键
    def slowestKey(self, releaseTimes: List[int], keysPressed: str) -> str:
        cnt, n = collections.defaultdict(int), len(releaseTimes)
        cnt[keysPressed[0]] = releaseTimes[0]
        for i in range(1, n):
            cnt[keysPressed[i]] = max(cnt[keysPressed[i]], releaseTimes[i] - releaseTimes[i - 1])
        res = sorted(list(cnt.items()), key=lambda x: (x[1], ord(x[0])), reverse=True)
        return res[0][0]

    # 49. 字母异位词分组
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        # 利用n位26进制的数判断是否是异位词
        def num(s: str) -> tuple:
            cnt = [0] * 26
            for i in range(s.__len__()):
                cnt[ord(s[i]) - ord('a')] += 1
            return tuple(cnt)

        cnt = collections.defaultdict(list)
        for s in strs:
            cnt[num(s)].append(s)
        return list(cnt.values())

    # 70.爬楼梯
    def climbStairs(self, n: int) -> int:
        # dp[i] 代表爬到第i层的方法
        # dp[i] = dp[i-1] + dp[i-2]
        # dp[0] = 1 dp[1] = 1 dp[2] = dp[1] + dp[0]
        dp = [1, 1] + [0] * (n - 1)
        for i in range(2, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[n]

    # 306. 累加数
    def isAdditiveNumber(self, num: str) -> bool:
        def DFS(a: int, b: int, num: str) -> bool:
            if num.__len__() == 0:
                return True
            for i in range(len(num)):
                c = int(num[:i + 1])
                if (num[:i + 1].startswith('0') and i > 0) or a + b < c:
                    break
                if a + b == c:
                    return DFS(b, c, num[i + 1:])
            return False

        for i in range(len(num) - 2):
            for j in range(i + 1, len(num) - 1):
                if (num[:i + 1].startswith('0') and i > 0) or (num[i + 1:j + 1].startswith('0') and j > i + 1):
                    continue
                if DFS(int(num[:i + 1]), int(num[i + 1:j + 1]), num[j + 1:]):
                    return True
        return False

    # 236. 二叉树的最近公共祖先
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if root is None:
            return None
        if root.val == p.val or root.val == q.val:
            return root
        l = self.lowestCommonAncestor(root.left, p, q)
        r = self.lowestCommonAncestor(root.right, p, q)
        return root if l is not None and r is not None else (l if l is not None else r)

    # 19. 删除链表的倒数第 N 个结点
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        if head.next is None:
            return None
        f, s = ListNode(0), ListNode(0)
        res = s
        f.next = head
        s.next = head
        for i in range(n):
            f = f.next
        while f is not None and f.next is not None:
            f = f.next
            s = s.next
        s.next = s.next.next
        return res.next

    # 45. 跳跃游戏 II
    def jump(self, nums: List[int]) -> int:
        k, res, n = 0, 0, len(nums)
        dis = 0
        # for i in range(n):
        #     if k >= n - 1:
        #         break
        #     if i > dis:
        #         res += 1
        #         dis = k
        #     k = max(k, i + nums[i])
        # return res + 1 if n != 1 else 0
        # n-1 能够避免只有一个的情况
        for i in range(n - 1):
            k = max(k, i + nums[i])
            if i == dis:
                res += 1
                dis = k
        return res

    # LCP 09. 最小跳跃次数
    def minJump(self, jump: List[int]) -> int:
        # dp[i] 存储跳跃到i点需要的最少次数
        n = len(jump)
        dp = [0] + [n] * (n - 1)
        res = 0
        for i in range(n):
            for j in range(i + 1):
                x = dp[i] + 2 if j != i else dp[i] + 1
                if j + jump[j] >= n:
                    res = x
                    break
                dp[j + jump[j]] = min(x, dp[j + jump[j]])
        return res


s = Solution()
a = ListNode.createListNode([1, 4, 5])
b = ListNode.createListNode([1, 3, 4])
c = ListNode.createListNode([2, 6])
print(s.minJump([2, 5, 1, 1, 1, 1]))
