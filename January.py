import collections
from heapq import heapify, heappop, heappush
import heapq
from pickletools import long1
from queue import PriorityQueue
from sys import setprofile, version_info
from time import time
from typing import Collection, Dict, List, Literal
from xml.dom import INDEX_SIZE_ERR

from pkg_resources import EggMetadata
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

    # 1036. 逃离大迷宫
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        # # 洪泛法模拟找到一条通路到达target(超时)
        # n = pow(10, 6)
        # # visit 数组太大，利用map记录访问过的节点
        # visit, que = collections.defaultdict(int), [source]
        # dx, dy = [0, 0, 1, -1], [1, -1, 0, 0]
        # # 初始化visit
        # for block in blocked:
        #     visit[(block[0], block[1])] = 1
        # visit[(target[0], target[1])] = 2

        # while que.__len__() != 0:
        #     cur = que.pop(0)
        #     for i in range(4):
        #         x, y = cur[0] + dx[i], cur[1] + dy[i]
        #         if x < 0 or x >= n or y < 0 or y >= n or visit[(x, y)] == 1:
        #             continue
        #         if visit[(x, y)] == 2:
        #             return True
        #         if visit[(x, y)] == 0:
        #             que.append([x, y])
        #             visit[(x, y)] = 1
        # return False

        # 带上界的BFS
        # 通过blocked数组判断是否是否将source 和 target 进行了分割
        # 两种情况，blocked 自身把 source 或者 target 围起来，通过边界将他们围起来
        # 带上界的BFS。经过数学证明：一个包围圈内最多有n(n-1)/2个非障碍位，n表示blocked的数量
        # 如果循环在n(n-1)/2之前停下且没有遇见target 表示不可达，如果超过了n(n-1)/2，则表示可达
        BOUND = 10**6
        FOUND, VALID, NOTFOUND = 1, 0, -1
        hashblock = set((pos[0], pos[1]) for pos in blocked)

        def check(s: List[int], e: List[int]) -> int:
            dx, dy = [0, 0, 1, -1], [1, -1, 0, 0]
            visit, que = collections.defaultdict(int), [s]
            # 计数经过的非障碍物位置
            n = len(blocked)
            cnt = n * (n - 1) // 2
            visit[(e[0], e[1])] = 2
            visit[(s[0], s[1])] = 1
            while len(que) > 0 and cnt > 0:
                cur = que.pop(0)
                for nx, ny in zip(dx, dy):
                    x, y = cur[0] + nx, cur[1] + ny
                    if x < 0 or x >= BOUND or y < 0 or y >= BOUND or visit[(x, y)] == 1 or (x, y) in hashblock:
                        continue
                    if visit[(x, y)] == 2:
                        return FOUND
                    cnt -= 1
                    visit[(x, y)] = 1
                    que.append((x, y))

            return NOTFOUND if cnt > 0 else VALID

        if (res := check(source, target)) == FOUND:
            return True
        elif res == NOTFOUND:
            return False
        else:
            res = check(target, source)
            if res == NOTFOUND:
                return False
        return True

    # LCP 09. 最小跳跃次数
    def minJump(self, jump: List[int]) -> int:
        # 动态规划 dp[i] 表示到达i时要跳出去的最小跳跃次数
        # i 这个节点可以作为初始节点，也可以作为中间节点，分别讨论并更新dp表
        # 1. 计算从i节点往右跳，跳出机器需要花费的最小距离
        #   i+jump[i]>=n -> dp[i] = 1
        #   i+jump[i]<n -> dp[i] = dp[i+jump[i]]+1
        # 2. i作为中间节点时（即j节点跳到i节点然后跳出去），更新[i,n-1]的dp表，更新dp表只需要更新dp[j]>=dp[i]+1 的那一部分
        #   dp[j] = min(dp[i]+1,dp[j]])
        n = len(jump)
        dp = [0] * (n - 1) + [1]
        for i in range(n - 2, -1, -1):
            # dp[i] 直接更新而不需要判断最小值，因为是从重点遍历到起点，dp[i] 永远是最小跳出次数
            dp[i] = 1 if i + jump[i] >= n else dp[i + jump[i]] + 1
            for j in range(i + 1, n):
                if dp[j] < dp[i] + 1:
                    break
                dp[j] = dp[i] + 1
        return dp[0]

    # 1306. 跳跃游戏 III
    def canReach(self, arr: List[int], start: int) -> bool:
        # 题目要求能否到达0的任意位置（即指定位置）
        # 利用BFS 更新能够到达的节点
        n = len(arr)
        que = collections.deque()
        visit = [0] * n
        que.append(start)
        visit[start] = 1
        while len(que) != 0:
            cur = que.popleft()
            if cur + arr[cur] < n and visit[cur + arr[cur]] == 0:
                que.append(cur + arr[cur])
                visit[cur + arr[cur]] = 1
            if cur - arr[cur] >= 0 and visit[cur - arr[cur]] == 0:
                que.append(cur - arr[cur])
                visit[cur - arr[cur]] = 1

        for i in range(n):
            if arr[i] == 0 and visit[i] == 1:
                return True
        return False

    # 1345. 跳跃游戏 IV
    # 每一步，你可以从下标 i 跳到下标：
    # i + 1 满足：i + 1 < arr.length
    # i - 1 满足：i - 1 >= 0
    # j 满足：arr[i] == arr[j] 且 i != j
    # 请你返回到达数组最后一个元素的下标处所需的 最少操作次数 。
    def minJumps(self, arr: List[int]) -> int:
        # 每一个节点i都是从 i+1 , i-1 ,j 三的个地方跳过来的，找到最小值就行了
        # 如果从i+1跳到i比i小的情况只能时 i+1这个节点在作为j的时候被更新过
        # 重点在于j，利用 map 记录所有值相同的下标，循环更新
        # cnt, n = collections.defaultdict(list), len(arr)
        # dp = [0] + [n] * (n - 1)
        # for i in range(n):
        #     cnt[arr[i]].append(i)
        # for i in range(n):
        #     if i - 1 >= 0:
        #         dp[i] = min(dp[i], dp[i - 1] + 1)
        #     if i + 1 < n:
        #         dp[i] = min(dp[i + 1] + 1, dp[i])
        #     for j in cnt[arr[i]]:
        #         dp[j] = min(dp[j], dp[i] + 1)
        # return dp[-1]
        # dp 不行在于 dp 转换公式会产生循环依赖的问题，dp[i+1] 依赖于 dp[i] ,d[i]也依赖于dp[i+1]
        # BFS
        # 遍历 i-1 ,i+1 , j 这些数字并且进行step的更新
        cnt, n = collections.defaultdict(list), len(arr)
        for i in range(n):
            cnt[arr[i]].append(i)
        visit = [True] + [False] * (n - 1)
        que = collections.deque()
        que.append((0, 0))
        while len(que) != 0:
            i, step = que.popleft()
            for j in (cnt[arr[i]] + [i - 1, i + 1]):
                if 0 <= j < n and visit[j] == False:
                    if j == n - 1:
                        return step + 1
                    visit[j] = True
                    que.append((j, step + 1))
            # 防止多次遍历
            cnt[arr[i]].clear()
        return 0

    # 334. 递增的三元子序列
    def increasingTriplet(self, nums: List[int]) -> bool:
        # 贪心优先贪比start大的最小的那一个
        n, i = len(nums), 0
        start, mid, premid = 0, 0, 0
        while i < n:
            # 更新start
            if nums[i] < nums[start]:
                premid = mid if mid != start else premid
                start, mid = i, i
            # 更新mid
            if (nums[start] == nums[mid] and nums[i] > nums[start]) or nums[start] < nums[i] < nums[mid]:
                mid = i
            # 找到end
            if (nums[i] > nums[mid] and nums[mid] != nums[start]) or (nums[i] > nums[premid] and premid != 0):
                return True
            i += 1
        return False

    # 1340. 跳跃游戏 V
    def maxJumps(self, arr: List[int], d: int) -> int:
        # 只能跳跃到比自己矮的位置-> 从高位开始跳
        # dp[i] i 位置最多能够访问的下标数量 dp[i] = max(dp[i+x]+dp[i-x])+1
        # dp[i+x] dp[i-x] 必须要能够跳到,这一题不会存在相互依赖因为 加入dp[i] 能够跳到 dp[i+x] 那么 dp[i+x] 是无法跳到dp[i]的
        n = len(arr)
        dp = [1] * n
        s = sorted([(arr[i], i) for i in range(n)], key=lambda x: x[0])
        for k, i in s:
            # 左右比k小的最大值位置
            l, r = i - 1, i + 1
            # 找右边
            while r < n and arr[r] < arr[i] and r <= i + d:
                dp[i] = max(dp[r] + 1, dp[i])
                r += 1
            # 找左边
            while l >= 0 and arr[l] < arr[i] and l >= i - d:
                dp[i] = max(dp[l] + 1, dp[i])
                l -= 1
        return max(dp)

    # 1696. 跳跃游戏 VI
    def maxResult(self, nums: List[int], k: int) -> int:
        # dp[i]表示到达i位置时能够得到的最大分数之和
        # dp[i] = max(dp[i-k-1:i])+nums[i]
        # stack 维护 [i-k-1:i]内的最大值
        # return dp[n-1]
        n = len(nums)
        dp = [nums[0]] + [-float('inf')] * (n - 1)
        stack = collections.deque()
        for i in range(n):
            # 弹出越界元素
            while len(stack) != 0 and stack[-1][0] < i - k:
                stack.pop()
            # 更新dp
            dp[i] = max(dp[i], stack[0][1])
            # 压栈
            while len(stack) != 0 and dp[i] >= stack[-1][1]:
                stack.pop()
            stack.append((i, dp[i]))
        return dp[-1]

    # 373. 查找和最小的K对数字
    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        # k^2 解法
        # que = list()
        # heapq.heapify(que)
        # i = 0
        # while i < len(nums1) and i < k:
        #     j = 0
        #     while j < len(nums2) and j < k:
        #         heapq.heappush(que, [nums1[i] + nums2[j], [nums1[i], nums2[j]]])
        #         j += 1
        #     i += 1
        # return [heapq.heappop(que)[1] for i in range(min(k, len(que)))]
        # 多路归并
        # 假设 (i,j)是答案的点，那么(i+1,j),(i,j+1)是后续的待选点,(0,0)是必选点
        # 优先把(0,[1,k])放入备选点,之后的循环加入j就可以防止重复加入
        m, n = len(nums1), len(nums2)
        res = list()
        que = [(nums1[i] + nums2[0], i, 0) for i in range(min(k, m))]
        while que and len(res) < k:
            _, i, j = heappop(que)
            res.append([nums1[i], nums2[j]])
            if j + 1 < n:
                heappush(que, (nums1[i] + nums2[j + 1], i, j + 1))
        return res

    # 1716. 计算力扣银行的钱
    def totalMoney(self, n: int) -> int:
        # 计算有几周 -> 计算剩余几天
        if n <= 7:
            return int((1 + n) * n / 2)
        weeks, rest = n // 7, n % 7
        a = (1 + 7) * 7 / 2 * weeks + (1 + weeks - 1) * (weeks - 1) / 2 * 7
        b = (weeks + 1 + weeks + 1 + rest - 1) * rest / 2
        return int(a + b)

    # 1220. 统计元音字母序列的数目
    def countVowelPermutation(self, n: int) -> int:
        # 该问题可以转换成在指定矩阵中长度为n的路径有多少条
        # dp[i][j] 表示从i点开始，长度为j的路径有 dp[i][j] 条
        # dp[i][j] = sum(dp[k][j-1]) k in vowels[i]
        # vowels = [
        #     [1],
        #     [0, 2],
        #     [0, 1, 3, 4],
        #     [2, 4],
        #     [0]
        # ]
        # dp = [[0 for i in range(n + 1)] for j in range(vowels.__len__())]
        # # 初始化长度为一的时候的情况
        # for i in range(5):
        #     dp[i][1] = 1
        # res = 0
        # for j in range(1, n + 1):
        #     for i in range(vowels.__len__()):
        #         for k in range(vowels[i].__len__()):
        #             dp[i][j] += dp[vowels[i][k]][j - 1]
        # for i in range(vowels.__len__()):
        #     res += dp[i][n]
        #     res = res % (pow(10, 9) + 7)
        # return res
        # 简单dp
        mod = pow(10, 9) + 7
        dp = (1, 1, 1, 1, 1)
        for i in range(1, n):
            dp = (dp[1] % mod, (dp[0] + dp[2]) % mod, (dp[0] + dp[1] + dp[3] + dp[4]) % mod, (dp[2] + dp[4]) % mod, (dp[0]) % mod)
        return sum(dp) % mod

    # 539. 最小时间差
    def findMinDifference(self, timePoints: List[str]) -> int:
        times = sorted([int(time[:2]) * 60 + int(time[3:]) for time in timePoints])
        res = 2400
        for i in range(1, times.__len__()):
            res = min(res, times[i] - times[i - 1])
        res = min(times[-1] - times[0], res, times[0] + 1440 - times[-1])
        return res

    # 219. 存在重复元素 II
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        i, n = 0, len(nums)
        j = 1
        cnt = collections.defaultdict(int)
        while i < n and i <= k:
            if cnt[nums[i]] == 0:
                cnt[nums[i]] += 1
            else:
                return True
            i += 1
        cnt[nums[0]] -= 1
        while i < n:
            if cnt[nums[i]] == 0:
                cnt[nums[i]] += 1
            else:
                return True
            cnt[nums[j]] -= 1
            i += 1
            j += 1
        return False

    # 剑指 Offer 03. 数组中重复的数字
    def findRepeatNumber(self, nums: List[int]) -> int:
        n = len(nums)
        for i in range(n):
            while nums[i] != i:
                if nums[i] == nums[nums[i]]:
                    return nums[i]
                idx = nums[i]
                nums[i], nums[idx] = nums[idx], nums[i]
        return -1

    # 2029. 石子游戏 IX
    def stoneGameIX(self, stones: List[int]) -> bool:
        # alice 获胜条件：
        #   bob移除石头后，石头总价值能够被三整除
        # bob获胜条件：
        #   alice 移除石头之后，石头价值能够被三整除
        #   价值一直不能被三整除，直到所有石头被移除
        # 移除的石头的价值可以被归类为三种 0,1,2 (stones[i]%3)
        # 为了不出现移除石头之后的价值能够被三整除，那么1,2是不能先后被移除的。
        # 所以两者都安全的序列为 112121，221212
        # 为了能让alice获胜在两个序列中得到的条件就是 cnt2>=cnt1>0 , cnt1>=cnt2>0
        # 综合两个条件 -> cnt2>0 and cnt1>0
        # 先后手交换情况下，即在安全序列下，后手的alice需要赢则要满足 cnt1>cnt2+2 cnt2>cnt1+2
        # 综合cnt0的先后手交换情况，就可以通过cnt数量来进行判断
        cnt = [0, 0, 0]
        for i in stones:
            cnt[i % 3] += 1
        # 先后手未交换
        if cnt[0] % 2 == 0:
            return cnt[2] >= cnt[1] > 0 or cnt[1] >= cnt[2] > 0
        # 先后手交换
        return cnt[1] > cnt[2] + 2 or cnt[2] > cnt[1] + 2

    # 剑指 Offer 04. 二维数组中的查找
    def findNumberIn2DArray(self, matrix: List[List[int]], target: int) -> bool:
        # 利用二分查找，二分查找按区间进行查找
        # 若小于mid 则向左上半区进行查询，若大于mid则向右半区和下半区进行查询
        # bound 存储的是左上角和右下角的位置，用来指代半区
        def binarySerach(matrix: List[List[int]], bound: List[int], target: int) -> bool:
            a, b, c, d = bound
            if a > c or b > d:
                return False
            mid = [(a + c) >> 1, (b + d) >> 1]
            if target == matrix[mid[0]][mid[1]]:
                return True
            if target > matrix[mid[0]][mid[1]]:
                x = binarySerach(matrix, [a, mid[1] + 1, c, d], target)
                y = binarySerach(matrix, [mid[0] + 1, b, c, mid[1]], target)
                return x or y
            else:
                x = binarySerach(matrix, [a, b, c, mid[1] - 1], target)
                y = binarySerach(matrix, [a, mid[1], mid[0] - 1, d], target)
                return x or y

        if len(matrix) == 0 or len(matrix[0]) == 0:
            return False
        return binarySerach(matrix, [0, 0, len(matrix) - 1, len(matrix[0]) - 1], target)

    # 剑指 Offer 11. 旋转数组的最小数字
    def minArray(self, numbers: List[int]) -> int:
        n = len(numbers)
        l, r = 0, n - 1
        while l <= r:
            mid = (l + r) >> 1
            if numbers[mid] < numbers[r]:
                r = mid
            elif numbers[mid] == numbers[r]:
                r -= 1
            else:
                l = mid + 1
        return numbers[l]

    # 剑指 Offer 50. 第一个只出现一次的字符
    def firstUniqChar(self, s: str) -> str:
        # 一定要遍历所有的字符才能知道哪些节点是不重复的
        # 在遍历过程中利用有序map 存储所有的节点，
        # 如果有重复的数字则不添加到map中，并且将map中的元素删除，同时也得确保，之后不再被添加进来
        alpha = collections.OrderedDict()
        cnt = [0] * 26
        for i in s:
            index = ord(i) - ord('a')
            if cnt[index] == 0:
                alpha[i] = 1
                cnt[index] += 1
            else:
                if alpha.get(i) is not None:
                    alpha.pop(i)
        if len(alpha.items()) == 0:
            return ' '
        return alpha.popitem(False)[0]

    # 2045. 到达目的地的第二短时间
    def secondMinimum(self, n: int, edges: List[List[int]], time: int, change: int) -> int:
        # 构建无向图,节点值从1开始
        graph = [[] for i in range(n + 1)]
        for i in edges:
            x, y = i[0], i[1]
            graph[x].append(y)
            graph[y].append(x)
        que = collections.deque([(1, 0)])
        dis = [[float('inf'), float('inf')] for i in range(n + 1)]
        dis[1][0] = 0
        # 只要找到到达n节点的最短路，就可以找到开销最小的时间
        while dis[n][1] == float('inf'):
            cur = que.popleft()
            for i in graph[cur[0]]:
                d = cur[1] + 1
                if d < dis[i][0]:
                    dis[i][0] = d
                    que.append((i, d))
                elif dis[i][0] < d < dis[i][1]:
                    dis[i][1] = d
                    que.append((i, d))
        res = 0
        for i in range(dis[n][1]):
            if res % (change * 2) >= change:
                res += 2 * change - res % (2 * change)
            res += time
        return res


s = Solution()
# print(s.firstUniqChar("abaccdeff"))
# [5,1,5,5,2,5,4]
# [2,5,0,6,6]
# [2,5,0,1,2]
# [5, 1, 6]
# [20,100,10,12,5,13]
a = dict()
a[1] = 2
a[4] = 2
a[3] = 2
print(list(a.keys())[-1])
