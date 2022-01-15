import collections
from sys import setprofile, version_info
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
        s = [(arr[i], i) for i in range(n)]
        s.sort(key=lambda x: x[0])
        for k, i in s:
            # 需要找到区间内的 比 k 小的最大值
            l, r = i, i + 1
            for j in range(i + 1, i + d + 1 if i + d < n else n):
                if arr[j] >= k:
                    break
                if arr[j] > arr[r]:
                    r = j
            l = sorted(set(arr[(i - d if i - d >= 0 else 0): (i + d + 1 if i + d < n else n)]))
            t = l.index(k) - 1
            dp[i] = max(dp[i], dp[t] + 1 if t >= 0 else 0)
        return max(dp)


s = Solution()
a = ListNode.createListNode([1, 4, 5])
b = ListNode.createListNode([1, 3, 4])
c = ListNode.createListNode([2, 6])
print(s.maxJumps([6, 4, 14, 6, 8, 13, 9, 7, 10, 6, 12], 2))
# [5,1,5,5,2,5,4]
# [2,5,0,6,6]
# [2,5,0,1,2]
# [5, 1, 6]
# [20,100,10,12,5,13]
