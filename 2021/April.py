import functools
from Modules.NodeHealper import TreeNode
from typing import List


class Solution:
    # 1006. 笨阶乘
    def clumsy(self, N: int) -> int:
        def caculator(a: int, b: int, oper: str) -> int:
            if oper == "*":
                return a*b
            elif oper == "/":
                return a//b
            elif oper == "+":
                return a+b
            elif oper == "-":
                return a-b
            else:
                return 0
        operator = ["*", "/", "+", "-"]
        stack1, stack2 = [N], []
        index = 0
        for i in range(N-1, 0, -1):
            if operator[index % 4] == "*" or operator[index % 4] == "/":
                stack1.append(caculator(stack1.pop(), i, operator[index % 4]))
            else:
                stack1.append(i)
                stack2.append(operator[index % 4])
            index += 1
        for i in range(len(stack2)):
            a, b = stack1.pop(0), stack1.pop(0)
            stack1.insert(0, caculator(a, b, stack2[i]))
        return stack1.pop()

    # 80. 删除有序数组中的重复项 II
    def removeDuplicates2(self, nums: List[int]) -> int:
        count, i = 1, 1
        while i < len(nums):
            count = count+1 if nums[i] == nums[i-1] else 1
            if count > 2:
                nums.pop(i)
                i -= 1
            i += 1
        return len(nums)

    # 81. 搜索旋转排序数组 II
    def search(self, nums: List[int], target: int) -> bool:
        def binary_search(nums: list, target: int, L: int, r: int) -> bool:
            if L > r:
                return False
            mid = (L+r)//2
            if target < nums[mid]:
                return binary_search(nums, target, L, mid-1)
            elif nums[mid] == target:
                return True
            else:
                return binary_search(nums, target, mid+1, r)
        index, n = 0, len(nums)
        for i in range(1, n):
            if nums[i] < nums[i-1]:
                index = i
                break
        if target >= nums[0]:
            return binary_search(nums, target, 0, index-1 if index > 0 else n-1)
        return binary_search(nums, target, index if index > 0 else 0, n-1)

    # 153. 寻找旋转排序数组中的最小值
    def findMin(self, nums: List[int]) -> int:
        L, r = 0, len(nums)-1
        while L < r:
            if nums[L] <= nums[r]:
                return nums[L]
            else:
                mid = (L+r) >> 1
                if nums[mid] > nums[r]:
                    L = mid+1
                else:
                    r = mid
        return nums[L]

    # 154. 寻找旋转排序数组中的最小值 II
    def findMin2(self, nums: List[int]) -> int:
        L, R = 0, len(nums)-1
        while L < R:
            mid = (L+R) >> 1
            if nums[mid] > nums[R]:
                L = mid+1
            elif nums[mid] < nums[R]:
                R = mid
            else:
                R -= 1
        return nums[L]

    # 179 最大数
    def largestNumber(self, nums: List[int]) -> str:
        nums.sort(key=functools.cmp_to_key(lambda x, y: 1 if str(x)+str(y) < str(y)+str(y) else -1))
        res = "".join(list(map(str, nums)))
        return "0" if res.startswith("0") else res

    # 783. 二叉搜索树节点最小距离
    def minDiffInBST(self, root: TreeNode) -> int:
        minVal = 100000

        def DFS(root: TreeNode, pre: int):
            nonlocal minVal
            if root is None:
                return
            DFS(root=root.left, pre=pre)

            if pre != -1:
                minVal = min(minVal, root.val-pre)
            pre = root.val

            DFS(root=root.right, pre=pre)

        DFS(root=root, pre=-1)
        return minVal

    # 213. 打家劫舍 II
    def rob(self, nums: List[int]) -> int:
        # dp 表示到第i家能够得到的最大现金 最后一家和第一家不能同时抢
        # 针对与第i家 有两种状态：抢，不抢
        # dp[i] = max(dp[i-1],dp[i-2]+nums[i])
        n = len(nums)
        dp1, dp2 = [nums[i] for i in range(n)], [nums[i] for i in range(n)]
        if n == 1:
            return nums[0]
        if n == 2:
            return max(nums[0], nums[-1])
        # 第一个不选
        for i in range(2, n):
            dp1[i] = max(dp1[i], dp1[i-1])
            if i >= 3:
                dp1[i] = max(dp1[i-1], dp1[i-2]+nums[i])
        # 最后一个不选
        dp2[1] = max(dp2[0], dp2[1])
        for i in range(2, n-1):
            dp2[i] = max(dp2[i-1], dp2[i-2]+nums[i])

        return max(dp1[-1], dp2[-2])

    # 26 删除有序数组中的重复项
    def removeDuplicates(self, nums: List[int]) -> int:
        i = 1
        while i < len(nums):
            if nums[i] == nums[i-1]:
                nums.pop(i)
            else:
                i += 1
        return len(nums)

    # 28 实现str()
    def strStr(self, haystack: str, needle: str) -> int:
        def kmp_getNext(origin: str) -> list:
            res = [0 for i in range(origin.__len__())]
            k, res[0] = -1, -1
            for i in range(1, origin.__len__()):
                while k > -1 and origin[k+1] != origin[i]:
                    k = res[k]
                if origin[i] == origin[k+1]:
                    k += 1
                res[i] = k
            return res

        if needle.__len__() == 0:
            return 0
        next, k = kmp_getNext(needle), -1
        res = 1
        for i in range(haystack.__len__()):
            while k > -1 and needle[k+1] != haystack[i]:
                k = next[k]
            if needle[k+1] == haystack[i]:
                k += 1
            if k+1 >= needle.__len__():
                res = i - needle.__len__()+1
                break
        return res if k+1 >= needle.__len__() else -1

    # 368.最大整除法
    def largestDivisibleSubset(self, nums: List[int]) -> List[int]:
        # 第i个位置能够得到的最大值
        nums.sort()
        dp, res = [[nums[i]] for i in range(nums.__len__())], 0
        for i in range(1, nums.__len__()):
            for j in range(0, i):
                if nums[i] % dp[j][0] == 0 and nums[i] % dp[j][-1] == 0:
                    dp[i] = dp[j] + [nums[i]] if dp[i].__len__() < len(dp[j]+[nums[i]])else dp[i]
            if dp[res].__len__() < dp[i].__len__():
                res = i
        return dp[res]

    # 377.组合总数
    def combinationSum4(self, nums: List[int], target: int) -> int:
        dp = [1]+[0]*target
        for i in range(1, target+1):
            for num in nums:
                if num <= i:
                    dp[i] += dp[i-num]
        return dp[target]

    # 897.递增顺序搜索树
    def increasingBST(self, root: TreeNode) -> TreeNode:
        res = TreeNode()
        index = res

        def inOrder(root: TreeNode):
            nonlocal index
            if root is None:
                return
            inOrder(root=root.left)
            index.right = TreeNode(root.val)
            index = index.right
            inOrder(root=root.right)

        inOrder(root=root)
        return res.right

    # 1011. 在 D 天内送达包裹的能力
    def shipWithinDays(self, weights: List[int], D: int) -> int:
        def count(target: int, weights: List[int]) -> int:
            d, cur = 1, 0
            for i in weights:
                cur += i
                if cur > target:
                    d += 1
                    cur = i
            return d
        L, r = max(weights), sum(weights)
        while L < r:
            mid = (L+r) >> 1
            days = count(mid, weights)
            if days > D:
                L = mid+1
            else:
                r = mid
        return L

    # 938.二叉搜索树的范围和
    def rangeSumBST(self, root: TreeNode, low: int, high: int) -> int:
        if not root:
            return 0
        if root.val > high:
            return self.rangeSumBST(root.left, low, high)
        if root.val < low:
            return self.rangeSumBST(root.right, low, high)
        return root.val + self.rangeSumBST(root.left, low, high) + self.rangeSumBST(root.right, low, high)


s = Solution()
# nums = [1, 2, 3, 4, 5, 6]
# print(bisect.bisect(nums, 5, 0, 3))

root = TreeNode.createTreeNode([10, 5, 15, 3, 7, None, 18])
print(s.rangeSumBST(root, 7, 15))
