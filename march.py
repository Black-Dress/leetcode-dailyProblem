import bisect
# Definition for singly-linked list.


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class NumMatrix:

    def __init__(self, matrix: [[int]]):
        if len(matrix) == 0 or len(matrix[0]) == 0:
            return
        self.row = len(matrix)
        self.col = len(matrix[0])
        # 修改每一格的值
        for i in range(self.row-1, -1, -1):
            for j in range(self.col-1, -1, -1):
                down = matrix[i+1][j] if i+1 < self.row else 0
                right = matrix[i][j+1] if j+1 < self.col else 0
                num = matrix[i+1][j+1] if i+1 < self.row and j+1 < self.col else 0
                matrix[i][j] += (down+right-num)
        self.matrix = matrix

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        A = self.matrix[row1][col1]
        B = self.matrix[row1][col2+1] if col2+1 < self.col else 0
        C = self.matrix[row2+1][col1] if row2+1 < self.row else 0
        D = self.matrix[row2+1][col2+1] if row2+1 < self.row and col2+1 < self.col else 0
        return A-B-C+D

        # Your NumMatrix object will be instantiated and called as such:
        # obj = NumMatrix(matrix)
        # param_1 = obj.sumRegion(row1,col1,row2,col2)


class NestedInteger:
    def __init__(self):
        self.integer = -1
        self.integerList = [NestedInteger]
        self.isInt = False

    def isInteger(self) -> bool:
        return self.isInt
        """
        @return True if this NestedInteger holds a single integer, rather than a nested list.
        """

    def getInteger(self) -> int:
        return self.integer
        """
        @return the single integer that this NestedInteger holds, if it holds a single integer
        Return None if this NestedInteger holds a nested list
        """

    def getList(self) -> []:
        return self.integerList
        """
        @return the nested list that this NestedInteger holds, if it holds a nested list
        Return None if this NestedInteger holds a single integer
        """


class NestedIterator:

    def __init__(self, nestedList: [NestedInteger]):
        nestedList.reverse()
        self.stack = nestedList

    def next(self) -> int:
        item = self.stack.pop()
        return item.getInteger()

    def hasNext(self) -> bool:
        if len(self.stack) == 0:
            return False
        # 跳过所有空的数组
        while len(self.stack) and not self.stack[-1].isInteger():
            item = self.stack.pop()
            item.getList().reverse()
            self.stack.extend(item.getList())
        return len(self.stack) != 0


class Solution:
    # 338. 比特位计数
    def countBits(self, num: int) -> [int]:
        # def getNum(n: int):
        #     count = 0
        #     while (n):
        #         count += 1
        #         n = n & (n-1)
        #     return count
        # res = []
        # for i in range(0, num+1):
        #     res.append(getNum(i))
        # return res

        # 奇数比上一个偶数多了一个一，即最末尾的一
        # 偶数和上一个偶数的一是一样多的
        res = [0 for i in range(num+1)]
        for i in range(1, num+1):
            if i & 1:
                res[i] = res[i-1]+1
            else:
                res[i] = res[i//2]
        return res

    # 354. 俄罗斯套娃信封问题
    def maxEnvelopes(self, envelopes: [[int]]) -> int:
        if not envelopes:
            return 0

        n = len(envelopes)
        envelopes.sort(key=lambda x: (x[0], -x[1]))

        f = [envelopes[0][1]]
        for i in range(1, n):
            if (num := envelopes[i][1]) > f[-1]:
                f.append(num)
            else:
                index = bisect.bisect_left(f, num)
                f[index] = num

        return len(f)

    # 503. 下一个更大元素 II
    def nextGreaterElements(self, nums: [int]) -> [int]:
        if nums is None or len(nums) == 0:
            return[]
        n = len(nums)
        res, stack = [-1]*n, [(nums[0], 0)]
        index = 0
        for i in range(1, 2*n):
            index = (index+1) % n
            while len(stack) != 0 and nums[index] > stack[-1][0]:
                res[stack[-1][1]] = nums[index]
                stack.pop()
            else:
                stack.append((nums[index], index))
        return res

    # 132.分割回文串II
    def minCut(self, s: str) -> int:
        n = len(s)
        # g[i][j] 表示[i~j]是否是回文
        g = [[True]*n for i in range(n)]

        # 回文预处理，在O(1)时间内判断是否是回文
        for i in range(n-1, -1, -1):
            for j in range(i+1, n,):
                g[i][j] = g[i+1][j-1] and (s[i] == s[j])
        # dp[i]代表前i个最小分割数
        dp = [n] * n
        for i in range(n):
            if(g[0][i]):
                dp[i] = 0
            else:
                for j in range(i):
                    # 已经判断了0～i，且还需要判断 i～i所以j+1
                    if g[j+1][i] is True:
                        dp[i] = min(dp[i], dp[j]+1)
        return dp[n-1]

    # 1047. 删除字符串中的所有相邻重复项
    def removeDuplicates(self, S: str) -> str:
        stack = []
        for i in range(len(S)):
            if(len(stack) == 0 or S[i] != stack[-1]):
                stack.append(S[i])
                continue
            stack.pop()
        return ''.join(stack)

    # 227. 基本计算器 II
    def calculate(self, s: str) -> int:
        num, cur = [], 0
        pre = '+'
        for i in range(len(s)):
            if s[i].isdigit():
                cur = cur*10+int(s[i])
            if i == len(s)-1 or s[i] in '+-*/':
                if pre == '+':
                    num.append(cur)
                elif pre == '-':
                    num.append(-cur)
                elif pre == '*':
                    num.append(num.pop()*cur)
                else:
                    num.append(int(num.pop()/cur))
                pre = s[i]
                cur = 0
        return sum(num)

    # 92. 反转链表 II
    def reverseBetween(self, head: ListNode, left: int, right: int) -> ListNode:
        stack = []
        index = ListNode(0, head)
        for i in range(0, left-1):
            index = index.next
        i = left
        while i <= right and index.next is not None:
            stack.append(index.next)
            index.next = index.next.next
            i += 1
        res = index
        while len(stack) != 0:
            temp = stack.pop()
            temp.next = index.next
            index.next = temp
            index = index.next

        return res.next if left <= 1 else head

    # 73 矩阵置零
    def setZeroes(self, matrix: [[int]]) -> None:
        def set(matrix: [[int]], i: int, j: int):
            for k in range(len(matrix[i])):
                matrix[i][k] = None if matrix[i][k] != 0 else 0
            for k in range(len(matrix)):
                matrix[k][j] = None if matrix[k][j] != 0 else 0
        n, m = len(matrix), len(matrix[0])
        for i in range(n):
            for j in range(m):
                if matrix[i][j] == 0:
                    set(matrix, i, j)
        for i in range(n):
            for j in range(m):
                if matrix[i][j] is None:
                    matrix[i][j] = 0

    # 1 的位数
    def hammingWeight(self, n: int) -> int:
        res = 0
        while n != 0:
            n &= (n-1)
            res += 1
        return res

    # 132模式
    def find132pattern(self, nums: [int]) -> bool:
        n = len(nums)
        if n < 3:
            return False
        # dp记录第0～i最小的数和离i最近的最大的数的位置
        dp = [[i, i] for i in range(n)]
        for i in range(1, n):
            dp[i][0] = dp[i][0] if nums[dp[i][0]] < nums[dp[i-1][0]] else dp[i-1][0]
            if nums[i] < nums[i-1]:
                dp[i][1] = i-1
            else:
                j = i-1
                while nums[dp[j][1]] < nums[i]:
                    j -= 1
                dp[i][1] = dp[j][1]
            # nums[i] 要小于之前的比他大的值，并且要大于最大值之前的最小值
            if nums[i] < nums[dp[i][1]] and nums[i] > nums[dp[dp[i][1]][0]]:
                return True
        return False

    # 82. 删除排序链表中的重复元素
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        buckt, index = [0 for i in range(200)], ListNode()
        while head is not None:
            buckt[head.val+100] += 1
            head = head.next
        res = index
        for i in range(200):
            if buckt[i] == 1:
                index.next = ListNode(i-100)
                index = index.next
        return res.next

    # 61. 旋转链表
    def rotateRight(self, head: ListNode, k: int) -> ListNode:
        if(head is None or k == 0):
            return head
        n, index = 0, head
        # 获得链表长度
        while index is not None:
            index = index.next
            n += 1
        k, begin = k % n, head
        if k == 0:
            return head
        # index 移动到起点
        for i in range(n-k):
            begin = begin.next
        f, r = head, begin
        # r 移动到可以拼接的位置
        while r.next is not None:
            r = r.next
        while f != begin:
            r.next = f
            f = f.next
            r = r.next
        r.next = None
        return begin


def createListNode(nums: [int]) -> ListNode:
    res = ListNode()
    index = res
    for i in range(len(nums)):
        index.next = ListNode(nums[i])
        index = index.next
    return res.next


def sout(head: ListNode) -> []:
    res = []
    while head is not None:
        res.append(head.val)
        head = head.next
    return res


s = Solution()
print(sout(s.rotateRight(createListNode([1, 2, 3, 4, 5]), 3)))
# [1,4,0,-1,-2,-3,-1,-2]
# [3,5,0,3,4]
# head = ListNode(1)
# index = head
# for i in range(2, 6):
#     index.next = ListNode(i)
#     index = index.next

# index = s.reverseBetween(head, 2, 3)
# while index is not None:
#     print(index.val)
#     index = index.next
# s.setZeroes([[0, 1, 2, 0], [3, 4, 5, 2], [1, 3, 1, 5]])
# print(s.hammingWeight(11111111111111111111111111111101))


# p = NestedInteger()
# p.isInt = True
# p.integer = 1
# q = NestedInteger()
# q.isInt = True
# q.integer = 2

# a = NestedInteger()
# a.integerList = [p, q]

# b = NestedInteger()
# b.isInt = True
# b.integer = 1

# c = NestedInteger()
# c.integerList = [q, p]

# d = NestedInteger()
# d.integerList = []

# e = NestedInteger()
# e.integerList = [d]

# iterator = NestedIterator([d, d, e, a])
# while iterator.hasNext():
#     print(iterator.next())
