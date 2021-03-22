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


s = Solution()

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
print(s.hammingWeight(11111111111111111111111111111101))
