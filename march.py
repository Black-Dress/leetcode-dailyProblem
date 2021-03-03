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


s = Solution()
print(s.countBits(5))
