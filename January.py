class Solution:
    # 509
    def fib(self, n: int) -> int:
        if n < 2:
            return n
        res = [0 for i in range(n+1)]
        res[0], res[1] = 0, 1
        for i in range(2, n+1):
            res[i] = res[i-1]+res[i-2]
        return res[-1]


s = Solution()
print(s.fib(3))
