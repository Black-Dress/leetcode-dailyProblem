from typing import List
from queue import PriorityQueue


class Solution:
    # 1446. 连续字符
    def maxPower(self, s: str) -> int:
        maxnum, cur = 1, 1
        for i in range(1, len(s)):
            if s[i] == s[i - 1]:
                cur += 1
            else:
                cur = 1
            maxnum = max(cur, maxnum)
        return maxnum

    # 1005. K 次取反后最大化的数组和
    def largestSumAfterKNegations(self, nums: List[int], k: int) -> int:
        minqueue = PriorityQueue()
        for num in nums:
            minqueue.put(num)
        for i in range(k):
            item = minqueue.get()
            minqueue.put(-item)
        return sum(minqueue.queue)

    # 1816. 截断句子
    def truncateSentence(self, s: str, k: int) -> str:
        index = 0
        while k != 0:
            k -= 1 if s[index] == ' ' or index == len(s) - 1 else 0
            index += 1
        return s[:index].strip()


s = Solution()
print(s.truncateSentence('hello a b', 2))
