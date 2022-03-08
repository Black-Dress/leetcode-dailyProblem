from typing import List


class Solution:
    # 2055. 蜡烛之间的盘子
    def platesBetweenCandles(self, s: str, queries: List[List[int]]) -> List[int]:
        # 利用栈可以轻松得到查询的结果
        # 有没有方法初始化初始字符串，并用cnt进行存储，表示某区间段的结果
        # def check(query:List[int])->int:
        #     stack,res = [],0
        #     for i in range(query[0],query[1]+1):
        #         if s[i] == '|':
        #             res += i-stack[-1]-1 if stack else 0
        #             stack.append(i)
        #     return res
        # return [check(i) for i in queries]

        # 存储所有蜡烛位置，利用二分查找离边界最近的蜡烛位置
        cnt = [i for i in range(len(s)) if s[i] == '|']

        # 二分查找最近的蜡烛
        # def index(i: int, l: int, r: int, isL: bool) -> int:
        #     while l <= r:
        #         mid = (l + r) >> 1
        #         if i == cnt[mid]:
        #             return mid
        #         if i > cnt[mid]:
        #             l = mid + 1
        #         if i < cnt[mid]:
        #             r = mid - 1
        #     return r if isL else l
        # res = []
        # for i in queries:
        #     l, r = index(i[0], 0, len(cnt) - 1, False), index(i[1], 0, len(cnt) - 1, True)
        #     res.append(max(cnt[r] - cnt[l] - (r - l), 0) if l in range(0, len(cnt)) and r in range(0, len(cnt)) else 0)
        # return res

        # 前缀和，利用两个数组存储i位置左边和右边最近的蜡烛，用一个数组存储i位置之前有多少个盘子
        n = len(s)
        l, r, candles = [0] * n, [0] * n, [0] * n
        index = -1
        for i in range(n):
            if s[i] == '|':
                index = i
            l[i] = index
        index = -1
        for i in range(n - 1, -1, -1):
            if s[i] == '|':
                index = i
            r[i] = index
        index = 0
        for i in range(n):
            if s[i] == '|':
                index += 1
                candles[i] = index
        res = []
        for a, b in queries:
            x, y = r[a], l[b]
            res.append(0 if x < 0 or y < 0 else max(y - x - (candles[y] - candles[x]), 0))
        return res


s = Solution()
print(s.platesBetweenCandles("*|||*", [[1, 4]]))
