from collections import defaultdict
from typing import Collection, List
import math


class Solution:
    # 回旋镖的数量，找到两个点的中点
    # 暴力法
    def numberOfBoomerangs(self, points: List[List[int]]) -> int:
        cnt, res = defaultdict(int), 0
        for i in range(points.__len__()):
            cnt.clear()
            for j in range(points.__len__()):
                distance = math.pow(points[i][0]-points[j][0], 2)+math.pow(points[i][1]-points[j][1], 2)
                cnt[distance] += 1
            for (k, v) in cnt.items():
                res += v*(v-1)
        return res

    # 524. 通过删除字母匹配到字典里最长单词
    def findLongestWord(self, s: str, dictionary: List[str]) -> str:
        # 先判断 s 是否含有字典中的某一个值
        # 只要源字符串包含了匹配串的所有字符，并且按照匹配串的顺序 则成功匹配
        # 双指针匹配
        def match(s: str, ss: str) -> bool:
            i, j, p, q = 0, s.__len__()-1, 0, ss.__len__()-1
            while i <= j:
                while s[i] != ss[p] and i < j:
                    i += 1
                p += 1 if s[i] == ss[p] else 0
                i += 1
                while s[j] != ss[q] and i < j:
                    j -= 1
                q -= 1 if s[j] == ss[q] else 0
                j -= 1
                if p > q:
                    return True
            return False
        dictionary.sort()
        dictionary.sort(key=lambda x: len(x), reverse=True)
        for i in dictionary:
            if match(s, i):
                return i
        return ""


s = Solution()
print(s.findLongestWord("abcdefgc", ["acc", "acbcc", "accc", "abc"]))
