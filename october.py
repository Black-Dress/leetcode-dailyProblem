from collections import defaultdict
from typing import Collection, List


class Solution:
    # 187. 重复的DNA序列
    def findRepeatedDnaSequences(self, s: str) -> List[str]:
        target, table = s[0:10], defaultdict(int)
        table[target] = 1
        res = list()
        for i in range(10, s.__len__()):
            target = target[1:] + s[i]
            table[target] += 1
        for k, v in table.items():
            if v > 1:
                res.append(k)
        return res


s = Solution()
print(s.findRepeatedDnaSequences("AAAAAAAAAAAAA"))
