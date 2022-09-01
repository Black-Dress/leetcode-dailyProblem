# 给你一个聊天记录，共包含 n 条信息。
# 给你两个字符串数组 messages 和 senders ，其中 messages[i] 是 senders[i] 发出的一条 信息 。
# 一条 信息 是若干用单个空格连接的 单词 ，信息开头和结尾不会有多余空格。
# 发件人的 单词计数 是这个发件人总共发出的 不重复单词数 。注意，一个发件人可能会发出多于一条信息。
# 请你返回发出单词数 最多 的发件人名字。如果有多个发件人发出最多单词数，请你返回 字典序 最大的名字。
# 注意：字典序里，大写字母小于小写字母。"Alice" 和 "alice" 是不同的名字。

# 输入：messages = ["Hello userTwooo","Hi userThree","Wonderful day Alice","Nice day userTwooo"],
# senders = ["Alice","userTwo","userThree","Alice"]
# 输出："Alice"
# 解释：Alice 总共发出了 2 + 3 = 5 个单词，但有一个单词是重复的，因此单词计数是4。
# userTwo 发出了 2 个单词。
# userThree 发出了 3 个单词。
# 由于 Alice 发出单词数最多，所以我们返回 "Alice" 。

# 给你一个仅由数字组成的字符串 s 。
# 请你判断能否将 s 拆分成两个或者多个 非空子字符串 ，使子字符串的 数值 按 降序 排列，且每两个 相邻子字符串 的数值之 差 等于 1 。
# 例如，字符串 s = "0090089" 可以拆分成 ["0090", "089"] ，数值为 [90,89] 。这些数值满足按降序排列，且相邻值相差 1 ，这种拆分方法可行。
# 另一个例子中，字符串 s = "001" 可以拆分成 ["0", "01"]、["00", "1"] 或 ["0", "0", "1"] 。然而，所有这些拆分方法都不可行，因为对应数值分别是 [0,1]、[0,1] 和 [0,0,1] ，都不满足按降序排列的要求。
# 如果可以按要求拆分 s ，返回 true ；否则，返回 false 。
# 子字符串 是字符串中的一个连续字符序列。
# 示例 1：
# 输入：s = "1234"
# 输出：false
# 解释：不存在拆分 s 的可行方法。
# 示例 2：
# 输入：s = "050043"
# 输出：true
# 解释：s 可以拆分为 ["05", "004", "3"] ，对应数值为 [5,4,3] 。
# 满足按降序排列，且相邻值相差 1 。
# 示例 3：
# 输入：s = "9080701"
# 输出：false
# 解释：不存在拆分 s 的可行方法。
# 示例 4：
# 输入：s = "10009998"
# 输出：true
# 解释：s 可以拆分为 ["100", "099", "98"] ，对应数值为 [100,99,98] 。
# 满足按降序排列，且相邻值相差 1 。

from collections import defaultdict
from operator import truediv
from typing import Counter, List


def solution(messages: List[str], senders: List[str]):
    res = defaultdict(set)
    name, size = "", 0
    # 统计完成
    for i in range(len(messages)):
        words = list(messages[i].split(" "))
        for j in words:
            res[senders[i]].add(j)
    # 遍历获得结果
    for k, v in res.items():
        if len(v) > size:
            name = k
            size = len(v)
        elif len(v) == size:
            name = max(name, k)
    # 返回结果
    return name


def solution2(s: str) -> bool:
    def dfs(start: int, s: str) -> bool:
        if s == "":
            return True
        for i in range(len(s)):
            if int(s[:i + 1]) == start - 1:
                return dfs(int(s[:i + 1]), s[i + 1:])
        return False

    for i in range(len(s) - 1):
        start = int(s[:i + 1])
        if dfs(start, s[i + 1:]):
            return True
    return False


print(solution2("050043"))
