import collections
from typing import Collection, List


class Solution:
    # BFS
    def LCP07_BFS(self, relation: List[List[int]], k: int, n: int) -> int:
        edges = collections.defaultdict(list)
        for i in relation:
            edges[i[0]].append(i[1])
        steps, queue = 0, collections.deque([0])
        while queue.__len__() and steps < k:
            steps += 1
            for i in range(queue.__len__()):
                to = edges[queue.popleft()]
                queue.extend(to)
        res = 0
        if steps == k:
            while queue.__len__():
                res += 1 if queue.popleft() == n - 1 else 0
        return res

    # DFS
    def LCP07_DFS(self, edges: dict, level: int, cur: int, k: int, n: int, res: List):
        if level == k:
            res[0] += 1 if cur == n - 1 else 0
            return
        for i in edges[cur]:
            self.LCP07_DFS(edges, level + 1, i, k, n, res)

    # LCP 07
    def numWays(self, n: int, relation: List[List[int]], k: int) -> int:
        # return self.LCP07_BFS(relation, k, n)
        edges, res = collections.defaultdict(list), [0]
        for i in relation:
            edges[i[0]].append(i[1])
        self.LCP07_DFS(edges, 0, 0, k, n, res)
        return res[0]


s = Solution()
print(s.numWays(5, [[0, 2], [2, 1], [3, 4], [2, 3], [1, 4], [2, 0], [0, 4]], 3))
