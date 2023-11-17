# -*- coding: utf-8 -*-
"""
@Version ： python3.9
@Time    ： 2023/11/13 1:35 PM
@Author  ： kai.wang
@Email   :  kai.wang@westwell-lab.com
"""

import collections
import heapq
import json
import math
import time
import multiprocessing as mp
from heapq import heappop, heappush
from itertools import combinations

import cv2
import numpy as np
from copy import deepcopy
from typing import List, Tuple, Dict


class Node(object):
    def __init__(self, x: float, y: float, idx: int, node_ids: List):
        self.x = x
        self.y = y
        self.to_ids = [(n, 1) for n in node_ids]
        self.idx = idx

    def __lt__(self, other):
        return other.idx > self.idx  # 升序


class GenTestGraph(object):
    def __init__(self):
        self.master_lane = 2  # 主干道车道
        self.block_lane = 10  # 堆场车道
        self.graph: List[Node] = []

    def gen_graph(self) -> None:
        """

        :return:
        """
        node_list = []
        max_idx = self.block_lane + 1 + self.master_lane * 2
        idx = 0
        for i in range(1, max_idx):
            per_list = []
            for j in range(1, max_idx):
                # 上边主干道
                if i <= self.master_lane:
                    direct = ((-1, 0), (1, 0), (0, - 1), (0, 1))  # 上下左右
                # 下边
                elif i >= max_idx - self.master_lane:
                    direct = ((-1, 0), (1, 0), (0, - 1), (0, 1))  # 上下左右
                # 左边
                elif j < self.master_lane:
                    direct = ((-1, 0), (1, 0), (0, - 1), (0, 1))  # 上下左右
                # 右边
                elif j > max_idx - self.master_lane:
                    direct = ((-1, 0), (1, 0), (0, - 1), (0, 1))  # 上下左右
                # 堆场里的
                else:
                    direct = ((-1, 0), (1, 0), None, None)

                per_list.append((j + j * 3, i + i * 3, idx, direct))
                idx += 1

            node_list.append(per_list)

        # 将节点处理为类
        for i, per_list in enumerate(node_list):
            for j, val in enumerate(per_list):
                x, y, idx, direct = val
                node_ids = []
                for d in direct:
                    if d is None:
                        continue
                    idx1, idx2 = d
                    if len(node_list) > i + idx1 >= 0 and 0 <= j + idx2 < len(per_list):
                        node_ids.append(node_list[i + idx1][j + idx2][2])
                self.graph.append(Node(x, y, idx, node_ids))

        for i, node in enumerate(self.graph):
            if i != node.idx:
                raise
            # print(node.idx, node.x, node.y, node.node_ids)

    def draw(self, paths: List[List[int]], wait_status):
        # cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('frame', 1280, 720)
        long = int(len(self.graph) ** 0.5)
        c = 45
        frame = np.ones((long * c * 5, long * c * 5, 3), np.uint8) * 255
        wait = True
        j = -1

        for node in self.graph:
            x, y = node.x, node.y
            cv2.circle(frame, (x * c, y * c), c, (224, 224, 224), -1)
            for node_id, weight in node.to_ids:
                near_node = self.graph[node_id]
                cv2.line(frame, (x * c, y * c), (near_node.x * c, near_node.y * c), (0, 0, 0), 2)
                cv2.line(frame, (near_node.x * c, near_node.y * c), (x * c, y * c), (0, 0, 0), 2)

        # 绘制单车
        for idx, path in enumerate(paths):
            color = self.assign_colour(idx)
            cv2.circle(frame, (self.graph[path[0]].x * c, self.graph[path[0]].y * c), c, color, -1)
            cv2.circle(frame, (self.graph[path[-1]].x * c, self.graph[path[-1]].y * c), c, color, -1)

        while True:

            # 绘制路径
            if 0 <= j:
                for idx, path in enumerate(paths):
                    for i in range(1, len(path)):
                        if i != j:
                            continue
                        cur_node, pre_node = self.graph[path[i]], self.graph[path[i - 1]]
                        # 绘制带箭头的直线
                        cv2.arrowedLine(frame, (pre_node.x * c, pre_node.y * c), (cur_node.x * c, cur_node.y * c), self.assign_colour(idx), thickness=c // 2, tipLength=0.1)

            frame = deepcopy(frame)
            cv2.imshow('frame', frame)
            if wait_status == 1:
                if wait:
                    cv2.waitKey(0)
                    wait = False

                    k = cv2.waitKey(500) & 0xFF
                    if k == ord('q'):
                        break
            elif wait_status == 0:
                k = cv2.waitKey(0) & 0xFF
                if k == ord('q'):
                    break

            j += 1

    def assign_colour(self, x):
        x = hash(str(x + 42))
        return x & 0xFF, (x >> 8) & 0xFF, (x >> 16) & 0xFF


class Dijkstra(object):

    def dijkstra(self, graph: List[Node], start: int, end: int, ct: 'Constraints') -> Tuple[int, List[int]]:
        """
        dijkstra 算法，需要传入对应的开始节点和结束节点
        :param graph:
        :param start:
        :param end:
        :param ct:
        :return:
        """
        start, end = graph[start], graph[end]
        n = len(graph)
        dist = [float('inf')] * n
        dist[start.idx] = 0
        prev = [(-1, 0, 1)] * n
        pq = [(0, start, 1)]

        while pq:
            d, u, count = heapq.heappop(pq)
            if u.idx == end.idx:
                path = []
                while u != -1:
                    path.append(u.idx)
                    u = prev[u.idx][0]

                return d, path[::-1]
            if d > dist[u.idx]:
                continue
            for node_id, w in u.to_ids:
                v = graph[node_id]
                if v.idx in ct.agent_constraints[count]:
                    continue
                if ct.obstacle.get(v.idx) and count > ct.obstacle[v.idx]:
                    continue
                if dist[u.idx] + w <= dist[v.idx]:
                    dist[v.idx] = dist[u.idx] + w
                    prev[v.idx] = (u, w, count + 1)
                    heapq.heappush(pq, (dist[v.idx], v, count + 1))
        return -1, []


class Constraints:

    def __init__(self):
        #
        self.agent_constraints = collections.defaultdict(set)
        self.obstacle = {}

    '''
    Deepcopy self with additional constraints
    '''

    def fork(self, window: Dict) -> 'Constraints':
        agent_constraints_copy = deepcopy(self.agent_constraints)
        for k, vs in window.items():
            for v in vs:
                agent_constraints_copy[k].add(v)

        new_constraints = Constraints()
        new_constraints.agent_constraints = agent_constraints_copy
        return new_constraints

    def rebuild_obstacle(self, node: 'CTNode', agent):
        self.obstacle.clear()
        for k, v in node.solution.items():
            if agent == k:
                continue
            self.obstacle[v[-1]] = len(v)


class CTNode:

    def __init__(self, constraints: Constraints,
                 solution: Dict):
        self.constraints = constraints
        self.solution = solution
        self.cost = self.sic(solution)

    # Sum-of-Individual-Costs heuristics
    @staticmethod
    def sic(solution):
        return sum(len(v) for k, v in solution.items())

    def __lt__(self, other):
        return self.cost < other.cost

    def __str__(self):
        return str(self.constraints.agent_constraints)


class CBS(object):
    def __init__(self, graph):
        self.dij = Dijkstra()
        self.graph = graph
        self.max_iter = 200
        self.max_process = 10

    def plan(self, agents: List):
        """

        :param agents:
        :return:
        1. 任务指派, agents已指定
        2. 计算agent最优路径
        3. 计算冲突
        4. 添加约束树
        5. 循环计算
        """
        constraints = Constraints()

        solution = {}
        for s, e in agents:
            p = self.dij.dijkstra(self.graph, s, e, constraints)
            solution[(s, e)] = p[1]

        open = []
        if all(len(path) != 0 for path in solution.values()):
            # Make root node
            node = CTNode(constraints, solution)
            # Min heap for quick extraction
            open.append(node)

        manager = mp.Manager()
        iter_ = 0
        while open and iter_ < self.max_iter:
            iter_ += 1

            results = manager.list([])

            processes = []

            print(f"第{iter_}次计算, {len(open)}个分支")

            # Default to 10 processes maximum
            for _ in range(self.max_process if len(open) > self.max_process else len(open)):
                p = mp.Process(target=self.search_node, args=[heappop(open), results])
                processes.append(p)
                p.start()

            for p in processes:
                p.join()

            for result in results:
                if len(result) == 1:
                    print("CBS success...")
                    return result[0]
                if result[0]:
                    heappush(open, result[0])
                if result[1]:
                    heappush(open, result[1])

        return []

    def search_node(self, best: CTNode, results: List):
        """
        1. 计算冲突节点, 无冲突返回结果
        2. 计算冲突时间窗
        3. fork 约束节点
        4. 重计算并更新路径
        5. 将结果放入到优先级队列,等待下一次循环
        :param best:
        :param results:
        :return:
        """
        ai, aj, start_conflict, end_conflict, window = self.validate_paths(best)
        if ai is None:
            results.append((best,))
            return
        at_i, at_j = best.constraints, best.constraints
        if window:
            at_i = best.constraints.fork(window)
            window2 = self.safe_distance(best.solution[aj], best.solution[ai])[2]
            at_j = best.constraints.fork(window2)

        best.constraints.rebuild_obstacle(best, ai)
        path1 = self.dij.dijkstra(self.graph, *ai, at_i)[1]

        best.constraints.rebuild_obstacle(best, aj)
        path2 = self.dij.dijkstra(self.graph, *aj, at_j)[1]
        # print(path1, ai, window)
        # print(path2, aj, window2)

        # Replace old paths with new ones in solution
        solution_i = best.solution
        solution_j = deepcopy(best.solution)
        solution_i[ai] = path1
        solution_j[aj] = path2

        node_i = None
        if all(len(path) != 0 for path in solution_i.values()):
            node_i = CTNode(at_i, solution_i)

        node_j = None
        if all(len(path) != 0 for path in solution_j.values()):
            node_j = CTNode(at_j, solution_j)

        results.append((node_i, node_j))

    def validate_paths(self, node: CTNode):
        agents = sorted(node.solution.items(), key=lambda x: len(x[1]), reverse=True)
        agents = [agent[0] for agent in agents]
        for ai, aj in combinations(agents, 2):
            start_conflict, end_conflict, window = self.safe_distance(node.solution[ai], node.solution[aj])

            if start_conflict != -1:
                return ai, aj, start_conflict, end_conflict, window
        return None, None, -1, -1, {}

    def safe_distance(self, p1, p2) -> Tuple:
        """
        1个时间窗内占据起始点和终点
        :param p1:
        :param p2:
        :return:
        """
        max_len = min(len(p1), len(p2))
        conflict = {0: {p1[0], }}
        for i in range(1, max_len):
            conflict[i] = {p1[i - 1], p1[i]}

        start, end, window = -1, -1, {}
        for i in range(max_len):
            if start == -1:
                if p2[i] in conflict[i]:
                    start = i
            elif end == -1:
                if p2[i] not in conflict[i]:
                    end = i
            else:
                break

        if start != -1:
            window = {i: conflict[i] for i in range(start, end)}

        # 车辆到达后的冲突
        if len(p1) > len(p2):
            for i in range(len(p2), len(p1)):
                if p1[i] == p2[-1]:
                    start = 1

        return start, end, window


if __name__ == '__main__':
    """
    解决已有路径的问题
    """
    obj = GenTestGraph()
    obj.gen_graph()
    cbs = CBS(obj.graph)
    se = [(1, 195), (13, 1), (187, 21), (33, 187), (14, 28), (27, 29), (140, 30), (100, 181)]
    se = [(1, 195), (13, 1), (187, 21), (33, 187), (14, 28), (27, 29), (140, 30), (100, 181)]
    # se = [(1, 195), (13, 1), (187, 21), (33, 187), (100, 194), (23, 191)]
    # se = [
    #     (1, 195), (13, 1), (187, 21), (33, 167),
    #     (100, 194), (23, 182), (55, 77), (67, 188),
    #     (28, 178), (70, 163)
    # ]
    node: CTNode = cbs.plan(se)
    result = list(node.solution.values())
    for r in result:
        print(r)
    obj.draw(result, wait_status=0)
