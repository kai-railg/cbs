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
            cv2.circle(frame, (self.graph[path[-1]].x * c, self.graph[path[-1]].y * c), c, color, 10)

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
    def __init__(self):
        self.window = collections.defaultdict(bool)
        self.forbid = []
        # self.window = {
        #     "1": {
        #         "1": True
        #     }
        # }

    def dijkstra(self, graph: List[Node], start: Node, end: Node) -> Tuple[int, List[int]]:
        """
        dijkstra 算法，需要传入对应的开始节点和结束节点
        :param graph:
        :param start:
        :param end:
        :return:
        """
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
                    # if u != -1:
                    #     print(u.idx, prev[u.idx][2])

                res = path[::-1]
                for i in range(1, len(res)):
                    self.window[res[i - 1]] = res[i]
                # print("window", {k: v for k, v in self.window.items() if v})
                # print({k: path[k] for k in range(len(path))})
                # print("window", dict(self.window))

                return d, path[::-1]
            if d > dist[u.idx]:
                continue
            for node_id, w in u.to_ids:
                v = graph[node_id]
                if self.window[v.idx] == u.idx:
                    continue
                elif dist[u.idx] + w < dist[v.idx]:
                    dist[v.idx] = dist[u.idx] + w
                    prev[v.idx] = (u, w, count + 1)
                    heapq.heappush(pq, (dist[v.idx], v, count + 1))
        return -1, []


if __name__ == '__main__':
    """
    对长路径的方向限制
    """
    obj = GenTestGraph()
    obj.gen_graph()
    paths = []
    se = [(1, 195), (13, 1), (187, 21), (33, 187)]
    dij = Dijkstra()
    for start, end in se:
        result = dij.dijkstra(obj.graph, obj.graph[start], obj.graph[end])
        if result[1]:
            paths.append(result[1])
        print(result)
    obj.draw(paths, 0)
