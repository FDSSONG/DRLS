#!/usr/bin/python
# -*- coding: UTF-8 -*-

from Node import *
from Graph import *
import json
import numpy as np
import random
import sys
import copy
import math

# sys.path.append("../resource/")
from schedule import Schedule
from param import *

"""
    调度环境类，用于管理网络拓扑、流量调度、时隙分配等操作。

    主要功能：
    - 维护图结构（Graph）
    - 处理 TT（时间触发）流的调度
    - 记录流量信息，如流量路径、时间分配等
    - 计算链路利用率
    - 提供调度查询功能
    - 进行回滚、删除操作
"""
class Environment:
    def __init__(self, data=None):
        """
             初始化环境，包括网络拓扑、时间管理和 TT 流调度相关参数。

             参数：
             - data: 预设的调度数据（可选），如果为空，则从 `tt_flow.json` 读取调度数据。

             主要属性：
             - self.graph: Graph 类的实例，表示网络拓扑。
             - self.time: int，当前调度时间。
             - self.tt_queries: list，存储所有 TT 流调度信息（从文件或 `data` 读取）。
             - self.schedule: 主要的调度管理器。
             - self.current_stream_schedule: 当前流的调度状态管理器。
             """
        self.graph = Graph(data)        # 初始化网络拓扑
        self.time = 0                   # 当前调度时间
        self.depth = 8                  # 记录缓冲区占用率的深度
        self.decay = 0.5                # 递减系数
        self.tt_query_id = -1           # 当前 TT 流 ID
        self.reschedule_queries = []    # 需要重新调度的查询
        self.reschedule_pos = -1        # 重新调度的查询索引
        self.cur_tt_flow_id = -1        # 当前 TT 流 ID
        self.reschedule = 0             # 重新调度标志

        # 记录已经访问过的节点
        self.valid_edges = {}
        self.visited_node = []
        # 记录当前TT流的周期
        self.tt_flow_cycle = args.global_cycle  # TT 流的周期
        self.tt_flow_start = -1                 # TT 流起始节点
        self.tt_flow_end = -1                   # TT 流目标节点
        self.tt_flow_end = -1                   # TT 流目标节点
        self.delay = -1                         # TT 流延迟
        self.tt_flow_lenth = 1                  # TT 流长度
        self.tt_flow_deadline = -1              # TT 流截止时间
        self.tt_flow_length = -1
        # 记录当前TT流的所有发出时间
        self.tt_flow_time_record = []
        # 记录每条流使用的链路和时隙
        self.tt_flow_to_edge = {}
        # 记录每个链路包含的流
        self.edge_to_tt_flow = {}
        # 初始化 `edge_to_tt_flow` 映射
        for i in range(len(self.graph.nodes)):
            for j in range(len(self.graph.nodes)):
                self.edge_to_tt_flow[(i, j)] = set()

        # 奖励参数（用于强化学习）
        self.stop_parameter = 10
        self.delay_parameter = 0.000001
        self.lasting_parameter = 0

        # 读取调度流量数据
        self.data = data
        if self.data is None:
            # 如果没有提供数据，从 JSON 文件读取调度流量任务
            self.tt_queries = json.load(open(args.data_path + 'tt_flow.json', encoding='utf-8'))
        else:
            # 此处有 6 万条 TT-调度信息
            self.tt_queries = self.data.tt_flow

        self.schedule = Schedule()                  # 主要调度管理器
        self.current_stream_schedule = Schedule()   # 当前流调度状态
        self.enforce_next_query()                   # 执行下一次调度查询

    def edge_usage(self):
        """
              计算网络边的利用率。

              返回：
              - float：利用率 = (所有边的可用时隙数量) / (总时隙数量)
        """
        total = len(self.graph.edges) * args.global_cycle   # 总时隙数
        cur = 0
        for edge in self.graph.edges.values():
            cur += sum(edge.time_slot_available)            # 可用时隙总数
        return cur / total                                  # 计算利用率

    def enforce_next_query(self, rool_back=False):
        """
         执行下一个调度查询，将 TT 流信息加载到环境中。

         参数：
         - rool_back: bool，是否回滚查询（默认为 False）。

         逻辑：
         1. 更新 `cur_tt_flow_id`，选择下一个调度流。
         2. 读取该流的起始点、终点、周期、截止时间和长度等信息。
         3. 设置图中对应节点的状态。
         4. 调用调度器 `sche_start` 进行流调度。
         """

        # 重置所有节点的状态
        for node in self.graph.nodes.values():
            node.is_source_node = 0
            node.is_destination_node = 0

        # print(self.reschedule_pos, self.reschedule_queries)
        if self.reschedule_pos + 1 < len(self.reschedule_queries):
            self.reschedule_pos += 1
            self.cur_tt_flow_id = self.reschedule_queries[self.reschedule_pos]
            if self.reschedule == 0:
                self.reschedule = 2
        elif not rool_back:
            self.tt_query_id += 1
            self.cur_tt_flow_id = self.tt_query_id
            self.reschedule = 0
        else:
            self.cur_tt_flow_id = self.tt_query_id

        # print(self.tt_queries[self.cur_tt_flow_id])

        # 读取 TT 流的参数
        self.tt_flow_start = self.tt_queries[self.cur_tt_flow_id][0]
        self.tt_flow_end = self.tt_queries[self.cur_tt_flow_id][1]
        self.tt_flow_cycle = self.tt_queries[self.cur_tt_flow_id][2]
        self.tt_flow_deadline = self.tt_queries[self.cur_tt_flow_id][3]
        self.tt_flow_length = self.tt_queries[self.cur_tt_flow_id][4]

        # 将秒转换成时隙
        self.tt_flow_deadline = int(self.tt_flow_deadline * args.slot_per_millisecond)
        self.tt_flow_cycle = int(self.tt_flow_cycle * args.slot_per_millisecond)
        # 当前报文横跨的 slot 数量，用2**17代替10**6/8
        self.tt_flow_length = int(
            math.ceil(self.tt_flow_lenth * 1.0 / (args.link_rate * 2 ** 17) * args.slot_per_millisecond))

        # print("query", self.cur_tt_flow_id, self.tt_flow_start, self.tt_flow_end, self.tt_flow_cycle,
        #       self.tt_flow_deadline, self.tt_flow_lenth)

        # 设置源节点和目标节点
        self.graph.nodes[self.tt_flow_start].set_source_node()
        self.graph.nodes[self.tt_flow_end].set_destination_node()
        self.visited_node = [self.tt_flow_start]
        self.tt_flow_time_record = [-1]
        self.tt_flow_to_edge[self.cur_tt_flow_id] = []

        # 刷新所有边的状态
        for edge in self.graph.edges.values():
            edge.refresh()

        # 开始进行调度
        self.schedule.sche_start(self.graph.nodes[self.tt_flow_start].id, self.cur_tt_flow_id, cycle=self.tt_flow_cycle)
        self.current_stream_schedule.reset()
        self.current_stream_schedule.sche_start(self.graph.nodes[self.tt_flow_start].id, self.cur_tt_flow_id, cycle=self.tt_flow_cycle)

    def enforce_specific_query(self, query, rool_back=False):
        """
        强制执行特定的 TT 流调度查询。
        该方法允许手动插入一个新的 TT 流（query）到调度环境中，并执行调度。
        参数：
        - query: list，包含 TT 流的相关参数 `[start_node, end_node, cycle, deadline, length]`。
        - rool_back: bool，是否进行回滚（默认为 False）。
        逻辑：
        1. 重置所有节点的状态，确保新的 TT 流不会受到之前调度的影响。
        2. 分配新的 `tt_query_id`，确保 TT 流 ID 唯一。
        3. 如果 `self.tt_queries` 长度小于 `cur_tt_flow_id`，则扩展 `self.tt_queries` 并添加 `query`。
        4. 读取 `query` 中的 TT 流参数（起始点、终点、周期、截止时间、长度）。
        5. 将 TT 流的时间参数转换为时隙（slot），便于调度计算：
           - `tt_flow_deadline` 和 `tt_flow_cycle` 以 `args.slot_per_millisecond` 进行转换。
           - `tt_flow_length` 计算所需时隙数量，采用 `2**17` 近似替代 `10**6/8`。
        6. 设置图中源节点和目标节点状态。
        7. 记录 TT 流的相关信息，并清空之前的 `visited_node`。
        8. 刷新所有边的状态，确保调度时边的状态是最新的。
        9. 调用 `schedule.sche_start()` 开始调度，更新 `current_stream_schedule`。

        """

        # 重置所有节点状态
        for node in self.graph.nodes.values():
            node.is_source_node = 0
            node.is_destination_node = 0

        # 递增查询 ID
        self.tt_query_id += 1
        self.cur_tt_flow_id = self.tt_query_id

        # 如果 `self.tt_queries` 长度不足，则扩展列表
        if len(self.tt_queries) <= self.cur_tt_flow_id:
            self.tt_queries.append(query)
            self.tt_queries[self.cur_tt_flow_id] = query
        else:
            self.tt_queries[self.cur_tt_flow_id] = query

        # print(self.tt_queries[self.cur_tt_flow_id])

        # 解析 TT 流参数
        self.tt_flow_start = self.tt_queries[self.cur_tt_flow_id][0]    # 源节点
        self.tt_flow_end = self.tt_queries[self.cur_tt_flow_id][1]      # 目标节点
        self.tt_flow_cycle = self.tt_queries[self.cur_tt_flow_id][2]    # 调度周期
        self.tt_flow_deadline = self.tt_queries[self.cur_tt_flow_id][3] # 截止时间
        self.tt_flow_length = self.tt_queries[self.cur_tt_flow_id][4]   # 传输数据长度

        # 将秒转换成时隙
        self.tt_flow_deadline = int(self.tt_flow_deadline * args.slot_per_millisecond)  # 截止时间换算成时隙
        self.tt_flow_cycle = int(self.tt_flow_cycle * args.slot_per_millisecond)        # 周期换算成时隙
        # 当前报文横跨的 slot 数量，用2**17代替10**6/8
        self.tt_flow_length = int(
            math.ceil(self.tt_flow_lenth * 1.0 / (args.link_rate * 2 ** 17) * args.slot_per_millisecond))
        # 计算报文跨越的时隙数，替代 `10**6/8` 计算

        # print("query", self.cur_tt_flow_id, self.tt_flow_start, self.tt_flow_end, self.tt_flow_cycle,
        #       self.tt_flow_deadline, self.tt_flow_lenth)

        # 设置源节点和目标节点
        self.graph.nodes[self.tt_flow_start].set_source_node()
        self.graph.nodes[self.tt_flow_end].set_destination_node()
        # 初始化 TT 流记录
        self.visited_node = [self.tt_flow_start]            # 记录已访问的节点
        self.tt_flow_time_record = [-1]                     # 记录 TT 流的调度时间点
        self.tt_flow_to_edge[self.cur_tt_flow_id] = []      # 记录 TT 流经过的边信息

        # 刷新所有边的状态
        for edge in self.graph.edges.values():
            edge.refresh()

        # 开始进行调度
        self.schedule.sche_start(self.graph.nodes[self.tt_flow_start].id, self.cur_tt_flow_id, cycle=self.tt_flow_cycle)
        self.current_stream_schedule.reset()
        self.current_stream_schedule.sche_start(self.graph.nodes[self.tt_flow_start].id, self.cur_tt_flow_id, cycle=self.tt_flow_cycle)


    def translate_data_to_heuristic_inputs(self):
        """
        转换调度环境数据为启发式算法输入。

        该方法用于提取当前 TT 流的信息，并获取图的邻接边矩阵，
        供启发式调度算法使用。

        返回：
        - self.tt_flow_start: int，TT 流的起始节点 ID。
        - self.tt_flow_end: int，TT 流的目标节点 ID。
        - self.tt_flow_cycle: int，TT 流的调度周期（时隙单位）。
        - edge_mat: np.array，图的邻接边矩阵，表示边的连通关系。
        - self.tt_flow_length: int，TT 流的数据长度（时隙单位）。
        - self.tt_flow_deadline: int，TT 流的截止时间（时隙单位）。
        """
        edge_mat = self.graph.adjacent_edge_matrix
        return self.tt_flow_start, self.tt_flow_end, self.tt_flow_cycle, edge_mat, \
               self.tt_flow_length, self.tt_flow_deadline

    def translate_data_to_inputs(self):
        """
        转换调度环境数据为强化学习（RL）模型或其他调度策略的输入。

        该方法为调度决策提供输入数据，包括：
        - 计算可用边（valid_edges）。
        - 生成策略输入矩阵（policy_inputs），用于训练或推理调度决策。
        - 计算时间输入（time_inputs），用于时间敏感性决策。

        返回：
        - valid_edges: dict，可用边的字典，key 为边的 ID，value 为边对象。
        - policy_inputs: np.array，策略输入数据矩阵，每行对应一条边的特征。
        - time_inputs: 预留的时间特征输入（目前为 0）。
        - self.tt_flow_cycle: int，TT 流的周期（时隙单位）。
        - self.time % args.global_cycle: int，当前调度时间相对于全局周期的位置。
        - self.tt_flow_length: int，TT 流长度（时隙单位）。
        - self.tt_flow_deadline: int，TT 流的截止时间（时隙单位）。
        """
        self.valid_edges = {}                   # 存储当前可用的边
        edge_num = len(self.graph.edges)        # 获取边的总数

        # 初始化策略输入矩阵（每行对应一条边，列数等于 policy_input_dim）
        policy_inputs = np.zeros([edge_num, args.policy_input_dim])

        # 预留的时间特征输入，目前设为 0（可扩展）
        time_inputs = 0#np.zeros([edge_num, args.time_input_dim])

        # 遍历所有边，确定有效边
        for edge in self.graph.edges.values():
            if edge.start_node.is_source_node and edge.end_node.id not in self.visited_node \
                    and len(edge.time_slot_status[self.tt_flow_cycle]) > 0:
                edge.is_source_edge = 1
                self.valid_edges[edge.id] = edge
            if edge.end_node.is_destination_node:
                edge.is_destination_edge = 1

        total_usage = self.edge_usage()     # 计算整个网络的边利用率

        # 遍历所有边，计算策略输入
        for edge in self.graph.edges.values():
            # is_destination = edge.is_destination_edge
            is_source = edge.is_source_edge     # 记录该边是否是源边
            # 时延应该 从发出时间算，而不是从0时刻开始算
            # 计算当前流的起始时间（从 TT 流的第一条时间记录计算）
            start_time = -1
            offset = -1
            if len(self.tt_flow_time_record) > 1:
                start_time = self.tt_flow_time_record[-1] - self.tt_flow_time_record[1]
                offset = self.tt_flow_time_record[-1] % args.global_cycle

            # 查找该边上的可用时间隙
            time_slot, score = edge.find_time_slot(start_time, offset, self.tt_flow_cycle, self.tt_flow_length,
                                                   self.tt_flow_deadline)
            if time_slot < 0:
                is_source = 0 # 将超时因素也考虑进去  # 若无可用时隙，则该边不作为源边

            time_slot_num = self.avaiable_time_slot_number(edge)    # 获取该边可用时间隙数量
            # policy_inputs[edge.id, 0] = is_destination
            # policy_inputs[edge.id, 1] = end_node_buffer
            # policy_inputs[edge.id, 2] = time_slot_num
            # 生成策略输入特征（每条边的 8 个特征）
            policy_inputs[edge.id, 0] = self.graph.distance_edge_matrix[edge.id, self.tt_flow_end]          # 边到目标节点的距离
            policy_inputs[edge.id, 1] = is_source  # 不是邻边也可能是因为没有可用时隙                              # 该边是否为源边（但不一定是邻边）
            policy_inputs[edge.id, 2] = edge.end_node.id in self.visited_node                               # 目标节点是否已访问过
            policy_inputs[edge.id, 3] = time_slot_num / self.tt_flow_cycle                                  # 该边可用时隙比例
            policy_inputs[edge.id, 4] = sum(edge.time_slot_available) / args.global_cycle - total_usage     # 计算当前链路资源剩余情况
            policy_inputs[edge.id, 5] = (score + 120) / 120                                                 # 归一化调度得分
            policy_inputs[edge.id, 6] = (self.tt_flow_deadline - self.time) / self.tt_flow_deadline          # 归一化剩余截止时间
            policy_inputs[edge.id, 7] = sum(self.graph.reachable_edge_matrix[1][edge.id]) / len(self.graph.edges) * 3    # 可达边数量归一化
            # print((score + 120) / 120, (self.tt_flow_delay - self.time) / self.tt_flow_delay, sum(self.graph.reachable_edge_matrix[1][edge.id]) / 5)
        return self.valid_edges, policy_inputs, time_inputs, self.tt_flow_cycle, self.time % args.global_cycle, \
               self.tt_flow_length, self.tt_flow_deadline


    # 计算节点在 最近 depth 个时间单位内的缓冲区占用率，并使用衰减因子减少较远时间单位的影响。
    # 适用于需要考虑 节点的缓冲区稳定性 作为调度策略的一部分。
    def accumulated_buffer_occupation_rate(self, node):
        """
        计算给定节点的累积缓冲区占用率，考虑时间衰减因子。

        该方法通过遍历 `self.depth` 个时间单位，计算节点 `node` 的缓冲区占用情况，
        并使用 `decay` 系数进行时间加权，以便反映 **近期缓冲区占用情况对当前时刻的影响**。
        参数：
        - node: Node，当前需要计算缓冲区占用率的节点。
        逻辑：
        1. `occupation_rate` 初始化为 0，用于累加计算的占用率。
        2. `decay_t` 初始化为 1，每次迭代后乘以 `self.decay`，用于对更早的数据赋予较低的权重。
        3. `time_t` 计算当前时间相对于 `global_cycle` 的偏移，确保周期性时间处理。
        4. 在 `self.depth` 个时间单位内：
           - 累加 `node.buffer_avaiable[time_t]`，并乘以当前 `decay_t`。
           - `time_t` 更新为 `(time_t + 1) % args.global_cycle`，向前推进一个时间单位。
           - `decay_t` 乘以 `self.decay`，减少对较早时间单位的影响。
        5. 返回最终计算的 **累积缓冲区占用率**。
        为什么循环 `depth` 次？
        - `depth` 代表最近 `depth` 个时间单位内的缓冲区变化趋势。
        - 采用衰减因子 `decay` 给予较近的时间单位更大的权重，而较远的时间单位较小的权重。
        - 这样可以反映节点 **近期的缓冲区状态是否稳定**，有助于调度决策。

        返回：
        - occupation_rate: float，累积缓冲区占用率。
        """
        occupation_rate = 0
        decay_t = 1
        time_t = self.time % args.global_cycle
        # 此处为什么循环 depth 次？
        for i in range(self.depth):
            occupation_rate += decay_t * node.buffer_avaiable[time_t]
            time_t = (time_t + 1) % args.global_cycle
            decay_t *= self.decay
        return occupation_rate

    # 计算边 edge 在当前 TT 流周期 self.tt_flow_cycle 内的可用时间隙数量。
    # 适用于 调度决策过程中选择时隙可用性高的边。
    def avaiable_time_slot_number(self, edge):
        """
         计算边 `edge` 在当前 TT 流周期内的可用时隙数量。

         该方法返回 `edge.time_slot_status[self.tt_flow_cycle]` 的长度，
         即该边在当前 TT 流周期 `self.tt_flow_cycle` 内剩余可用的时间隙数。

         参数：
         - edge: Edge，当前需要计算可用时隙数量的边。

         逻辑：
         1. `edge.time_slot_status[self.tt_flow_cycle]` 记录了该边在 `self.tt_flow_cycle` 周期内的可用时间隙。
         2. 计算其长度，即可用时间隙的数量。
        """
        return len(edge.time_slot_status[self.tt_flow_cycle])

    # 返回奖励值和是否结束，1表示当前TT流调度结束，0表示调度未结束，-1表示调度失败
    def step(self, edge, time_slot, LD_score, heuristic=False):
        # print(LD_score)

        """
           执行一步调度操作。
           该方法用于分配时隙给特定边 `edge`，并根据调度结果计算奖励值。
           参数：
           - edge: Edge，当前选择的边。
           - time_slot: int，分配给 `edge` 的时间隙。
           - LD_score: float，路径调度评分（可能用于强化学习）。
           - heuristic: bool，是否使用启发式方法（默认 False）。
           逻辑：
           1. 如果 `time_slot > -1`，说明调度成功：
              - 更新 `schedule` 和 `current_stream_schedule`。
           2. 如果 `time_slot < 0` 或 `edge.id` 不在 `valid_edges`，则判断调度失败：
              - 计算负奖励 `reward`。
              - 根据 `time_slot` 失败原因，设置 `reason`：
                - `-1`: "No buffer"
                - `-2`: "No time slot"
                - `-3`: "timeout"
                - 其他: "Visited edge or Not adjacent edge"
              - 结束函数，返回负奖励。
           3. 调度成功，执行以下操作：
              - `occupy_time_slot()` 占用该时隙。
              - 计算 `self.time`，并更新 `tt_flow_time_record`。
              - 记录已访问节点，防止重复访问。
              - 记录流量信息到 `tt_flow_to_edge` 和 `edge_to_tt_flow`。
           4. 如果到达目标节点，计算最终奖励：
              - 计算 `delay`（调度延迟）。
              - 计算 `reward`，考虑：
                - 停止参数 `stop_parameter`。
                - `delay_parameter` 的延迟惩罚。
                - `edge_usage()` 计算网络利用率。
                - `LD_score` 影响最终分数。
              - 结束调度，返回奖励和 `1` 作为结束标志。
           5. 如果未到目标节点，则：
              - 更新 `source_node`，确保 `edge.end_node` 为新的源节点。
              - 刷新所有边状态。
              - 返回 `0`（表示调度仍在进行）。
           返回：
           - reward: float，奖励值。
           - int，状态标识：
             - `1`：调度完成。
             - `0`：调度仍在进行。
             - `-1`：调度失败。
           - reason: str，失败原因或 "Success"。
           """

        if time_slot > -1:
            self.schedule.update(edge.end_node.id, time_slot)
            self.current_stream_schedule.update(edge.end_node.id, time_slot)
        else:
            self.schedule.update(-1, -1)
            self.current_stream_schedule.update(-1, -1)

        reason = "Success"
        # 调度失败
        if (time_slot < 0 or edge.id not in self.valid_edges) and not heuristic:
            reward = -1 * self.stop_parameter - LD_score / 24 - \
                     1 / (self.tt_query_id + 1) * self.lasting_parameter
            if time_slot == -1:
                reason = "No buffer"
            elif time_slot == -2:
                # total_edge_usage = self.edge_usage()
                # usages = []
                # for edge in self.graph.edges.values():
                #     usages.append(sum(edge.time_slot_available) / args.global_cycle)
                # print(sum(edge.time_slot_available) / args.global_cycle, min(usages), total_edge_usage)
                reason = "No time slot"
                reward -= 10
                # print(reason, reward)
            elif time_slot == -3:
                reason = "timeout"
            elif edge.id not in self.valid_edges:
                reason = "Visited edge or Not adjacent edge"
                reward -= 10
            # print(reason, reward, self.valid_edges)
            self.time = 0
            return reward, -1, reason

        # 无论调度是否结束，都需要占用当前时隙
        edge.occupy_time_slot(time_slot, self.tt_flow_cycle, self.tt_flow_length)

        # 无论调度是否结束，都需要记录发出时间
        offset = self.time % args.global_cycle
        self.time = self.time + (time_slot - offset + args.global_cycle) % args.global_cycle + 1
        self.tt_flow_time_record.append(self.time)
        # 将当前选中的边加入集合，此后不能再次进入这个节点
        self.visited_node.append(edge.end_node.id)
        # 记录链路的流的信息
        self.tt_flow_to_edge[self.cur_tt_flow_id].append(
            [(edge.start_node.id, edge.end_node.id), time_slot, self.tt_flow_cycle, self.tt_flow_length])
        self.edge_to_tt_flow[(edge.start_node.id, edge.end_node.id)].add(self.cur_tt_flow_id)

        # 调度成功
        if edge.end_node.is_destination_node:
            delay = self.tt_flow_time_record[-1] - self.tt_flow_time_record[1]
            reward = 1 * self.stop_parameter - delay * self.delay_parameter - 0.1 * len(self.tt_flow_time_record) + \
                     10 * (sum(edge.time_slot_available) / args.global_cycle - self.edge_usage()) - LD_score / 24
            # 记录上一个调度
            # self.last_schedule = copy.deepcopy(self.schedule.sche)
            self.time = 0
            self.visited_node = []
            return reward, 1, reason

        # 中间步骤，更新节点和边信息
        for node in self.graph.nodes.values():
            node.is_source_node = 0
        edge.end_node.is_source_node = 1
        # print(edge.start_node.id, edge.end_node.id)
        for edge in self.graph.edges.values():
            edge.refresh()
        return 0, 0, reason

    def roll_back(self, number):
        self.reschedule_queries = []
        self.reschedule_pos = -1
        for i in reversed(range(number)):
            cur_id = self.cur_tt_flow_id - i
            if cur_id not in self.tt_flow_to_edge:
                print(cur_id)
                continue
            self.delete_tt_flow(cur_id)
            for info in self.edge_to_tt_flow.values():
                if cur_id in info:
                    info.remove(cur_id)
        self.tt_query_id = self.cur_tt_flow_id
        self.reschedule = 1

    def delete_tt_flow(self, tt_flow_id, reschedule=True):
        # print("delete flow", tt_flow_id)
        for info in self.tt_flow_to_edge[tt_flow_id]:
            edge_tuple = info[0]
            edge_id = self.graph.node_to_edge[edge_tuple]
            time_slot = info[1]
            cycle = info[2]
            self.graph.edges[edge_id].reset_time_slot(time_slot, cycle)

        self.schedule.delete_by_tt_flow_id(tt_flow_id)
        self.current_stream_schedule.reset()
        self.tt_flow_to_edge.pop(tt_flow_id)

        if not reschedule:
            for info in self.edge_to_tt_flow.values():
                if tt_flow_id in info:
                    info.remove(tt_flow_id)

        if reschedule:
            self.reschedule_queries.append(tt_flow_id)

    def delete_edge(self, edge_id):
        print(self.edge_to_tt_flow[edge_id])
        for tt_flow_id in self.edge_to_tt_flow[edge_id]:
            self.delete_tt_flow(tt_flow_id)
        self.edge_to_tt_flow[edge_id] = set()
        self.graph.delete_edge(edge_id, self.tt_flow_to_edge)

    def delete_node(self, node_id):
        node_num = len(self.graph.nodes)
        for i in range(node_num):
            if self.graph.adjacent_node_matrix[i][node_id] == 1:
                self.delete_edge((i, node_id))
            if self.graph.adjacent_node_matrix[node_id][i] == 1:
                self.delete_edge((node_id, i))

    def reset(self):
        self.graph.reset()
        self.schedule.reset()
        self.current_stream_schedule.reset()
        self.tt_query_id = 0
        self.enforce_next_query()
        self.visited_node = []


def main():
    env = Environment(Graph())
    env.enforce_next_query()
    valid_edges, policy_inputs, time_inputs = env.translate_data_to_inputs()
    for edge in valid_edges:
        print(edge.id, edge.is_source_edge, edge.time_slot_available)
    print(policy_inputs)
    print(time_inputs)


if __name__ == '__main__':
    main()
