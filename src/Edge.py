import numpy as np
from param import *


class Edge:
    def __init__(self, index, start_node, end_node):
        self.id = index                         # 边的唯一标识符(ID)。
        self.start_node = start_node            # 起始节点。
        self.end_node = end_node                # 终止节点。
        self.is_source_edge = self.start_node.is_source_node                # 记录该边是否连接到源节点。
        self.is_destination_edge = self.end_node.is_destination_node        # 记录该边是否连接到目标节点。
        self.global_cycle = args.global_cycle                               # 全局时隙循环周期
        self.time_slot_available = np.ones([self.global_cycle])             # 记录时隙是否可用，初始化时全部置为 1（可用）。
        self.time_slot_status = {}                                          # 记录时隙状态，存储不同周期的可用时隙。
        self.max_cycle = max(args.tt_flow_cycles)                           # 计算 tt_flow_cycles（时间触发流周期）的最大值。
        # self.queue_status = [0 for i in range(8)]
        # 计算不同周期对应的可用时隙集合
        for cycle in [i * args.slot_per_millisecond for i in args.tt_flow_cycles]:
            self.time_slot_status[cycle] = {i for i in range(cycle)}


    def occupy_single_time_slot(self, time_slot, cycle):
        """
            占用单个时间隙，并在多个周期中标记该时间隙为不可用。
            参数：
            - time_slot: int，要占用的时隙索引。
            - cycle: int，该时隙的重复周期。

              逻辑：
            1. 遍历 `self.time_slot_status`，对于所有周期，移除 `time_slot` 在模运算后的索引，确保该时隙不可用。
            2. 在 `self.global_cycle` 范围内，每隔 `cycle` 个时隙，标记 `self.time_slot_available[pos] = 0`，即不可用。
            3. 断言 `self.time_slot_available[pos] == 1`，确保在占用之前该时隙是可用的。
        """
        for key in self.time_slot_status:
            pos = time_slot % key
            if pos in self.time_slot_status[key]:
                self.time_slot_status[key].remove(pos)
        pos = time_slot
        while pos < self.global_cycle:
            assert self.time_slot_available[pos] == 1
            for key in self.time_slot_status:
                if pos in self.time_slot_status[key]:
                    self.time_slot_status[key].remove(pos)
            self.time_slot_available[pos] = 0
            pos += cycle

    def occupy_time_slot(self, time_slot, cycle, length):
        """
            预留多个连续的时间隙。
            参数：
            - time_slot: int，起始时间隙索引。
            - cycle: int，时隙的重复周期。
            - length: int，占用的连续时间隙数量。
            逻辑：
            1. 通过 `occupy_single_time_slot` 依次占用从 `time_slot` 开始的 `length` 个时隙。
            2. 这些时隙的占用周期为 `cycle`，即在整个 `global_cycle` 中按 `cycle` 间隔重复占用。
            示例：
            若 `time_slot=3, cycle=10, length=2`，则：
            - 第 `3` 和 `4` 个时隙会被占用。
            - 在 `global_cycle` 内，每 `10` 个时隙重复占用 `3` 和 `4` 号时隙。
         """
        # print("occupy", time_slot, cycle, length)
        for k in range(length):
            self.occupy_single_time_slot(time_slot + k, cycle)

    # start_time表示目前已经使用的时间，offser表示目前所在的时隙
    def find_time_slot(self, start_time, offset, cycle, length, deadline):
        """
        在指定的周期 `cycle` 内寻找一个合适的时间隙，使得：
        - 该时间隙及其后续 `length` 个时隙是可用的
        - 符合 `deadline` 约束
        - 计算一个评分（score）来选择最优的时间隙

        参数：
        - start_time: int，开始寻找时间隙的时间点。
        - offset: int，当前任务的时间偏移量。
        - cycle: int，时间隙的周期。
        - length: int，连续需要占用的时间隙个数。
        - deadline: int，可接受的最大延迟时间。

         逻辑：
        1. 遍历 `cycle` 内所有可能的时间隙 `i`：
           - 检查 `i` 及 `length` 个后续时间隙是否可用。
        2. 计算该时间隙 `i` 产生的 `delay`（延迟时间）。
           - 如果超出 `deadline`，则跳过。
        3. 计算评分 `score`：
           - 考虑其他周期 `key` 中的时间隙使用情况，减少冲突。
           - 计算 `(deadline - delay)`，如果 delay 越大，惩罚越重。
        4. 选择得分最高的时间隙 `time_solt`。

        返回：
        - time_solt: int，找到的最佳时间隙索引，如果找不到，返回 -3。
        - max_score: float，对应的评分。
        """
        max_score = -120
        time_solt = -2      # 默认没有找到合适的时间隙
        # print(self.time_slot_status, cycle)
        for i in range(cycle):
            # 检查从 i 开始的 length 个时间隙是否都可用
            flag = True
            for k in range(length):
                if i + k not in self.time_slot_status[cycle]:
                    flag = False
            if not flag:
                continue

            # 计算当前时间隙的延迟
            if i > offset:
                delay = start_time + i - offset
            else:
                delay = start_time + i - offset + args.global_cycle
            if start_time == -1:
                delay = 0

            # 如果延迟超过 deadline，则跳过
            if delay > deadline:
                if time_solt < 0:
                    time_solt = -3
                continue

            # 计算评分
            score = 0
            for k in range(length):
                for key in self.time_slot_status:
                    pos = (i + k) % key
                    if pos in self.time_slot_status[key] and key != cycle:
                        score -= args.global_cycle / key    # 通过周期长度权衡时间隙占用情况
            score -= args.global_cycle * 2 / (deadline - delay + 1)     # 给予靠近deadline的时间更大惩罚
            # print(i, start_time, deadline - delay, score)

            # 选择评分最高的时间隙
            if time_solt < 0 or max_score < score:
                time_solt = i
                max_score = score
        return time_solt, max_score

    # 找最快的时隙
    def find_time_slot_fast(self, start_time, offset, cycle, length, max_delay):
        """
          在周期 `cycle` 内寻找最小延迟的时间隙，以满足 `max_delay` 约束。
          参数：
          - start_time: int，任务开始时间。
          - offset: int，当前任务的时间偏移量。
          - cycle: int，时间隙周期。
          - length: int，连续需要占用的时间隙个数。
          - max_delay: int，允许的最大延迟时间。
          逻辑：
          1. 遍历 `cycle` 内所有可能的时间隙 `i`：
             - 检查 `i` 及 `length` 个后续时间隙是否可用。
          2. 计算该时间隙 `i` 产生的 `delay`（延迟时间）。
             - 如果 `delay > max_delay`，跳过该时间隙。
          3. 选择 `delay` 最小的时间隙 `time_solt`。
          返回：
          - time_solt: int，找到的最小延迟的时间隙索引，如果找不到，返回 -2。
          - -delay: int，返回负值，表示最优解的延迟时间（负号仅用于排序权重）。
          """
        min_delay = -120
        time_solt = -2
        delay = -1
        # print(self.time_slot_status, cycle)
        for i in range(cycle):
            flag = True
            for k in range(length):
                if i + k not in self.time_slot_status[cycle]:
                    flag = False
            if not flag:
                continue
            if i > offset:
                delay = start_time + i - offset
            else:
                delay = start_time + i - offset + args.global_cycle
            if start_time == -1:
                delay = 0
            if delay > max_delay:
                continue
            if time_solt == -2 or delay < min_delay:
                time_solt = i
                min_delay = delay
        return time_solt, -delay

    def find_time_slot_LD_old(self, start_time, cycle, length, max_delay):
        """
        在给定周期 `cycle` 内查找合适的时间隙，以最小化延迟并优化调度。

        参数：
        - start_time: int，任务开始时间。
        - cycle: int，时间隙的周期长度。
        - length: int，占用的连续时隙数量。
        - max_delay: int，允许的最大延迟时间。

        逻辑：
        1. 遍历 `cycle` 内所有可能的时间隙 `i`：
           - 检查 `i` 及 `length` 个后续时间隙是否可用。
        2. 计算当前时间隙 `i` 产生的 `delay`（延迟时间）。
           - 若 `delay > max_delay`，跳过该时间隙。
        3. 计算评分 `score`：
           - 避免与其他周期 `key` 冲突（权重为 `global_cycle * 4 / key`）。
           - 对 `delay` 进行惩罚，使得更早的时间隙优先选择。
        4. 选择评分最高的时间隙 `time_solt`。

        返回：
        - time_solt: int，找到的最佳时间隙索引，若找不到，返回 -2。
        - max_score: float，对应的评分。

        """
        max_score = -120
        time_solt = -2
        # print(self.time_slot_status, cycle)
        for i in range(cycle):
            flag = True
            for k in range(length):
                if i + k not in self.time_slot_status[cycle]:
                    flag = False
            if not flag:
                continue
            delay = (i - start_time + args.global_cycle) % args.global_cycle
            if delay == 0:
                delay = args.global_cycle
            if start_time == -1:
                delay = 0
            if delay > max_delay:
                continue
            score = 0
            for k in range(length):
                for key in self.time_slot_status:
                    pos = (i + k) % key
                    if pos in self.time_slot_status[key] and key != cycle:
                        score -= args.global_cycle * 4 / key
            score -= delay
            if time_solt == -2 or max_score < score:
                time_solt = i
                max_score = score
        # print(max_time_slot_score)
        if time_solt == -3:
            print(start_time, cycle)
        return time_solt, max_score

    def reset_time_slot(self, time_slot, cycle=args.global_cycle):
        """
            释放已占用的时间隙，使其重新可用。

            参数：
            - time_slot: int，要释放的起始时间隙。
            - cycle: int，该时间隙的重复周期，默认为全局周期 `args.global_cycle`。

            逻辑：
            1. 遍历 `self.time_slot_status`，恢复 `time_slot` 在不同周期 `key` 中的可用性：
               - 若 `key >= cycle`，则 `time_slot` 在 `cycle` 递增的间隔下恢复可用。
               - 否则，取 `time_slot % key` 判断是否可以恢复，并添加回 `self.time_slot_status[key]`。
            2. 遍历 `self.global_cycle`，恢复 `self.time_slot_available` 中对应的时隙。

            备注：
            - `self.time_slot_status[key]` 记录了周期 `key` 内可用的时间隙。
            - `self.time_slot_available` 记录整个 `global_cycle` 内的可用性。
            """
        # print("reset", time_slot, cycle)
        for key in self.time_slot_status:
            if key >= cycle:
                pos = time_slot
                while pos < key:
                    self.time_slot_status[key].add(pos)
                    pos += cycle
            else:
                pos = time_slot % key
                if self.judge(pos, key):
                    self.time_slot_status[key].add(pos)
        pos = time_slot
        while pos < self.global_cycle:
            assert self.time_slot_available[pos] == 0
            self.time_slot_available[pos] = 1
            pos += cycle

    def judge(self, time_slot, cycle):
        """
         判断某个时间隙 `time_slot` 在周期 `cycle` 内是否完全可用。

         参数：
         - time_slot: int，要检查的时间隙。
         - cycle: int，该时间隙的重复周期。

         逻辑：
         1. 遍历 `self.global_cycle` 范围内的 `time_slot`：
            - 若发现任何 `self.time_slot_available[pos] == 0`，说明不可用，返回 False。
         2. 如果所有 `pos` 都是可用的，返回 True。

         返回：
         - bool: 该时间隙是否完全可用。
         """
        pos = time_slot
        while pos < self.global_cycle:
            if self.time_slot_available[pos] == 0:
                return False
            pos += cycle
        return True

    def refresh(self):
        """
         刷新边的状态，使其同步起点和终点节点的信息。

         逻辑：
         1. `is_source_edge`: 该边的起始节点是否为源节点。
         2. `is_destination_edge`: 该边的终点节点是否为目标节点。

         备注：
         - 这个方法主要用于确保 `Edge` 的 `is_source_edge` 和 `is_destination_edge` 状态同步更新。
         - 当网络拓扑发生变化时，应调用此方法进行刷新。
         """
        self.is_source_edge = self.start_node.is_source_node
        self.is_destination_edge = self.end_node.is_destination_node

    def reset(self):
        """
            彻底重置当前边的时隙状态，并刷新边的节点信息。
            逻辑：
            1. `self.time_slot_available` 重新初始化为全可用状态（全为1）。
            2. 调用 `self.refresh()`，同步更新 `is_source_edge` 和 `is_destination_edge` 状态。
            备注：
            - 该方法适用于清空所有时隙的占用情况，使边恢复到初始状态。
            - 若不希望 `refresh()` 影响当前状态，可将其注释掉。
            """
        self.time_slot_available = [1 for _ in range(self.global_cycle)]
        self.refresh()
