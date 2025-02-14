import tensorflow as tf
from tensorflow import keras
from keras import layers as tl
import time
from gcn import GraphCNN
from agent import Agent
from Environment import *
from Data.Random import *
from Data.A380 import *
from Data.Ladder import *

# import sys
# 系统路径进入到上层目录，可以引用上层目录的库
# sys.path.append("..")

from param import *
from utils import *
from tf_op import *

class ActorAgent(Agent):
    def __init__(self, policy_input_dim, time_input_dim, hid_dims, output_dim,
                 max_depth, eps=1e-6, act_fn=leaky_relu,
                 optimizer=tf.keras.optimizers.Adam, scope='actor_agent'):

        Agent.__init__(self)

        self.policy_input_dim = policy_input_dim
        self.time_input_dim = time_input_dim
        self.hid_dims = hid_dims
        self.output_dim = output_dim
        self.max_depth = max_depth
        self.eps = eps
        self.act_fn = act_fn
        self.optimizer = optimizer
        self.scope = scope
        # update
        self.success_exps = []
        self.fail_exps = []
        self.flow_count_record = {}

        # node input dimension: [total_num_nodes, num_features]
        self.policy_inputs = tf.keras.Input(shape=(args.policy_input_dim,), dtype=tf.float32)

        # 8维
        self.gcn_policy = GraphCNN(
            self.policy_inputs, self.policy_input_dim, self.hid_dims,
            self.output_dim, self.max_depth, self.act_fn, "policy")

        # map gcn_outputs and raw_inputs to action probabilities
        # node_act_probs: [batch_size, total_num_nodes]
        # job_act_probs: [batch_size, total_num_dags]
        # gsn只有一个输出，是整图编码
        self.policy_act_probs = self.actor_network(
            self.policy_inputs, self.gcn_policy.outputs, self.act_fn)

        # draw action based on the probability (from OpenAI baselines)
        # node_acts [batch_size, 1]
        logits = tf.math.log(self.policy_act_probs)
        noise = tf.random.uniform(tf.shape(logits))
        # 这处对 noise 连用两个log好奇怪
        self.policy_acts = tf.argmax(logits - tf.math.log(-tf.math.log(noise)), 1)

        # Selected action for node, 0-1 vector ([batch_size, total_num_nodes])
        self.edge_selected_vec = tf.keras.Input(shape=(None,), dtype=tf.float32)
        # Selected action for job, 0-1 vector ([batch_size, num_jobs, num_limits])

        # advantage term (from Monte Calro or critic) ([batch_size, 1])
        self.adv = tf.keras.Input(shape=(1,), dtype=tf.float32)

        # select node action probability
        # node_act_probs：一个全连接网络
        # node_act_vec：一个占位符
        # reduction_indices指定reduce_sum求和的轴维度
        # keep_dims求和后是否保持原有维度
        # node_act_vec是一个表示节点选取的向量，除了选出来的节点为1外，其他位置都为0
        self.selected_edge_prob = tf.reduce_sum(tf.multiply(self.policy_act_probs, self.edge_selected_vec))

        # actor loss due to advantge (negated)
        # reduce_sum：计算张量某一维的和
        # multiply：矩阵各对应位置元素相乘
        # self.eps：一个常量，感觉是用来防止和为0的
        # self.adv：一个占位符
        self.edge_act_loss = tf.multiply(tf.math.log(self.selected_edge_prob + 1e-6), -self.adv)
        # prob on each job
        # self.prob_each_job = tf.reshape(
        #     tf.sparse_tensor_dense_matmul(self.gsn.summ_mats[0],
        #         tf.reshape(self.node_act_probs, [-1, 1])),
        #         [tf.shape(self.node_act_probs)[0], -1])

        # define combined loss
        # self.act_loss = self.adv_loss

        # todo修改
        # get training parameters
        self.edge_params = self.gcn_policy.trainable_variables

        # actor gradients
        self.edge_act_gradients = tf.gradients(self.edge_act_loss, self.edge_params)

        # adaptive learning rate
        self.lr_rate = tf.placeholder(tf.float32, shape=[])

        # actor optimizer
        # self.act_opt = self.optimizer(self.lr_rate).minimize(self.act_loss)

        # apply gradient directly to update parameters
        self.apply_edge_grads = self.optimizer(self.lr_rate).\
            apply_gradients(zip(self.edge_act_gradients, self.edge_params))

        self.env = None

        # network paramter saver
        self.saver_policy = tf.train.Saver(self.edge_params, max_to_keep=args.num_saved_models)
        self.sess.run(tf.global_variables_initializer())

    # gsn只有一个量，
    def actor_network(self, policy_inputs, gcn_policy_outputs, act_fn):

        # takes output from graph embedding and raw_input from environment
        batch_size = 1 # tf.shape(policy_inputs)[0]

        # (1) reshape node inputs to batch format
        policy_inputs_reshape = tf.reshape(
            policy_inputs, [batch_size, -1, args.policy_input_dim])

        # (4) reshape gcn_outputs to batch format
        gcn_policy_outputs_reshape = tf.reshape(
            gcn_policy_outputs, [batch_size, -1, self.output_dim])

        # (4) actor neural network
        with tf.name_scope("policy"):
            # -- part A, the distribution over nodes --
            policy_input = tf.concat([policy_inputs_reshape, gcn_policy_outputs_reshape], axis=2)

            # 构建节点网络结构
            # 第一层：输出维度为32，激活函数为 act_fn
            policy_hid_0 = tl.Dense(32, activation=act_fn)(policy_input)
            # 第二层：输出维度为16，激活函数为 act_fn
            policy_hid_1 = tl.Dense(16, activation=act_fn)(policy_hid_0)
            # 第三层：输出维度为8，激活函数为 act_fn
            policy_hid_2 = tl.Dense(8, activation=act_fn)(policy_hid_1)
            # 输出层：输出维度为1，不使用激活函数
            policy_outputs = tl.Dense(1, activation=None)(policy_hid_2)

            # 重塑输出形状为 (batch_size, total_num_nodes)
            policy_outputs = tf.reshape(policy_outputs, [batch_size, -1])

            # 重塑输出形状为 (batch_size, total_num_nodes)
            policy_outputs = tf.reshape(policy_outputs, [batch_size, -1])

            # 使用绝对值处理DNN输出
            # policy_abs = tf.abs(policy_outputs)
            # policy_max = tf.reduce_max(policy_abs, keep_dims=True, axis=-1)
            # policy_outputs = tf.divide(policy_outputs, policy_max)
            # policy_outputs = tf.nn.softmax(policy_outputs, dim=-1)

            # 使用“减去最小值”方法处理网络输出
            policy_min = tf.reduce_min(policy_outputs, keepdims=True, axis=-1)
            policy_outputs = tf.subtract(policy_outputs, policy_min)
            policy_max = tf.reduce_max(policy_outputs, keepdims=True, axis=-1)
            policy_outputs = tf.divide(policy_outputs, policy_max)
            policy_outputs = tf.nn.softmax(policy_outputs, axis=-1)

            return policy_outputs

    def apply_edge_gradients(self, gradients, lr_rate):
        # print("apply_edge_gradients")
        self.sess.run(self.apply_edge_grads, feed_dict={
            i: d for i, d in zip(
                self.edge_act_gradients + [self.lr_rate],
                gradients + [lr_rate])
        })

    def invoke_model(self, manual):
        # implement this module here for training
        # (to pick up state and action to record)
        valid_edges, policy_inputs, time_inputs, cycle, time_offset, flow_length, max_delay = self.env.translate_data_to_inputs()
        if manual and len(valid_edges) == 0:
            print("manual error!")
            return policy_inputs, time_inputs, -1, -2, -1, -1, -1, False
        # invoke learning model
        # 使用深度网络得到节点和任务决策
        # 每一条合法边都需要被判断一次，每一次的可达性矩阵是不同的，这能够产生针对当前边的gcn
        res = []
        # print(time_inputs)
        edge_act_probs, edge_acts, gcn = self.sess.run(
            [self.policy_act_probs, self.policy_acts, self.gcn_policy.outputs],
            feed_dict={i: d for i, d in zip(
                [self.policy_inputs] + [self.gcn_policy.reachable_edge_matrix],
                [policy_inputs] + [self.env.graph.reachable_edge_matrix])
                       })
        # print("gcn \n", gcn)
        # print("policy inputs:\n", policy_inputs)
        # print("policy ", edge_act_probs)
        scope = range(len(policy_inputs))
        if manual:
            scope = valid_edges
        for edge_id in scope:
            res.append([edge_id, edge_act_probs[0, edge_id]])
        res = sorted(res, key=lambda edge_info: edge_info[1])
        # print("edge network: ", res)
        # 选边
        edge_info = res[-1]
        edge = self.env.graph.edges[edge_info[0]]
        # print("edge network: ", res, edge.id)
        # 计算策略网络梯度梯度：给出选中的边的mask
        edge_selected_mask = np.zeros([1, len(self.env.graph.edges)])
        edge_selected_mask[0, edge.id] = 1

        # 时延应该 从发出时间算，而不是从0时刻开始算
        start_time = -1
        offset = -1
        if len(self.env.tt_flow_time_record) > 1:
            start_time = self.env.tt_flow_time_record[-1] - self.env.tt_flow_time_record[1]
            offset = self.env.tt_flow_time_record[-1] % args.global_cycle
        time_slot, LD_score = edge.find_time_slot(start_time, offset, cycle, flow_length, max_delay)
        return policy_inputs, time_inputs, edge, time_slot, edge_selected_mask, cycle, LD_score, True

    def compute_edge_gradient(self, policy_inputs, gcn_policy_masks, edge, edge_selected_vec, adv):
        gradients, loss = self.sess.run([self.edge_act_gradients, self.edge_act_loss],
                                        feed_dict={i: d for i, d in zip(
                [self.policy_inputs] + [self.gcn_policy.reachable_edge_matrix] + [self.edge_selected_vec] + [self.adv], \
                [policy_inputs] + [gcn_policy_masks] + [edge_selected_vec] + [adv])
                        })
        # print("edge loss", edge_probs, loss)
        return gradients, loss

    def update(self):
        # 反向传播更新
        edge_gradients = []
        cur_exps = random.sample(self.success_exps, min(len(self.success_exps), 800))
        cur_exps.extend(random.sample(self.fail_exps, min(len(self.fail_exps), 200)))
        loss_record = []
        for exp in cur_exps:
            policy_input = exp[0]
            edge = exp[2]
            edge_vec = exp[6]
            mask = exp[8]
            adv = np.zeros([1])
            if exp[4] > 0:
                adv[0] = exp[4] * (1 + self.flow_count_record[exp[9]] - 0.5)
            else:
                adv[0] = exp[4]
            # print(exp[4], adv[0], self.flow_count_record[exp[9]])
            # print(adv[0], exp[4], ((1 + flow_count_record[exp[9]] - 0.5) ** 2))
            edge_gradient, edge_loss = self.compute_edge_gradient(policy_input, mask, edge, edge_vec, adv)
            loss_record.append(edge_loss)
            edge_gradients.append(edge_gradient)
        print("loss", sum(loss_record) / len(loss_record))

        edge_gradients = aggregate_gradients(edge_gradients)
        self.apply_edge_gradients(edge_gradients, 0.001)


def discount(rewards, gamma):
    res = [i for i in rewards]
    total = len(rewards)
    for i in reversed(range(len(rewards) - 1)):
        res[i] += gamma * res[-1] * (total - (len(rewards) - i - 1)) / total
    return res


def main():
    np.random.seed()
    tf.random.set_seed(0)

    # 配置GPU（例如开启内存自增长）
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    actor_agent = ActorAgent(
        args.policy_input_dim, 1028,
        args.hid_dims, args.output_dim, args.max_depth)

    # 加载已经训练好的模型，参数是模型位置，默认为models/zcm
    main_model = "../models/DRLS_Model/111203/policy/"
    actor_agent.saver_policy.restore(actor_agent.sess, main_model)

    node_nums = [15 for _ in range(50000)]
    index = 0
    min_usage = 1
    update_number = -1
    max_exps_number = 60000
    total_exps_cnt = 0
    model_count = 0
    for node_num in range(100000):
        index += 1
        print("train number", index, "last updated", update_number)
        data = DataGenerater()
        data.gene_all(node_num=15, eps=0.35, rand_min=5, rand_max=10, tt_num=60000,
                      delay_min=64, delay_max=512, pkt_min=72, pkt_max=1526, hop=1, dynamic=True)
        # data_gene = A380Generater()
        # data_gene.gene_all(rand_min=1000, rand_max=1000, tt_num=60000,
        #                    delay_min=64, delay_max=256, pkt_min=64, pkt_max=1526, dynamic=True)
        # data_gene = LadderGenerater()
        # node_num = 14
        # print("node number", node_num)
        # data_gene.gene_all(node_num=node_num, eps=0.35, rand_min=5, rand_max=10, tt_num=60000,
        #                    delay_min=64, delay_max=512, pkt_min=72, pkt_max=1526, hop=1, dynamic=True)

        actor_agent.env = Environment(data)
        start_time = time.time()
        runtime_threshold = 60
        done = 0
        flow_number = 0
        per_flow_cnt_total = 0
        per_flow_cnt_valid = 0
        time_record = {}
        manual = False # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!一定记得改回False
        while True:
            exps = []
            while done == 0:
                per_flow_cnt_total += 1
                per_flow_cnt_valid += 1
                # print("TT_flow: ", flow_cnt, "epoch: ", cnt, "index: ", index)
                policy_inputs, time_inputs, edge, time_slot, edge_selected_mask, cycle, LD_score, flag = actor_agent.invoke_model(manual)
                reward, done, reason = actor_agent.env.step(edge, time_slot, LD_score)
                # print(len(actor_agent.env.schedule.sche))
                # print("    reward: ", reward, "done: ", done)
                if flag:
                    exps.append([policy_inputs, time_inputs, edge, time_slot, reward, done, edge_selected_mask, cycle,
                                 actor_agent.env.graph.reachable_edge_matrix, index])
                else:
                    exps[-1][4] = reward

            cumulated_reward = np.array([exp[4] for exp in exps])
            cumulated_reward = discount(cumulated_reward, 0.8)
            for i in range(len(exps)):
                exps[-i - 1][4] = cumulated_reward[-i - 1]
            if done == 1:
                for exp in exps:
                    total_exps_cnt += 1
                    if len(actor_agent.success_exps) < max_exps_number:
                        actor_agent.success_exps.append(exp)
                    else:
                        k = random.randint(0, total_exps_cnt - 1)
                        if k < max_exps_number:
                            actor_agent.success_exps[k] = exp
            else: #if reason == "Visited edge or Not adjacent edge":
                print(reason, actor_agent.env.tt_query_id, flow_number, done)
                actor_agent.fail_exps.extend(exps)
            # print("cumulated reward", cumulated_reward, "done", done)

            cur_time = time.time()
            if done == -1:
                # if exps[-1][4] < -15:
                #     print(exps[-1][0])
                actor_agent.env.enforce_next_query()  # 调度下一条流
                done = 0
                per_flow_cnt_valid = 0
                # actor_agent.env.roll_back(1)
            elif done == 1: # 继续调度下一条流
                delay = actor_agent.env.tt_flow_time_record[-1] - actor_agent.env.tt_flow_time_record[0]
                usage = actor_agent.env.edge_usage()
                print("TT_flow", flow_number, "cycle", cycle, "usage", usage,  "use time", cur_time - start_time, "delay", delay, "reward", reward)
                time_record[flow_number] = [flow_number, cycle, per_flow_cnt_total, per_flow_cnt_valid, cur_time - start_time, delay]
                actor_agent.env.enforce_next_query()  # 调度下一条流
                start_time = time.time()
                flow_number += 1
                per_flow_cnt_total = 0
                per_flow_cnt_valid = 0
                done = 0
            # print(actor_agent.env.edge_usage(), len(success_exps), len(fail_exps))

            if (actor_agent.env.tt_query_id >= flow_number + 10) or actor_agent.env.tt_query_id == 59999:
                # actor_agent.env.schedule.show()
                break
        edge_usage = actor_agent.env.edge_usage()
        print(edge_usage, len(actor_agent.success_exps), len(actor_agent.fail_exps))
        actor_agent.flow_count_record[index] = 1 - edge_usage
        # 反向传播更新
        actor_agent.update()

        # if edge_usage < 0.5 or edge_usage < min_usage:
        #     min_usage = edge_usage
        #     actor_agent.saver_policy.save(actor_agent.sess, "../models/Ladder_Model-8/L/policy/")
        #     if edge_usage < 0.5:
        #         while os.path.exists(f'../models/Ladder_Model-8/L{model_count}'):
        #             model_count += 1
        #         os.mkdir(f'../models/Ladder_Model-8/L{model_count}')
        #         cur_model_path = "../models/Ladder_Model-8/L" + str(model_count) + "/policy/"
        #         print(cur_model_path)
        #         # actor_agent.saver_policy.save(actor_agent.sess, cur_model_path)
        #         time.sleep(5)
        #
        #     update_number = index


if __name__ == '__main__':
    main()
