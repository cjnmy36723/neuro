# config=utf-8
import random, math


class BackPropagationNetwork(object):
    """
        BP 神经网络。
    """
    # 权重集合
    weight_layers = []
    # 神经元集合
    neuro_layers = []
    # 误差值，如果真实的误差比该值小，则表示可以收敛。
    difference = 0.0001
    # 学习率，即：每次修正误差时，移动的距离。
    rate = 0.1
    # 是否已完成训练
    completed_epoch = False

    def __init__(self, input_count, *neuro_count):
        # 前一层神经元的个数。
        previously = 0

        # 初始化神经元和权重。
        for index in range(len(neuro_count)):
            # 定义每层神经元的个数
            neuro = neuro_count[index]
            # self.neuro_layers.append([neuro])
            neuro_rows = []
            weight_rows = []
            for loop in range(neuro):
                neuro_rows.append(0)
                if index == 0:
                    weight_cols = []
                    for jump in range(input_count):
                        # 初始化权重值是 0 至 1 之间的小数。
                        weight_cols.append(random.random())
                    weight_rows.append(weight_cols)
                else:
                    weight_cols = []
                    for jump in range(previously):
                        # 初始化权重值是 0 至 1 之间的小数。
                        weight_cols.append(random.random())
                    weight_rows.append(weight_cols)

            self.neuro_layers.append(neuro_rows)
            self.weight_layers.append(weight_rows)
            previously = neuro

    def __neuro_handler(self, *properties):
        """
            计算神经元输出值。
        :param properties:
            特征集合。
        :return:
            无。
        """
        for index_layer in range(len(self.neuro_layers)):
            for index_number in range(len(self.neuro_layers[index_layer])):
                # 神经元输入信号加权求和。
                input_sum = 0
                if index_layer == 0:
                    for property_index_number in range(len(properties)):
                        # 输入参数，样本的特征集合。
                        input_property = properties[property_index_number]
                        # 权重。
                        input_weight = self.weight_layers[index_layer][index_number][property_index_number]
                        # 加权求和。
                        input_sum += input_property * input_weight
                else:
                    # 前一层的神经元集合。
                    left_neuro_layer = self.neuro_layers[index_layer - 1]
                    for property_index_number in range(len(left_neuro_layer)):
                        # 输入参数，上一层的所有神经元的输出值的集合。
                        input_property = left_neuro_layer[property_index_number]
                        # 权重。
                        input_weight = self.weight_layers[index_layer][index_number][property_index_number]
                        # 加权求和。
                        input_sum += input_property * input_weight
                # 计算神经元的输出值，使用 Sigmoid 函数做为激活函数。
                self.neuro_layers[index_layer][index_number] = 1 / (1 + math.exp(-1 * input_sum))

    def __get_output(self):
        """
            获得输出层的值。
        :return:
        """
        return self.neuro_layers[len(self.neuro_layers)-1]

    @staticmethod
    def __calculate_difference(test_list, real_list):
        """
            误差计算。
            计算公式：误差值 = （测量值 - 真实值）^2
            为什么要用平方？
            误差统计中一般只需要使用误差的绝对值，而绝对值函数有跳变，不光滑。
            对绝对值进行平方就可以消除函数不光滑，方便后面各种积分微分等的运算。
        :param test_list:
            测试值。
        :param real_list:
            真实值。
        :return:
        """
        difference = 0
        for index in range(len(test_list)):
            test = test_list[index]
            real = real_list[index]
            difference += math.pow(test - real, 2)

    def __update_weight(self, properties, output_list):
        """
            权重修正。
            输出层→隐藏层：误差偏导数 = -(输出值-样本值) * 激活函数的导数
            隐藏层→隐藏层：误差偏导数 = (右层每个节点的误差偏导数加权求和) * 激活函数的导数
            误差偏导数全部计算好后，就可以更新权重了：
            输入层：权重增加 = 输入值 * 右层对应节点的误差偏导数 * 学习率
            隐藏层：权重增加 = 当前节点的 Sigmoid * 右层对应节点的误差偏导数 * 学习率
            偏移值的权重增加 = 右层对应节点的误差偏导数 * 学习率
            学习率是一个预先设置好的参数，用于控制每次更新的幅度
        :param properties:
        :param output_list:
        :return:
        """
        pass

    def Epoch(self, samples, output):
        """
            训练神经网络。
        :param samples:
        :param output:
        :return:
        """

        if self.completed_epoch:
            return

        e = 0
        for index in range(len(samples)):
            # 样本数据进行神经网络计算。
            self.__neuro_handler(samples[index])
            # 误差计算。
            e += self.__calculate_difference(self.__get_output(), output[index])
            # 权重修正。
            self.__update_weight(samples[index], output[index])

        e /= len(samples)

        self.completed_epoch = e < self.difference

network = BackPropagationNetwork(2, 10, 5)
print("----------------------------")
print(network.neuro_layers)
print(network.weight_layers)
print("----------------------------")
