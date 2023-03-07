import networkx as nx
import tensorflow as tf
import numpy as np

from utils import hypergraph_to_input_data, network_to_hypergraph
from objects import TimeDist, SizeDist

# all with traffic
def get_traffic_matrix(matrix):
    result_matrix = []
    for row in matrix:
        result_row = []
        for element in row:
            if element:
                result_row.append({'AggInfo': {'AvgBw': 591.646, 'PktsGen': 0.593295, 'TotalPktsGen': 34247.953875},
                                   'Flows': [{'TimeDist': TimeDist(0),
                                              'TimeDistParams': {'EqLambda': 596.717, 'AvgPktsLambda': 0.596717, 'ExpMaxFactor': 10.0},
                                              'SizeDist': SizeDist(2),
                                              'SizeDistParams': {'AvgPktSize': 1000.0, 'PktSize1': 300.0, 'PktSize2': 1700.0},
                                              'AvgBw': 591.646,
                                              'PktsGen': 0.593295,
                                              'TotalPktsGen': 34247.953875,
                                              'ToS': 1.0}]})
            else:
                result_row.append({'AggInfo': {'AvgBw': 0.0, 'PktsGen': 0.0, 'TotalPktsGen': 0.0},'Flows': []})
        result_matrix.append(result_row)
    return np.matrix(result_matrix)

# all direct connections
def get_routing_matrix(matrix):
    result_matrix = []
    i = 0
    for row in matrix:
        j = 0
        result_row = []
        for element in row:
            if i == j:
                result_row.append([i])
            else:
                result_row.append([i, j])
            j += 1
        result_matrix.append(result_row)
        i += 1
    return np.array(result_matrix, dtype=object)

# its not important
def get_performance_matrix(matrix):
    result_matrix = []
    for i in range(0, len(matrix)):
        result_row = []
        for j in range(0, len(matrix)):
            if i == j:
                result_row.append({'AggInfo': {'PktsDrop': 0.0, 'AvgDelay': -1.0, 'AvgLnDelay': -1.0, 'p10': -1.0, 'p20': -1.0, 'p50': -1.0, 'p80': -1.0, 'p90': -1.0, 'Jitter': -1.0}, 'Flows': [{'PktsDrop': 0.0, 'AvgDelay': -1.0, 'AvgLnDelay': -1.0, 'p10': -1.0, 'p20': -1.0, 'p50': -1.0, 'p80': -1.0, 'p90': -1.0, 'Jitter': -1.0}]})
            else:
                result_row.append({'AggInfo': {'PktsDrop': 0.0, 'AvgDelay': 0.103026, 'AvgLnDelay': -2.61639, 'p10': 0.03, 'p20': 0.03, 'p50': 0.118485, 'p80': 0.169999, 'p90': 0.169999, 'Jitter': 0.00552},
                                    'Flows': [{'PktsDrop': 0.0, 'AvgDelay': 0.103026, 'AvgLnDelay': -2.61639, 'p10': 0.03, 'p20': 0.03, 'p50': 0.118485, 'p80': 0.169999, 'p90': 0.169999, 'Jitter': 0.00552}]})
        result_matrix.append(result_row)
    return np.matrix(result_matrix)

# all direct connections
def get_graph(matrix):
    g = nx.MultiDiGraph()
    for i in range(0, len(matrix)):
        g.add_node(i, queueSizes="32,32", schedulingPolicy="WFQ", levelsQoS=2, schedulingWeights="50,50")
    i = 0
    n = [0] * len(matrix)
    for row in matrix:
        j = 0
        for element in row:
            if element:
                g.add_edge(i, j, weight=1, key=0, port=n[i], bandwidth=10000)
                n[i] += 1
            j += 1
        i += 1
    return nx.DiGraph(g)

def generator(M):
    G = get_graph(M)
    T = get_traffic_matrix(M)
    R = get_routing_matrix(M)
    P = get_performance_matrix(M)
    HG = network_to_hypergraph(G=G, R=R, T=T, P=P)
    ret = hypergraph_to_input_data(HG)
    yield ret

def input_fn(M):
    ds = tf.data.Dataset.from_generator(generator,
                                        args=[M],
                                        output_signature=(
                                            {"traffic": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                                             "packets": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                                             "length": tf.TensorSpec(shape=None, dtype=tf.int32),
                                             "model": tf.TensorSpec(shape=None, dtype=tf.int32),
                                             "eq_lambda": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                                             "avg_pkts_lambda": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                                             "exp_max_factor": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                                             "pkts_lambda_on": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                                             "avg_t_off": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                                             "avg_t_on": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                                             "ar_a": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                                             "sigma": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                                             "capacity": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                                             "queue_size": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                                             "policy": tf.TensorSpec(shape=None, dtype=tf.int32),
                                             "priority": tf.TensorSpec(shape=None, dtype=tf.int32),
                                             "weight": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                                             "link_to_path": tf.RaggedTensorSpec(shape=(None, 1), dtype=tf.int32),
                                             "queue_to_path": tf.RaggedTensorSpec(shape=(None, 1), dtype=tf.int32),
                                             "queue_to_link": tf.RaggedTensorSpec(shape=(None, 1), dtype=tf.int32),
                                             "path_to_queue": tf.RaggedTensorSpec(shape=(None, None, 2), dtype=tf.int32,
                                                                                  ragged_rank=1),
                                             "path_to_link": tf.RaggedTensorSpec(shape=(None, None, 2), dtype=tf.int32,
                                                                                 ragged_rank=1)
                                             }
                                            , tf.TensorSpec(shape=None, dtype=tf.float32)
                                        ))

    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds
