import numpy as np

from lstm import LstmParam, LstmNetwork

class ToyLossLayer:
    """
    Computes square loss with first element of hidden layer array.
    """
    @classmethod
    def loss(self, pred, label):
        return (pred[0] - label) ** 2

    @classmethod
    def bottom_diff(self, pred, label):
        diff = np.zeros_like(pred)
        diff[0] = 2 * (pred[0] - label)
        return diff

def example_0():
    # learns to repeat simple sequence from random inputs
    np.random.seed(0)

    # parameters (mem_cell_ct and x_dim) for input data dimension and lstm cell count
    mem_cell_ct = 100

    #x_list (== input_val_arr) is an input sequence where each x is a real valued vector
    # x_dim give the dimension of each vector in the sector
    x_dim = 50

    concat_len = x_dim + mem_cell_ct
    lstm_param = LstmParam(mem_cell_ct, x_dim) 
    lstm_net = LstmNetwork(lstm_param)

    # y_list is the target sequence
    y_list = [-0.5,0.2,0.1, -0.5]

    #training set are examples where each example is (an input sequence, target_sequence) pair
    input_val_arr = [np.random.random(x_dim) for _ in y_list]
    print "input_val-arr : ", input_val_arr
    print "len(input_val-arr) : ", len(input_val_arr)
    print "len(input_val-arr[0]) : ", len(input_val_arr[0])

    print "\n"
    i =0
    for cur_iter in range(100):
        print "cur iter: ", cur_iter
        for ind in range(len(y_list)):
            print "i : ", i
            i +=1
            print "ind : ", ind, "  - len(input_val_arr[ind]) : ", len(input_val_arr[ind])
            lstm_net.x_list_add(input_val_arr[ind])
            print "y_pred[%d] : %f" % (ind, lstm_net.lstm_node_list[ind].state.h[0])
            #print "lstm_net.lstm_node_list[ind] : ", lstm_net.lstm_node_list[ind]
            print "len(lstm_net.lstm_node_list) : ", len(lstm_net.lstm_node_list)
            #print  "lstm_net.lstm_node_list[ind].state : ", lstm_net.lstm_node_list[ind].state
            print "len(lstm_net.lstm_node_list[ind].state.h) : ", len(lstm_net.lstm_node_list[ind].state.h)
            #print "lstm_net.lstm_node_list[ind].state.h : ", lstm_net.lstm_node_list[ind].state.h
            print "\n"

        #computation of the global loss that we want to minimize at each time step
        loss = lstm_net.y_list_is(y_list, ToyLossLayer)
        print "loss: ", loss
        print "len(lstm_net.lstm_node_list) : ", len(lstm_net.lstm_node_list)
        print "\n\n\n"
        #update of the parameters
        lstm_param.apply_diff(lr=0.1)

        lstm_net.x_list_clear()

if __name__ == "__main__":
    example_0()

