class Argparse():
    def __init__(self):
        self.mode = 'char2'
        self.emb_dim = 300
        self.kernel_num = 500
        self.output_size = 50
        self.batch_size = 32
        self.kernel_sizes = [3, 4, 5]
        self.n_fold = 10
        self.dropout = 0.4
        self.epoch = 50
        self.lr = 1e-4
        self.checkpoint_path = 'Checkpoint/50_user_100_posts/'
        self.data_path = '../datasets/twitter/50_user_100_posts/'
