class Argparse():
    def __init__(self):
        self.mode = 'char1'
        self.output_size = 50
        self.batch_size = 32
        self.n_fold = 10
        self.epoch = 50
        self.lr = 1e-3
        self.checkpoint_path = 'Checkpoint/50_user_100_posts/'
        self.data_path = '../datasets/twitter/50_user_100_posts/'
