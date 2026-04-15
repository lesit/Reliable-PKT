seq_len_space = [50, 75, 100]

class ModelConfig:
    def __init__(self):
        self.model_type = "code_dkt"

        self.att_dim_org = True

        self.batch_size = 16
        self.epochs = 100
        self.learning_rate = 0.00005

        self.max_seq_len = 50   # 50, 75, 100

        self.n_problems = 10

        self.max_code_len = 100

        self.dropout = 0.2
        
        self.emb_size = 50
        self.hidden = 128

        self.has_w0 = False
        
        self.rnn_num_layers = 1

        self.problem_rank_dim = 32
        self.code_knowledge_dim = 768

