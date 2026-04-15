# -*- coding: utf-8 -*-
"""
ECKT Model Implementation
This model is built upon the ECKT (Educational Code Knowledge Tracing) framework.

References:
- Code-DKT Implementation: https://github.com/YangAzure/Code-DKT/blob/main/src/c2vRNNModel.py
- Note: Modified the input layer to support BERT embeddings and problem difficulty ranks.
"""

import torch
import torch.nn as nn

class ECKT(nn.Module):
    def __init__(self,
                 model_type,
                 input_info,
                 input_dim, hidden_dim, rnn_num_layers, output_dim,
                 emb_size,
                 dropout,
                 max_code_len, 
                 num_problems,
                 problem_rank_dim, code_knowledge_dim,
                 att_dim_org,
                 has_w0,
                 logger):
        super(ECKT, self).__init__()
        
        self.model_type = model_type
        self.input_dim = input_dim
        self.max_code_len = max_code_len
        self.code_knowledge_dim = code_knowledge_dim
        self.logger = logger

        input_index_len_dict = input_info["input_index_len"]
        ast_node_vocab_size = input_info["code_ast_path_info"]["node_vocab_size"]
        ast_path_vocab_size = input_info["code_ast_path_info"]["path_vocab_size"]

        self.logger.info(f"ECKT.model_type:{model_type}. max_code_len:{max_code_len}")

        self.input_index_len_dict = input_index_len_dict
        score_idx_len = input_index_len_dict["score"]
        self.score_indices = score_idx_len[0], score_idx_len[0]+score_idx_len[1]

        rnn_input_sizes = [input_dim]

        rnn_class = nn.LSTM
        if self.model_type == "dkt":
            pass
        elif self.model_type in ["codedkt", "eckt"]:
            self.logger.info(f"att_dim_org:{att_dim_org}. has_w0:{has_w0}. dropout:{dropout}")

            self.embed_nodes = nn.Embedding(ast_node_vocab_size, emb_size)
            self.embed_paths = nn.Embedding(ast_path_vocab_size, emb_size)

            self.embed_dropout = nn.Dropout(dropout)

            code_embed_concat_size = emb_size*3 # start node, path node, end node

            rnn_input_sizes.append(code_embed_concat_size+input_dim)

            self.path_transformation_layer = nn.Linear(input_dim+code_embed_concat_size, input_dim+code_embed_concat_size)
            self.attention_layer = nn.Linear(input_dim+code_embed_concat_size, 1)

            if att_dim_org:
                self.attention_softmax = nn.Softmax(dim=1)  # code-dkt version
            else:
                self.attention_softmax = nn.Softmax(dim=2)  # modify version

            if has_w0:
                self.W_0 = nn.Linear(input_dim+code_embed_concat_size, input_dim+code_embed_concat_size)
            else:
                self.W_0 = None

            code_idx_len = input_index_len_dict["code"]
            difficulty_idx_len = input_index_len_dict["difficulty"]
            knowledge_idx_len = input_index_len_dict["knowledge"]

            self.code_indices = code_idx_len[0], code_idx_len[0]+code_idx_len[1]
            self.difficulty_indices = difficulty_idx_len[0], difficulty_idx_len[0]+difficulty_idx_len[1]
            self.knowledge_indices = knowledge_idx_len[0], knowledge_idx_len[0]+knowledge_idx_len[1]

            if self.model_type == "eckt":            
                rnn_class = nn.GRU
                # apply ECKT: difficulty embedding
                self.diff_embedding = nn.Embedding(num_problems, problem_rank_dim)

                rnn_input_sizes.append(problem_rank_dim)
                rnn_input_sizes.append(code_knowledge_dim)

        self.rnn_input_sizes = rnn_input_sizes
        self.rnn = rnn_class(sum(rnn_input_sizes),
                        hidden_dim,
                        num_layers=rnn_num_layers,
                        dropout = dropout if rnn_num_layers>1 else 0,
                        batch_first=True)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sig = nn.Sigmoid()

    def forward(self, x): # shape of input: [batch_size, length, questions*2 + c2vnodes + problem_rank(1) + code_knowledge_dim]
        rnn_first_part = x[:, :, self.score_indices[0]:self.score_indices[1]]

        if self.model_type == "dkt":
            rnn_input = rnn_first_part

        elif self.model_type in ["codedkt", "eckt"]:
            c2v_input = x[:, :, self.code_indices[0]:self.code_indices[1]].reshape(
                x.size(0), x.size(1), self.max_code_len, 3).long() # (b,l,c,3)

            rnn_attention_part = torch.stack([rnn_first_part]*self.max_code_len, dim=-2) # (b,l,c,2q)

            starting_node_index = c2v_input[:,:,:,0] # (b,l,c,1)
            ending_node_index = c2v_input[:,:,:,2] # (b,l,c,1)
            path_index = c2v_input[:,:,:,1] # (b,l,c,1)

            starting_node_embed = self.embed_nodes(starting_node_index) # (b,l,c,1) -> (b,l,c,ne)   : c=100, ne=50 or 100
            ending_node_embed = self.embed_nodes(ending_node_index) # (b,l,c,1) -> (b,l,c,ne)
            path_embed = self.embed_paths(path_index) # (b,l,c,1) -> (b,l,c,pe)

            full_embed = torch.cat((starting_node_embed, ending_node_embed, path_embed, rnn_attention_part), dim=3) # (b,l,c,2ne+pe+2q)
            full_embed = self.embed_dropout(full_embed) 

            full_embed_transformed = torch.tanh(self.path_transformation_layer(full_embed)) # (b,l,c,2ne+pe+2q)
            context_weights = self.attention_layer(full_embed_transformed) # (b,l,c,1)
            attention_weights = self.attention_softmax(context_weights) # (b,l,c,1)
            code_vectors = torch.sum(torch.mul(full_embed, attention_weights), dim=2) # (b,l,2ne+pe+2q)

            if self.W_0 is None:
                e_t = code_vectors
            else:
                e_t = self.W_0(code_vectors)

            if self.model_type == "codedkt": # Code-DKT
                rnn_input = torch.cat((rnn_first_part, e_t), dim=2)
            else:   # ECKT
                diff_rank = x[:, :, self.difficulty_indices[0]:self.difficulty_indices[1]]
                kn_emb = x[:, :, self.knowledge_indices[0]:self.knowledge_indices[1]]

                assert self.knowledge_indices[1] == x.shape[2]

                # difficulty embedding
                x_hat_t = self.diff_embedding(diff_rank.long().squeeze(-1))
                
                # knowledge embedding
                c_t = kn_emb

                rnn_input = torch.cat((rnn_first_part, e_t, x_hat_t, c_t), dim=2)

        out, hn = self.rnn(rnn_input)  # shape of out: [batch_size, length, hidden_size]
        res = self.sig(self.fc(out))  # shape of res: [batch_size, length, question]
        return res
