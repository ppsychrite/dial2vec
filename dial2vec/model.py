import torch.nn as nn 
import torch 

class PoolingAverage(nn.Module):
    def __init__(self, eps=1e-12):
        super(PoolingAverage, self).__init__()
        self.eps = eps

    def forward(self, hidden_states, attention_mask):
        mul_mask = lambda x, m: x * torch.unsqueeze(m, dim=-1)
        reduce_mean = lambda x, m: torch.sum(mul_mask(x, m), dim=1) / (torch.sum(m, dim=1, keepdims=True) + self.eps)

        avg_output = reduce_mean(hidden_states, attention_mask)
        return avg_output

    def equal_forward(self, hidden_states, attention_mask):
        mul_mask = hidden_states * attention_mask.unsqueeze(-1)
        avg_output = torch.sum(mul_mask, dim=1) / (torch.sum(attention_mask, dim=1, keepdim=True) + self.eps)
        return avg_output
    

class DialogueTransformer(nn.Module):
    def __init__(self, model, config, tokenizer, logger):
        super(DialogueTransformer, self).__init__()
        self.bert = model 
        self.config = config
        self.tokenizer = tokenizer

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.labels_data = None
        self.sample_nums = 10
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.avg = PoolingAverage(eps=1e-6)
        self.logger = logger  

    def forward(self, data, strategy='mean_by_role', output_attention=False):


        if len(data) == 7:
            input_ids, attention_mask, token_type_ids, role_ids, turn_ids, position_ids, labels = data
        else:
            input_ids, attention_mask, token_type_ids, role_ids, turn_ids, position_ids, labels, guids = data

        input_ids = input_ids.view(input_ids.size()[0] * input_ids.size()[1], input_ids.size()[-1])
        attention_mask = attention_mask.view(attention_mask.size()[0] * attention_mask.size()[1], attention_mask.size()[-1])
        token_type_ids = token_type_ids.view(token_type_ids.size()[0] * token_type_ids.size()[1], token_type_ids.size()[-1])
        role_ids = role_ids.view(role_ids.size()[0] * role_ids.size()[1], role_ids.size()[-1])
        turn_ids = turn_ids.view(turn_ids.size()[0] * turn_ids.size()[1], turn_ids.size()[-1])
        position_ids = position_ids.view(position_ids.size()[0] * position_ids.size()[1], position_ids.size()[-1])

        one_mask = torch.ones_like(role_ids)
        zero_mask = torch.zeros_like(role_ids)
        role_a_mask = torch.where(role_ids == 0, one_mask, zero_mask)
        role_b_mask = torch.where(role_ids == 1, one_mask, zero_mask)

        a_attention_mask = (attention_mask * role_a_mask)
        b_attention_mask = (attention_mask * role_b_mask)

        self_output, pooled_output = self.encoder(input_ids, attention_mask, token_type_ids, position_ids, turn_ids, role_ids)

        q_self_output = self_output * a_attention_mask.unsqueeze(-1)
        r_self_output = self_output * b_attention_mask.unsqueeze(-1)

        self_output = self_output * attention_mask.unsqueeze(-1)
        w = torch.matmul(q_self_output, r_self_output.transpose(-1, -2))

        if turn_ids is not None:
            view_turn_mask = turn_ids.unsqueeze(1).repeat(1, self.config.max_position_embeddings, 1)
            view_turn_mask_transpose = view_turn_mask.transpose(2, 1)
            view_range_mask = torch.where(abs(view_turn_mask_transpose - view_turn_mask) <= 1000,
                                          torch.ones_like(view_turn_mask),
                                          torch.zeros_like(view_turn_mask))
            filtered_w = w * view_range_mask

        q_cross_output = torch.matmul(filtered_w.permute(0, 2, 1), q_self_output)
        r_cross_output = torch.matmul(filtered_w, r_self_output)

        q_self_output = self.avg(q_self_output, a_attention_mask)
        q_cross_output = self.avg(q_cross_output, b_attention_mask)
        r_self_output = self.avg(r_self_output, b_attention_mask)
        r_cross_output = self.avg(r_cross_output, a_attention_mask)

        self_output = self.avg(self_output, attention_mask)
        q_self_output = q_self_output.view(-1, self.sample_nums, self.config.hidden_size)
        q_cross_output = q_cross_output.view(-1, self.sample_nums, self.config.hidden_size)
        r_self_output = r_self_output.view(-1, self.sample_nums, self.config.hidden_size)
        r_cross_output = r_cross_output.view(-1, self.sample_nums, self.config.hidden_size)

        self_output = self_output.view(-1, self.sample_nums, self.config.hidden_size)
        pooled_output = pooled_output.view(-1, self.sample_nums, self.config.hidden_size)

        output = self_output[:, 0, :]
        q_output = q_self_output[:, 0, :]
        r_output = r_self_output[:, 0, :]
        #q_contrastive_output = q_cross_output[:, 0, :]
        #r_contrastive_output = r_cross_output[:, 0, :]

        logit_q = []
        logit_r = []
        for i in range(self.sample_nums):
            cos_q = self.calc_cos(q_self_output[:, i, :], q_cross_output[:, i, :])
            cos_r = self.calc_cos(r_self_output[:, i, :], r_cross_output[:, i, :])
            logit_r.append(cos_r)
            logit_q.append(cos_q)

        logit_r = torch.stack(logit_r, dim=1)
        logit_q = torch.stack(logit_q, dim=1)

        loss_r = self.calc_loss(logit_r, labels)
        loss_q = self.calc_loss(logit_q, labels)

        if strategy not in ['mean', 'mean_by_role']:
            raise ValueError('Unknown strategy: [%s]' % strategy)

        output_dict = {'loss': loss_r + loss_q,
                       'final_feature': output if strategy == 'mean' else q_output + r_output,
                       'q_feature': q_output,
                       'r_feature': r_output,
                       'attention': w}

        return output_dict

    def encoder(self, *x):
        input_ids, attention_mask, token_type_ids, position_ids, _, _ = x    

        output = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            output_hidden_states=True,
                            return_dict=True)
        all_output = output['hidden_states']
        pooler_output = output['pooler_output']
        return all_output[-1], pooler_output

    def calc_cos(self, x, y):
        cos = torch.cosine_similarity(x, y, dim=1)
        cos = cos / 1.0 # cos = cos / 2.0
        return cos

    def calc_loss(self, pred, labels):
        loss = -torch.mean(self.log_softmax(pred) * labels)
        return loss

    def get_result(self):
        return self.result

    def get_labels_data(self):
        return self.labels_data
