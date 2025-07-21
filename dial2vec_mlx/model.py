import mlx 
import mlx.core as mx 
import mlx.nn as nn 
from mlx.nn.losses import cosine_similarity_loss


class PoolingAverage(nn.Module): 
    def __init__(self, eps = 1e12): 
        super().__init__()

        self.eps = eps 

    def __call__(self, 
                 hidden_states: mx.array, 
                 attention_mask: mx.array): 

        a = mx.sum(hidden_states * mx.expand_dims(attention_mask, axis = -1), axis = 1)
        b = mx.sum(attention_mask, axis = 1, keepdims = True)

        return a / b + self.eps

class DialogueTransformer(nn.Module):

    def __init__(self, bert):
        super().__init__()

        self.model = bert
        self.averager = PoolingAverage(eps = 1e-6)
        self.log_softmax = nn.LogSoftmax()

        self.hidden_size = self.model.config.hidden_size 
        self.sample_nums = 10 


    def __call__(self, 
                 input_ids, 
                 attention_mask, 
                 position_ids, 
                 role_ids,
                 labels, 
                 strategy = 'mean_by_role',
    ):


        # Reshape batches together 
        new_shape = (input_ids.shape[0] * input_ids.shape[1], input_ids.shape[-1])
        input_ids = mx.reshape(input_ids, new_shape)
        attention_mask = mx.reshape(attention_mask, new_shape)
        position_ids = mx.reshape(attention_mask, new_shape)
        role_ids = mx.reshape(role_ids, new_shape)

        one_mask = mx.ones_like(role_ids)
        zero_mask = mx.zeros_like(role_ids)

        role_a_mask = mx.where(role_ids == 0, one_mask, zero_mask)
        role_b_mask = mx.where(role_ids == 1, one_mask, zero_mask)

        a_attention_mask = (attention_mask * role_a_mask)
        b_attention_mask = (attention_mask * role_b_mask)


        self_output, pooled_output = self.encode(input_ids, 
                                                 attention_mask, 
                                                 position_ids)


        q_self_output = self_output * mx.expand_dims(a_attention_mask, axis = -1)
        r_self_output = self_output * mx.expand_dims(b_attention_mask, axis = -1)

        self_output = self_output * mx.expand_dims(attention_mask, axis = -1)


        w = mx.matmul(q_self_output, mx.transpose(r_self_output, axes = [0, -1, -2]))

        # Turn id check is deprecated in ModernBERT, skip it 

        q_cross_output = mx.matmul(
            mx.transpose(w, axes = [0, 2, 1]),
            q_self_output
        )

        r_cross_output = mx.matmul( 
            w, 
            r_self_output
        )

        q_self_output = self.averager(q_self_output, a_attention_mask)
        q_cross_output = self.averager(q_cross_output, b_attention_mask)
        r_self_output = self.averager(r_self_output, b_attention_mask)
        r_cross_output = self.averager(r_cross_output, a_attention_mask)



        self_output = self.averager(self_output, attention_mask)
        
        new_shape = (-1, self.sample_nums, self.hidden_size)
        
        q_self_output = q_self_output.reshape(new_shape)
        q_cross_output = q_cross_output.reshape(new_shape)
        r_self_output = r_self_output.reshape(new_shape)
        r_cross_output = r_cross_output.reshape(new_shape)


        self_output = self_output.reshape(new_shape)
        #pooled_output = pooled_output.reshape(new_shape)

        output = self_output[:, 0, :]
        q_output = q_self_output[:, 0, :]
        r_output = r_self_output[:, 0, :]

        logit_q = [] 
        logit_r = []  
        for i in range(self.sample_nums): 
            cos_q = self.calc_cos(q_self_output[:, i, :], q_cross_output[:, i, :])
            cos_r = self.calc_cos(r_self_output[:, i, :], r_cross_output[:, i, :])
            logit_r.append(cos_r)
            logit_q.append(cos_q)

        logit_r = mx.stack(logit_r, axis = 1)
        logit_q = mx.stack(logit_q, axis = 1)

        loss_r = self.calc_loss(logit_r, labels)
        loss_q = self.calc_loss(logit_q, labels)

        output_dict = {
            'final_feature': output if strategy == 'mean' else q_output + r_output,
            'q_feature' : q_output, 
            'r_feature' : r_output,
            'attention' : w,
            'loss' : loss_r + loss_q
        }

        return output_dict

    def encode(self, *x):
        input_ids, attention_mask, position_ids = x

        output = self.model(input_ids = input_ids,
                            attention_mask = attention_mask, 
                            position_ids = position_ids,
                            return_dict = False,
                            output_hidden_states = True)



        return output.hidden_states[0][-1], output.pooler_output 


    def calc_cos(self, x, y) -> float:
        cos = cosine_similarity_loss(x, y)
        cos = cos / 1.0 
        return cos 
    
    def calc_loss(self, pred, labels):
        loss = -mx.mean(self.log_softmax(pred) * labels)
        return loss

