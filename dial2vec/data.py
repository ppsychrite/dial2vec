import codecs 
from transformers import AutoConfig, AutoTokenizer

""" 
    Tokenized Session Data passed to transformer. 
"""
class Session():
    def __init__(self, input_ids, input_mask, segment_ids, role_ids, label_id, turn_ids=None, position_ids=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.role_ids = role_ids
        self.turn_ids = turn_ids
        self.position_ids = position_ids
        self.label_id = label_id

        self.batch_size = len(self.input_ids)



""" 
    Given data stored in order\tsessions\tlabel format, parse it and return list of "Session"
        where order is 1s and 0s representing the order of the dialogue. 
    
    
"""
def get_sessions(file_path : str, tokenizer : AutoTokenizer, config : AutoConfig) -> list:
    max_seq_length = 512 #config.max_position_embeddings
    features = []

    f = codecs.open(file_path, "r", "utf")

    examples = [] 
    for line in f: 
        line = [s.strip() for s in line.split('\t') if s.strip()]
        role, session, label = line[0], line[1], line[2]
        examples.append((role, session, label))

    for example in examples: 
        samples = example[1].split("|")
        roles = [int(r) for r in example[0].split("|")] \
                if example[0].find("#") != -1 \
                else [int(r) for r in example[0]]
    
        sample_input_ids = []
        sample_segment_ids = []
        sample_role_ids = []
        sample_input_mask = []
        sample_turn_ids = []
        sample_position_ids = []

        for t, s in enumerate(samples):
            text_tokens = []
            text_turn_ids = []
            text_role_ids = []

            texts = s.split("#")

            # bert-token:     [cls]  token   [sep]  token
            # roberta-token:   <s>   token   </s>   </s> token
            text_tokens.append(tokenizer.cls_token)
            text_turn_ids.append(0)
            text_role_ids.append(roles[0])

            for i, text in enumerate(texts): 

                tokenized = tokenizer.tokenize(text)
                text_tokens.extend(tokenized)
                text_turn_ids.extend([i] * len(tokenized))
                text_role_ids.extend([roles[i]] * len(tokenized))

                if i != (len(text) - 1): 
                    text_tokens.append(tokenizer.sep_token)
                    text_turn_ids.append(i)
                    text_role_ids.append(roles[i])


            text_tokens = text_tokens[:max_seq_length]
            text_turn_ids = text_turn_ids[:max_seq_length]
            text_role_ids = text_role_ids[:max_seq_length]

            text_input_ids = tokenizer.convert_tokens_to_ids(text_tokens)

            # Pad to be the max sequence length for the transformer. 
            text_input_ids += [tokenizer.pad_token_id] * (max_seq_length - len(text_tokens))
            text_input_mask = [1] * len(text_tokens) + [0] * (max_seq_length - len(text_tokens))
            text_segment_ids = [0] * max_seq_length
            text_position_ids = list(range(len(text_tokens))) + [0] * (max_seq_length - len(text_tokens))
            text_turn_ids += [0] * (max_seq_length - len(text_tokens))
            text_role_ids += [0] * (max_seq_length - len(text_tokens))

            assert len(text_input_ids) == max_seq_length
            assert len(text_input_mask) == max_seq_length
            assert len(text_segment_ids) == max_seq_length
            assert len(text_position_ids) == max_seq_length
            assert len(text_turn_ids) == max_seq_length
            assert len(text_role_ids) == max_seq_length

            sample_input_ids.append(text_input_ids)
            sample_turn_ids.append(text_turn_ids)
            sample_role_ids.append(text_role_ids)
            sample_segment_ids.append(text_segment_ids)
            sample_position_ids.append(text_position_ids)
            sample_input_mask.append(text_input_mask) 


        n_neg = 9
        label_id = [1] + [0] * n_neg
        session = Session(input_ids=sample_input_ids,
                                    input_mask=sample_input_mask,
                                    segment_ids=sample_segment_ids,
                                    role_ids=sample_role_ids,
                                    turn_ids=sample_turn_ids,
                                    position_ids=sample_position_ids,
                                    label_id=label_id)
        features.append(session)
    f.close()
    return features