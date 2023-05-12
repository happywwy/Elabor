import argparse
import json
from tqdm import tqdm
import numpy as np
import os
import torch
import transformers
import pdb
import random
from torch import cuda
import torch.nn.functional as F
device = 'cuda' if cuda.is_available() else 'cpu'
ANSWERS = ['yes', 'no']

def prompt_format(prompt_path: str, query: str):
    with open(prompt_path) as f:
        context_string = f.read().strip('\n')
    if query is not None:
        context_string = context_string.replace('{question}', query)
    return context_string


def checker(args, answer, pred):
    if args.task == 'numersense':
        if answer == pred:
            return 1
        if answer in ['no', 'zero'] and pred in ['no', 'zero']:
            return 1
        return 0
    return 1 if answer == pred else 0


def process_item(args, data, tokenizer_inf, model_inf):
    cand_index = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)']
    with torch.no_grad():
        new_data = []
        for item in data:
            query = item['query']
            
            if 'cands' in item:
                cands = item['cands']
            else:
                cands = ANSWERS
            answer = item['answer']
            cands_text = ' '.join([c + ' ' + t for (c,t) in zip(cand_index[:len(cands)], cands)])
            answer_id = cand_index[cands.index(item["answer"])]

            knowledges = item['knowledges'] if 'knowledges' in item else []
            answer_ind = cands.index(answer)
            
            # knowledges = random.sample(knowledges, min(10, len(knowledges)))
            # input_inf = [k + ' ' + query.replace('<mask>', '<extra_id_0>') for k in knowledges]
            # input_inf = tokenizer_inf.batch_encode_plus(
            #                     input_inf,
            #                     max_length=model_params['MAX_SOURCE_INF_LENGTH'],
            #                     pad_to_max_length=True,
            #                     truncation=True,
            #                     padding="max_length",
            #                     return_tensors="pt",
            #                 )
            
            # score_cands = torch.Tensor().to(device)
            # for cand in cands:
            #     label = tokenizer_inf('<extra_id_0> %s <extra_id_1>' % cand, return_tensors='pt').input_ids.to(device)
            #     logits = model_inf(input_ids=input_inf["input_ids"].to(device), 
            #                     attention_mask=input_inf["attention_mask"].to(device), 
            #                     labels=label.repeat(input_inf["input_ids"].size(0), 1)).logits
            #     score_inf = []
            #     for l in range(1, label.size(-1)-1):
            #         score_inf.append(logits[:, l, label[0,l]]) # NK
            #     score_inf = torch.mean(torch.stack(score_inf), dim=0) # NK
            #     score_cands = torch.cat([score_cands, score_inf.unsqueeze(1)], dim=1)
            # score_cands = torch.softmax(score_cands, dim=1) # Nk x Nc
            # top_inds = torch.argmax(score_cands, dim=1) # Nk
            # topk_inds = (top_inds == answer_ind).nonzero().flatten()

            # score_pos = score_cands[:, answer_ind]
            # if answer_ind < score_cands.size(1) - 1:
            #     score_neg = torch.mean(torch.cat([score_cands[:, :answer_ind], score_cands[:, answer_ind+1:]], dim=1), dim=1)
            # else:
            #     score_neg = torch.mean(score_cands[:, :answer_ind], dim=1)
            # score_final = score_pos - score_neg
            # topk_scores, topk_inds = torch.topk(score_final, k=min(args.topk, score_final.size(0)))

            # score_kg, score_inf = [], []
            # source_kg = tokenizer_kg(query, return_tensors='pt').input_ids.to(device)
            # for k in knowledges:
            #     input_kg = tokenizer_kg(query + ' ' + k, return_tensors='pt').input_ids.to(device)
            #     label_kg = input_kg.clone().detach()
            #     label_kg[0, :len(source_kg)] = -100
            #     loss_kg = model_kg(input_ids=input_kg.to(device), labels=label_kg).loss.item()
            #     # logits = model_kg(input_ids=input_kg.to(device), labels=label_kg).logits
            #     # score = []
            #     # for l in range(source_kg.size(-1), label_kg.size(-1)):
            #     #     score.append(logits[0, l, label_kg[0, l]])
            #     # score = torch.mean(torch.tensor(score).to(device), dim=0).item()
            #     score_kg.append(-loss_kg)

            #     input_inf = tokenizer_inf(k + ' ' + query.replace('<mask>', '<extra_id_0>'), return_tensors='pt').input_ids.to(device)
            #     label_inf = tokenizer_inf('<extra_id_0> %s <extra_id_1>' % answer, return_tensors='pt').input_ids.to(device)
            #     loss_inf = model_inf(input_ids=input_inf, labels=label_inf).loss.item()
            #     score_inf.append(-loss_inf)

            # score_kg = torch.softmax(torch.tensor(score_kg).to(device), dim=0)
            # score_inf = torch.softmax(torch.tensor(score_inf).to(device), dim=0)

            # score_all = score_inf * score_kg

            # topk_scores, topk_inds = torch.topk(score_inf, k=min(args.topk, score_inf.size(0)))
            # topk_knowledges = [knowledges[k] for k in topk_inds]
            # item["topk_knowledges"] = topk_knowledges
            # new_data.append(item)

            scores_all, topk_inds = [], []
            for knowledge in knowledges:
                # input_ids = tokenizer_inf(knowledge + ' ' + query.replace('<mask>', '<extra_id_0>'), return_tensors='pt').input_ids.cuda()
                # label = tokenizer_inf('<extra_id_0> %s <extra_id_1>' % answer, return_tensors='pt').input_ids.cuda()
                input_ids = tokenizer_inf(query + ' \\n ' + cands_text + ' \\n ' + knowledge, return_tensors='pt').input_ids.cuda()
                label = tokenizer_inf(answer_id, return_tensors='pt').input_ids.cuda()
                loss = model_inf(input_ids=input_ids, labels=label).loss.item()
                scores_all.append(-loss)

            scores_all = torch.softmax(torch.tensor(scores_all).to(device), dim=0)
            topk_scores, topk_inds = torch.topk(scores_all, k=min(args.topk, scores_all.size(0)))
            # topk_inds = (scores_all >= 0.1).nonzero().flatten().tolist()

            topk_knowledges = [knowledges[k] for k in topk_inds]
            item["topk_knowledges"] = topk_knowledges
            new_data.append(item)
    return new_data



def data_batch(args, model_params, data, tokenizer_kg, tokenizer_inf, model_inf):
    source_inf, source_kg, target_inf, target_kg, input_kg = [], [], [], [], []
    for item in data:
        topk_scores, topk_knowledge_ind = process_item(args, item, tokenizer_inf, model_inf)
        knowledges, query = item["knowledges"], item["query"]
        topk_knowledges = [knowledges[k] for k in topk_knowledge_ind]
        source_inf.extend([k + ' ' + query.replace('<mask>', '<extra_id_0>') for k in topk_knowledges])
        target_inf.extend(['<extra_id_0> %s <extra_id_1>' % item['answer'] for k in topk_knowledges])
        source_kg.extend([query for k in knowledges])
        target_kg.extend([k for k in knowledges])
        input_kg.extend([query + ' ' + k for k in knowledges])

    input_kg = tokenizer_kg.batch_encode_plus(
        input_kg,
        max_length=model_params['MAX_INPUT_KG_LENGTH'],
        pad_to_max_length=True,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    source_kg = tokenizer_kg.batch_encode_plus(
        source_kg,
        max_length=model_params['MAX_SOURCE_KG_LENGTH'],
        pad_to_max_length=True,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    target_kg = tokenizer_kg.batch_encode_plus(
            target_kg,
            max_length=model_params['MAX_TARGET_KG_LENGTH'],
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

    source_inf = tokenizer_inf.batch_encode_plus(
        source_inf,
        max_length=model_params['MAX_SOURCE_INF_LENGTH'],
        pad_to_max_length=True,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    target_inf = tokenizer_inf.batch_encode_plus(
            target_inf,
            max_length=model_params['MAX_TARGET_INF_LENGTH'],
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

    return {
            "input_kg_ids": input_kg["input_ids"].to(dtype=torch.long),
            "input_kg_mask": input_kg["attention_mask"].to(dtype=torch.long),
            "source_kg_ids": source_kg["input_ids"].to(dtype=torch.long),
            "source_kg_mask": source_kg["attention_mask"].to(dtype=torch.long),
            "target_kg_ids": target_kg["input_ids"].to(dtype=torch.long),
            "target_kg_mask": target_kg["attention_mask"].to(dtype=torch.long),
            "source_inf_ids": source_inf["input_ids"].to(dtype=torch.long),
            "source_inf_mask": source_inf["attention_mask"].to(dtype=torch.long),
            "target_inf_ids": target_inf["input_ids"].to(dtype=torch.long)
        }



def data_batch_kg(args, model_params, data, tokenizer_kg):
    source_inf, source_kg, target_inf, target_kg, input_kg = [], [], [], [], []
    for item in data:
        # topk_scores, topk_knowledge_ind = process_item(args, item, tokenizer_inf, model_inf)
        knowledges, query = item["topk_knowledges"], item["query"]
        # topk_knowledges = [knowledges[k] for k in topk_knowledge_ind]
        # source_inf.extend([k + ' ' + query.replace('<mask>', '<extra_id_0>') for k in topk_knowledges])
        # target_inf.extend(['<extra_id_0> %s <extra_id_1>' % item['answer'] for k in topk_knowledges])
        source_kg.extend([query for k in knowledges])
        target_kg.extend([k for k in knowledges])
        input_kg.extend([query + ' ' + k for k in knowledges])

        # source_inf.append(knowledges + ' ' + query)
        # target_inf.append('<extra_id_0> %s <extra_id_1>' % item['answer'])
        # source_kg.append(query)
        # target_kg.append(knowledges)
        # input_kg.append(query + ' ' + knowledges)

    
    if input_kg != []:
        input_kg = tokenizer_kg.batch_encode_plus(
            input_kg,
            max_length=model_params['MAX_INPUT_KG_LENGTH'],
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_kg = tokenizer_kg.batch_encode_plus(
            source_kg,
            max_length=model_params['MAX_SOURCE_KG_LENGTH'],
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        target_kg = tokenizer_kg.batch_encode_plus(
                target_kg,
                max_length=model_params['MAX_TARGET_KG_LENGTH'],
                pad_to_max_length=True,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )


        return {
                "input_kg_ids": input_kg["input_ids"].to(dtype=torch.long),
                "input_kg_mask": input_kg["attention_mask"].to(dtype=torch.long),
                "source_kg_ids": source_kg["input_ids"].to(dtype=torch.long),
                "source_kg_mask": source_kg["attention_mask"].to(dtype=torch.long),
                "target_kg_ids": target_kg["input_ids"].to(dtype=torch.long),
                "target_kg_mask": target_kg["attention_mask"].to(dtype=torch.long)
            }
    
    else:
        return None


def data_batch_inf(args, model_params, data, tokenizer_inf):
    source_inf, target_inf = [], []
    cand_index = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)']    
    
    for item in data:
        if 'cands' in item:
            cands = item['cands']
        else:
            cands = ANSWERS
        cands_text = ' '.join([c + ' ' + t for (c,t) in zip(cand_index[:len(cands)], cands)])
        answer_id = cand_index[cands.index(item["answer"])]
        knowledges, query = item["gen_knowledges"], item["query"]
        source_inf.extend([query + ' \\n ' + cands_text + ' \\n ' + k for k in knowledges])
        target_inf.extend([answer_id for k in knowledges])


    source_inf = tokenizer_inf.batch_encode_plus(
        source_inf,
        max_length=model_params['MAX_SOURCE_INF_LENGTH'],
        pad_to_max_length=True,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    target_inf = tokenizer_inf.batch_encode_plus(
            target_inf,
            max_length=model_params['MAX_TARGET_INF_LENGTH'],
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

    return {
            "source_inf_ids": source_inf["input_ids"].to(dtype=torch.long),
            "source_inf_mask": source_inf["attention_mask"].to(dtype=torch.long),
            "target_inf_ids": target_inf["input_ids"].to(dtype=torch.long)
        }


def train_distill(tokenizer, model, data, optimizer):

    model.train()
    ids = data["input_kg_ids"].to(device, dtype=torch.long)
    ids[data["input_kg_ids"] == tokenizer.pad_token_id] = 0
    mask = data["input_kg_mask"].to(device, dtype=torch.long)
    source_mask = data["source_kg_mask"].to(device, dtype=torch.long)
    lm_labels = data["input_kg_ids"].clone().detach()
    lm_labels[data["input_kg_ids"] == tokenizer.pad_token_id] = -100
    for i, m in zip(range(lm_labels.size(0)), source_mask):
        lm_labels[i, :m.sum()] = torch.tensor([-100 for j in range(m.sum())]).to(device)
    lm_labels = lm_labels.to(device, dtype=torch.long)
    outputs = model(input_ids=ids,
        attention_mask=mask,
        labels=lm_labels
    )

    loss = outputs.loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def train_inf(tokenizer, model, data, optimizer):

    model.train()
    y = data["target_inf_ids"].to(device, dtype=torch.long)
    ids = data["source_inf_ids"].to(device, dtype=torch.long)
    ids[data["source_inf_ids"] == tokenizer.pad_token_id] = 0
    mask = data["source_inf_mask"].to(device, dtype=torch.long)
    y[data["target_inf_ids"] == tokenizer.pad_token_id] = -100

    outputs = model(input_ids=ids,
        attention_mask=mask,
        labels=y
    )
    loss = outputs.loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def _score_cands(tokenizer, model, source, cands):
    with torch.no_grad():
        input_ids = tokenizer(source, return_tensors='pt').input_ids.cuda()
        scores = []
        for i in range(len(cands)):
            label = tokenizer(cands[i], return_tensors='pt').input_ids.cuda()
            loss = model(input_ids=input_ids, labels=label).loss.item()
            scores.append(-loss)
        probs = F.softmax(torch.tensor(scores), dim=0)
        return probs


def scores_for_query(tokenizer, model, query, knowledges, cands):
    n = len(knowledges)
    scores_ = []
    # For UnifiedQA, use non-capitalized ids such as (a), (b), ...
    cand_index = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)']
    cands_text = ' '.join([c + ' ' + t for (c,t) in zip(cand_index[:len(cands)], cands)])
    if n == 0:
        source = query + ' \\n ' + cands_text
        scores = _score_cands(tokenizer, model, source, cand_index[:len(cands)])
        scores_.append(scores)
    for i in range(0, n, 1):
        source = query + ' \\n ' + cands_text + ' \\n ' + knowledges[i]
        scores = _score_cands(tokenizer, model, source, cand_index[:len(cands)])
        scores_.append(scores)

    return torch.stack(scores_)


def generate_knowledge(tokenizer_kg, model_kg, data, gen_num):
    pbar = tqdm(data, total=len(data))
    model_kg.eval()
    new_data = []

    with torch.no_grad():
        for item in pbar:
            query = item['query']
            answer = item['answer']
            input_ids = tokenizer_kg.encode(query, return_tensors="pt").to(device)
            generated_ids = model_kg.generate(
                    input_ids = input_ids,
                    do_sample=True, 
                    max_length=50+input_ids.size(-1),
                    temperature=0.7,
                    top_k=50, 
                    top_p=0.95,
                    num_return_sequences=gen_num,
                    repetition_penalty=1.2, 
                    early_stopping=True
                    )

            knowledges = [tokenizer_kg.decode(g[input_ids.size(-1):], skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            item["gen_knowledges"] = ['.'.join(k.split('.')[:-1]) + '.' for k in knowledges]            
            new_data.append(item)
    
    return new_data


def test(tokenizer_inf, model_inf, test_data):
    num, den = 0, 0
    pbar = tqdm(test_data, total=len(test_data))
    model_inf.eval()
    new_data = []
    cand_index = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)']
    
    for item in pbar:
        if 'cands' in item:
            cands = item['cands']
        else:
            # CSQA2 does not have "cands"
            cands = ANSWERS
        query = item['query']
        cands_text = ' '.join([c + ' ' + t for (c,t) in zip(cand_index, cands)])
        knowledges = item["gen_knowledges"]
        scores = scores_for_query(tokenizer_inf, model_inf, query, knowledges, cands)
        scores, max_ind = torch.max(scores, dim=0)

        p = scores.argmax().item()
        max_knowledge = max_ind[p].item()
        pred = cands[p]
        item['pred'] = pred
        item['gen_knowledges'] = knowledges

        if 'answer' in item:
            answer = item['answer']
            ok = cmp(answer, pred)
            item['ok'] = ok
        if 'ok' in item:
            num += item['ok']
            den += 1
            pbar.set_postfix({'acc': num / den})
        new_data.append(item)

    return num / den, new_data
        

def cmp(answer, pred):
    if answer == pred:
        return 1
    if answer in ['no', 'zero'] and pred in ['no', 'zero']:
        return 1
    return 0

class CosineWithRestarts(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing with restarts.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer

    T_max : int
        The maximum number of iterations within the first cycle.

    eta_min : float, optional (default: 0)
        The minimum learning rate.

    last_epoch : int, optional (default: -1)
        The index of the last epoch.

    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 T_max: int,
                 eta_min: float = 0.,
                 last_epoch: int = -1,
                 factor: float = 1.) -> None:
        # pylint: disable=invalid-name
        self.T_max = T_max
        self.eta_min = eta_min
        self.factor = factor
        self._last_restart: int = 0
        self._cycle_counter: int = 0
        self._cycle_factor: float = 1.
        self._updated_cycle_len: int = T_max
        self._initialized: bool = False
        super(CosineWithRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Get updated learning rate."""
        # HACK: We need to check if this is the first time get_lr() was called, since
        # we want to start with step = 0, but _LRScheduler calls get_lr with
        # last_epoch + 1 when initialized.
        if not self._initialized:
            self._initialized = True
            return self.base_lrs

        step = self.last_epoch + 1
        self._cycle_counter = step - self._last_restart

        lrs = [
            (
                self.eta_min + ((lr - self.eta_min) / 2) *
                (
                    np.cos(
                        np.pi *
                        ((self._cycle_counter) % self._updated_cycle_len) /
                        self._updated_cycle_len
                    ) + 1
                )
            ) for lr in self.base_lrs
        ]

        if self._cycle_counter % self._updated_cycle_len == 0:
            # Adjust the cycle length.
            self._cycle_factor *= self.factor
            self._cycle_counter = 0
            self._updated_cycle_len = int(self._cycle_factor * self.T_max)
            self._last_restart = step

        return lrs


def Trainer(ds, args, tokenizer_kg, tokenizer_inf, model_kg, model_inf, test_data, model_params):

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(model_params["SEED"])  # pytorch random seed
    np.random.seed(model_params["SEED"])  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer_kg = torch.optim.Adam(params=model_kg.parameters(), lr=model_params["LEARNING_RATE_KG"])
    optimizer_inf = torch.optim.Adam(params=model_inf.parameters(), lr=model_params["LEARNING_RATE_INF"])
    scheduler_kg = CosineWithRestarts(optimizer_kg, T_max=5, factor=2, eta_min=1e-5)
    scheduler_inf = CosineWithRestarts(optimizer_inf, T_max=5, factor=2, eta_min=1e-5)

    # logging
    print("[Data]: Reading data...\n")
    train_data = [ds[i:i + model_params["TRAIN_ALTERNATE_SIZE"]] for i in range(0, len(ds), model_params["TRAIN_ALTERNATE_SIZE"])]
    # Training loop
    print("[Initiating Fine Tuning]...\n")
    with open(args.result_path, "w") as f:       
        
        best_acc = 0.0
        for epoch in range(model_params["TRAIN_EPOCHS"]): 
            new_data = [] # for recording knowledge filtering
            loss_all, loss_all_kg, loss_all_inf = 0.0, 0.0, 0.0
            # batched data for alternative training
            for it, iter_data in enumerate(train_data):
                model_inf.eval()
                iter_data_kg = process_item(args, iter_data, tokenizer_inf, model_inf)
                new_data.extend(iter_data_kg)
                # use answer model to filter knowledge and distill knowledge to the generator
                batch_data_kg = [iter_data_kg[i:i + model_params["TRAIN_BATCH_SIZE"]] for i in range(0, len(iter_data_kg), model_params["TRAIN_BATCH_SIZE"])]
                loss_iter_kg = 0               
                for batch_ind, batch in enumerate(batch_data_kg):
                    training_set = data_batch_kg(args, model_params, batch, tokenizer_kg)
                    if training_set:
                        loss_kg = train_distill(tokenizer_kg, model_kg, training_set, optimizer_kg)
                        loss_iter_kg += loss_kg
                print("Epoch "+str(epoch)+" Iter "+str(it)+": Distillation Loss "+str(loss_iter_kg))
                loss_all += loss_iter_kg
                loss_all_kg += loss_iter_kg

                # use the distilled model to generate knowledge and learn answer model
                model_kg.eval()
                iter_data_inf = generate_knowledge(tokenizer_kg, model_kg, iter_data, args.num_knowledge) 
                batch_data_inf = [iter_data_inf[i:i + model_params["TRAIN_BATCH_SIZE"]] for i in range(0, len(iter_data_inf), model_params["TRAIN_BATCH_SIZE"])]
                loss_iter_inf = 0
                for batch_ind, batch in enumerate(batch_data_inf):
                    training_set = data_batch_inf(args, model_params, batch, tokenizer_inf)
                    loss_inf = train_inf(tokenizer_inf, model_inf, training_set, optimizer_inf)
                    loss_iter_inf += loss_inf
                print("Epoch "+str(epoch)+" Iter "+str(it)+": Inference Loss "+str(loss_iter_inf))
                loss_all += loss_iter_inf
                loss_all_inf += loss_iter_inf

            scheduler_kg.step()
            scheduler_inf.step()

            # validate
            model_kg.eval()
            model_inf.eval()
            test_data = generate_knowledge(tokenizer_kg, model_kg, test_data, args.num_knowledge) 
            acc, new_test_data = test(tokenizer_inf, model_inf, test_data)
            f.write("Epoch " + str(epoch) + " Accuracy: " + str(acc) + "\n")
            print("Epoch " + str(epoch) + " Accuracy: " + str(acc))
            print("Epoch " + str(epoch) + " Loss: " + str(loss_all))
            print("Epoch " + str(epoch) + " KG Loss: " + str(loss_all_kg) + " INF Loss: " + str(loss_all_inf))
            f.write("Epoch " + str(epoch) + " Loss: " + str(loss_all) + "\n")
            f.write("Epoch " + str(epoch) + " KG Loss: " + str(loss_all_kg) + " INF Loss: " + str(loss_all_inf) + "\n")

            if acc > best_acc:
                best_acc = acc
                torch.save(model_kg, f'model/{args.task}/kg_{args.model_type_kg.split("/")[-1]}.pth')
                torch.save(model_inf, f'model/{args.task}/inf_{args.model_type_inf.split("/")[-1]}.pth')
                with open(args.output_path, "w") as fw:
                    json.dump(new_test_data, fw)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='obqa', choices=['obqa', 'csqa', 'csqa2', 'qasc'])
    parser.add_argument('--model-type-kg', type=str, default='gpt2-large')
    parser.add_argument('--model-type-inf', type=str, default='t5-large')
    parser.add_argument('--train-path', type=str, default='data/obqa/knowledge_gpt3.train.obqa.json')
    parser.add_argument('--test-path', type=str, default='data/obqa/dev.obqa.json')
    parser.add_argument('--aggfunc', type=str, default='best_prob', choices=['best_score', 'best_prob'])
    parser.add_argument('--num-knowledge', type=int, default=20)
    parser.add_argument('--topk', type=int, default=3)
    args = parser.parse_args()
    args.output_path = f'data/{args.task}/inference/inference_{args.model_type_inf.split("/")[-1]}.{args.train_path.split("/")[-1]}'
    args.result_path = f'result/{args.task}/result_{args.model_type_inf.split("/")[-1]}.txt'

    model_params = {
    "TRAIN_BATCH_SIZE": 5,  # batch size within each alternative training loop
    "TRAIN_ALTERNATE_SIZE": 100, # training alternation size
    "TRAIN_EPOCHS": 10,  # number of training epochs
    "LEARNING_RATE_KG": 1e-5,  # learning rate
    "LEARNING_RATE_INF": 1e-5,  # learning rate
    "MAX_INPUT_KG_LENGTH": 150,  # max length of all input text
    "MAX_SOURCE_KG_LENGTH": 80,  # max length of input question
    "MAX_TARGET_KG_LENGTH": 50,  # max length of target knowledge
    "MAX_SOURCE_INF_LENGTH": 150,  # max length of all input text
    "MAX_TARGET_INF_LENGTH": 10,  # max length of output answer text
    "SEED": 42,  # set seed for reproducibility
    }   

    # load training data containing generated knowledge from GPT-3
    with open(args.train_path) as f:
        train_data = json.load(f)

    with open(args.test_path) as f:
        test_data = json.load(f)

    # initialize two LMs, one for knowledge generation (kg), one for answer inference (inf)
    tokenizer_inf = transformers.T5Tokenizer.from_pretrained(args.model_type_inf)
    tokenizer_kg = transformers.GPT2Tokenizer.from_pretrained(args.model_type_kg)
    if tokenizer_kg.pad_token is None:
        tokenizer_kg.add_special_tokens({'pad_token': '[PAD]'})
    model_kg = transformers.GPT2LMHeadModel.from_pretrained(args.model_type_kg, pad_token_id=tokenizer_kg.eos_token_id)
    model_kg = model_kg.to(device)
    model_inf = transformers.T5ForConditionalGeneration.from_pretrained(args.model_type_inf)
    model_inf = model_inf.to(device)


    # training and inference
    Trainer(train_data, args, tokenizer_kg, tokenizer_inf, model_kg, model_inf, test_data, model_params=model_params) 




if __name__ == '__main__':
    main()

