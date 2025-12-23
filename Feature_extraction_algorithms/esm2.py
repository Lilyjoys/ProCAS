from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from transformers import T5EncoderModel, T5Tokenizer
from transformers import BertTokenizer, BertModel
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_sequence(sequences):
    return [' '.join(list(seq)) for seq in sequences]


def batchify(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

# ======== ESM-1b ========
def get_esm1b_features(sequences, model_path="LLM/esm1b", batch_size=8):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device).eval()
    sequences = preprocess_sequence(sequences)

    all_features = []
    for batch in batchify(sequences, batch_size):
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            features = (outputs.last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
            all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0)


# ======== ProtT5-XL========
def get_prott5_features(sequences, model_path="LLM/prot_t5", batch_size=4):
    tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_path).to(device).eval()
    sequences = [" ".join(list(seq.upper())) for seq in sequences]

    all_embeddings = []
    for batch in batchify(sequences, batch_size):
        tokens = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state

        for i in range(len(batch)):
            seq_len = attention_mask[i].sum()
            emb = embeddings[i, :seq_len].mean(dim=0)
            all_embeddings.append(emb.cpu().numpy())

    return np.stack(all_embeddings)


# ======== ProtBERT========
def get_protbert_features(sequences, model_path="LLM/ProtBert", batch_size=8):
    tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False)
    model = BertModel.from_pretrained(model_path).to(device).eval()
    sequences = [" ".join(list(seq.upper())) for seq in sequences]

    all_embeddings = []
    for batch in batchify(sequences, batch_size):
        tokens = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state

        mask = attention_mask.unsqueeze(-1)
        summed = torch.sum(embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        mean_embeddings = summed / counts
        all_embeddings.append(mean_embeddings.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


# ======== Mistral-Peptide ========
def get_mistral_peptide_features(sequences, model_path="LLM/mixtral", batch_size=8):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device).eval()

    all_embeddings = []
    for batch in batchify(sequences, batch_size):
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(device)

        with torch.no_grad():
            outputs = model(input_ids)
            hidden_states = outputs[0]

        pooled = torch.max(hidden_states, dim=1)[0]
        all_embeddings.append(pooled.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


# ======== ESM-2 ========
def esm2(sequences, batch_size=128):
    tokenizer = AutoTokenizer.from_pretrained('Feature_extraction_algorithms/esm2_t6_8M_UR50D')
    model = AutoModelForMaskedLM.from_pretrained('Feature_extraction_algorithms/esm2_t6_8M_UR50D').to(device).eval()

    results = []
    for batch in batchify(sequences, batch_size):
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            batch_embeddings = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()
            results.append(batch_embeddings)

    return np.concatenate(results, axis=0)


if __name__ == '__main__':
    sequences = ["TKQKEVITAQDTVIKAKYAEVAKHKEQNNDSQLKIKELDHNISKHKREA",
                 "APGNYLISVKYGGPNHIVGSPFKAKVTGQRLVSPGSANETSSILVESVT"]

    esm1b_feat = get_esm1b_features(sequences, batch_size=128)
    print("ESM-1b:", esm1b_feat.shape)
    print(esm1b_feat[0])

    prott5_feat = get_prott5_features(sequences, batch_size=128)
    print("prott5:", prott5_feat.shape)
    print(prott5_feat[0])

    bert_feat = get_protbert_features(sequences, batch_size=128)
    print("bert:", bert_feat.shape)
    print(bert_feat[0])

    mistral_feat = get_mistral_peptide_features(sequences, batch_size=128)
    # Mistral-Peptide-v1-423M
    print("mistral_feat:", mistral_feat.shape)

    esm2_feat = esm2(sequences, batch_size=128)
    print("esm2:", esm2_feat.shape)