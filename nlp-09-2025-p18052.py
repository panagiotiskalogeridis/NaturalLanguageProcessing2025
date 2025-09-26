import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import spacy
import spacy.cli
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import torch


#Reproducibility seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

#spaCy for sentence splitting
nlp = spacy.load("en_core_web_sm")

#transformer models
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Loaded SentenceTransformer: all-MiniLM-L6-v2")

device_idx = 0 if torch.cuda.is_available() else -1

bart_pipeline = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=device_idx
)
print("Loaded BART")

tokenizerGrammar = AutoTokenizer.from_pretrained("vennify/t5-base-grammar-correction", use_fast=False)
modelGrammar = AutoModelForSeq2SeqLM.from_pretrained("vennify/t5-base-grammar-correction")
modelGrammar.to("cpu")
print("Loaded T5 grammar-correction")


#Parrot T5 paraphraser
parrotTokenizer = AutoTokenizer.from_pretrained("prithivida/parrot_paraphraser_on_T5")
parrotModel = AutoModelForSeq2SeqLM.from_pretrained("prithivida/parrot_paraphraser_on_T5")
parrotModel.to("cpu")

print("Loaded Parrot")

def parrot_paraphrase(text, max_length=256, num_beams=5):
    input_text = "paraphrase: " + text
    inputs = parrotTokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to("cpu") for k, v in inputs.items()}
    outputs = parrotModel.generate(**inputs, max_length=max_length, num_beams=num_beams, early_stopping=True)
    return parrotTokenizer.decode(outputs[0], skip_special_tokens=True)

text1 = """Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives. Hope you too, to enjoy it as my deepest wishes. Thank your message to show our words to the doctor, as his next contract checking, to all of us. I got this message to see the approved message. In fact, I have received the message from the professor, to show me, this, a couple of days ago. I am very appreciated the full support of the professor, for our Springer proceedings publication"""
text2 = """During our final discuss, I told him about the new submission — the one we were waiting since last autumn, but the updates was confusing as it not included the full feedback from reviewer or maybe editor? Anyway, I believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation. We should be grateful, I mean all of us, for the acceptance and efforts until the Springer link came finally last week, I think. Also, kindly remind me please, if the doctor still plan for the acknowledgments section edit before he sending again. Because I didn’t see that part final yet, or maybe I missed, I apologize if so. Overall, let us make sure all are safe and celebrate the outcome with strong coffee and future targets"""
original_texts = [text1, text2]


def grammarCorrection(text, max_length=256):
    prefix = "grammar: "
    inputs = tokenizerGrammar(prefix + text, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to("cpu") for k, v in inputs.items()}
    outputs = modelGrammar.generate(**inputs, max_length=max_length, num_beams=5, early_stopping=True)
    return tokenizerGrammar.decode(outputs[0], skip_special_tokens=True)

def bartFunction(text, min_length=15, max_length=60):
    out = bart_pipeline(text, max_length=max_length, min_length=min_length, do_sample=False)
    return out[0]['summary_text']

def getSentenceList(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def sbertSimilarityCalc(a, b):
    emb_a = semantic_model.encode(a, convert_to_tensor=True)
    emb_b = semantic_model.encode(b, convert_to_tensor=True)
    return util.cos_sim(emb_a, emb_b).item()

def perp(n_samples):
    if n_samples <= 6:
        return max(2, n_samples - 1)
    return min(30, max(5, (n_samples - 1) // 3))

#Reconstruction
def customReconstruction(text):
    sents = getSentenceList(text)
    first_two = " ".join(sents[:2])
    return grammarCorrection(first_two)

def reconstructTexts(text):
    sents = getSentenceList(text)
    bart_out = bartFunction(text, min_length=20, max_length=100)
    parrot_out = parrot_paraphrase(text)
    t5_corrected = grammarCorrection(text)

    return bart_out, parrot_out, t5_corrected


embeddings_for_viz = []
labels_for_viz = []

for idx, original in enumerate(original_texts, start=1):
    print("Original text:\n", original, "\n")

    aRecon = customReconstruction(original)
    orig2Sent = " ".join(getSentenceList(original)[:2])
    aSimilarity = sbertSimilarityCalc(orig2Sent, aRecon)
    print("A. first two sentences:")
    print("ORIGINAL:", orig2Sent)
    print("RECONSTRUCTED:", aRecon)
    print("SBERT similarity:", aSimilarity, "\n")

    bart_recon, parrot_recon, t5_recon = reconstructTexts(original)
    print("B. reconstructions:")
    print("BART:", bart_recon)
    print("Parrot:", parrot_recon)
    print("T5 corrected:", t5_recon)

    #similarities
    sim_bart = sbertSimilarityCalc(original, bart_recon)
    sim_parrot = sbertSimilarityCalc(original, parrot_recon)
    sim_t5 = sbertSimilarityCalc(original, t5_recon)

    print("\nSBERT similarities:")
    print("BART:", sim_bart)
    print("Parrot:", sim_parrot)
    print("T5:", sim_t5, "\n")

    # Embeddings for vizualization
    emb_orig = semantic_model.encode(original)
    emb_bart = semantic_model.encode(bart_recon)
    emb_parrot = semantic_model.encode(parrot_recon)
    emb_t5 = semantic_model.encode(t5_recon)

    embeddings_for_viz.extend([emb_orig, emb_bart, emb_parrot, emb_t5])
    labels_for_viz.extend([f"Original_{idx}", f"BART_{idx}", f"Parrot_{idx}", f"T5_{idx}"])


all_emb_np = np.vstack(embeddings_for_viz)
n_comp = min(2, all_emb_np.shape[1])
pca = PCA(n_components=n_comp, random_state=SEED)
emb_pca = pca.fit_transform(all_emb_np)

perp = perp(len(labels_for_viz))
tsne = TSNE(n_components=2, perplexity=perp, random_state=SEED, init='pca')
emb_tsne = tsne.fit_transform(emb_pca)

# plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(emb_pca[:, 0], emb_pca[:, 1], s=100)
for i, lab in enumerate(labels_for_viz):
    axes[0].annotate(lab, (emb_pca[i, 0], emb_pca[i, 1]), fontsize=9)
axes[0].set_title("PCA (2D)")

axes[1].scatter(emb_tsne[:, 0], emb_tsne[:, 1], s=100)
for i, lab in enumerate(labels_for_viz):
    axes[1].annotate(lab, (emb_tsne[i, 0], emb_tsne[i, 1]), fontsize=9)
axes[1].set_title(f"t-SNE (perplexity={perp})")

plt.tight_layout()
plt.show()
