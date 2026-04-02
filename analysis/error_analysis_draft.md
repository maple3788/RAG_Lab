# QASPER error analysis draft

*Auto-generated from `analysis/qasper_traces.jsonl` for review (Tech Lead). Regenerate after exporting fresh traces.*

*Note: No row had `failure_bucket: none` in this file — showing the **best-scoring** rows by (EM, gold_hit, F1). Re-export with a real LLM (`ollama` / `gemini`) for textbook “success” cases.*

## Success cases (3)

#### Success 1

- **ID:** `qasper:b5e4866f0685299f1d7af267bbcc4afe2aab806f`
- **Question:** what is the source of the news sentences?
- **Retrieval stage:** `gold_in_final` · **Failure bucket:** `generation`
- **Gold aliases (sample):** ilur.am; In order to evaluate the models trained on generated data, we manually annotated a named entities dataset comprising 534; In order to evaluate the models trained on generated data, we manually annotated a named entities dataset comprising 534; links between Wikipedia articles to generate sequences of named-entity annotated tokens; This approach uses links between Wikipedia articles to generate sequences of named-entity annotated tokens. … (+1 more)
- **Prediction:** [1] 53453 tokens and 2566 sentences selected from over 250 news texts from ilur.am. This dataset is comparable in size with the test sets of other languages (Table TABREF10 ). Included sentences are f
- **Metrics:** EM=0.0 · token_f1=0.5 · gold_hit=1.0
- **First final chunk preview:** 53453 tokens and 2566 sentences selected from over 250 news texts from ilur.am. This dataset is comparable in size with the test sets of other languages (Table TABREF10 ). Included sentences are from political, sports, local and world news (Figures FIGREF8 , FIGREF9 ), covering the period between August 2012 and July 2018. The dataset provides annotations for 3 popular named entity classes: people…

#### Success 2

- **ID:** `qasper:9a05a5f4351db75da371f7ac12eb0b03607c4b87`
- **Question:** which datasets did they experiment with?
- **Retrieval stage:** `gold_in_final` · **Failure bucket:** `generation`
- **Gold aliases (sample):** Europarl; MultiUN; We evaluate our cross-lingual pre-training based transfer approach against several strong baselines on two public datats; We evaluate our cross-lingual pre-training based transfer approach against several strong baselines on two public datats; Europarl BIBREF31 … (+1 more)
- **Prediction:** [1] we use BLEU as the automatic metric for translation evaluation. The statistics of Europarl and MultiUN corpora are summarized in Table TABREF18. For Europarl corpus, we evaluate on French-English-
- **Metrics:** EM=0.0 · token_f1=0.4556962025316456 · gold_hit=1.0
- **First final chunk preview:** we use BLEU as the automatic metric for translation evaluation. The statistics of Europarl and MultiUN corpora are summarized in Table TABREF18. For Europarl corpus, we evaluate on French-English-Spanish (Fr-En-Es), German-English-French (De-En-Fr) and Romanian-English-German (Ro-En-De), where English acts as the pivot language, its left side is the source language, and its right side is the targe…

#### Success 3

- **ID:** `qasper:a313e98994fc039a82aa2447c411dda92c65a470`
- **Question:** How do they match words before reordering them?
- **Retrieval stage:** `ranking` · **Failure bucket:** `ranking`
- **Gold aliases (sample):** CFILT-preorder system; We use the CFILT-preorder system for reordering English sentences to match the Indian language word order. It contains t; We use the CFILT-preorder system for reordering English sentences to match the Indian language word order. It contains t
- **Prediction:** [1] two re-ordering systems: (1) generic rules that apply to all Indian languages BIBREF17 , and (2) hindi-tuned rules which improve the generic rules by incorporating improvements found through an er
- **Metrics:** EM=0.0 · token_f1=0.6818181818181818 · gold_hit=0.0
- **First final chunk preview:** two re-ordering systems: (1) generic rules that apply to all Indian languages BIBREF17 , and (2) hindi-tuned rules which improve the generic rules by incorporating improvements found through an error analysis of English-Hindi reordering BIBREF28 . These Hindi-tuned rules have been found to improve reordering for many English to Indian language pairs BIBREF29 . In this section, we describe the resu…

## Failure cases (3)

#### Failure — retrieval (no gold in retrieve_k pool)

- **ID:** `qasper:ef4dba073d24042f24886580ae77add5326f2130`
- **Question:** What accuracy does the proposed system achieve?
- **Retrieval stage:** `retrieval` · **Failure bucket:** `retrieval`
- **Gold aliases (sample):** F1 scores of 85.99 on the DL-PS data,  75.15 on the EC-MT data and 71.53 on the EC-UQ data; FLOAT SELECTED: Table 2: Main results on the DL-PS data.; FLOAT SELECTED: Table 3: Main results on the EC-MT and EC-UQ datasets.; F1 of 85.99 on the DL-PS dataset (dialog domain); 75.15 on EC-MT and 71.53 on EC-UQ (e-commerce domain)
- **Prediction:** [1] by definition. In particular, our goal is not to simply minimize the new objective. Actually, we aim for a saddle point, finding the parameters $\Theta $ and $\Theta ^{\prime }$ satisfying the fol
- **Metrics:** EM=0.0 · token_f1=0.14814814814814814 · gold_hit=0.0
- **First final chunk preview:** by definition. In particular, our goal is not to simply minimize the new objective. Actually, we aim for a saddle point, finding the parameters $\Theta $ and $\Theta ^{\prime }$ satisfying the following conditions: $$\begin{split} \hat{\Theta } &= \mathop {arg~min}_{\Theta }\text{R}(\Theta , \Theta ^{\prime }, \mathbf {X}, \mathbf {\bar{y}}, \bar{z}) \\ \hat{\Theta }^{\prime } &= \mathop {arg~max}…

#### Failure — ranking (gold in pool, not in final_k)

- **ID:** `qasper:b6f15fb6279b82e34a5bf4828b7b5ddabfdf1d54`
- **Question:** which multilingual approaches do they compare with?
- **Retrieval stage:** `ranking` · **Failure bucket:** `ranking`
- **Gold aliases (sample):** We compare our approaches with related approaches of pivoting, multilingual NMT (MNMT) BIBREF19, and cross-lingual trans; The results show that our approaches consistently outperform other approaches across languages and datasets, especially ; Table TABREF19 and TABREF26 report zero-shot results on Europarl and Multi-UN evaluation sets, respectively. We compare ; multilingual NMT (MNMT) BIBREF19
- **Prediction:** [1] we use BLEU as the automatic metric for translation evaluation. The statistics of Europarl and MultiUN corpora are summarized in Table TABREF18. For Europarl corpus, we evaluate on French-English-
- **Metrics:** EM=0.0 · token_f1=0.16666666666666666 · gold_hit=0.0
- **First final chunk preview:** we use BLEU as the automatic metric for translation evaluation. The statistics of Europarl and MultiUN corpora are summarized in Table TABREF18. For Europarl corpus, we evaluate on French-English-Spanish (Fr-En-Es), German-English-French (De-En-Fr) and Romanian-English-German (Ro-En-De), where English acts as the pivot language, its left side is the source language, and its right side is the targe…

#### Failure — generation (gold in final context, wrong answer)

- **ID:** `qasper:f5e6f43454332e0521a778db0b769481e23e7682`
- **Question:** what are the pivot-based baselines?
- **Retrieval stage:** `gold_in_final` · **Failure bucket:** `generation`
- **Gold aliases (sample):** pivoting; pivoting$_{\rm m}$; We compare our approaches with related approaches of pivoting, multilingual NMT (MNMT) BIBREF19, and cross-lingual trans; Although it is challenging for one model to translate all zero-shot directions between multiple distant language pairs o; Table TABREF19 and TABREF26 report zero-shot results on Europarl and Multi-UN evaluation sets, respectively. We compare  … (+6 more)
- **Prediction:** [1] we use BLEU as the automatic metric for translation evaluation. The statistics of Europarl and MultiUN corpora are summarized in Table TABREF18. For Europarl corpus, we evaluate on French-English-
- **Metrics:** EM=0.0 · token_f1=0.17777777777777778 · gold_hit=0.0
- **First final chunk preview:** we use BLEU as the automatic metric for translation evaluation. The statistics of Europarl and MultiUN corpora are summarized in Table TABREF18. For Europarl corpus, we evaluate on French-English-Spanish (Fr-En-Es), German-English-French (De-En-Fr) and Romanian-English-German (Ro-En-De), where English acts as the pivot language, its left side is the source language, and its right side is the targe…
