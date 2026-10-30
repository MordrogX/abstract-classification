# Classifying Research Abstracts: Science vs Business vs Social Science

## 1. Problem

Predict the field of a peer-reviewed publication (**Science**, **Business**, **Social Science**) using only its abstract. This is a baseline multi-class text-classification task.

## 2. Dataset

* **Source:** [OpenAlex](https://openalex.org) API (free, open scholarly metadata)
* **Size:** 1,500 English abstracts (500 per class), articles published 2000 onward, abstracts of 50+ words
* **Labels:** derived from each paper's OpenAlex *primary topic → field*:
  * Science = Agricultural and Biological Sciences, Biochemistry, Genetics and Molecular Biology, Chemistry, Computer Science, Earth and Planetary Sciences, Energy, Engineering, Environmental Science, Immunology and Microbiology, Materials Science, Mathematics, Medicine, Neuroscience, Pharmacology, Toxicology and Pharmaceutics, Physics and Astronomy, Health Professions
  * Business = Business, Management and Accounting
  * Social Science = Social Sciences
* **File:** `dataset.csv` (columns: `id, title, abstract, year, openalex_field, label`), created by the notebook
* **Demo data:** `demo_test_data.csv` (~100 records from the held-out test set)

## 3. Method

1. **Pre-processing:** lowercase → remove HTML/numbers/punctuation → tokenize → remove stop-words → lemmatize
2. **Features:** TF-IDF (unigrams + bigrams, 6,000 features), fitted on training data only
3. **Split:** 80% train / 20% test, stratified, `random_state=42`
4. **Models:** Logistic Regression, Decision Tree, Naive Bayes, kNN (cosine), SVM (linear), Random Forest
5. **Evaluation:** accuracy, macro precision/recall/F1, 5-fold cross-validation, confusion matrices, per-class F1, top words per class (matplotlib only)

## 4. Results

| Model               | Accuracy | Precision | Recall | F1    |
| ------------------- | -------- | --------- | ------ | ----- |
| Logistic Regression | 0.859    | 0.864     | 0.859  | 0.860 |
| Decision Tree       | 0.669    | 0.672     | 0.669  | 0.670 |
| Naive Bayes         | 0.848    | 0.860     | 0.848  | 0.849 |
| kNN                 | 0.781    | 0.783     | 0.780  | 0.780 |
| SVM                 | 0.840    | 0.843     | 0.840  | 0.841 |
| Random Forest       | 0.825    | 0.829     | 0.825  | 0.825 |

**Best model:** Logistic Regression (macro F1 = 0.860; accuracy = 0.859).

**Observations:** Logistic Regression performed best, with Naive Bayes close behind; the Decision Tree was the weakest baseline. The best model misclassified 38 of 269 test abstracts. Examples of errors include confusion between Business and Social Science, and between Science and Social Science, consistent with overlap in the subject areas and automatically assigned labels.

## 5. Limitations

Labels come from OpenAlex's automatic topic assignment and are not hand-verified; Business and Social Science overlap; only abstracts are used; models are untuned baselines.
