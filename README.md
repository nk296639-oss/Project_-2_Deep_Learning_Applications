# Project_-2_Deep_Learning_Applications

# Project 2, Week 5: Scene Classification with Transfer Learning

- **Dataset:** Intel Image Classification, 6 classes (buildings, forest, glacier, mountain, sea, street), **14,034 train / 3,000 test** images, each 150 × 150 × 3. The training folder is split 85/15 into **11,929 train / 2,105 validation**.
- **Model:** ResNet-18 pretrained on ImageNet, with the 1,000-class head replaced by a new 6-class head.
- **Feature extraction (3,078 trainable parameters):** val accuracy **0.905**
- **Fine-tuning `layer4` (about 8.4M trainable, 8,396,806):** val accuracy **0.933**
- **Test accuracy (used once):** **0.933** (2,798 / 3,000); weakest class: **glacier** (recall 0.881), confused with **mountain**
- **Most instructive error:** A grayscale old-town street photo labelled *buildings* was predicted as *street* with 0.98 confidence, because both a building and a street are clearly visible, so the label itself is ambiguous and the model's confident "mistake" is partly a labelling problem.
- **Next:** my own domain in Week 6: diabetic retinopathy grading from retina photos (full proposal at the end of this file).

---

## 1. Data

| Class | Train | Test |
|---|---:|---:|
| buildings | 2,191 | 437 |
| forest | 2,271 | 474 |
| glacier | 2,404 | 553 |
| mountain | 2,512 | 525 |
| sea | 2,274 | 510 |
| street | 2,382 | 501 |
| **Total** | **14,034** | **3,000** |

The classes are roughly balanced (smallest 2,191, largest 2,512), so plain accuracy is a fair headline metric.

An image is a tensor of shape $H \times W \times C$:

$$150 \times 150 \times 3 = 67{,}500 \text{ numbers per image}, \quad \text{dtype } \texttt{uint8}, \text{ range } [0, 255]$$

**Split arithmetic**

$$n_{val}=\lfloor 0.15 \times 14{,}034 \rfloor = 2{,}105 \qquad n_{train}=14{,}034-2{,}105=11{,}929$$

Batch size is 64 for training, so one epoch takes $\lceil 11{,}929/64 \rceil = 187$ steps. The test set plays no part in any choice (epoch selection, comparing methods); it is evaluated once at the end.

**Preprocessing.** Pretrained ResNet-18 expects 224 × 224 inputs normalised with the ImageNet statistics:

$$x_{norm}=\frac{x-\text{mean}}{\text{std}},\qquad \text{mean}=[0.485,0.456,0.406],\;\text{std}=[0.229,0.224,0.225]$$

| Transform | Used on |
|---|---|
| `RandomResizedCrop(224, scale=(0.7,1.0))`, `RandomHorizontalFlip()`, `ColorJitter(brightness=0.2, contrast=0.2)` | train only |
| `Resize((224,224))` | validation + test |
| `ToTensor()`, `Normalize(MEAN, STD)` | all |

Rotating by 90° or flipping vertically was deliberately left out: an upside-down sea or street is not a natural scene.

## 2. Convolution and parameter counts

A convolution slides a kernel over the image and sums the products:

$$\text{out}(i,j)=\sum_{m=0}^{K-1}\sum_{n=0}^{K-1} x(i+m,\,j+n)\;k(m,n)$$

Hand check on the lecture example (vertical-edge kernel with rows `[-1, 0, 1]`, image rows `[0, 0, 10, 10, 10]`): each row gives $(-1)(0)+(0)(0)+(1)(10)=10$, so the top-left output is $10+10+10=30$. PyTorch printed three rows of `[30, 30, 0]`, which matches.

$$\text{Output size}=\left\lfloor\frac{W-K+2P}{S}\right\rfloor+1 \qquad\qquad \text{Parameters}=(K\cdot K\cdot C_{in}+1)\cdot C_{out}$$

| Layer | Calculation | Result |
|---|---|---:|
| Conv 3×3, 3 → 64 | $(3\cdot3\cdot3+1)\cdot64 = 28\cdot64$ | **1,792** |
| Dense 67,500 → 512 | $67{,}500\cdot512+512$ | **34,560,512** |
| Ratio | $34{,}560{,}512 / 1{,}792$ | **≈ 19,286× more** |
| Output, $K=3, P=1, S=1$ (224 input) | $\frac{224-3+2}{1}+1$ | 224 |
| Output, $K=3, P=1, S=2$ (224 input) | $\lfloor 111.5 \rfloor + 1$ | 112 |

Weight sharing is why the convolution is so small: one 3×3 filter is reused at every position, so its size does not depend on the image size and it recognises a feature wherever it appears.

## 3. Training method

Each batch goes through the four boxes of training:

| Step | Code | Maths |
|---|---|---|
| Forward | `logits = model(xb)` | $\hat y=\text{softmax}(z)$ |
| Loss | `F.cross_entropy(logits, yb)` | $L=-\log p_{true}$ |
| Backward | `loss.backward()` | $\frac{\partial L}{\partial w}=\frac{\partial L}{\partial \hat y}\cdot\frac{\partial \hat y}{\partial z}\cdot\frac{\partial z}{\partial w}$ |
| Step | `optimizer.step()` | $w \leftarrow w-\alpha\,\nabla L$ |

$$\text{softmax}(z)_k=\frac{e^{z_k}}{\sum_j e^{z_j}}$$

The loop keeps the weights of the best validation epoch.

### Stage 1: feature extraction (epochs 1-3)

The body is frozen and only the new head is trained (Adam, learning rate $10^{-3}$):

$$\text{new head}: 512\times6+6 = 3{,}078 \qquad \frac{3{,}078}{11{,}179{,}590}=0.0275\% \text{ of all parameters}$$

Check of the total: $11{,}689{,}512 - (512\cdot1000+1000) + 3{,}078 = 11{,}179{,}590$.

| Epoch | Train loss | Train acc | Val loss | Val acc |
|---:|---:|---:|---:|---:|
| 1 | 0.601 | 0.807 | 0.331 | 0.895 |
| 2 | 0.339 | 0.885 | 0.309 | 0.895 |
| 3 | 0.308 | 0.892 | 0.274 | **0.905** |

### Stage 2: fine-tuning `layer4` (epochs 4-6)

`layer4` is unfrozen with discriminative learning rates: $10^{-4}$ for the pretrained `layer4` and $10^{-3}$ for the new head.

$$\text{layer4} = 8{,}396{,}806-3{,}078 = 8{,}393{,}728 \qquad \frac{8{,}393{,}728}{11{,}179{,}590}\approx 75.1\%$$

| Epoch | Train loss | Train acc | Val loss | Val acc |
|---:|---:|---:|---:|---:|
| 4 | 0.260 | 0.906 | 0.213 | 0.926 |
| 5 | 0.166 | 0.940 | 0.214 | 0.932 |
| 6 | 0.120 | 0.956 | 0.208 | **0.933** |

Fine-tuning improved validation accuracy by $0.933-0.905 = 0.028$ (+2.8 points) and lowered validation loss from 0.274 to 0.208.

**Learning curves.** Training loss keeps falling in epochs 5-6 ($0.260 \to 0.166 \to 0.120$) while validation loss stays flat (about 0.21), so the train-validation gap widens. That is a mild sign of overfitting at most, because validation loss never rose. Training accuracy was below validation accuracy in the early epochs because training images are augmented and the model is in `train()` mode.

> Note: this run used the CPU (`Device: cpu`), so epochs took about 830 s (feature extraction) and 1,025 s (fine-tuning) instead of roughly 1 minute on a T4 GPU. The results are valid, only slower.

## 4. Test results (used once)

$$\text{accuracy}=\frac{401+469+487+464+501+476}{3000}=\frac{2798}{3000}=\mathbf{0.933} \qquad \text{errors}=202$$

$$\text{precision}=\frac{TP}{TP+FP} \qquad \text{recall}=\frac{TP}{TP+FN} \qquad F_1=\frac{2\cdot P\cdot R}{P+R}$$

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| buildings | 0.937 | 0.918 | 0.927 | 437 |
| forest | 0.998 | 0.989 | 0.994 | 474 |
| glacier | 0.894 | **0.881** | 0.887 | 553 |
| mountain | 0.901 | **0.884** | 0.892 | 525 |
| sea | 0.947 | 0.982 | 0.964 | 510 |
| street | 0.928 | 0.950 | 0.939 | 501 |

Test accuracy (0.933) matches validation accuracy (0.933), so there is no sign of overfitting to the validation set.

**Largest confusions** (true → predicted):

| Confusion | Count |
|---|---:|
| mountain → glacier | 49 |
| glacier → mountain | 47 |
| buildings → street | 34 |
| street → buildings | 24 |
| glacier → sea | 16 |

$$\text{glacier}\leftrightarrow\text{mountain}: 47+49=96 \qquad \text{buildings}\leftrightarrow\text{street}: 34+24=58$$

The weakest class is **glacier** (recall $487/553=0.881$), confused with **mountain**. The easiest class is forest (recall $469/474=0.989$). My prediction before training, that glacier and mountain would be confused most, was correct.

### Error analysis

The first 8 errors in the gallery are all true *buildings* images (7 predicted as street, 1 as mountain):

1. *Buildings → mountain (0.69):* a stone house in a green valley with a stream. The scene is dominated by landscape, so the model followed the landscape.
2. *Buildings → street (0.98, grayscale):* an old-town street with a wooden building. Genuinely ambiguous, since building and street are both clearly visible.
3. *Buildings → street (0.85):* an interior long hall with rails. This looks nothing like a building facade, so the label may be wrong.

Many of the confident errors (0.90-1.00) are therefore ambiguous or noisy labels, not simple model failures.

### Inside one prediction

For the first test image (true class *buildings*) the logits were $[7.21, -8.42, -4.16, -4.74, -5.10, -1.86]$:

$$p_{buildings}=\frac{e^{7.21}}{\sum_j e^{z_j}}=\frac{1352.9}{1353.08}=0.99986 \qquad L=-\log(0.99986)\approx 0.00014$$

## 5. Saved model

The fine-tuned weights are saved as `resnet18_intel.pt`.

$$11{,}179{,}590 \text{ parameters} \times 4 \text{ bytes} \approx 44.7 \text{ MB} \quad (\text{file: } 44.8 \text{ MB})$$

This is under GitHub's 100 MB limit. To reload it, build `models.resnet18()`, replace `model.fc` with `nn.Linear(512, 6)` first, then call `load_state_dict(...)`.

## 6. Results summary

| Stage | Trainable params | Val accuracy | Test accuracy |
|---|---:|---:|---:|
| Feature extraction (head only) | 3,078 | 0.905 | - |
| Fine-tuning (`layer4` + head) | 8,396,806 | 0.933 | **0.933** |

Weakest class: glacier (recall 0.881), confused with mountain.

## 7. Check your understanding (short answers)

- **Why do networks without nonlinearities collapse?** $W_2(W_1x+b_1)+b_2=(W_2W_1)x+(W_2b_1+b_2)$ is a single linear layer, so depth adds nothing without an activation such as ReLU.
- **Softmax of logits $[3,1,0]$:** $e^3=20.086$, sum $=23.804$, so $p_1 = 20.086/23.804 = 0.844$.
- **Conv 3×3, 64 → 128:** $(3\cdot3\cdot64+1)\cdot128 = 577\cdot128 = 73{,}856$ parameters.
- **56×56 input, $K=3$, $P=1$, $S=2$:** $\lfloor (56-3+2)/2 \rfloor + 1 = 27+1 = 28$, so a 28×28 output.
- **Why skip connections help:** for $y=F(x)+x$, $\partial y/\partial x=\partial F/\partial x+1$, so the gradient always has a direct path back, while a plain deep sigmoid network can shrink the signal to about $0.25^{20}\approx 9\times10^{-13}$.
- **When to train from scratch:** with very large labelled datasets, with images unlike everyday photos (radar, spectrograms, microscopy), or when a special small architecture is needed.

---

## Week 6 Proposal: Diabetic retinopathy grading (retina photos)

**Domain:** Medical images (fundus photographs).

**Problem:** Grade diabetic retinopathy severity from 0 (none) to 4 (proliferative) from one retina photo.

**Who benefits:** Diabetic patients in places with few eye specialists, since screening could flag who needs referral. Missing severe cases is the costly error, so I will report recall per grade.

**Dataset:** APTOS 2019 Blindness Detection (Kaggle competition data), about 3,660 labelled images. Check the competition rules for allowed use, as this is not a standard open licence.

**Plan:** `GRAYSCALE = False`. Resize to 224×224 after cropping the black borders. Use the standard two-stage training with class weights. Report macro-F1, per-grade recall, a confusion matrix, and Grad-CAM. I will also report quadratic weighted kappa, because the grades are ordered.

**Expected challenges:**

- Heavy imbalance (most images are grade 0) and image quality that varies by camera and clinic.
- The grades are ordinal, so confusing grade 0 with grade 4 is worse than confusing 2 with 3. The confusion matrix should show this.
- Lighting and camera differences can act as shortcuts.
- Grades 1 and 2 are subtle, and even graders disagree, so some errors may be label noise.

- # Week 6: Referable Diabetic Retinopathy Screening with ResNet-18

**Owner:** Muhammad Umar Khan  |  **Task:** binary classification of fundus photographs, *referable DR* (grade ≥ 2) vs *non-referable* (grade 0–1)  |  **Backbone:** ResNet-18 (ImageNet weights, transfer learning)

---

## Summary card

**Who benefits:** eye-care teams and patients in diabetic-retinopathy (DR) screening settings, for example a clinic or tele-ophthalmology programme where few ophthalmologists must review many retinal photographs. The model is meant to help prioritise which photographs a specialist should look at first.

**Why it matters:** a missed referable case (false negative) means untreated DR that can progress to vision loss; a false alarm (false positive) costs an unnecessary referral. On the test set the model made **12 misses and 32 false alarms in 506 images** (see Results).

**Dataset:** APTOS 2019 Blindness Detection, [kaggle.com/c/aptos2019-blindness-detection](https://www.kaggle.com/c/aptos2019-blindness-detection), used through a Kaggle re-upload (`mariaherrerot/aptos2019`) whose `train_1.csv`, `valid.csv` and `test.csv` were merged and **re-split** by this notebook. **3,662 labelled images** (3,528 after de-duplication), **5 DR grades** (0 none, 1 mild, 2 moderate, 3 severe, 4 proliferative) collapsed to **2 classes**. Labels: severity grades assigned by clinicians according to the competition description. **Licence:** not stated in the notebook; check the competition rules / dataset page before reuse or redistribution.

**Data audit:** imbalance ratio **1.46** (2175 / 1487; printed as 1.5), 1.54 after de-duplication; **exact duplicates: 251 files in 123 groups, 134 files removed**; **test images with a near-duplicate in train: 0 by construction** (every pair with fingerprint distance < 0.10 is forced into the same group; this is a count of the grouping rule, not a separate measurement); split by **near-duplicate group** (no patient id exists), **0 shared groups** between train/val, train/test and val/test.

**Imbalance handling:** class-weighted cross-entropy with $w_c = N/(K\,n_c)$; weights **non-referable 0.890, referable 1.141**.

**Results (test, used once):** **macro-F1 0.879** (validation 0.919); per-class recall at the default threshold 0.5: **non-referable 0.777, referable 0.956**. At the recall-targeted threshold **t = 0.464 (set on validation)**: **sensitivity 0.967** (95 % CI 0.944–0.981) and **specificity 0.770** (95 % CI 0.693–0.832); PPV 0.917, NPV 0.899, AUC 0.956.

**Grad-CAM:** on the 9 test images inspected (5 correct, 4 errors) the model mostly looks at the **inside of the retinal disc**: the posterior pole, the area around the optic disc and macula, and, in the correct grade 3 and 4 cases, the regions with yellow exudates. In several errors the heat is pushed to the **periphery or the edge of the field**. **Shortcut risk:** no hospital label, text or marker is visible in these 9 images and the heat does not sit on the black background; this was a visual check only (no quantitative border-attention test, no masking experiment), so shortcut learning is *not ruled out*.

**Limitations:** one data source (a single competition dataset, images of very different resolution, 640×480 up to 4288×2848, suggesting several cameras); no patient ids, so leakage control relies on an image-fingerprint heuristic; only 5 training epochs on CPU and validation macro-F1 was still rising; one seed; very different class mix per split (referable 15 % in validation vs 73 % in test); the binary rule hides the grade 1 vs grade 2 boundary where most errors occur. **Not for** clinical diagnosis, unsupervised use, population screening without re-validation, or use on other cameras, populations or other eye diseases.

---

## 1. Pipeline

| Step | What is done | Key value |
|---|---|---|
| Load | merge `train_1`, `valid`, `test` csv, link images | 2930 + 366 + 366 = **3662** |
| Pre-process | resize once to 256×256, md5, 32×32 fingerprint | 329 s (≈ 11 images/s) |
| Exact duplicates | drop repeated files, drop contradictory ones | 3662 → **3528** |
| Near-duplicates | group images with fingerprint distance < 0.10 | **1998 groups** |
| Split | `StratifiedGroupKFold` (stratify by grade, group by near-dup group) | **1970 / 1052 / 506** |
| Train | ResNet-18, head only (2 ep) then last block (3 ep), select on val macro-F1 | best val macro-F1 **0.919** |
| Threshold | highest threshold with validation recall ≥ 0.95 | **t = 0.464** |
| Explain | Grad-CAM on `layer4` | 7×7 map, up-sampled ×32 |

Inputs are 224×224 RGB; colour is kept (no hue or saturation jitter) because it carries lesion information. Training augmentation: random resized crop (scale 0.8–1.0), horizontal and vertical flips, rotation up to 20°, brightness and contrast ±15 %.

---

## 2. Data audit with calculations

### 2.1 Class balance

Label rule: $y_i = \text{referable}$ if grade $g_i \ge 2$, else non-referable.

$$n_{\text{referable}} = 999 + 193 + 295 = 1487, \qquad n_{\text{non-referable}} = 1805 + 370 = 2175, \qquad 1487 + 2175 = 3662$$

$$\text{IR} = \frac{n_{\max}}{n_{\min}} = \frac{2175}{1487} = 1.46, \qquad \text{prevalence} = \frac{1487}{3662} = 40.6\,\%$$

After de-duplication the classes are $1107 + 892 + 139 = 2138$ non-referable and $863 + 160 + 367 = 1390$ referable, so $\text{IR} = 2138/1390 = 1.54$.

### 2.2 Exact duplicates

251 files fall into 123 identical-file groups, so the redundant copies are $251 - 123 = 128$. Six groups carry contradictory (binary) labels, and for them even the copy that would be kept is dropped:

$$128 + 6 = 134 \text{ removed}, \qquad 3662 - 134 = 3528 \text{ images remain}$$

### 2.3 Near-duplicates and groups

Each image gets a standardised $32 \times 32$ grey fingerprint $f \in \mathbb{R}^{1024}$:

$$f = \frac{a - \mu_a}{\sigma_a + \varepsilon}, \qquad \varepsilon = 10^{-6}$$

Distance between two images (mean absolute difference), and the linking rule:

$$d(i,j) = \frac{1}{1024}\sum_{k=1}^{1024}\left|f_{i,k} - f_{j,k}\right|, \qquad \text{link if } d(i,j) < 0.10$$

Connected components (union-find) form the groups, and a group is never split across train, validation and test.

$$3528 - 1581 = 1947 \text{ singletons}, \qquad 1947 + 51 = 1998 \text{ groups}$$

$$\text{images in multi-image groups} = \frac{1581}{3528} = 44.8\,\%, \qquad \text{mean multi-image group size} = \frac{1581}{51} \approx 31$$

The median nearest-neighbour distance is 0.109 (above the 0.10 threshold), which is consistent with about 45 % of images being grouped. There are 20,748 linked pairs but only 51 groups, so linking is transitive (chaining), and **20 groups mix different grades**. The threshold is therefore somewhat loose: some groups probably merge different eyes. That errs on the safe side for leakage but makes the split uneven.

### 2.4 Split

| Split | Images | Share of 3528 | Non-referable | Referable | Referable prevalence |
|---|---|---|---|---|---|
| train | 1970 | 55.8 % | 1107 | 863 | $863/1970 = 43.8\,\%$ |
| validation | 1052 | 29.8 % | 892 | 160 | $160/1052 = 15.2\,\%$ |
| test | 506 | 14.3 % | 139 | 367 | $367/506 = 72.5\,\%$ |

$$1970 + 1052 + 506 = 3528 \qquad \text{shared groups: train–val } 0,\ \text{train–test } 0,\ \text{val–test } 0$$

The ideal shares would be 66.7 / 16.7 / 16.7 %; the sizes differ because groups cannot be divided (one very large group lands in validation).

---

## 3. Imbalance handling

$$w_c = \frac{N}{K\,n_c}, \qquad N = 1970,\; K = 2$$

$$w_{\text{non-ref}} = \frac{1970}{2 \times 1107} = 0.890, \qquad w_{\text{ref}} = \frac{1970}{2 \times 863} = 1.141$$

Check: each class carries the same total weight, $1107 \times 0.8898 = 863 \times 1.1413 = N/K = 985$. The positive class is up-weighted by $1107/863 = 1.28\times$. The weights enter the loss as

$$\mathcal{L} = -\frac{\sum_i w_{y_i}\,\log p_{i,y_i}}{\sum_i w_{y_i}}, \qquad p_{i,c} = \frac{e^{z_{i,c}}}{\sum_k e^{z_{i,k}}}$$

A balanced sampler (`USE_SAMPLER = True`, $P(i) \propto 1/n_{c(i)}$) is implemented but was **not** used.

---

## 4. Model and training

* ResNet-18 loaded from the local ImageNet checkpoint (`resnet18-f37072fd.pth`, 102 tensors, none skipped), final layer replaced by a 2-class linear layer.
* **Phase 1:** body frozen, Adam, learning rate $10^{-3}$, 2 epochs. **Phase 2:** `layer4` unfrozen (learning rate $10^{-4}$, head $10^{-3}$), 3 epochs.
* Model selection by validation **macro-F1**, not accuracy.

$$\text{trainable head} = 512 \times 2 + 2 = 1026 \text{ parameters} = \frac{1026}{11{,}177{,}538} \approx 0.009\,\%\ \text{of the network}$$

| Phase | Epoch | Val macro-F1 |
|---|---|---|
| head only | 1 | 0.806 |
| head only | 2 | 0.818 |
| fine-tune | 1 | 0.887 |
| fine-tune | 2 | 0.898 |
| fine-tune | 3 | **0.919** |

$$\text{gain from fine-tuning } \texttt{layer4}: \; 0.919 - 0.818 = +0.101$$

Total training time $176 + 173 + 224 + 223 + 222 = 1018\ \text{s} \approx 17\ \text{min}$ on CPU.

---

## 5. Results (test set, used once)

### 5.1 Default decision rule (argmax, threshold 0.5)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| non_referable | 0.871 | 0.777 | 0.821 | 139 |
| referable_DR | 0.919 | 0.956 | 0.937 | 367 |

Metric definitions and worked values:

$$P_c = \frac{TP_c}{TP_c+FP_c}, \qquad R_c = \frac{TP_c}{TP_c+FN_c}, \qquad F1_c = \frac{2P_cR_c}{P_c+R_c}$$

$$F1_{\text{non-ref}} = \frac{2 \times 0.871 \times 0.777}{0.871 + 0.777} = 0.821, \qquad F1_{\text{ref}} = \frac{2 \times 0.919 \times 0.956}{0.919 + 0.956} = 0.937$$

$$\text{macro-F1} = \frac{0.821 + 0.937}{2} = \mathbf{0.879}, \qquad \text{weighted-F1} = \frac{139 \times 0.821 + 367 \times 0.937}{506} = 0.905$$

$$\text{accuracy} = \frac{108 + 351}{506} = 0.907$$

The confusion matrix reconstructed from recall × support is $TN \approx 108,\ FP \approx 31,\ FN \approx 16,\ TP \approx 351$ (it reproduces the printed precisions, $351/382 = 0.919$ and $108/124 = 0.871$). Weighted-F1 is higher than macro-F1 because 72.5 % of the test images belong to the easier referable class; the weak point is the **non-referable recall of 0.777**.

### 5.2 Screening operating point (threshold set on validation)

The threshold is the highest value that still gives validation recall ≥ 0.95, then applied unchanged to the test set:

$$\hat y_i = \mathbb{1}\!\left[p_i(\text{referable}) \ge t\right], \qquad t = 0.464$$

Test counts: $TP = 355,\ FN = 12,\ TN = 107,\ FP = 32$.

$$\text{Sensitivity} = \frac{355}{355 + 12} = \mathbf{0.967}, \qquad \text{Specificity} = \frac{107}{107 + 32} = \mathbf{0.770}$$

$$\text{PPV} = \frac{355}{355 + 32} = 0.917, \qquad \text{NPV} = \frac{107}{107 + 12} = 0.899, \qquad \text{AUC} = 0.956$$

95 % Wilson confidence intervals (small test set, so they are wide):

$$\hat p \in \frac{\hat p + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}, \; z = 1.96 \;\Rightarrow\; \text{sensitivity } [0.944, 0.981], \; \text{specificity } [0.693, 0.832]$$

F1 at this operating point, computed from the counts (the notebook does not print it): $F1_{\text{ref}} = \frac{2 \cdot 355}{2 \cdot 355 + 32 + 12} = 0.942$, $F1_{\text{non-ref}} = \frac{2 \cdot 107}{2 \cdot 107 + 12 + 32} = 0.830$, so **macro-F1 ≈ 0.886**.

Moving from threshold 0.5 to 0.464 removed 4 misses ($FN$ 16 → 12) at the price of one extra false alarm ($FP$ 31 → 32): sensitivity $351/367 = 0.956 \to 0.967$, specificity $0.777 \to 0.770$. The validation target (0.95) was met on test (0.967).

### 5.3 Where the errors are (share of test images flagged, by grade)

| Grade | n | Flagged | Count flagged | Reading |
|---|---|---|---|---|
| 0 none | 90 | 0.044 | 4 | false alarms |
| 1 mild | 49 | 0.571 | 28 | false alarms (not referable by the label rule) |
| 2 moderate | 266 | 0.966 | 257 | 9 missed |
| 3 severe | 33 | 1.000 | 33 | 0 missed |
| 4 proliferative | 68 | 0.956 | 65 | 3 missed |

$$4 + 28 + 257 + 33 + 65 = 387 = TP + FP, \qquad FP = 4 + 28 = 32, \qquad FN = 9 + 0 + 3 = 12$$

Every false alarm is a grade 0 or grade 1 eye, and 28 of the 32 are grade 1: the model's difficulty is the mild-to-moderate boundary, as expected clinically.

### 5.4 Effect of prevalence (why the test PPV is optimistic)

PPV depends on how common the disease is:

$$\text{PPV} = \frac{\text{Sens}\cdot\pi}{\text{Sens}\cdot\pi + (1-\text{Spec})(1-\pi)}$$

On the test set ($\pi = 0.725$): $\;\dfrac{0.967 \times 0.725}{0.967 \times 0.725 + 0.230 \times 0.275} = \dfrac{0.7011}{0.7644} = 0.917$.

In a general population with $\pi = 2\,\%$: $\;\dfrac{0.967 \times 0.02}{0.967 \times 0.02 + 0.230 \times 0.98} = \dfrac{0.0193}{0.2447} = 0.079$.

So at 2 % prevalence only about **8 %** of flagged patients would truly be referable, even though the PPV looks like 92 % on this test set. Prevalence-dependent numbers (accuracy, PPV, weighted F1) must not be carried over to a different population.

---

## 6. Grad-CAM

Hooks on the last convolution block (`layer4`) give the activations $A^k$ and the gradients of the class score $y^c$:

$$\alpha_k^c = \frac{1}{Z}\sum_i\sum_j \frac{\partial y^c}{\partial A^k_{ij}}, \qquad L^c_{\text{Grad-CAM}} = \mathrm{ReLU}\!\Big(\sum_k \alpha_k^c A^k\Big)$$

For a 224×224 input ResNet-18 has total stride 32, so $Z = (224/32)^2 = 7 \times 7 = 49$ and the map is up-sampled by a factor of 32 (bilinear) and divided by its maximum, so values lie in $[0,1]$. Because the map is so coarse it can show *regions*, not individual lesions.

The figure uses one correct prediction per DR grade plus 4 random errors (9 images). Observations:

* **Correct, non-referable (grade 0):** broad attention over the retina and the optic disc.
* **Correct, referable (grades 2, 3, 4):** attention on the posterior pole and macular region; in grade 3 the hot area covers the yellow exudate spots.
* **Errors:** attention moves to the edge or periphery of the field (for example one grade 1 image predicted as referable has its strongest heat at the left border, and a grade 4 image predicted as non-referable has heat on the lower border), which suggests the wrong cue is being used in those cases.

**Next checks to make before trusting the model** (not done in this notebook): quantify the fraction of heat in the border region, $\sum_{\text{border}} L / \sum_{\text{all}} L$ over the whole test set; mask the border or any camera marker and re-evaluate; test on images from a second camera or clinic.

---

## 7. Limitations

1. **Single source.** All images come from one competition dataset; resolutions vary from 640×480 to 4288×2848 (median 2144×1536), so several cameras or protocols are mixed and performance on other devices is unknown.
2. **No patient identifier.** Leakage control uses a 32×32 fingerprint with a loose threshold (20,748 pairs, 20 mixed-grade groups). Leakage between eyes of the same patient that look different is not controlled.
3. **Uneven and shifted splits.** Referable prevalence is 15.2 % in validation, 43.8 % in train and 72.5 % in test. The threshold chosen on a low-prevalence validation set may not transfer; the test PPV is high partly because the test set is mostly positive.
4. **Small test set.** 139 negatives give a specificity interval of 0.693–0.832; one seed only, no repeated runs.
5. **Under-trained.** Only 5 epochs on CPU and validation macro-F1 was still increasing.
6. **Binary simplification.** Collapsing five grades into two hides the grade 1 / grade 2 boundary, where all 32 false alarms come from.
7. **Image preparation.** Photos are squeezed to square 256×256 (median aspect 1.40 → 1.00, about 50× fewer pixels), which can erase small lesions such as microaneurysms.
8. **Labels.** Grades carry inter-observer variability, and identical images with contradictory labels existed in the data (6 groups, removed).
9. **Grad-CAM evidence is anecdotal** (9 images, visual only).

**Not for:** clinical diagnosis or treatment decisions; stand-alone, unsupervised screening; use on other cameras, populations or age groups without local validation; grading severity (it is a referable / non-referable screen); detecting other retinal diseases.

---

## 8. Files and reproducibility

| File | Content |
|---|---|
| `week6-blindness.ipynb` | full pipeline, all numbers above |
| `splits.csv` | split (train / val / test), grade, label and group of each image |
| `aptos_model.pt` | weights, class order, backbone, image size, flags |

Fixed settings: `SEED = 42`, `IMG_SIZE = 224`, `CACHE_SIZE = 256`, `NEAR_DUP_THR = 0.10`, `TARGET_RECALL = 0.95`, `REFERABLE_MIN_GRADE = 2`. The operating threshold **t = 0.464 is not stored in the checkpoint**; record it separately before any use.
