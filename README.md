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


# Week 7: Text and Multimodal AI

Text tools (spam classifier, named-entity recognition, regular-expression fields, OCR, privacy masking) connected to my **Week 6 image model** (APTOS 2019 fundus photos, referable diabetic retinopathy) in one decision-support record.

> **Decision support only.** Nothing here is validated clinically. A qualified human makes every decision. All patient names, numbers and notes in this project are invented.

## Summary

| Item | Result |
|---|---|
| Text classifier | SMS Spam Collection, **5,572** texts (747 spam, 13.4 %), macro-F1 **0.974** (word 1-2 grams) vs **0.982** (character 2-5 grams) |
| Top positive (spam) features | call, txt, free, text, reply, uk, to, www, mobile, claim, stop, your |
| Top negative (ham) features | my, me, ok, that, it, but, ll, gt, lt, can, what, da |
| NER on my domain (diabetic eye-screening reports) | out-of-the-box F1 **0.667** -> with entity ruler **0.889** on the 5 held-out sentences (0.526 -> 1.000 on the 5 sentences the rules were written from) |
| NER on the lab sentences | F1 **0.483** -> **0.839** |
| Regex fields extracted | CNIC, mobile, amount, date, e-mail, invoice no.; domain: fundus file id, HbA1c, eye (OD/OS/OU), DR grade, MRN, age |
| False positives tested | **7 negative tests, 0 false positives** (10 of 10 domain tests pass); 3 loose patterns shown and fixed |
| OCR | CER on the 3 lab pairs: raw **0.062 / 0.059 / 0.143** (mean 0.088) -> **0.000** after correction rules; synthetic invoice 0.000 -> 0.000 (too clean to test the rules) |
| Pipeline | image model + text evidence -> JSON record (example below) |
| Privacy | CNIC, phone, e-mail (and MRN) masked **before** extraction and logging; leak check: 0 of 5 fragments found in the record |

## Inputs

| Input | Used for |
|---|---|
| SMS Spam Collection (`uciml/sms-spam-collection-dataset`, `spam.csv`, Latin-1) | Part 1 (spam classifier) |
| spaCy `en_core_web_sm` (3.8.16) | NER |
| `aptos_model.pt` (Week 6: ResNet18, 2 classes `non_referable` / `referable_DR`, 224 x 224, colour) | image evidence |
| APTOS 2019 fundus photos (Week 6 mirror, original source: Kaggle competition *APTOS 2019 Blindness Detection*) | example photos |
| EasyOCR (needs internet the first time) | invoice OCR |

The licence of each Kaggle upload was not checked; check it and cite the original sources before reuse.

---

## 1. Text classifier: spam vs ham

**Data.** 4,825 ham + 747 spam = 5,572 messages. Spam share $747/5572 = 0.134$, so an "always ham" model already scores $4825/5572 = 0.866$ accuracy; accuracy is therefore not used to compare models. Imbalance ratio $4825/747 = 6.46$.

**Split.** 80 / 20, stratified, `random_state=42`: train 4,457 (598 spam), test 1,115 (149 spam, 966 ham).

**Features and model.**

$$tf'(t,d) = 1 + \ln tf(t,d), \qquad idf(t) = \ln\frac{1+N}{1+df(t)} + 1, \qquad \mathbf{x}_d = \frac{tf' \cdot idf}{\lVert tf' \cdot idf \rVert_2}$$

$$p(\text{spam}\mid \mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w}^{\top}\mathbf{x} + b)}}$$

Balanced class weights $w_c = N_{\text{train}} / (K\, n_c)$:

$$w_{\text{ham}} = \frac{4457}{2 \times 3859} = 0.577, \qquad w_{\text{spam}} = \frac{4457}{2 \times 598} = 3.727$$

### Word vs character model

| Model | Features | Macro-F1 | Spam precision | Spam recall | Spam F1 | Errors (of 1,115) |
|---|---|---|---|---|---|---|
| Word 1-2 grams (`min_df=2`) | 12,128 | **0.974** | 0.979 | 0.933 | 0.955 | 13 |
| Character 2-5 grams (`char_wb`) | 36,449 | **0.982** | 0.986 | 0.953 | 0.969 | 9 |

**Working (word model).** Recall $0.933 = TP/149 \Rightarrow TP = 139$, $FN = 10$; precision $0.979 = 139/(139+FP) \Rightarrow FP = 3$; $TN = 963$.

$$F1_{\text{spam}} = \frac{2 \times 139}{142 + 149} = 0.955, \quad F1_{\text{ham}} = 0.993, \quad \text{macro-F1} = \frac{0.955 + 0.993}{2} = 0.974$$

**Working (character model).** $TP = 142$, $FN = 7$, $FP = 2$: $F1_{\text{spam}} = \frac{2 \times 142}{144 + 149} = 0.969$, macro-F1 $= \frac{0.969 + 0.995}{2} = 0.982$.

**Reading the comparison.** The macro-F1 gap is $0.982 - 0.974 = 0.008$, i.e. $13 - 9 = 4$ messages ($0.36\,\%$ of the test set) on one split: too small to call a real difference. The case for characters is robustness: after padding, "prize" and "pr1ze" each give 18 character n-grams and share 6, so

$$J = \frac{6}{18 + 18 - 6} = 0.20 \quad (\text{a word model sees an unseen token: overlap } 0).$$

The cost is a vocabulary $36449/12128 = 3.0\times$ larger.

### Top features (word model)

A coefficient $w_j > 0$ multiplies the spam odds by $e^{w_j}$ per unit of feature value.

| Direction | Terms |
|---|---|
| Spam-like | call, txt, free, text, reply, uk, **to**, www, mobile, claim, stop, **your** |
| Ham-like | my, me, ok, that, it, but, ll, **gt**, **lt**, can, what, **da** |

Most spam terms are real calls to action. Shortcut-like terms: *to* and *your* (function words), and the artefacts *gt* / *lt* (HTML escapes `&gt;`, `&lt;`) and *da* (a chat particle). The shortcut causes the false positive "Waiting for your call." with $p = 0.853$.

### Robustness to spelling tricks (threshold 0.5)

| Message | Expected | Word $p$ | Char $p$ |
|---|---|---|---|
| `W1N a FR33 pr1ze!! txt CLAIM now` | spam | 0.99 | 0.85 |
| `kal milte hain, 5 baje` | ham | 0.19 | 0.12 |
| `C0ngrats!! U w0n Rs 25000 pr1ze. Cl@im by txt 5588 now` | spam | 0.81 | 0.88 |
| `Mubarak ho! Aap ne 50000 ka inaam jeeta hai, abhi 8000 par SMS karein` | spam | **0.32 (missed)** | 0.53 |
| `Yaar meeting kal subah 9 baje hai, notes bhej dena` | ham | 0.13 | 0.12 |

Neither model was trained on Roman Urdu, so these rows are untrained behaviour, not evidence; the character model passes the fourth row by only $0.53 - 0.50 = 0.03$.

### Error analysis (word model, 13 errors)

| Direction | Count | Share |
|---|---|---|
| False negatives (spam scored below 0.5) | 10 | $10/149 = 6.7\,\%$ of spam |
| False positives (ham scored 0.5 or higher) | 3 | $3/966 = 0.31\,\%$ of ham |

| Category (my judgement) | Count | Share of errors |
|---|---|---|
| Model error (fixable) | 8 | 62 % |
| Genuinely ambiguous | 4 | 31 % |
| Label error | 1 | 8 % |

Error rate $13/1115 = 1.17\,\%$; counting only the 8 model errors, $8/1115 = 0.72\,\%$. Closest misses: $p = 0.4645$ ("Call me NOW" premium-number message) and $p = 0.4248$.

---

## 2. Named-entity recognition (NER)

Scoring is by exact match of text **and** label:

$$P = \frac{TP}{TP + FP}, \quad R = \frac{TP}{TP + FN}, \quad F1 = \frac{2\,TP}{\#\text{predicted} + \#\text{gold}}$$

| Test set | Gold | Out of the box (P / R / F1) | With entity ruler (P / R / F1) |
|---|---|---|---|
| Lab sentences (6) | 16 | 0.538 / 0.438 / **0.483** | 0.867 / 0.812 / **0.839** |
| My domain, sentences 1-5 (rules written from these) | 20 | 0.556 / 0.500 / **0.526** | 1.000 / 1.000 / **1.000** |
| **My domain, sentences 6-10 (held out)** | 18 | 0.667 / 0.667 / **0.667** | 0.889 / 0.889 / **0.889** |

**Working.**

* Lab, before: $TP = 0.438 \times 16 = 7$, predicted $= 7/0.538 = 13$, $F1 = \frac{2 \times 7}{13 + 16} = \frac{14}{29} = 0.483$. After: $TP = 13$, predicted $= 15$, $F1 = \frac{26}{31} = 0.839$; gain $+0.356$ (the lecture, with its original names, reports $0.533 \to 0.875$).
* Domain held-out, before: $TP = 12$, $F1 = \frac{24}{36} = 0.667$. After: $TP = 16$, $FP = FN = 2$, $F1 = \frac{32}{36} = 0.889$.
* **Optimism gap:** $1.000 - 0.889 = 0.111$. The first number alone would overstate the rules.

**Domain and rules.** Ten sentences imitating diabetic eye-screening reports. Rules: `MONEY` (`PKR` / `Rs.` + number), `DATE` (`dd/mm/yyyy`), `ORG` for PIMS, a "Capitalised words + Hospital / Society / Centre / Clinic / Institute / Bank" pattern, four cities, and a `strip_titles` step that removes `Dr.` / `Patient` from person spans.

**What the rules did on the held-out set.** 12 of 18 entities were already correct before any rule (all 4 organisations, all 3 cities, 2 dates, 3 of 5 persons); the rules fixed 4 (the date `05/12/2026` and the amounts `PKR 12,500`, `PKR 24,000`, `Rs. 80,000`): $12 + 4 = 16 = TP$. So the held-out set barely tests the hospital-suffix pattern and the city list; the gain comes from the **format** rules.

**Remaining errors:** two person names, *Amna Javed* (tagged `LOC`) and *Danish Akram* (tagged `ORG`); no rule covers open-class person names. With only $20 + 18 = 38$ gold entities a trained model is not possible; next step would be annotating a few hundred real report sentences.

---

## 3. Regex field extraction

| Field | Pattern (case-insensitive, except where noted) |
|---|---|
| `cnic` | `\b\d{5}-\d{7}-\d\b` |
| `mobile` | `(?:\+92[- ]?\|\b0)3\d{2}[- ]?\d{7}\b` |
| `amount` | `\b(?:PKR\|Rs\.?)\s?([\d,]+(?:\.\d{1,2})?)` |
| `date` | `\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b` |
| `email` | `\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b` |
| `invoice` | `\b(?:Invoice\s+)?No\.?\s*(\d{3,})\b` |
| `fundus_file` | `\b[0-9a-f]{12}\.(?:png\|jpe?g)\b` (APTOS `id_code`) |
| `hba1c` | `\bHbA1c\s?(?:of\s\|[:=]\s?)?(\d{1,2}(?:\.\d)?)\s?%` (percent only) |
| `eye` | `(?-i:\b(?:OD\|OS\|OU)\b)` (**case-sensitive**) |
| `dr_grade` | `\bDR\s+grade\s?([0-4])\b` |
| `mrn` | `\bMRN[- ]?\d{6,8}\b` |
| `age` | `\baged?\s+(\d{1,2}\|1[01]\d)\b` |

**Tests.** Lab: 5 of 5 pass (3 positive, 2 negative). Domain: **10 of 10 pass**, of which **7 are negative tests** (must return nothing):

| Negative test | Why it must not match |
|---|---|
| `Take the tablet od for one month` | lowercase *od* = once daily, not the right eye |
| `Grade 3 students had an HbA1c test` | "grade" without "DR"; no value |
| `HbA1c 53 mmol/mol` | IFCC units, not a percentage |
| `Photo abcdefabcdef12.png and 12345.png` | 14 hex characters / too short for an id |
| `BP 140/90 mmHg and glucose 7 mmol/L` | not a date |
| `MRN 12 was seen in age 2026` | too few digits; not an age |
| `Doctors 500 mg twice daily` | "Rs" inside "Doctors" is not an amount |

**False positives found and fixed** (probe: `Doctors 500 mg; HbA1c 53 mmol/mol; Grade 3 students`):

| Pattern | Loose version returns | Fixed version returns |
|---|---|---|
| `amount` without `\b` | `500` | none |
| `hba1c` without `%` | `53` | none |
| `dr_grade` without "DR" | `3` | none |

Each loose pattern gave exactly 1 false hit; the fixes remove it and keep all 3 positive tests. **Known limit:** an uppercase prescription abbreviation (`metformin OD`) would still match `eye`.

---

## 4. OCR and correction rules

$$d_{i,j} = \min\big(d_{i-1,j} + 1,\; d_{i,j-1} + 1,\; d_{i-1,j-1} + [a_i \ne b_j]\big), \qquad \text{CER} = \frac{S + D + I}{N_{\text{ref}}}$$

`fix_ocr` turns a letter `O` into `0` only when it touches a digit, `l`/`I` between digits into `1`, and `N0.` into `No.`.

| Reference | OCR output | Edits | $N_{\text{ref}}$ | CER raw | CER after rules |
|---|---|---|---|---|---|
| `Invoice No. 4417` | `Invoice N0. 4417` | 1 | 16 | $1/16 = 0.062$ | 0.000 |
| `Total: PKR 12,500` | `Total: PKR 12,5O0` | 1 | 17 | $1/17 = 0.059$ | 0.000 |
| `Due 30/09/2026` | `Due 3O/09/2O26` | 2 | 14 | $2/14 = 0.143$ | 0.000 |

Mean CER of the three pairs $= (0.062 + 0.059 + 0.143)/3 = 0.088$; pooled $= 4/47 = 0.085$; both fall to $0.000$.

**Synthetic invoice (EasyOCR).** Reference length 124 characters; CER raw $0/124 = 0.000$, after rules $0.000$; field accuracy over `mobile`, `amount`, `date`, `invoice` $4/4 = 1.00 \to 1.00$. The image is clean rendered text, so it cannot show the benefit of the rules; the expected gain has to be measured on real scans with hand-typed reference text.

**CER alone misleads.** `PKR 4,5O0` has CER $1/9 = 0.111$, yet the amount is wrong (field accuracy for it is 0).

**Limit of the repair rule (checked by running it).** `PKR 4,5OO` (two adjacent letters, CER $2/9 = 0.222$) is returned unchanged, because the second `O` is preceded by an `O`, not a digit. A single `O` (`PKR 4,5O0`) is repaired.

---

## 5. Multimodal pipeline: image model + text evidence -> JSON record

```
 photo ──► Week 6 ResNet18 ──► P(referable_DR) ─────────────┐
                                                             ├─► decision + flag ─► JSON record
 note ──► mask_pii ──► NER + regex + urgent terms ──────────┘
          (CNIC, phone, e-mail, MRN)
```

**Image evidence.** $p_c = e^{z_c}/\sum_k e^{z_k}$; photos are resized to $256 \times 256$ (as in Week 6 caching), then $224 \times 224$. **Operating point** $t = 0.464$, chosen in Week 6 on the *validation* set as the highest threshold with recall $\ge 0.95$. In logit form $z_{\text{ref}} - z_{\text{non}} \ge \ln\frac{0.464}{0.536} = -0.144$: a photo is referred even if the model is slightly more confident of "non-referable". Week 6 test split at this threshold: sensitivity $355/367 = 0.967$, specificity $107/139 = 0.770$.

**Text evidence.** Domain NER entities, regex fields and urgent symptom terms (blurred vision, floaters, dark spots, vision loss, flashes of light, eye pain, ...) matched at the start of a word.

**Decision rule.** With $I = [p_{\text{referable}} \ge t]$ and $U = [\text{urgent terms found}]$:

$$\text{REFER} = I \lor U, \qquad \text{FLAG for human review} = I \oplus U$$

| $I$ | $U$ | Decision | Flag |
|---|---|---|---|
| 0 | 0 | ROUTINE | no |
| 1 | 0 | REFER | yes |
| 0 | 1 | REFER | yes |
| 1 | 1 | REFER | no |

### Example record (invented patient; CNIC, phone and MRN masked before extraction)

```json
{
  "image": "e4f12411fd85.png",
  "image_probabilities": {"non_referable": 0.001, "referable_DR": 0.999},
  "note_masked": "Patient Adeel Hussain, aged 46, type 2 diabetes for 12 years, has had blurred vision for 3 weeks with floaters in the OD and one sudden dark spot on 10/11/2026. Fundus photograph shows DR grade 4. HbA1c 9.4%. Referred from Mayo Hospital Lahore. [MRN]. CNIC [CNIC], family mobile [PHONE].",
  "text_evidence": {
    "entities": [["Adeel Hussain","PERSON"],["46","DATE"],["2","CARDINAL"],["12 years","DATE"],
                 ["3 weeks","DATE"],["OD","ORG"],["one","CARDINAL"],["10/11/2026","DATE"],
                 ["DR","GPE"],["4","CARDINAL"],["HbA1c 9.4%","PERCENT"],["Mayo Hospital","ORG"],
                 ["Lahore","GPE"],["CNIC","ORG"],["CNIC","ORG"]],
    "fields": {"date": ["10/11/2026"], "hba1c": ["9.4"], "eye": ["OD"], "dr_grade": ["4"], "age": ["46"]},
    "urgent_terms": ["blurred vision", "floater", "dark spot"]
  },
  "decision": "REFER",
  "reasons": ["image model: P(referable_DR) = 1.00",
              "note mentions: blurred vision, floater, dark spot"],
  "flags": [],
  "disclaimer": "Decision support only; a qualified human decides."
}
```

**Reading it.** $p = 0.999 \ge 0.464$ (margin $0.535$) so $I = 1$; three urgent terms so $U = 1$; REFER, flag $= 1 \oplus 1 = 0$. No `cnic`, `mobile` or `mrn` field exists because those values were masked first.

### When the sources disagree

| Case | $P(\text{referable\_DR})$ | $I$ | Urgent terms | $U$ | Decision | Flag |
|---|---|---|---|---|---|---|
| A: grade-0 photo + urgent note | 0.055 (below $t$ by 0.409) | 0 | blurred vision, floater, vision loss | 1 | REFER | **yes** |
| B: grade-4 photo + routine note | 0.999 (above $t$ by 0.535) | 1 | none | 0 | REFER | **yes** |

Missing referable retinopathy costs far more than an extra review, so either source alone refers the case and the disagreement is flagged for a human; neither source is silently dropped.

### Entity quality of the record

Against my own gold list for the note (6 entities: person, hospital, city, two durations, one date): spaCy returned 15 entities and found all 6, so $R = 6/6 = 1.00$, $P = 6/15 = 0.40$, $F1 = \frac{2 \times 6}{15 + 6} = 0.571$. Of the 9 extra entities, 5 are harmless number-like labels (`46`, `2`, `one`, `4`, `HbA1c 9.4%`) and 4 are wrong as data (`OD` as `ORG`, `DR` as `GPE`, `CNIC` as `ORG` twice): $4/15 = 26.7\,\%$.

---

## 6. Privacy

`mask_pii` replaces CNIC -> `[CNIC]`, mobile -> `[PHONE]`, e-mail -> `[EMAIL]`, MRN -> `[MRN]` **before** any extraction, logging or JSON output. Reason: extraction copies text into new places (entities, fields, JSON, logs); the field extractor returns the matched text itself (`MRN 4471823`, the full CNIC), so masking afterwards cannot recall what was already copied.

| Check | Result |
|---|---|
| `CNIC 35202-1234567-1, call 0300-1234567, a@b.pk, MRN 4471823` | `CNIC [CNIC], call [PHONE], [EMAIL], [MRN]` (4 of 4 removed) |
| Leak check on the JSON record (full CNIC, phone, MRN and the two digit runs `35202`, `1234567`) | **0 of 5 fragments found** |

**Known gaps (not fixed in this notebook):**

* The patient's **name** is not masked: it remains in `note_masked` and in the entities, so the record is not fully de-identified.
* The label word *CNIC* stays in the text and spaCy tags it as `ORG` twice (see example).
* The leak check only tests the five listed fragments.

---

## Limitations

* **Example photos.** In this run they were taken from the mirror's `test.csv`, not from the Week 6 `splits.csv` test split. Week 6 re-split all images, so they may have been in training, and $0.999$ may be optimistic. Week 6 itself has no patient id and its test split was unbalanced ($367/506 = 72.5\,\%$ referable), so its PPV does not carry to a clinic.
* **Spam.** One split, one seed; the word/character gap is 4 messages. Roman Urdu was never trained on. 5 of the 13 errors are ambiguous or label errors.
* **NER.** 38 gold entities in total; the held-out set barely tests the organisation rules.
* **OCR.** The only full-image test is a clean synthetic invoice; `fix_ocr` does not repair adjacent look-alike letters.

## Reproducing

Run the notebook top to bottom on Kaggle with the inputs above attached (`spam.csv`, APTOS photos, `aptos_model.pt`; `splits.csv` from Week 6 if available, otherwise the mirror's `test.csv` is used and the notebook prints that). Internet is needed for the first EasyOCR run. Seeds: `random_state=42` for the spam split.
