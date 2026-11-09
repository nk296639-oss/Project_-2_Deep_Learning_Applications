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
