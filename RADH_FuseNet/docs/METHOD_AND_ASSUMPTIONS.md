# RADH-FuseNet: manuscript-derived implementation specification

## Status and provenance

Source: user-supplied `Manuscript_INF(20261011-134341).docx`, titled *Multi-level Reliability Aware Fusion with Dynamic Hypergraph Modeling and Wearable Sensing for Long-Term Driver Affective State Analysis*. Audited on 2026-10-11. Relevant sources are §§3.2–3.7, §4.1, Tables 2–4, Figures 6–11 and Algorithms 1–2. Equation numbers below refer to the manuscript.

The document contains no native OMML equations. Equations are MathType WMF images with embedded legacy glyph positions. Equations were inspected after position-preserving extraction; ordinary Word paragraph extraction omits them. Architecture figures and both algorithm images were also inspected. Minor special-glyph rendering defects were resolved from symbol-font records and the surrounding notation; the method is not inferred solely from blank text extraction.

Official repository: https://github.com/Wang-Qing-sys/RADH_FuseNet . Direct official GitHub HTTP access succeeded, status 200. At inspection the default branch was `main`, commit `bdd1395dc92f756bad11c5ca45bf6091d142afc5`, and the tree contained exactly one file, `README.md`. The official raw README, https://raw.githubusercontent.com/Wang-Qing-sys/RADH_FuseNet/main/README.md , contained only `# RADH_FuseNet`. The repository exists but did not provide source code, configuration, data, checkpoints, or a license in its inspected tree. The API was rate-limited and the web-index fetch failed; these failures are not evidence of repository nonexistence.

**Deliverable classification:** a manuscript-based reference implementation with explicitly documented choices, not recovered official code or a demonstrated reproduction of reported performance. No real dataset was supplied with the manuscript.

## 1. Input, targets and source roles

- Synchronized analysis window: five separately encoded streams, ordered PPG, PPS, RSP, EHS, RHS. Each has T samples plus sample/source availability masks. No raw input concatenation across sources.
- PPG/PPS are independent cardiovascular measurements; RSP is respiration; EHS/RHS supply environmental/respiratory humidity context. The manuscript alternates between emotional/electrocardiac pulse terminology for PPG and respiratory/relative humidity terminology for RHS.
- Classification has C=5: activated, deactivated, unpleasant, pleasant, neutral. The manuscript's class numbers are 1–5; a zero-based implementation must declare its mapping.
- Three physiological class heads (PPG/PPS/RSP), plus one global head. EHS/RHS must not be independent affective voters, although their representations enter the contextual hypergraph.
- Continuous output: stress-related affective trajectory for H future steps, associated predictive variance and Student-t predictive intervals. H, target units, target-generation procedure and forecast timestep are not supplied.
- Equations 1, 3 and 4 give the joint mapping to class probabilities p, uncertainty u, future trajectory y and variance. Equation 2: `x_obs_m(t) = a_m(t)[x_m(t)+n_m(t)+delta_m(t)]`. The prose additionally describes temporal shift, which is not explicitly present in Eq. 2.

## 2. Signal correction and quality (§3.3)

Preprocessing described: common time axis, anti-alias resampling, waveform-preserving pulse filtering plus wavelet denoising, slow baseline removal for humidity, local landmark cross-correlation for alignment, robust normalization, explicit short-gap masks, and no synthetic low-frequency filling of long gaps. Filter families/orders/cutoffs, alignment limits, gap thresholds and normalization fit/application protocol are not specified.

Exact humidity equations:

- (5) `beta_t = sigmoid(g_beta([c_RHS_t, c_EHS_t, Delta c_RHS_t, Delta c_EHS_t]))`.
- (6) `x_tilde_RSP_t = x_RHS_t - beta_t*x_EHS_t`.

The `c` variables are context descriptors; prose mentions local values, first differences and robust statistics but does not give their exact vector construction. Beta is bounded in (0,1). Prose requires a coefficient total-variation penalty and a respiration-preservation term, with no exact preservation formula. Corrected RSP is robust-standardized before encoding. Eq. 6 constructs respiration from the humidity pair, whereas Algorithm 1's `CorrectRSP(raw_RSP; rho_h)` wording is less precise. Handling absent EHS or RHS must be declared; using the original RSP as a fallback is an implementation choice, not a supplied equation. Compensation is disabled for WESAD.

Quality descriptor per source/window includes robust amplitude range, short-time energy, kurtosis, spectral concentration, sample-jump rate, saturation ratio, missing-sample ratio, autocorrelation consistency and expected-band energy. Pulse sources add peak-interval regularity and contour continuity; humidity adds baseline drift and residual coupling. No exact descriptor formulas, bands or thresholds are supplied.

- (7) `q_m,t = sigmoid(w_q^T phi_m,t + b_q)`.

SQI is technical waveform quality, not final learned reliability. Must retain masks in descriptor extraction and prevent missing placeholders from appearing clean.

## 3. Source encoders and shared/private factors (§3.4)

Independent parameters per source, shared topology. Stem: 1-D Conv(kernel 3, width 96) then layer normalization. Three same-width/same-temporal-resolution branches:

1. Local morphology: two residual blocks, depthwise kernels 3,7,15, pointwise mixing; residual and average-pooling paths discussed.
   - (8) `zL_m = LN(z0_m + W_L[DWConv3(z0_m) || DWConv7(z0_m) || DWConv15(z0_m)])`.
2. Spectral: 64-point STFT **or** CWT; learned complex convolution/gate and two residual blocks. Source-specific informative frequency gating, quality conditioning described.
   - (9) `ZF_m = F(z0_m)`.
   - (10) `zF_m = F^{-1}(G_m ⊙ ZF_m)`.
   - (11) `G_m = sigmoid(WF_m)`.
   No STFT hop/window/padding/centering, CWT wavelet/scales, exact complex kernels or quality-gate connection are supplied.
3. Selective state space: four blocks, state size 16, expansion 2, depthwise temporal convolution before causal selective scan.
   - (12) `h_m,t = Abar_m,t*h_m,t-1 + Bbar_m,t*x_m,t`.
   - (13) `zS_m,t = C_m,t*h_m,t + D_m,t*x_m,t`.
   No A/B/C/D parameterization, continuous-to-discrete rule, delta network, initialization or convolution kernel is supplied. A diagonal input-dependent recurrence is a defensible explicit implementation choice; the manuscript does not specify a particular Mamba release.

Adaptive branch fusion:
- (14) `alpha_m = softmax(g_m([Pool(zL_m), Pool(zF_m), Pool(zS_m)]))`.
- (15) `z_m = alpha_m,L*zL_m + alpha_m,F*zF_m + alpha_m,S*zS_m`.
The prose additionally puts current SQI into the gating network, although Eq. 14 omits it. It also later claims learned reliability controls branch gating, but reliability is estimated after branch fusion; no feedback or second pass is defined. SQI-conditioned gating plus downstream reliability is the clean acyclic interpretation and must be documented.

Shared/private decomposition:
- (16) `s_m = Ws_m*z_m`, shared width 64.
- (17) `p_m = Wp_m*z_m`, private width 32.
- (18) `xhat_m = D_m([s_m || p_m])`.
- (19) `L_ssp = sum_m ||x_m-xhat_m||_1 + lambda_orth sum_m ||s_m^T p_m||_F^2`.

Use L1 reconstruction if following Eq. 19, with masks in a concrete implementation. Pooling axes, decoder architecture and scaling/normalization of orthogonality are unspecified. Contrastive consistency of synchronized clean shared factors is described, with no exact loss or temperature.

Reliability:
- (20) `r_m,t = sigmoid(f_rho([q_m,t, e_m,t, a_m,t, cos(s_m,t,s_-m,t)]))`.
- `e` is reconstruction residual; `s_-m` is a consensus formed from remaining sources. Consensus weighting and residual reduction are unspecified.
- Clean/mild/severe corruption provides reliability ranking; source prediction correctness provides calibration; adjacent-window EMA is mentioned without a coefficient or state/reset protocol.
- Availability must force unavailable incidences/evidence to zero. Eq. 20 alone only supplies a as a network input and does not mathematically force r=0; multiplying by availability is a necessary explicit operational choice.
- Ranking supervision encourages reliability to decrease with corruption. A generic sigmoid MLP does not mathematically guarantee the manuscript's claimed monotonicity for every input.

## 4. Reliability-controlled dynamic hypergraph (§3.5)

- (21) `v_m,k = Pool(s_m[(k-1)P+1:kP])`, P=32; K=T/P. Shared factors, source identity and temporal patch position define nodes. Eq. 21 uses only s; Algorithm 1 writes `Patch({s_m,p_m})`, an inconsistency.
- Priors: cardiovascular `{PPG,PPS}`, respiratory `{RSP}`, environmental `{EHS,RHS}` and a global edge within a temporal region. Figure 9 explicitly shows source sets `{1,2}`, `{3}`, `{4,5}`.
- Adaptive top-k=2 semantic edges from same or neighboring patches. Search radius, self-inclusion, similarity transformation and exact edge indexing are not specified.
- (22) `H_v,e = A_prior_v,e*r_v*exp(-||s_v-c_e||_2^2/tau) + A_adapt_v,e*r_v*TopKSim(s_v,c_e)`.
- (23) `c_e = sum_(v in e) a_v,e*W_v*s_v`.
- (24) `a_v,e = softmax_v((Wq*s_v)^T*(Wk*c_e)/sqrt(d) + lambda_r*log(r_v+epsilon))`.
- (25) `m_i^l = r_i sum_(e containing i) sum_(j in e) [b_i,e*a_j,e/sqrt(d_i*d_j)]*W_e*h_j^l`.
- (26) `h_i^(l+1) = GRU/GAT(h_i^l,m_i^l)`.

Two hypergraph layers, adaptive memberships recomputed after each. Low-reliability receiving nodes rely more on the local residual path. Definition/normalization of reverse coefficient b, degrees d, tau, lambda_r, edge weights and which GRU/GAT variant is chosen are absent. Eq. 23/24 also requires an initial edge centroid because the attention depends on c_e while c_e depends on attention; initializing c_e by member pooling is a reasonable disclosed choice. Dynamic top-k is piecewise differentiable, not a guaranteed end-to-end differentiable discrete selector.

Algorithm 1 pools modality nodes for each k into one fused temporal token. Exact source-pooling rule is not supplied. Empty edges/all-missing inputs require explicit safe handling.

## 5. Cross-scale temporal fusion (§3.6)

- (27) `H^(s) = SSM_s(Down_s(Z_fused))`, s in `{1,2,4}`.
- Scales original/half/quarter resolution; block depths 1,2,6, respectively. Causal temporal paths.
- (28) softmax gate g_omega over pooled scale features. The printed equation erroneously lists H1,H2,H3,H4; Eq. 27, Eq. 29, Figure 10 and Table 2 agree on only scales 1,2,4.
- (29) `H_temp = sum_(s in {1,2,4}) omega_s*Up_s(H^(s))`.

Prose adds mean reliability and recent variation to the context gate. Down/upsampling operators and those statistics are unspecified. A claimed causal implementation must avoid coarse buckets incorporating future samples relative to the output timestamp; merely using causal SSM blocks does not make arbitrary averaging/interpolation or a whole-sequence context gate causal. Window-level predictions at the end of the input window are easier to interpret honestly than sample-level streaming claims.

## 6. Evidential classification and regression (§3.6)

Each physiological head and global head:
- (30) `e_c^m = softplus(o_c^m)`.
- (31) `alpha_c^m = e_c^m + 1`.
- `S_m = sum_c alpha_c^m`.
- (32) `b_c^m = e_c^m/S_m`.
- (33) `u_m = C/S_m`.
- (34) `bbar_c^m = r_m*b_c^m`.
- (35) `ubar_m = 1-r_m*(1-u_m)`.

Thus absent sources should be vacuous opinions b=0,u=1. The global head's discount/reliability definition is not specified. Figure 11 instead shows alpha interpolation, `alpha_bar=r*alpha+(1-r)*alpha_ignorance`, which is not generally equivalent to Eqs. 34–35; prioritize the numbered equations and prose's mass transfer to uncertainty.

For two already-discounted opinions A,B:
- (36) `K(A,B) = sum_(i != j) bbar_i^A*bbar_j^B`.
- (37) `btilde_c = bbar_c^A*bbar_c^B + bbar_c^A*ubar_B + bbar_c^B*ubar_A`.
- (38) `utilde = ubar_A*ubar_B + eta(K)*K`.

Low-conflict prose requires normalized conjunctive combination; high conflict transfers some conflict to uncertainty. Eta(K), final normalization and multiclass/head combination order are not supplied. A defensible explicit choice is eta=0 below threshold and eta=1 above (Dempster/Yager switch), then normalize btilde/utilde to unit total. Setting eta=0 gives denominator 1-K; eta=1 gives denominator 1. Threshold selection and a safe K≈1 fallback need documentation. Multi-opinion order matters for a threshold-based rule; specify deterministic/reliability-based order. Recover class probability as Dirichlet mean, equivalently `p=b+u/C` for a normalized opinion. Conversion back to alpha needs protection when u≈0.

Regression head gives gamma,nu,alpha,beta per horizon, with positive nu/beta and alpha>1 for finite predictive variance:
- (39) `p(y|gamma,nu,alpha,beta) = Student-t(gamma, 2*alpha, beta*(1+nu)/(nu*alpha))`.
The final argument is not labelled as scale or squared scale. The standard NIG marginal uses it as squared Student-t scale; implement `scale=sqrt(beta*(1+nu)/(nu*alpha))` and disclose this convention. Predictive variance is `beta*(1+nu)/(nu*(alpha-1))`. Credible/predictive intervals must use Student-t quantiles, not unlabelled Gaussian approximations.

## 7. Training and configuration (§3.7, Algorithms 1–2)

Stage A: random masked-span reconstruction and clean/mild/severe corruption views; `L_A=L_rec+lambda_rank*L_rank`; select best validation encoder checkpoint.

Stage B: initialize from Stage A, supervised full-network optimization with modality dropout and Gaussian noise, impulsive spikes, contiguous dropout, baseline drift, temporal shift and humidity variation. Exact distributions/severities are absent.

(40) `L=L_EDL + lambda_reg*L_reg + lambda_rec*L_rec + lambda_orth*L_orth + lambda_con*L_con + lambda_rel*L_rel + lambda_cal*L_cal + lambda_tv*L_tv`.

Only Eq. 19 defines component loss forms. EDL classification, regression NLL/evidence regularizer, contrastive loss, reliability rank, calibration, coefficient/temporal TV and respiratory preservation are not mathematically defined. Common evidential digamma CE+wrong-class KL, Student-t NLL+evidence penalty, InfoNCE, hinge ranking, BCE correctness calibration and masked TV may be selected, but are implementation assumptions.

Fixed optimizer settings: AdamW, lr 0.0002, weight decay 0.01, cosine decay, five warm-up epochs, gradient-norm clip 1.0, dropout 0.1. Curriculum described qualitatively: reconstruction/reliability early, task/evidence/calibration later. No epoch counts, batch size, loss weights, exact schedule, seeds or checkpoint tie-breaker.

Validation search (not all default values): d `{64,96,128}`; P `{8,16,32,64}`; source SSM depth `{2,4,6}`; graph layers `{1,2,3}`; top-k `{1,2,3}`; modality dropout `{.1,.2,.3}`; conflict threshold `{.2,.4,.6}`; loss weights log scale `.01–1`.

Validation criteria include macro-F1, MAE, ECE and risk-coverage area. Algorithm 2 says final test once per seed/split with locked hyperparameters. No particular multimetric selection formula is supplied.

## 8. Data protocol and evaluation limitations

Dataset A: 300 drivers, more than 300,000 unspecified physiological data points; stationary vehicle with engine running, ordinary roads, viaducts, tunnels. Five synchronized sources. Driver-disjoint train/validation/test. **No raw data, sampling rates, durations, subject IDs, splits/ratios, window size, stride/overlap, class counts, annotation procedure, continuous target units or files are provided.** No data-access URL or reproducible manifest is in the method.

Dataset B/WESAD: 15 subjects; ECG or BVP, EDA, respiration, temperature and motion as available; no humidity correction. Table 4 says “LOSO or subject separated,” leaving the exact protocol unresolved. Sensor selection, resampling rate, label mapping and continuous-regression target are not given. Do not invent a mapping from WESAD conditions to the five Dataset A affect classes, and do not invent continuous targets. A separate explicit WESAD configuration/adapter can have its own justified task definition.

The text says training and validation statistics are computed from their corresponding partitions, without explaining whether validation normalization is fitted independently or transformed using training fit. For leakage-safe standard evaluation, fit global preprocessors on training only and apply to validation/test; per-window normalization may use only that window and must be declared. Split by subject before creating overlapping windows and keep sessions/chronology intact for forecasts.

Metrics: accuracy, precision, recall, macro-F1 and balanced accuracy. Continuous predictions/calibration described through MAE, expected calibration error, risk-coverage and uncertainty/error association; not all formulas/bins/reductions are supplied. Robustness includes missing source(s), Gaussian/impulsive corruption, shifts up to 30 s and relative humidity shifts, but units/severity schedules/source selection/random draws are insufficiently specified to recreate experiments.

**Reported-score inconsistencies:** §4.3 prose says Dataset A accuracy 96.2%, F1 95.8%; Table 5 says 95.2%, 94.7%; Table 6's five per-class F1 values average 91.8% (printed macro 0.92); Table 10 all-sources F1 is 98.2%. Distinct splits or run provenance are not supplied. WESAD Table 5 reports 90.0% F1. None of these are verified results of a new implementation and none should be promised.

## 9. Required reference-implementation disclosures and checks

1. Label every chosen window length, sampling rate, horizon, batch size, epochs, split ratios, seeds, conflict rule/threshold, loss formula/weight, corruption severity and unavailable-source fallback as a configurable engineering default when absent from the manuscript.
2. Distinguish smoke-test/generated fixture data from real sensor data. Never describe synthetic test performance as paper reproduction.
3. Never load missing values as trustworthy zero signals. Mask preprocessing, reconstruction, source availability, graph incidence and modality evidence. Specify behavior when all physiological sources or all sources are missing; the safe final opinion is uniform with maximum ignorance when no evidence is available.
4. Test finite values/gradients, class probability sums, opinion mass conservation, positive NIG parameters, batch size 1, non-divisible T/P, CPU forward/backward, missing channels/all-missing, save/load, training-to-evaluation normalization and subject split disjointness.
5. If pure-PyTorch SSM scan, dense hypergraph bookkeeping, STFT implementation, gate conditioning, pooling, no sequence EMA, or simplified preprocessing differs from the manuscript's prose, record the difference. Do not claim exact sparse complexity, streaming causality, embedded-device latency or clinical/driving safety validation without tests.
6. Request actual dataset/schema and annotation/split details before claiming empirical reproducibility. The package can be complete and runnable while that scientific result remains unestablished.

## 10. 本代码包的落实口径

- `configs/paper.json` 中可直接由 Table 2 支持的结构参数与缺失的工程参数分开理解；文件名不表示已复现论文成绩。
- STFT 路径选择论文允许的 STFT，而不是同时实现另一个 CWT 版本。
- 选择性 SSM 是可移植 PyTorch 的显式对角状态更新，不是作者未公开的软件或某一官方 Mamba 版本。
- 冲突函数选择阈值分段的 Dempster/Yager 组合；这补足了文中未定义的 eta(K)，不能当作作者公开的准确设置。
- 归一化以训练集拟合值为准；验证集和测试集不另拟合全局统计量。可选窗口内稳健标准化与全局训练归一化的作用不同。
- 不通过随机 batch 维护跨样本的可靠性 EMA。在线相邻窗口状态需要调用者显式传递并在驾驶员/会话边界重置。
- 当前数据预处理是掩码感知、抗混叠重采样及显式可选滤波/对齐工具；不能声称重现了论文未公开的波形去噪参数。未经提供的波形小波阈值及标签制作流程不会被当作已知事实。
- 本包不含稿件私有数据、作者训练权重、论文表格的重建数字或各个比较模型的独立复现。
- 实际编码器采用两次轻量分支门控：先以 SQI 得到临时共享特征和重建残差，估计可靠性后再次门控已计算的三个分支。此选择使“可靠性控制分支”可执行且无循环方程，但两次门控不是稿件明确指定的步骤。
- 湿度补偿仅在原始 RSP、EHS、RHS 对应样点均有效时应用 Eq.6；缺湿度时保留有效的原 RSP。环境数据不能让缺失生理通道变成“可用”。
- 没有生理通道时，分类输出被折扣为均匀意见；连续预测同时提供有效性标记，不把无生理观测的回归数值作为有效预测。
