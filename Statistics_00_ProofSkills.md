# 수리통계학 식 전개 패턴 총정리

## 랜덤표본 분포 변환

### 균등분포에서 지수분포로
$X_i \sim U(0,1)$ 독립이면, $Y_i = -\theta \log X_i \sim \text{Exp}(\theta)$

따라서 $\sum Y_i \sim \text{Gamma}(n, \theta)$

### 균등분포에서 카이제곱분포로
$Z_i = -\log X_i \sim \text{Exp}(1)$ (표준 지수분포)

$-2\sum_{i=1}^n \log X_i = 2\sum Z_i \sim \chi^2(2n)$

### 베타-감마 관계
$U \sim \text{Beta}(\alpha, \beta)$이면, $U = \frac{Y_1}{Y_1+Y_2}$로 표현 가능  
(단, $Y_1 \sim \text{Gamma}(\alpha,\theta)$, $Y_2 \sim \text{Gamma}(\beta,\theta)$ 독립)

특히 $\text{Beta}(\theta, 1)$의 경우, $X \sim \text{Beta}(\theta,1)$ ⟹ $-\log X \sim \text{Exp}(1)$

### F-분포와의 연결

$$F = \frac{\chi^2_{2m}/2m}{\chi^2_{2n}/2n} = \frac{Y_1/m}{Y_2/n}$$

(단, $Y_1 \sim \chi^2_{2m}$, $Y_2 \sim \chi^2_{2n}$ 독립)


### 증명
$X_i \sim U(0,1)$ 독립이면, $Y_i = -\theta \log X_i$는 $Y_i \sim \text{Exp}(\theta)$가 된다.

$$P(Y_i \le y) = P(-\theta \log X_i \le y) = P(X_i \ge e^{-y/\theta}) = 1 - e^{-y/\theta}$$

따라서 $Y_i$의 확률밀도함수(pdf)는

$$f_{Y_i}(y) = \frac{d}{dy}P(Y_i \le y) = \frac{1}{\theta} e^{-y/\theta}$$

즉, $Y_i \sim \text{Exp}(\theta)$가 된다.

이때 gamma 분포의 정의에 따라, $Y_i$의 합인 $\sum Y_i$는 $\text{Gamma}(n, \theta)$이 된다. 따라서

$$-\theta \log\sum X_i = \sum Y_i \sim \text{Gamma}(n, \theta)$$

$-2\sum \log X_i$의 $\chi^2$ 분포 유도: 앞에서 $Y_i=-\theta\log X_i\sim \mathrm{Exp}(\theta)$ (scale $\theta$) 이므로

$$S:=\sum_{i=1}^n Y_i \sim \mathrm{Gamma}(n,\theta).$$

감마분포의 스케일 성질 $cS\sim \mathrm{Gamma}(n,c\theta)$ ($c>0$)를 쓰면

$$2S=-2\theta\sum_{i=1}^n\log X_i \sim \mathrm{Gamma}(n,2\theta).$$

여기서 $\chi^2$와의 정확한 연결은 다음과 같다.

$$\chi^2(\nu)\equiv \mathrm{Gamma}\!\left(\frac{\nu}{2},\,2\right).$$

따라서 $\mathrm{Gamma}(n,2\theta)$가 $\chi^2(2n)$와 **동일**하려면 $\theta=1$이어야 한다.  
일반 $\theta$에 대해 $\chi^2$ 피벗은 $\theta$로 나눈 형태이다:

$$\frac{2S}{\theta}=\frac{-2\theta\sum \log X_i}{\theta}=-2\sum_{i=1}^n \log X_i.$$

그런데 $Z_i:=-\log X_i\sim \mathrm{Exp}(1)$, 독립이므로

$$\sum_{i=1}^n Z_i \sim \mathrm{Gamma}(n,1)\quad\Rightarrow\quad 2\sum_{i=1}^n Z_i \sim \mathrm{Gamma}(n,2)=\chi^2(2n).$$

즉 최종적으로

$$-2\sum_{i=1}^n \log X_i \sim \chi^2(2n),$$

그리고 동치로

$$-2\theta\sum_{i=1}^n \log X_i \sim \mathrm{Gamma}(n,2\theta).$$

**베타-감마 관계 증명**: $Y_1 \sim \text{Gamma}(\alpha,\theta)$, $Y_2 \sim \text{Gamma}(\beta,\theta)$ 독립이면, $U = \frac{Y_1}{Y_1+Y_2}$는 $\text{Beta}(\alpha,\beta)$를 따른다.

$Y_1$과 $Y_2$의 결합확률밀도함수(pdf)는 

$$f_{Y_1,Y_2}(y_1,y_2) = \frac{1}{\Gamma(\alpha)\theta^\alpha} y_1^{\alpha-1} e^{-y_1/\theta} \cdot \frac{1}{\Gamma(\beta)\theta^\beta} y_2^{\beta-1} e^{-y_2/\theta}$$

$$= \frac{1}{\Gamma(\alpha)\Gamma(\beta)\theta^{\alpha+\beta}} y_1^{\alpha-1} y_2^{\beta-1} e^{-(y_1+y_2)/\theta}$$

$U = \frac{Y_1}{Y_1+Y_2}$와 $V = Y_1 + Y_2$로 변수변환을 하면, 역변환은 $Y_1 = UV$, $Y_2 = (1-U)V$가 된다. 이때 야코비안은

$$J = \begin{vmatrix}\frac{\partial Y_1}{\partial U} & \frac{\partial Y_1}{\partial V} \\ \frac{\partial Y_2}{\partial U} & \frac{\partial Y_2}{\partial V}\end{vmatrix} = \begin{vmatrix}V & U \\ -V & 1-U\end{vmatrix} = V$$

따라서 $U$의 확률밀도함수(pdf)는 다음과 같이 계산된다.

$$f_U(u) = \int_0^\infty f_{Y_1,Y_2}(uv,(1-u)v) \cdot J \, dv$$

$$= \int_0^\infty \frac{1}{\Gamma(\alpha)\Gamma(\beta)\theta^{\alpha+\beta}} (uv)^{\alpha-1} ((1-u)v)^{\beta-1} e^{-v/\theta} \cdot v \, dv$$

$$= \frac{u^{\alpha-1}(1-u)^{\beta-1}}{\Gamma(\alpha)\Gamma(\beta)\theta^{\alpha+\beta}} \int_0^\infty v^{\alpha+\beta-1} e^{-v/\theta} dv$$

$$= \frac{u^{\alpha-1}(1-u)^{\beta-1}}{\Gamma(\alpha)\Gamma(\beta)\theta^{\alpha+\beta}} \cdot \Gamma(\alpha+\beta) \theta^{\alpha+\beta}$$

$$= \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} u^{\alpha-1}(1-u)^{\beta-1}$$

따라서 $U$는 $\text{Beta}(\alpha,\beta)$를 따른다. 특히 $\text{Beta}(\theta,1)$의 경우, $X \sim \text{Beta}(\theta,1)$이면 $-\log X \sim \text{Exp}(1)$이 된다. 왜냐하면 $\text{Beta}(\theta,1)$의 확률밀도함수(pdf)는

$$f_X(x) = \frac{\Gamma(\theta+1)}{\Gamma(\theta)\Gamma(1)} x^{\theta-1}(1-x)^{0} = \theta x^{\theta-1}$$

따라서 $Y = -\log X$의 확률밀도함수(pdf)는 다음과 같이 계산된다.

$$f_Y(y) = f_X(e^{-y}) \cdot \left| \frac{d}{dy} e^{-y} \right| = \theta e^{-\theta y}$$

즉 $Y$는 $\text{Exp}(1)$를 따른다.

**F-분포와의 연결 증명**: $Y_1 \sim \chi^2_{2m}$, $Y_2 \sim \chi^2_{2n}$ 독립이면, $F = \frac{Y_1/m}{Y_2/n}$는 $F(m,n)$을 따른다.

$Y_1$과 $Y_2$의 확률밀도함수(pdf)는 각각

$$f_{Y_1}(y_1) = \frac{1}{2^m \Gamma(m)} y_1^{m-1} e^{-y_1/2}$$

$$f_{Y_2}(y_2) = \frac{1}{2^n \Gamma(n)} y_2^{n-1} e^{-y_2/2}$$

따라서 $F = \frac{Y_1/m}{Y_2/n}$와 $V = Y_2$로 변수변환을 하면, 역변환은 $Y_1 = mF \cdot \frac{V}{n}$, $Y_2 = V$가 된다. 이때 야코비안은

$$J = \begin{vmatrix}\frac{\partial Y_1}{\partial F} & \frac{\partial Y_1}{\partial V} \\ \frac{\partial Y_2}{\partial F} & \frac{\partial Y_2}{\partial V}\end{vmatrix} = \begin{vmatrix}\frac{mV}{n} & \frac{mF}{n} \\ 0 & 1\end{vmatrix} = \frac{mV}{n}$$

따라서 $F$의 확률밀도함수(pdf)는 다음과 같이 계산된다.

$$f_F(f) = \int_0^\infty f_{Y_1,Y_2}\left(\frac{mV}{n}f, V\right) \cdot J \, dV$$

$$= \int_0^\infty \frac{1}{2^m \Gamma(m)} \left(\frac{mV}{n}f\right)^{m-1} e^{-\frac{mV}{2n}f} \cdot \frac{1}{2^n \Gamma(n)} V^{n-1} e^{-V/2} \cdot \frac{mV}{n} \, dV$$

$$= \frac{m^m f^{m-1}}{n^m 2^{m+n} \Gamma(m) \Gamma(n)} \int_0^\infty V^{m+n-1} e^{-\frac{V}{2}\left(1+\frac{mf}{n}\right)} dV$$

$$= \frac{m^m f^{m-1}}{n^m 2^{m+n} \Gamma(m) \Gamma(n)} \cdot \Gamma(m+n) \left(\frac{2}{1+\frac{mf}{n}}\right)^{m+n}$$

$$= \frac{\Gamma(m+n)}{\Gamma(m)\Gamma(n)} \left(\frac{m}{n}\right)^m \frac{f^{m-1}}{\left(1+\frac{mf}{n}\right)^{m+n}}$$

따라서 $F$는 $F(m,n)$을 따른다.

## 혼합분포 (Mixture Distribution)

### 기본 정의

확률변수 $X$가 혼합분포를 따를 때:

$$X \sim (1-\epsilon)F_1 + \epsilon F_2$$

여기서 $\epsilon \in [0,1]$은 혼합 비율(mixing weight)이다.

### 평균 (Mean)

$$E[X] = (1-\epsilon)\mu_1 + \epsilon\mu_2$$

**증명**: 전확률 공식(law of total probability)

$$E[X] = E[E[X|Z]] = P(Z=1)E[X|Z=1] + P(Z=2)E[X|Z=2] = (1-\epsilon)\mu_1 + \epsilon\mu_2$$

(단, $Z$는 어느 분포에서 샘플링할지 결정하는 지시변수)

### 분산 (Variance)

$$\text{Var}(X) = (1-\epsilon)\text{Var}(X|F_1) + \epsilon\text{Var}(X|F_2) + (1-\epsilon)\epsilon(\mu_1-\mu_2)^2$$

**증명**: 조건부 분산 공식

$$\text{Var}(X) = E[\text{Var}(X|Z)] + \text{Var}(E[X|Z])$$

첫 번째 항:
$$E[\text{Var}(X|Z)] = (1-\epsilon)\sigma_1^2 + \epsilon\sigma_2^2$$

두 번째 항:
$$\text{Var}(E[X|Z]) = \text{Var}((1-\epsilon)\mu_1 + \epsilon\mu_2)$$
$$= (1-\epsilon)\epsilon(\mu_1-\mu_2)^2$$

따라서 합하면 위 공식을 얻는다.

### 해석

- **첫 두 항**: 각 성분 내 분산 (within-component variance)
- **세 번째 항**: 성분 간 평균 차이로 인한 추가 분산 (between-component variance)

### 활용

- Outlier/anomaly detection: $\epsilon$가 작을 때, 컨섯 $F_2$로 이상치 모델링
- EM 알고리즘의 기초 구조
- Robust 통계: 오염된 분포 모델


## 정규분포 관련

### 표준정규분포의 주요 모멘트

$Z \sim N(0,1)$일 때, 표준정규분포의 확률밀도함수는

$$\phi(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2/2}$$

**1차 모멘트 (평균)**

$$E[Z] = \int_{-\infty}^{\infty} z \phi(z) dz = 0$$

(대칭성: $z\phi(z)$는 홀함수)

**2차 모멘트**

$$E[Z^2] = \int_{-\infty}^{\infty} z^2 \phi(z) dz = 1$$

표준정규분포의 분산은 1. 그런데 평균이 0이므로 $E[Z^2] = 1$.

따라서 $\text{Var}(Z) = E[Z^2] - (E[Z])^2 = 1$

**4차 모멘트**

$$E[Z^4] = \int_{-\infty}^{\infty} z^4 \phi(z) dz = 3$$

### 일반정규분포로의 확장

$X \sim N(\mu, \sigma^2)$이면, $Z = \frac{X-\mu}{\sigma}$로 표준화하여

$$E[X] = \mu, \quad \text{Var}(X) = \sigma^2$$

$$E[(X-\mu)^4] = 3\sigma^4$$

### 활용

- 표본모멘트와 모수의 MME (Method of Moments Estimator)
- 정규성 검정 (kurtosis = 3)
- 이차형식 $\sum Z_i^2 \sim \chi^2(n)$ 유도의 기초


## 1. 평균 중심화 (Mean-Centering Trick)

가장 많이 등장하는 전개이다.

$$x_i=(x_i-\bar{x})+\bar{x}$$

이를 이용하면 $\sum (x_i-\bar{x})=0$ 가 성립한다.

### 대표 활용

* 공분산 전개
* 표본분산 공식
* 회귀식 전개
* ANOVA 분해

### 예

$$\sum (x_i-\bar{x})(y_i-\bar{y}) 
= \sum (x_i y_i - x_i \bar{y} - \bar{x} y_i + \bar{x}\bar{y}) = \sum x_i y_i - n\bar{x}\bar{y}$$

## 5. 합을 평균으로 바꾸기 (Sum-Mean Conversion)

수리통계에서 거의 항상 등장한다.

$$\sum x_i = n\bar{x}$$

$$\sum (x_i-\bar{x}) = 0$$

### 사용

* 회귀식
* 공분산
* MLE
* 분산 계산

## 6. 공분산 전개 공식 (Covariance Expansion)

$$\text{Cov}(X,Y) = E[XY]-E[X]E[Y]$$

표본 버전

$$\sum (x_i-\bar{x})(y_i-\bar{y}) = \sum x_i y_i - n\bar{x}\bar{y}$$

### 예
표본 상관계수 계산 시 필수

## 7. 분산 전개 공식 (Variance Expansion)

$$\text{Var}(X) = E[X^2]-E[X]^2$$

표본 버전

$$\sum (x_i-\bar{x})^2 = \sum x_i^2 - n\bar{x}^2$$

### 예
표본분산: $s^2 = \frac{1}{n}\sum x_i^2 - \bar{x}^2$

## 8. 이차형식 전개 (Quadratic Form Expansion)

다변량 통계에서 자주 등장한다.

$$\mathbf{x}'A\mathbf{x} = \sum_i\sum_j a_{ij}x_ix_j$$

### 활용

* 다변량 정규분포
* Wishart 분포
* 회귀분석

## 9. 직교분해 (Orthogonal Decomposition)

정규분포에서 매우 중요한 구조이다.

$$|\mathbf{x}|^2 = |P\mathbf{x}|^2 + |(I-P)\mathbf{x}|^2$$

### 사용

* 카이제곱 분포
* 회귀분석
* ANOVA

## 10. Projection Matrix Trick

회귀분석 핵심 공식

$$H=X(X'X)^{-1}X'$$

성질

$$H^2=H$$

$$(I-H)^2=I-H$$

### 활용

* residual 분산
* 영향점 분석
* F-test

책 문맥(가중 제곱합 → 투영행렬 → χ² 분포)과 정확히 일치하도록 **노트 형태로 정리**한다. 불필요한 일반화 없이, 해당 풀이에서 실제로 쓰이는 구조만 남긴다.

## Quadratic Form Decomposition 
아래 식이 성립한다.

$$\frac{n(\hat\theta^0-\theta^0)^2}{\theta^0} = \frac{(w^T Z)^2}{w^T w}$$

이때 

$$\hat\theta = (\hat\theta_1,\dots,\hat\theta_k)^T, \quad \hat\theta^0 = \frac{\sum_{i=1}^k n_i \hat\theta_i}{\sum_{i=1}^k n_i}, \quad w = (\sqrt{n_1},\dots,\sqrt{n_k})^T,\quad n = \sum_{i=1}^k n_i$$

**표준화 변수 정의:**

$$Z_i = \frac{\sqrt{n_i}(\hat\theta_i-\theta^0)}{\sqrt{\theta^0}},
\quad Z = (Z_1,\dots,Z_k)^T$$

**핵심 분해 (가중 제곱합)**

$$\sum_{i=1}^k n_i(\hat\theta_i-\hat\theta_i^0)^2
= \sum_{i=1}^k n_i(\hat\theta_i-\theta^0)^2 - n(\hat\theta^0-\theta^0)^2$$

* 좌변: "각 그룹별 편차"
* 우변: "전체 편차 − 평균 방향 성분"

**벡터 형태로 변환**  
(1) 전체 제곱합

$$\frac{1}{\theta^0} \sum_{i=1}^k n_i(\hat\theta_i-\theta^0)^2
= Z^T Z$$

(2) 평균 방향 성분

$$\hat\theta^0-\theta^0
= \frac{1}{n}\sum n_i(\hat\theta_i-\theta^0)
= \frac{\sqrt{\theta^0}}{n} w^T Z \\
\therefore \frac{n(\hat\theta^0-\theta^0)^2}{\theta^0} = \frac{(w^T Z)^2}{w^T w}$$

$$\therefore \frac{1}{\theta^0}
\sum_{i=1}^k n_i(\hat\theta_i-\hat\theta_i^0)^2
= Z^T Z - \frac{(w^T Z)^2}{w^T w} 
=Z^T\left(I - \frac{w w^T}{w^T w}\right)Z$$

**해석 (핵심 구조)**

$$P_w = \frac{w w^T}{w^T w}$$

* $P_w$: $(w)$ 방향으로의 직교투영
* $(I - P_w)$: 그 직교보공간으로의 투영

따라서 $Z^T(I - P_w)Z$ 는

> "전체 벡터 (Z)에서 평균 방향((w))을 제거한 잔차 제곱합"

왜 자유도가 (k-1)인가?
* $Z \sim N(0, I_k)$
* $A = I - \frac{w w^T}{w^T w}$

성질:
* $A^2 = A$ (idempotent)
* $\text{rank}(A)=k-1$

따라서 $Z^T A Z \sim \chi^2(k-1)$  
* $w=(\sqrt{n_1},\dots,\sqrt{n_k})$  → "가중 평균 방향"
* 제거되는 성분 → "공통 평균 이동"
* 남는 성분 → "집단 간 차이"

이 구조는 ANOVA의 "between vs within decomposition"과 완전히 동일한 선형대수 표현이다.
즉 이 식을 이해하면 이후 LRT, score test, Wald test에서 등장하는 모든 χ² 구조를 거의 동일한 방식으로 해석할 수 있다.


## 26. 기댓값과 대각합(trace) 연산 순서 교환 (Expectation-Trace Interchange)

선형연산자 $\operatorname{trace}$와 기댓값 $E[\cdot]$는 교환 가능하다.

$$E[\operatorname{trace}(\mathbf{A})] = \operatorname{trace}(E[\mathbf{A}])$$

### 증명

$\mathbf{X} = (X_{ij})$를 $p \times p$ 확률행렬이라 하면

$$\operatorname{trace}(\mathbf{X}) = \sum_{i=1}^{p} X_{ii}$$

따라서

$$E[\operatorname{trace}(\mathbf{X})] = E\left[\sum_{i=1}^{p} X_{ii}\right] = \sum_{i=1}^{p} E[X_{ii}] = \operatorname{trace}(E[\mathbf{X}])$$

**핵심**: 대각합은 선형연산자(linear operator)이므로, 선형성(linearity of expectation)에 의해 기댓값과 교환 가능.

### 일반화: 선형연산자와 기댓값

모든 선형연산자 $L$에 대해

$$E[L(\mathbf{X})] = L(E[\mathbf{X}])$$

**예시**
- $\operatorname{trace}(\mathbf{X})$: 선형 ✓
- $\|\mathbf{X}\|_F^2 = \operatorname{trace}(\mathbf{X}^\top\mathbf{X})$: 이차형식이므로 선형 아님 ✗

### 적용 예시

회귀잔차 분산 계산:

$$E[\operatorname{trace}[(I - \Pi) \mathbf{e} \mathbf{e}^\top]] = \operatorname{trace}[E[(I - \Pi) \mathbf{e} \mathbf{e}^\top]]$$

### 관련 개념

- **선형성**: $E[a\mathbf{X} + b\mathbf{Y}] = aE[\mathbf{X}] + bE[\mathbf{Y}]$
- **Cyclic property of trace**: $\operatorname{trace}(\mathbf{ABC}) = \operatorname{trace}(\mathbf{BCA})$
- **Trace-Inner product**: $\operatorname{trace}(\mathbf{A}^\top\mathbf{B}) = \langle \mathbf{A}, \mathbf{B} \rangle_F$


## 11. Orthogonal Basis Decomposition

정규벡터에서 매우 중요하다.

$$Q'\mathbf{X}$$

여기서 $Q$는 orthogonal matrix이다.

### 활용

* χ² 분포 유도
* Wishart 분포
* 표본상관계수 분포

## 13. Gram-Schmidt Orthogonalization

상관을 제거하는 방법이다.

예

$$W=\frac{Y-\rho X}{\sqrt{1-\rho^2}}$$

### 활용

* 표본상관계수 분포
* 다변량 정규변환

## 14. χ² 분해 (Chi-Square Decomposition)

정규변수 제곱합

$$\sum Z_i^2 \sim \chi^2(n)$$

또한

$$\sum (Z_i-\bar{Z})^2 \sim \chi^2(n-1)$$

### 예
$Z_i \sim N(0,1)$ 독립 → $\sum Z_i^2 \sim \chi^2(n)$

## 15. 분산분해 (ANOVA Decomposition)

$$SST = SSR + SSE$$

즉

$$\sum (y_i-\bar{y})^2 = \sum (\hat{y}_i-\bar{y})^2 + \sum (y_i-\hat{y}_i)^2$$
### 증명

$\sum (y_i-\bar{y})^2 = \sum (y_i-\hat{y}_i+\hat{y}_i-\bar{y})^2$

$= \sum [(y_i-\hat{y}_i)+(\hat{y}_i-\bar{y})]^2$

$= \sum (y_i-\hat{y}_i)^2 + 2\sum(y_i-\hat{y}_i)(\hat{y}_i-\bar{y}) + \sum(\hat{y}_i-\bar{y})^2$

회귀잔차 성질에 의해 $\sum(y_i-\hat{y}_i)(\hat{y}_i-\bar{y})=0$이므로

$\sum (y_i-\bar{y})^2 = \sum (y_i-\hat{y}_i)^2 + \sum(\hat{y}_i-\bar{y})^2$


### 의미
총변동 = 회귀변동 + 잔차변동

## 16. 로그우도 전개 (Log-Likelihood Expansion)

MLE에서 자주 등장한다.

$$\log L(\theta) = \sum \log f(x_i|\theta)$$

이후 미분으로 극값을 찾는다.

$$\frac{d}{d\theta}\log L(\theta) = 0$$

### 예
정규분포 MLE: $\hat{\mu}=\bar{x}, \hat{\sigma}^2=\frac{1}{n}\sum (x_i-\bar{x})^2$

## 19. Delta Method

비선형 통계량 분포 계산

$$g(\hat{\theta}) \approx g(\theta) + g'(\theta)(\hat{\theta}-\theta)$$

따라서

$$\sqrt{n}(g(\hat{\theta})-g(\theta)) \xrightarrow{d} N(0, [g'(\theta)]^2\sigma^2)$$

### 예
$\hat{p}$가 정규분포 거의 따를 때, $\arcsin\sqrt{\hat{p}}$의 분포 계산

## 20. Conditioning Trick

정규분포에서 매우 자주 사용된다.

$$E[X] = E[E[X|Y]]$$


$$\text{Var}(X) 
= E(X^2) - E(X)^2 \\
= E[E(X^2|Y)] - E[E(X|Y)]^2 \\
= E[E(X^2|Y) - E(X|Y)^2] + E[E(X|Y)^2] - E[E(X|Y)]^2 \\
= E[\text{Var}(X|Y)] + \text{Var}(E[X|Y])$$


## 22. 대칭성 이용 (Symmetry)

$$\int_{-\infty}^{\infty} x \phi(x)dx = 0 \quad (\phi: \text{대칭 pdf})$$

### 활용

* 기댓값 계산 단순화
* 모멘트 계산


## 23. 부등식의 기댓값 보존 (Jensen's Inequality & Monotone Expectation)

**핵심**: 모든 $t > 0$에 대해 $f(t) \ge g(t)$이면, 양의 확률변수 $X$에 대해

$$E[f(X)] \ge E[g(X)]$$

(기댓값이 존재할 때)

### 예시

$-\log t \ge 1 - t$ (모든 $t > 0$)이면, $X > 0$에 대해

$$E[-\log X] \ge E[1-X] = 1 - E[X]$$

### 일반 원리: Monotone Expectation

$$f \le g \text{ (pointwise)} \implies E[f(X)] \le E[g(X)]$$

조건: $f, g$가 가측(measurable)이고 기댓값이 존재해야 함.

### 주요 활용

* Jensen's Inequality: 볼록/오목함수 이용
* 확률수렴 증명: 부등식 → 확률 상계
* Information theory: KL divergence 비음성 증명
* MLE 수렴: 로그우도 하한 제시

### 관련 개념

- **Jensen's Inequality** ($\phi$ 볼록): $\phi(E[X]) \le E[\phi(X)]$
- **Fatou's Lemma**: $\liminf$ 하에서 기댓값 순서 보존
- **Dominated/Monotone Convergence**: 극한 순서 바꾸기


## 23. 기댓값의 미분적분 (Differentiation Under Integration)

조건이 만족될 때, 미분과 적분 순서를 바꿀 수 있다.

$$\frac{d}{d\theta}E[g(X;\theta)] = E\left[\frac{\partial}{\partial\theta}g(X;\theta)\right]$$

### 주요 조건

* Dominated Convergence Theorem (DCT)
* Monotone Convergence Theorem (MCT)

### 활용

* Score function 유도
* Fisher Information 계산
* MLE 최적성 증명
* 점근이론

### 예

$$\frac{d}{d\theta}\int g(x;\theta)f(x)dx = \int \frac{\partial g(x;\theta)}{\partial\theta}f(x)dx$$

### 특수 경우: Leibniz Rule

$$\frac{d}{d\theta}\int_{a(\theta)}^{b(\theta)} g(x;\theta)dx = \int_{a(\theta)}^{b(\theta)} \frac{\partial g}{\partial\theta}dx + g(b;\theta)b'(\theta) - g(a;\theta)a'(\theta)$$


## 24. 사건 포함관계로 확률을 쪼개는 트릭 (Event Inclusion Bound)

확률수렴, Slutsky류 증명에서 자주 쓰는 트릭.  
핵심은 **복잡한 사건을 더 다루기 쉬운 두 사건의 합집합으로 포함시키는 것**이다.

### 기본 형태

임의의 $\varepsilon>0$, $M>0$에 대해  

$$\{|X_n Z_n|>\varepsilon\}
\subset
\{|X_n|>M\} \cup \left\{|Z_n|>\frac{\varepsilon}{M}\right\}$$

다르게 표현하면,

$$
\{|X_n Z_n|>\varepsilon\} \subset \{|X_n Z_n| > \varepsilon, |X_n| \le M\} \cup \{|X_n Z_n| > \varepsilon, |X_n| > M\} \\
\subset \{|Z_n| > \varepsilon/M\} \cup \{|X_n| > M\}
$$

### 확장: 합/차/몫 사건 분해

곱뿐 아니라 합, 차, 몫 등도 비슷하게 분해 가능.

- **합/차**

  $$  \{|X_n + Y_n| > \varepsilon\} \subset \{|X_n| > \varepsilon/2\} \cup \{|Y_n| > \varepsilon/2\} \\
  \{|X_n - Y_n| > \varepsilon\} \subset \{|X_n| > \varepsilon/2\} \cup \{|Y_n| > \varepsilon/2\}
  $$

- **몫**  
  $Y_n$이 0에 가까워지는 경우를 제외하면,

  $$ \left\{\left|\frac{X_n}{Y_n}\right| > \varepsilon\right\} \subset \{|X_n| > \varepsilon M\} \cup \{|Y_n| < 1/M\}$$

  (단, $|Y_n| > 1/M$로 제한)

### 왜 성립하는가

**대우(contrapositive)** 를 사용하면 간단하다.  
예를 들어, $|X_n| \le M$이고 $|Z_n| \le \frac{\varepsilon}{M}$이면  

$$|X_n Z_n| \le M \cdot \frac{\varepsilon}{M} = \varepsilon$$

즉, $|X_n Z_n| > \varepsilon$가 되려면 둘 중 하나는 반드시 조건을 벗어나야 한다.

### 바로 나오는 확률 상계

union bound를 적용하면

$$P(|X_n Z_n|>\varepsilon) \le P(|X_n|>M) + P\left(|Z_n|>\frac{\varepsilon}{M}\right)$$

합/차:

$$P(|X_n + Y_n| > \varepsilon) \le P(|X_n| > \varepsilon/2) + P(|Y_n| > \varepsilon/2)$$

몫:

$$P\left(\left|\frac{X_n}{Y_n}\right| > \varepsilon\right) \le P(|X_n| > \varepsilon M) + P(|Y_n| < 1/M)$$

### 전형적 사용 패턴

1. $M$ 또는 $\varepsilon/2$ 등 임계값을 잡아 각 항의 확률을 분리
2. 각각의 확률이 0으로 가는지 확인
3. union bound로 전체 사건의 확률을 상계

### 기억 포인트

> **합/차/곱/몫이 크려면, 적어도 하나의 성분이 충분히 커야 한다.**

복잡한 연산의 사건을 각각의 크기 조건으로 분해하는 것이 핵심.

### 관련 증명 습관

- 직접 증명보다 **대우**를 먼저 본다.
- 사건 포함관계를 만들면 곧바로 **확률 부등식**으로 바꾼다.
- 임의의 $M$을 도입해 **bounded part**와 **small remainder**를 분리한다.
- 이후 union bound, convergence in probability, tightness와 연결한다.

## 25. 

### 로그-가중 적분 공식 ($\mathrm{Beta}(\theta,1)$ 핵심 항)

$$\theta>0,\qquad \int_0^1 \log x \cdot \theta x^{\theta-1}\,dx = -\frac{1}{\theta}$$

즉, $X\sim \mathrm{Beta}(\theta,1)$이면

$$E[\log X]=-\frac{1}{\theta},\qquad E[-\log X]=\frac{1}{\theta}$$

### 빠른 유도 1 (치환)

$$u=x^\theta \;\Rightarrow\; du=\theta x^{\theta-1}dx,\quad \log x=\frac1\theta\log u \\
\int_0^1 \log x\cdot \theta x^{\theta-1}dx
= \frac1\theta\int_0^1 \log u\,du
= \frac1\theta(-1)
= -\frac1\theta.$$

### 빠른 유도 2 (파라미터 미분)

$$\int_0^1 x^{\theta-1}dx=\frac1\theta$$

양변을 $\theta$로 미분하면

$$\int_0^1 x^{\theta-1}\log x\,dx=-\frac1{\theta^2}.$$

여기에 $\theta$를 곱해

$$\int_0^1 \log x\cdot \theta x^{\theta-1}dx=-\frac1\theta.$$

### 활용 포인트

- Beta/Gamma 계열 로그우도 미분(Score) 계산
- Fisher Information 계산
- $(0,1)$ 구간 로그모멘트 계산의 기본 블록
