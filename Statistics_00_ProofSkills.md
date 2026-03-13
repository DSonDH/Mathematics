# 수리통계학 식 전개 패턴 총정리

## 1. 평균 중심화 (Mean-Centering Trick)

가장 많이 등장하는 전개이다.

$$x_i=(x_i-\bar{x})+\bar{x}$$

이를 이용하면

$$\sum (x_i-\bar{x})=0$$

가 성립한다.

### 대표 활용

* 공분산 전개
* 표본분산 공식
* 회귀식 전개
* ANOVA 분해

### 예

$$\sum (x_i-\bar{x})(y_i-\bar{y}) = \sum x_i y_i - n\bar{x}\bar{y}$$

## 2. 제곱합 분해 (Sum of Squares Decomposition)

수리통계에서 매우 중요하다.

$$\sum (x_i-a)^2 = \sum (x_i-\bar{x})^2 + n(\bar{x}-a)^2$$

### 의미

총 변동(total variation)을

* 표본 변동(sample variation)
* 평균 차이(mean deviation)

로 분해한다.

### 사용

* 표본평균의 최적성
* 분산분해
* ANOVA
* 최소제곱법

### 예
$a=0$일 때: $\sum x_i^2 = \sum (x_i-\bar{x})^2 + n\bar{x}^2$

## 3. 완전제곱식 (Completing the Square)

정규분포 계산에서 필수적인 기술이다.

$$ax^2+bx+c = a\left(x+\frac{b}{2a}\right)^2 + \left(c-\frac{b^2}{4a}\right)$$

### 주요 사용

* 정규분포 적분
* 조건부분포
* 베이즈 posterior
* MLE 유도

### 예

지수 $-\frac{1}{2}[(x-\mu)^2/\sigma^2]$ → 정규분포 표준형

## 4. 0을 의도적으로 추가하는 트릭 (Add-Zero Trick)

원하는 형태를 만들기 위해 0을 넣는다.

$$A = A + 0$$

예

$$\sum x_i y_i = \sum x_i y_i - n\bar{x}\bar{y} + n\bar{x}\bar{y}$$

이후 공분산 형태로 변형한다.

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

## 11. Orthogonal Basis Decomposition

정규벡터에서 매우 중요하다.

$$Q'\mathbf{X}$$

여기서 $Q$는 orthogonal matrix이다.

### 활용

* χ² 분포 유도
* Wishart 분포
* 표본상관계수 분포

## 12. 정규화 (Standardization)

모든 정규문제의 시작점이다.

$$Z=\frac{X-\mu}{\sigma}$$

### 활용

* Z-test
* t-test
* correlation distribution

### 예
$X \sim N(\mu, \sigma^2)$ → $Z \sim N(0,1)$

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

## 17. Jensen / Taylor 근사

점근이론에서 중요하다.

$$\sqrt{n}(\hat{\theta}-\theta) \xrightarrow{d} N(0, I(\theta)^{-1})$$

## 18. Slutsky Trick

점근분포 계산에서 등장한다.

$$X_n \xrightarrow{d} X,\quad Y_n \xrightarrow{p} c \implies X_n Y_n \xrightarrow{d} cX$$

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

$$\text{Var}(X) = E[\text{Var}(X|Y)] + \text{Var}(E[X|Y])$$

### 사용

* 조건부분포 계산
* 계층적 모형
* Gibbs sampling

## 21. 재가중화 (Reweighting / Importance Sampling)

$$E[g(X)] = \int g(x)\frac{f(x)}{q(x)}q(x)dx$$

### 활용

* Monte Carlo 추정
* 결측치 처리
* 표본재추출

## 22. 대칭성 이용 (Symmetry)

$$\int_{-\infty}^{\infty} x \phi(x)dx = 0 \quad (\phi: \text{대칭 pdf})$$

### 활용

* 기댓값 계산 단순화
* 모멘트 계산

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
$$
\{|X_n Z_n|>\varepsilon\}
\subset
\{|X_n|>M\} \cup \left\{|Z_n|>\frac{\varepsilon}{M}\right\}
$$

다르게 표현하면,

$$
\{|X_n Z_n|>\varepsilon\} \subset \{|X_n Z_n| > \varepsilon, |X_n| \le M\} \cup \{|X_n Z_n| > \varepsilon, |X_n| > M\} \\
\subset \{|Z_n| > \varepsilon/M\} \cup \{|X_n| > M\}
$$

### 확장: 합/차/몫 사건 분해

곱뿐 아니라 합, 차, 몫 등도 비슷하게 분해 가능.

- **합/차**
  $$
  \{|X_n + Y_n| > \varepsilon\} \subset \{|X_n| > \varepsilon/2\} \cup \{|Y_n| > \varepsilon/2\}
  $$
  $$
  \{|X_n - Y_n| > \varepsilon\} \subset \{|X_n| > \varepsilon/2\} \cup \{|Y_n| > \varepsilon/2\}
  $$

- **몫**  
  $Y_n$이 0에 가까워지는 경우를 제외하면,
  $$
  \left\{\left|\frac{X_n}{Y_n}\right| > \varepsilon\right\} \subset \{|X_n| > \varepsilon M\} \cup \{|Y_n| < 1/M\}
  $$
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