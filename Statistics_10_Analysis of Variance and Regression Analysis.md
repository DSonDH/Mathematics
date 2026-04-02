# 제10장 분산분석과 회귀분석 *(Analysis of Variance and Regression Analysis)*

## 일원분류모형의 분산분석 *(One-Way Analysis of Variance)*
여러 모집단의 평균을 비교할 때 가장 흔히 사용하는 모형으로 **일원분류 정규분포 모형(one-way normal model)** 을 둔다.

> 참고: 4장 일원분류모형(one-way classification model)

* 요인(인자, factor): 실험 조건을 구분하는 범주형 변수
    * $k$개의 수준(*level*)을 가지며, 각 수준의 적용을 처리(*treatment*)로 해석한다.
    * 수준(*level*):요인이 가질 수 있는 값이고, 인덱스 $i=1,\dots,k$로 색인
* 집합(group, 집단): 레벨 $i$에서의 관측치 $j=1,\dots,n_i$
    * 총 표본수 $n=\sum_{i=1}^k n_i$
* 주의: 선형회귀에서는 $X_{ij}$가 설명변수의 관측값이지만, 일원분류모형에서는 $X_{ij}$가 반응변수의 관측값이다.
    * $n_i$들은 평균은 $\mu_i$, 표준편차$\sigma$의 분포를 따른다
* 모형:
    
$$X_{ij}=\mu_i+e_{ij},\qquad e_{ij}\stackrel{iid}{\sim}N(0,\sigma^2) \\ -\infty<\mu_i<\infty,\quad \sigma^2>0$$

* 주된 목적: 가설 $H_0:\mu_1=\cdots=\mu_k$ (또는 $\alpha_1=\cdots=\alpha_k=0$)를 **ANOVA의 $F$-검정**으로 검정하고, 필요한 경우 대비(contrast) 및 동시신뢰구간으로 수준 간 차이를 정량화한다.

**모수의 재표현 *(overall mean and treatment effects)***  
전반적인 처리 평균(가중 평균)과 처리 효과를 다음과 같이 정의할 수 있다.

$$\bar\mu=\frac{1}{n}\sum_{i=1}^k n_i\mu_i,\qquad \alpha_i=\mu_i-\bar\mu$$

그러면 모형은

$$X_{ij}=\bar\mu+\alpha_i+e_{ij}, \quad \sum_{i=1}^k n_i\alpha_i=0$$

로도 표현된다. 여기서 $\bar\mu$는 전체 수준을 합친 전반 평균, $\alpha_i$는 수준 $i$의 처리 효과로 해석한다.

### 정리 10.1.1  일원분류 정규분포모형에서의 전역최소분산불편 추정량 *(UMVU estimators in one-way normal model)*
일원분류 정규분포모형에서 각 모수의 전역최소분산불편(UMVU, Uniformly Minimum Variance Unbiased) 추정량은 다음과 같다.

$$
\hat\mu_i=\bar X_i=\frac{1}{n_i}\sum_{j=1}^{n_i}X_{ij}\qquad(i=1,\dots,k) \\
\hat{\bar\mu}=\frac{1}{n}\sum_{i=1}^k n_i \hat\mu_i \quad (n = \sum_{i=1}^k n_i) \\
\hat\alpha_i=\hat\mu_i-\hat{\bar\mu}\qquad(i=1,\dots,k) \\
\hat\sigma^2=\frac{1}{n-k}\sum_{i=1}^k\sum_{j=1}^{n_i}(X_{ij}-\bar X_i)^2
$$

#### 증명
결합확률밀도함수는

$$
pdf(x;\theta)=\exp\!\left[-\frac{1}{2\sigma^2}\sum_{i=1}^k\sum_{j=1}^{n_i}(x_{ij}-\mu_i)^2-\frac{n}{2}\log(2\pi\sigma^2)\right],
\quad \theta=(\mu_1,\dots,\mu_k,\sigma^2)^t
$$

로 쓸 수 있으며, 이는 정리8.3.2 조건을 만족하는 지수족이다. 따라서 $\theta$에 대한 완비충분통계량을

$$
\Big(\bar X_1,\dots,\bar X_k,\ \sum_{i=1}^k\sum_{j=1}^{n_i}(X_{ij}-\bar X_i)^2\Big)^t
$$
    
로 잡을 수 있고, 이의 함수로 표현되는 불편추정량은 UMVUE가 된다. 위에 제시된 $\hat\mu_i,\hat{\bar\mu},\hat\alpha_i,\hat\sigma^2$는 모두 불편이며 위 완비충분통계량의 함수이므로 UMVUE이다.  
또한 모분산의 최대가능도 추정량은

$$
{\hat{\sigma^2}}^{MLE}=\frac{1}{n}\sum_{i=1}^k\sum_{j=1}^{n_i}(X_{ij}-\bar X_i)^2
$$

이며, $\sigma^2$만 분모가 $n$과 $n-k$로 달라짐을 확인할 수 있다.

### 정리 10.1.2  일원분류 정규분포모형에서의 표본분포에 관한 기본 정리 *(basic sampling distributions)*
(a) 집단별 표본평균 $\bar X_i$들은 서로 독립이며

$$\bar X_i\sim N\!\left(\mu_i,\frac{\sigma^2}{n_i}\right)\qquad(i=1,\dots,k)$$

(b)모분산 추정량 $\hat\sigma^2=\dfrac{1}{n-k}\sum_{i=1}^k\sum_{j=1}^{n_i}(X_{ij}-\bar X_i)^2$와 $(\bar X_1,\dots,\bar X_k)$는 서로 독립이다.

(c)

$$\frac{(n-k)\hat\sigma^2}{\sigma^2}\sim\chi^2(n-k)$$

#### 증명
정리 4.2.2 적용: 각 집단 $i$에서 $(X_{i1},\dots,X_{in_i})$는 $N(\mu_i,\sigma^2)$의 랜덤표본이므로 정규표본의 성질에 의해 $\bar X_i$와 $SS_i=\sum_{j=1}^{n_i}(X_{ij}-\bar X_i)^2$
는 독립이고 

$$\bar X_i\sim N(\mu_i,\sigma^2/n_i),\quad SS_i/\sigma^2\sim\chi^2(n_i-1)$$

또한 서로 다른 집단의 표본은 독립이므로 $(\bar X_1,\dots,\bar X_k)$는 서로 독립이다. 한편

$$(n-k)\hat\sigma^2=\sum_{i=1}^k\sum_{j=1}^{n_i}(X_{ij}-\bar X_i)^2=\sum_{i=1}^k SS_i$$

이므로 카이제곱분포의 가법성과 독립성으로부터 (b), (c)가 성립한다.

### 정리 10.1.3  일원분류 정규분포모형에서의 신뢰집합과 동시신뢰구간 *(confidence region & simultaneous confidence intervals)*
정리10.1.2로부터 모평균 $\mu_i$들의 선형결합에 대한 신뢰집합을 구할 수 있고, 이런 신뢰집합을 아래와 같이 **동시신뢰구간(simultaneous confidence interval)** 로 나타낼 수 있다.  

>참고: 동시신뢰구간  
>개별 신뢰구간을 $m$개 만들고 각각
>
>$$P(L_j\in CI_j)=1-\alpha\qquad (j=1,\dots,m)$$
>
>로 해석하더라도, 동시에 모두 포함될 확률(동시 포함확률, coverage)은 일반적으로 $1-\alpha$보다 작아진다. 예를 들어 서로 독립이라고 가정하면
>
>$$P(L_1\in CI_1,\dots,L_m\in CI_m)=(1-\alpha)^m<1-\alpha\qquad (m\ge 2)$$
>
>반면 **동시신뢰구간(simultaneous confidence intervals)** 은 구간을 더 넓혀
>
>$$P(L_1\in CI_1,\dots,L_m\in CI_m)=1-\alpha$$
>
>(또는 $\ge 1-\alpha$)를 보장하도록 만든 것이다.  
>대표적 구성 방법:
>* **Bonferroni**: 유한 개 $m$개의 구간을 동시에 보장(보통 보수적).
>* **Scheffé**: "모든 대비(contrast)"에 대해 동시 보장 가능.
>* **Tukey**: 쌍비교(pairwise) 전용으로 자주 사용.

계수가 r인 $k\times r$ 행렬 $C$에 대해 열공간을 $\mathrm{col}(C)=\{Ca:a\in\mathbb{R}^r\}$로 두고,

$$
D=\mathrm{Var}_{\mu,\sigma^2}(\hat\mu)/\sigma^2=\mathrm{diag}(1/n_i) \quad(\hat\mu=(\bar X_1,\dots,\bar X_k)^t)
$$

또한 $\mathrm{rank}(C)=r$이고 $C^\top D C$가 가역(invertible)이라 하자.
> **간단한 확인 $Var(\hat\mu)$**  
정리 10.1.2 (a)에 의해
>
>$$\bar X_i\sim N\!\left(\mu_i,\frac{\sigma^2}{n_i}\right),\quad
>\bar X_1,\dots,\bar X_k\ \text{서로 독립}$$
>
>따라서
>
>$$\mathrm{Var}(\bar X_i) = \frac{\sum_{j=1}^{n_i}\mathrm{Var}(X_{ij})}{n_i^2} =\frac{\sigma^2}{n_i},\quad \mathrm{Cov}(\bar X_i,\bar X_j)=0\ (i\neq j)$$
>
>즉 $\hat\mu=(\bar X_1,\dots,\bar X_k)^\top$의 공분산행렬은
>
>$$\mathrm{Var}(\hat\mu) =\mathrm{diag}\!\left(\frac{\sigma^2}{n_1},\dots,\frac{\sigma^2}{n_k}\right) =\sigma^2\,\mathrm{diag}(1/n_i),$$

라 하자. 그러면 일원분류정규분포모형에서 다음이 성립한다.

(a) (**신뢰집합 / confidence region**)  

$$
P_{\mu,\sigma^2}\!\left(
(C^t\mu-C^t\hat\mu)^t(C^tDC)^{-1}(C^t\mu-C^t\hat\mu)
\le r\hat\sigma^2\, F_\alpha(r,n-k)
\right)=1-\alpha
$$

- **무엇에 대한 신뢰집합?**  
    $C^\top\mu\in\mathbb R^r$ (즉, $\mu$의 **선형결합 $r$개를 한꺼번에**)에 대한 $(1-\alpha)$ 신뢰집합.
- **중심(center)**: $C^\top\hat\mu$  
    모수 $C^\top\mu$의 추정치(표본에서 계산되는 값).
- **거리(distance)**:  
    $(C^t\mu-C^t\hat\mu)^t(C^tDC)^{-1}(C^t\mu-C^t\hat\mu)$ 는  
    $C^\top\mu$가 $C^\top\hat\mu$에서 얼마나 떨어졌는지를 재는 **가중(공분산 보정) 제곱거리**(마할라노비스 거리 형태).
- **스케일(scale)과 임계값(threshold)**:  
    $\hat\sigma^2$로 미지의 $\sigma^2$를 대체하고, $F_\alpha(r,n-k)$를 써서 **커버리지 $1-\alpha$가 되도록** 반지름(경계)을 정함.
- **확률 해석(coverage)**:  
    위 부등식을 만족하는 영역을 $R$이라 하면, 반복 표본추출 시

    $$P_{\mu,\sigma^2}(C^\top\mu\in R)=1-\alpha$$

(b) 임의의 $c\in\mathrm{col}(C)$에 대해 동시로

$$
P_{\mu,\sigma^2}\!\left(
|c^t\mu-c^t\hat\mu|
\le \sqrt{c^\top D c\ \hat\sigma^2}\,\sqrt{rF_\alpha(r,n-k)},
\ \forall c\in\mathrm{col}(C)
\right)=1-\alpha
$$

- **무엇에 대한 구간?**: $c^\top\mu$ (즉 $\mu$의 **선형결합 1개**)에 대한 신뢰구간이지만, $c$를 $\mathrm{col}(C)$ 안에서 움직여도 **동시에(∀)** 성립하게 만든 **동시신뢰구간**.
- **중심(center)**: $c^\top\hat\mu$, 관심 모수 $c^\top\mu$의 추정치.
- **표준오차(SE) 역할**: $\sqrt{\sum_i \frac{c_i^2\hat\sigma^2}{n_i}}=\sqrt{c^\top D c\ \hat\sigma^2}$ (여기서 $D=\mathrm{diag}(1/n_i)$) — 선형결합 $c^\top\hat\mu$의 변동성을 반영.
- **임계값(동시 보정)**: $\sqrt{rF_\alpha(r,n-k)}$: 보통의 $t$-임계값 대신, **여러 방향($c$들)을 한꺼번에 보장**하기 위해 $F$-기반 반지름을 사용.
- **확률 해석(coverage)**: 반복 표본추출 시, **모든** $c\in\mathrm{col}(C)$에 대해 위 부등식이 동시에 성립할 확률이 $1-\alpha$.

#### 증명
$C$는 계수가 r인 $k\times r$ 행렬이다. 정리10.1.2 (a)와 다변량 정규분포 성질로부터

$$
C^t\hat\mu\sim N\!\left(C^t\mu,\ (C^tDC)\sigma^2\right) \\
\therefore (C^t\hat\mu-C^t\mu)^t(C^tDC\sigma^2)^{-1}(C^t\hat\mu-C^t\mu)\sim\chi^2(r)
$$

또한 정리 10.1.2로부터 $((n-k)\hat\sigma^2/\sigma^2\sim\chi^2(n-k))$이고 위 $\chi^2(r)$와 서로 독립이므로 F-분포의 정의로 (a)가 성립한다.  

(b)는 아래 등식으로부터, (a)를 다르게 표현한 것임을 알 수 있다.

$$
\max_{c: col(C), c\neq0}\frac{(c^t(\mu-\hat\mu))^2}{c^tDc}
= \max_{a\in R^r, a\neq 0}\frac{(a^tC^t(\mu-\hat\mu))^2}{a^tC^tDCa} 
= \max_{b\in R^r, b\neq 0}\frac{[b^t(C^tDC)^{-1/2}C^t(\mu-\hat\mu)]^2}{b^tb} \\
= (C^t\hat\mu-C^t\mu)^t(C^tDC)^{-1}(C^t\hat\mu-C^t\mu)
$$

### 대비 *(contrast)*
모평균들의 선형결합 $c_1\mu_1+\cdots+c_k\mu_k \quad (c_1+\cdots+c_k=0)$ 중 **계수합이 0인 경우** 를 **대비(contrast)** 라 한다. 대비는 처리 효과 비교에 자주 쓰이며, $\alpha_i=\mu_i-\bar\mu$를 사용하면

$$
c_1\mu_1+\cdots+c_k\mu_k=c_1\alpha_1+\cdots+c_k\alpha_k\qquad(c_1+\cdots+c_k=0)
$$

로도 쓸 수 있다. 특히 어떤 두 계수가 1, -1 $(c_i=1,c_j=-1)$이고 나머지 계수가 0인 대비는 기본대비라 하며 처리효과의 비교에 흔히 사용된다.  
이런 대비는 흔히
$\alpha_i = \mu_i -\bar\mu$를 사용하여 $c_1\alpha_1+\cdots+c_k\alpha_k\qquad(c_1+\cdots+c_k=0)$ 또는 $\alpha_i-\alpha_j$로 나타낸다.
  - 의의: 모든 $\mu_i$에 동일한 상수 $c$를 더하더라도, (예: 분산분석에서) 관심 대상이 되는 값/통계량은 변하지 않는다. 즉, 전체 평균 수준(location) 에 의존하지 않고 상대적 차이만 남는다.

이런 대비에 대한 동시신뢰구간은 아래와 같다
### 정리 10.1.4  일원분류 정규분포모형에서의 대비에 대한 동시신뢰구간 *(simultaneous CIs for contrasts)*
(a) 모든 대비 $c$ $(c_1+\cdots+c_k=0)$에 대해 동시로

$$
P_{\mu,\sigma^2}\!\left(
|c^t\alpha-c^t\hat\alpha| \le \sqrt{\sum_{i=1}^k \frac{c_i^2\hat\sigma^2}{n_i}}\, \sqrt{(k-1)F_\alpha(k-1,n-k)},\ \forall c:\sum c_i=0
\right) = 1-\alpha
$$

(b) 모든 $i\neq j$에 대해 동시로 (쉐페 동시신뢰구간, Scheffé simultaneous confidence intervals)

$$
P_{\mu,\sigma^2}\!\left(
|(\alpha_i-\alpha_j)-(\hat\alpha_i-\hat\alpha_j)|
\le \sqrt{\left(\frac1{n_i}+\frac1{n_j}\right)\hat\sigma^2}\,
\sqrt{(k-1)F_\alpha(k-1,n-k)},\ \forall i\neq j
\right) \ge 1-\alpha
$$

(c) $m=k(k-1)/2, \quad \alpha^*=\alpha/m$라 두면 (본페로니 동시신뢰구간, Bonferroni simultaneous confidence intervals)

$$
P_{\mu,\sigma^2}\!\left(|(\alpha_i-\alpha_j)-(\hat\alpha_i-\hat\alpha_j)|
\le \sqrt{\left(\frac1{n_i}+\frac1{n_j}\right)\hat\sigma^2}\,
t_{\alpha^*/2}(n-k),\ \forall i\neq j\right)\ge 1-\alpha
$$

#### 증명
(a)는 정리 10.1.3에서 $r=k-1$인 경우로 귀결된다.  
(b)는 (a)의 사건이 (b)의 사건을 포함함(특정 대비만 취한 부분사건)을 이용하면 된다.  
(c)에서 각 쌍 $(i,j)$ ($i<j$)에 대해 대비벡터를

$$
c^{(ij)}=(0,\dots,0,\underbrace{1}_{i\text{번째}},0,\dots,0,\underbrace{-1}_{j\text{번째}},0,\dots,0)^\top
\quad(\text{따라서 }\sum_{\ell=1}^k c_\ell^{(ij)}=0)
$$

로 두면

$$
c^{(ij)\top}\alpha=\alpha_i-\alpha_j,\quad c^{(ij)\top}\hat\alpha=\hat\alpha_i-\hat\alpha_j
$$

이고

$$
\sum_{\ell=1}^k\frac{(c_\ell^{(ij)})^2}{n_\ell} =\frac1{n_i}+\frac1{n_j}
$$

또한 정리 10.1.3 (b)를 $r=1$인 경우로 적용하면(즉 한 개의 선형결합에 대한 신뢰구간)

$$
P_{\mu,\sigma^2}\!\left(
\big|(\alpha_i-\alpha_j)-(\hat\alpha_i-\hat\alpha_j)\big| \le \sqrt{\left(\frac1{n_i}+\frac1{n_j}\right)\hat\sigma^2}\, t_{\alpha^*/2}(n-k)
\right) = 1-\alpha^*,
\quad \forall\, i\ne j
$$

가 성립한다. 이제 $m=k(k-1)/2, \quad \alpha^*=\alpha/m$로 두고 각 사건을 $A_{ij}$라 하면,
본페로니 부등식으로부터

$$
P\Big(\bigcap_{i=1}^{m}A_{i}\Big)\ge 1-\sum_{i=1}^{m}\big(1-P(A_{i})\big) = 1 - m\alpha^*=1-\alpha
$$

를 얻어 (c)가 따른다.

* (c)의 동시신뢰구간을 **본페로니(Bonferroni) 동시신뢰구간**이라 한다.
* (b)의 동시신뢰구간을 **셰페(Scheffé) 동시신뢰구간**이라 한다.
* 처리 개수가 많을수록 "쌍비교만" 할 때는 본페로니가 더 짧아지는 경향이 있으며, 셰페는 쌍비교보다 더 넓은 대비류에도 적용 가능하다는 장점이 있다.
* **표10.1.1 (생략)**

### 처리 평균의 유의성 검정 *(overall significance test / ANOVA F-test)*
처리 유의성 검정: 수준 간 평균 차이가 있는지 판단하는 대표 가설.

$$
H_0:\mu_1=\cdots=\mu_k \quad\text{vs}\quad H_1:\mu_1,\dots,\mu_k\text{가 모두 같지는 않다}
$$

이며, $\alpha_i$ 표현으로는

$$
H_0:\alpha_1=\cdots=\alpha_k=0 \quad\text{vs}\quad
H_1:\alpha_1,\dots,\alpha_k\text{가 모두 0은 아니다}
$$

### 정리 10.1.5  일원분류 정규분포모형에서의 유의성 검정 *(significance test in one-way normal model)*
최대가능도비 검정으로 다음을 정의한다.  
>(표기에서 **SS**는 *Sum of Squares* = 제곱합을 뜻한다.)
>* **SSB** (*Sum of Squares Between groups*): **집단 간 제곱합** (*between-group sum of squares*)
>* **SSW** (*Sum of Squares Within groups*): **집단 내 제곱합** (*within-group sum of squares*, 흔히 *SSE*라고도 표기)  
>
>참고로 전체 제곱합은 보통
>* **SST** (*Total Sum of Squares*): **전체 제곱합**
>으로 두며, 일원분산분석에서는 보통 \(SST = SSB + SSW\)로 분해된다.

$$
SSB=\sum_{i=1}^k n_i(\bar X_i-\bar X)^2,\quad
SSW=\sum_{i=1}^k\sum_{j=1}^{n_i}(X_{ij}-\bar X_i)^2=(n-k)\hat\sigma^2, \\
\bar X=\frac{1}{n}\sum_{i=1}^k\sum_{j=1}^{n_i}X_{ij}
$$

그러면 유의성 검정(최대가능도비 검정)의 검정통계량은

$$F_n=\frac{SSB/(k-1)}{SSW/(n-k)}$$

이고, 크기 $\alpha$인 기각역은

$$F_n\ge F_\alpha(k-1,n-k)$$

#### 증명
로그가능도는

$$
l(\theta)= -\frac{1}{2\sigma^2}\sum_{i=1}^k\sum_{j=1}^{n_i}(x_{ij}-\mu_i)^2-\frac{n}{2}\log(2\pi\sigma^2),
\quad \theta=(\mu_1,\dots,\mu_k,\sigma^2)^\top
$$

**(1) 최대가능도비 검정(LRT)와 제곱합 분해**  
정규모형의 로그가능도는

$$l(\theta) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^k\sum_{j=1}^{n_i}(x_{ij}-\mu_i)^2, \quad \theta=(\mu_1,\dots,\mu_k,\sigma^2)^\top$$

전체 모수공간 $\Omega$에서의 MLE:

$$\hat\mu_i = \bar x_i, \quad \hat\sigma_\Omega^2 = \frac{1}{n}\sum_{i=1}^k\sum_{j=1}^{n_i}(x_{ij}-\bar x_i)^2 = \frac{SSW}{n}$$

귀무가설 $H_0:\mu_1=\cdots=\mu_k(=\mu)$ 하에서의 MLE:

$$\hat\mu_0 = \bar x, \quad \hat\sigma_0^2 = \frac{1}{n}\sum_{i=1}^k\sum_{j=1}^{n_i}(x_{ij}-\bar x)^2$$

제곱합 분해 항등식을 이용하면:

$$\sum_{i=1}^k\sum_{j=1}^{n_i}(x_{ij}-\bar x)^2 = \sum_{i=1}^k\sum_{j=1}^{n_i}(x_{ij}-\bar x_i)^2 + \sum_{i=1}^k n_i(\bar x_i-\bar x)^2\\
\therefore \hat\sigma_0^2 = \frac{SSW+SSB}{n}, \quad \hat\sigma_\Omega^2 = \frac{SSW}{n}$$

**(2) 최대가능도비 검정통계량**  
MLE를 로그가능도에 대입하면:

$$l(\hat\theta_\Omega) = -\frac{n}{2}\log(2\pi\hat\sigma_\Omega^2) - \frac{n}{2}, \quad l(\hat\theta_0) = -\frac{n}{2}\log(2\pi\hat\sigma_0^2) - \frac{n}{2}$$

최대가능도비 검정통계량:

$$2\{l(\hat\theta_\Omega) - l(\hat\theta_0)\} = n\log\left(\frac{\hat\sigma_0^2}{\hat\sigma_\Omega^2}\right) = n\log\left(\frac{SSW+SSB}{SSW}\right) = n\log\left(1+\frac{SSB}{SSW}\right)$$

따라서 LRT의 기각역은 $\frac{SSB}{SSW}$가 큰 경우와 동치이고, 이는 (자유도 보정을 포함한) 통상적인 ANOVA의 $F$-검정통계량이다.

$$F_n = \frac{SSB/(k-1)}{SSW/(n-k)}$$

**(3) $H_0$ 하에서의 분포 및 독립성**  
한편 $(n\times 1)$ 벡터들로부터 $(n\times 1)$ 벡터로의 변환으로서

$$
(x_{ij}) \xrightarrow{A} (\bar x_i-\bar x)=A(x_{ij})
$$

로 정의된 변환 $A$를 생각하자. 그러면

$$
SSB=\sum_{i=1}^k\sum_{j=1}^{n_i}(\bar x_i-\bar x)^2
=(A(x_{ij}))^\top(A(x_{ij}))
$$

또한 $z_{ij}=\bar x_i-\bar x$로 두면

$$\bar z_i=\frac{1}{n_i}\sum_{j=1}^{n_i}z_{ij}=\bar x_i-\bar x,\quad \bar z=\frac{1}{n}\sum_{i=1}^k\sum_{j=1}^{n_i}z_{ij}=0 \\
\therefore A(z)=(\bar z_i-\bar z)_{ij}=(\bar x_i-\bar x)_{ij}=z$$

즉 $A^2=A$이다.

이제 임의의 $(n\times 1)$ 벡터 $(x_{ij}), (y_{ij})$에 대하여

$$
x^\top Ay
=\sum_{i=1}^k\sum_{j=1}^{n_i}x_{ij}(\bar y_i-\bar y)
=\sum_{i=1}^k n_i\bar x_i\bar y_i-n\bar x\bar y \\
x^\top A^\top y
=\sum_{i=1}^k\sum_{j=1}^{n_i}(\bar x_i-\bar x)y_{ij}
=\sum_{i=1}^k n_i\bar x_i\bar y_i-n\bar x\bar y
$$

으로 위아래 식이 같으므로

$$
A^2=A,\qquad A^\top=A,\qquad SSB=(X_{ij})^\top A(X_{ij})
$$

한편 $H_0:\mu_1=\cdots=\mu_k=\mu$ 인 경우에 $Z_{ij}=\frac{X_{ij}-\mu}{\sigma}$ 라고 하면

$$
Z_{ij}\stackrel{iid}{\sim}N(0,1),\qquad j=1,\dots,n_i,\ i=1,\dots,k \\
SSB=\sum_{i=1}^k n_i(\bar X_i-\bar X)^2
=\sigma^2\sum_{i=1}^k n_i(\bar Z_i-\bar Z)^2
=\sigma^2\sum_{i=1}^k\sum_{j=1}^{n_i}(\bar Z_i-\bar Z)^2
=\sigma^2 (Z_{ij})^\top A(Z_{ij})
$$

이므로, 정리 4.4.5로부터 $H_0:\mu_1=\cdots=\mu_k$ 하에서

$$
SSB/\sigma^2 \sim \chi^2(r),\\ r=\mathrm{trace}(A) 
=\sum_{i=1}^k\sum_{j=1}^{n_i}\left(\frac{1}{n_i}-\frac{1}{n}\right)
=k-1
$$

또한 SSW는 정리 10.1.1, 10.1.2에 따라

$$
\frac{SSW}{\sigma^2}
=\sum_{i=1}^k\sum_{j=1}^{n_i}\frac{(X_{ij}-\bar X_i)^2}{\sigma^2}
=\frac{(n-k)\hat\sigma^2}{\sigma^2}
\sim \chi^2(n-k)
$$

이고, $SSB$와 $SSW$는 서로 독립이다.  
따라서 $F$ 분포의 정의로부터 $H_0$ 하에서

$$
F_n =\frac{(SSB/\sigma^2)/(k-1)}{(SSW/\sigma^2)/(n-k)} \sim F(k-1,n-k)
$$

그러므로 크기 $\alpha$인 기각역은

$$
\boxed{\ F_n\ge F_\alpha(k-1,n-k)\ }
$$

### 처리 효과 모수의 다른 정의 *(alternative parametrization)*
처리 효과를 나타내는 모수로서

$$
\tilde\alpha_i=\mu_i-\tilde\mu,\qquad \tilde\mu=\frac{1}{k}\sum_{i=1}^k\mu_i
$$

로 두고 일원분류정규분포모형을

$$
X_{ij}=\tilde\mu+\tilde\alpha_i+e_{ij},\qquad \sum_{i=1}^k\tilde\alpha_i=0
$$

처럼 쓰기도 한다(가중치 $n_i$가 아닌 단순합 제약). 이때의 유의성 검정 귀무가설도 결국 $H_0:\mu_1=\cdots=\mu_k$와 동치이므로 동일한 $F$-검정으로 처리한다.

또한 $c_1 + \cdots + c_k = 0$ 일때 

$$ \sum_{i=1}^k c_i\tilde\alpha_i= \sum_{i=1}^k c_i\mu_i = \sum_{i=1}^k c_i\alpha_i $$

이므로, $\tilde\alpha_i = \mu_i-\tilde\mu, (i=1, \dots, k)$에 대한 대비 $c_1\tilde\alpha_1 + \cdots + c_k\tilde\alpha_k$의 동시신뢰구간도 $\alpha_i$에 대한 대비 $c_1\alpha_1 + \cdots + c_k\alpha_k$의 동시신뢰구간과 동일하다: 정리 10.1.4를 적용하면 된다.  

한편, $\tilde\alpha_i = \mu_i - \tilde\mu (i=1, \dots, k)$들의 일반적인 선형결합에 대한 추론은 이들을 $\mu_i$들의 선형결합으로 나타내어 정리 10.1.3으로 구할 수 있다. 같은 방법으로 $\alpha_i$들에 대한 추론도 할 수 있다.

#### 예 10.1.1  $\tilde\alpha_i$의 UMVU 및 동시신뢰구간 *(UMVU & simultaneous CIs for $\tilde\alpha_i$)*
* $\tilde\alpha_i=\mu_i-\tilde\mu$의 UMVU:

$$
\widehat{\tilde\alpha}_i=\hat \mu_i-\hat{\tilde \mu}=\bar X_i-\bar{\tilde X}, \quad \bar{\tilde X}=\frac{1}{k}\sum_{i=1}^k\bar X_i
$$

* $\widehat{\tilde\alpha}_i$에 대한 동시신뢰구간(정리 10.1.3 (b) 적용 형태):

$$
P_{\mu,\sigma^2}\!\left(
|\widehat{\tilde\alpha}_i|
\le
\sqrt{\frac{\big((k-1)^2n_i^{-1}+\sum_{j\ne i}n_j^{-1}\big)\hat\sigma^2}{k^2}}\,
\sqrt{(k-1)F_\alpha(k-1,n-k)},
\ \forall i
\right)\ge 1-\alpha
$$

* 한편 $\alpha_i=\mu_i-\bar\mu$에 대해서는(가중 평균 기준)

$$
P_{\mu,\sigma^2}\!\left(
|\hat\alpha_i|
\le
\sqrt{(n_i^{-1}-n^{-1})\hat\sigma^2}\,
\sqrt{(k-1)F_\alpha(k-1,n-k)},
\ \forall i
\right)\ge 1-\alpha
$$


## 이원분류모형의 분산분석 *(Two-Way Analysis of Variance)*
두 개의 요인(*factor*) $A, B$가 각각 $a, b$개의 수준(*level*)을 가질 때, 각 수준 조합을 **처리(treatment)** 라 하고, 처리 효과를 분석하기 위해 **이원분류 정규분포 모형(two-way normal model)** 을 설정한다.

### 모형 설정 *(Two-way normal model)*
* 요인 $A$: 수준 $i=1,\dots,a$
* 요인 $B$: 수준 $j=1,\dots,b$
* $n_{ij}$: $(i,j)$칸(cell)에서 처리가 반복되는 횟수 

모형은

$$
X_{ijk} = \mu_{ij} + e_{ijk},
\quad e_{ijk}\stackrel{iid}{\sim} N(0,\sigma^2),
\quad -\infty<\mu_{ij}<\infty,\quad \sigma^2>0
$$

### 정리 10.2.1 이원분류정규분포모형에서의 추정량과 표본분포 *(Estimators and sampling distributions in two-way normal model)*
**(a) 전역최소분산불편 추정량 *(UMVU (Uniformly Minimum Variance Unbiased) estimators)***  
* 모평균 추정량:
    
    $$\hat\mu_{ij} = \bar X_{ij\cdot} = \frac{1}{n_{ij}}\sum_{k=1}^{n_{ij}} X_{ijk} \quad (i=1,\dots,a,\ j=1,\dots,b)$$

* 모분산 추정량:
    
    $$
    \hat\sigma^2
    = \frac{1}{n-ab}
    \sum_{i=1}^a\sum_{j=1}^b\sum_{k=1}^{n_{ij}}
    (X_{ijk}-\bar X_{ij\cdot})^2,
    \qquad n=\sum_{i=1}^a\sum_{j=1}^b n_{ij}
    $$

    - ab가 빠지는 이유:
        - 모형을 가법모형(additive model, 상호작용 없음)으로 가정하면 A×B 상호작용항(ab)은 포함하지 않는다.
        - 또는 각 처리조합(셀)당 반복이 없어 상호작용을 따로 추정할 수 없는 경우, ab 성분이 오차(잔차)로 흡수되어 분해식/자유도에서 별도 항으로 나타나지 않는다

**(b) 표본평균**  
* $\hat\mu_{ij}=\bar X_{ij\cdot}$들은 서로 독립이고
    
$$\hat\mu_{ij}\sim N\!\left(\mu_{ij},\frac{\sigma^2}{n_{ij}}\right)$$
    
**(c) 모분산 추정량**  

$$\hat\sigma^2 \perp\!\!\!\perp\ (\hat\mu_{11},\dots,\hat\mu_{ab})$$

**(d) 분포**  

$$\frac{(n-ab)\hat\sigma^2}{\sigma^2}\sim\chi^2(n-ab)$$

#### 증명
각 셀 $(i,j)$에서 $(X_{ij1},\dots,X_{ijn_{ij}})$는 정규표본이므로, 정규표본의 기본 성질을 각 셀에 적용하면 바로 성립한다. 서로 다른 셀의 표본이 독립이라는 점을 이용하면 전체 결과가 따른다.

> 참고 (모분산의 최대가능도추정량)  
>이원분류 정규분포모형에서 모평균 $\mu_{ij}$의 최대가능도추정량은 각 셀의 표본평균
>
>$${\hat\mu_{ij}}^{\mathrm{MLE}}=\bar X_{ij\cdot}$$
>
>임이 명백하다. 또한 모분산의 최대가능도추정량은
>
>$$ {\hat\sigma^2}^{\mathrm{MLE}} =\frac{1}{n}\sum_{i=1}^a\sum_{j=1}^b\sum_{k=1}^{n_{ij}}(X_{ijk}-\bar X_{ij\cdot})^2$$
>
>로 주어진다. 따라서 정리 10.2.1의 불편추정량 $\hat\sigma^2$와는 분모만 $n$ vs. $n-ab$로 다르다.

### 정리 10.2.2 이원분류정규분포모형에서의 신뢰집합과 동시신뢰구간 *(Confidence regions and simultaneous confidence intervals)*
계수가 $p$인 $(ab)\times p$ 행렬 $C$에 대해, $C$의 열벡터공간(column space)을

$$\mathrm{col}(C)=\{Cx:\ x\in\mathbb{R}^p\}\subset\mathbb{R}^{ab}$$

로 둔다. 이때 $c\in\mathrm{col}(C)$는 $\mu=(\mu_{11},\dots,\mu_{ab})^\top$의 선형결합 $c^\top\mu$를 지정하는 계수벡터다.

$$ let \quad \hat\mu=(\hat\mu_{11},\dots, \hat\mu_{1b}; \dots; \hat\mu_{a1},\dots,\hat\mu_{ab})^t,
\quad D=\mathrm{Var}(\hat\mu)/\sigma^2=\mathrm{diag}(1/n_{ij})$$

**(a) 신뢰집합**  

$$
P_{\mu,\sigma^2}\!\left(
(C^t\mu-C^t\hat\mu)^t
(C^tDC)^{-1}
(C^t\mu-C^t\hat\mu)
\le p\hat\sigma^2 F_\alpha(p,n-ab)
\right)
=1-\alpha
$$

**(b) 동시신뢰구간**  

$$
P_{\mu,\sigma^2}\!\left(
|c^t\mu-c^t\hat\mu|
\le
\sqrt{\sum_{i,j}\frac{c_{ij}^2\hat\sigma^2}{n_{ij}}}\,
\sqrt{pF_\alpha(p,n-ab)},
\ \forall c\in\mathrm{col}(C)
\right)
=1-\alpha
$$

  - 이는 $\mathrm{col}(C)$에 속하는 **모든** 선형결합 $c^\top\mu$에 대해 동시에 커버리지를 보장하는 동시신뢰구간을 준다.
#### 증명
다변량 정규분포의 이차형식과 정리 10.2.1의 독립성, 그리고 F-분포의 정의를 그대로 적용하면 일원분류의 경우와 동일한 논리로 얻어진다.

### 균형된 이원분류모형 *(Balanced two-way ANOVA)*
일원분류의 경우와 같이 새로운 모수를

$$
n = n_{11} + \cdots + n_{ab},\\
\bar \mu_{..} = \frac{1}{n}\sum_{i=1}^a\sum_{j=1}^b n_{ij}\mu_{ij} \\
\mu_{i\cdot} = \frac{1}{n_{i\cdot}}\sum_{j=1}^b n_{ij}\mu_{ij}, \quad n_{i\cdot} = \sum_{j=1}^b n_{ij} \\
\mu_{\cdot j} = \frac{1}{n_{\cdot j}}\sum_{i=1}^a n_{ij}\mu_{ij}, \quad n_{\cdot j} = \sum_{i=1}^a n_{ij} \\
\alpha_i = \mu_{i\cdot} - \bar\mu_{..},\quad
\beta_j = \mu_{\cdot j} - \bar\mu_{..},\quad
\gamma_{ij} = \mu_{ij} - \mu_{i\cdot} - \mu_{\cdot j} + \bar\mu_{..}
$$

와 같이 정의한다. $\mu_{..}$는 전반 평균(overall mean), $\mu_{i\cdot}$는 요인 $A$의 수준 $i$의 평균, $\mu_{\cdot j}$는 요인 $B$의 수준 $j$의 평균, $\alpha_i$는 요인 $A$의 주효과(main effect), $\beta_j$는 요인 $B$의 주효과(main effect), $\gamma_{ij}$는 교호작용효과(interaction effect)로 해석된다.

* **주효과(main effect)** : 각 요인이 단독으로 평균에 미치는 영향으로, 위의 $\alpha_i$, $\beta_j$를 각각 요인 $A$, $B$의 주효과라 부른다.

* **교호작용효과(interaction effect)** : 가법모형(주효과의 합)로 설명되지 않는 비가법적(non-additive) 처리효과: $\gamma_{ij}=\mu_{ij}-\mu_{i\cdot}-\mu_{\cdot j}+\bar\mu_{..}$
    
    특히 **가법모형(additive model, 상호작용 없음)** 은 $\gamma_{ij}=0\ \ \forall i,j$ 인 경우를 의미한다.

반복 횟수가 모든 셀에서 같을 때: $n_{ij}=r,\quad n=rab$ 라 하면 **균형된 이원분류모형**이라 한다. 이 경우 해석과 분해가 아래와 같이 단순해진다.

$$
\bar\mu_{..}=\frac{1}{ab}\sum_{i=1}^a\sum_{j=1}^b\mu_{ij},
\quad \bar\mu_{i\cdot}=\frac{1}{b}\sum_{j=1}^b\mu_{ij},
\quad \bar\mu_{\cdot j}=\frac{1}{a}\sum_{i=1}^a\mu_{ij} \\
\alpha_i=\bar\mu_{i\cdot}-\bar\mu_{..},
\quad \beta_j=\bar\mu_{\cdot j}-\bar\mu_{..},
\quad \gamma_{ij}=\mu_{ij}-\bar\mu_{i\cdot}-\bar\mu_{\cdot j}+\bar\mu_{..}$$

* $\bar\mu_{..}$: 전반 평균(overall mean)
* $\bar\mu_{i\cdot}$: 요인 $A$의 수준 $i$의 평균: 수준 $i$의 평균이 전반 평균에서 얼마나 벗어나는지
* $\bar\mu_{\cdot j}$: 요인 $B$의 수준 $j$의 평균: 수준 $j$의 평균이 전반 평균에서 얼마나 벗어나는지
* $\alpha_i$: 요인 $A$의 **주효과(main effect)**
* $\beta_j$: 요인 $B$의 **주효과(main effect)**
* $\gamma_{ij}$: **교호작용효과(interaction effect)**

이때 모형과 제약조건은

$$
X_{ijk}=\bar\mu_{..}+\alpha_i+\beta_j+\gamma_{ij}+e_{ijk}, \quad e_{ijk} \sim N(0, \sigma^2) \text{ i.i.d.} \\
\sum_i\alpha_i=0,\quad
\sum_j\beta_j=0,\quad
\sum_i\gamma_{ij}=0,\quad
\sum_j\gamma_{ij}=0
$$

균형된 이원분류모형에서는 각 모수(주효과, 교호작용효과)가 셀 평균 $\mu_{ij}$들의 **선형결합**으로 표현된다. 따라서 관심 있는 효과(예: 주효과의 차이, 교호작용 유무)에 대한 추정과 추론은

$$
c^\top\mu\quad(\mu=(\mu_{11},\dots,\mu_{ab})^\top)
$$

꼴의 선형결합으로 정리한 뒤, 정리 10.2.2의 동시신뢰구간 결과를 그대로 적용하여 얻을 수 있다.  
특히 $n_{ij}=r$인 균형 설계에서 두 수준 $i,\ell$의 주효과 차이는

$$
\alpha_i-\alpha_\ell=\mu_{i\cdot}-\mu_{\ell\cdot}
$$

처럼 $\mu_{ij}$들의 선형결합(대비)으로 쓸 수 있다. 이에 대응하는 자연스러운 불편추정량은

$$
\hat\alpha_i=\bar X_{i\cdot\cdot}-\bar X_{\cdots},\qquad 
\bar X_{i\cdot\cdot}=\frac1b\sum_{j=1}^b\bar X_{ij\cdot},\qquad
\bar X_{\cdots}=\frac1{ab}\sum_{i=1}^a\sum_{j=1}^b\bar X_{ij\cdot}
$$

이고, 따라서 $\alpha_i-\alpha_\ell$의 추정량은 $\hat\alpha_i-\hat\alpha_\ell$가 된다.  

이제 정리 10.2.2를 (요인 $A$의 주효과 기본대비 전체를 한꺼번에 포함하도록 $C$를 잡아) 적용하면, 정리 10.1.4의 일원분류 대비 동시신뢰구간과 동일한 방식으로 **요인 $A$ 주효과의 모든 쌍비교 $(\alpha_i-\alpha_\ell)$에 대한 동시신뢰구간**을 얻는다. 이를 정리하면 다음과 같다.
### 정리 10.2.3 균형된 이원분류정규분포모형에서 주효과 기본대비의 동시신뢰구간 *(Simultaneous CIs for main-effect contrasts)*
(a) $n_{ij}=r$일 때, $\hat\alpha_i=\bar X_{i\cdot\cdot}-\bar X_{\cdots}$이고 $n_{i.}=rb, n=rab$이고 

$$
P_{\mu,\sigma^2}\!\left(
|(\alpha_i-\alpha_\ell)-(\hat\alpha_i-\hat\alpha_\ell)|
\le
\sqrt{\left(\frac1{n_{i.}}+\frac1{n_{\ell.}}\right)\hat\sigma^2}\,
\sqrt{(a-1)F_\alpha(a-1,n-ab)},
\ \forall i\neq\ell
\right)
\ge 1-\alpha
$$

(b) $a^*=\alpha/m, m = a(a-1)/2$라 하면 

$$
P_{\mu, \sigma^2} \left( {|(\alpha_i-\alpha_\ell)-(\hat\alpha_i-\hat\alpha_\ell)| \leq \sqrt{\frac1{n_{i.}}+\frac1{n_{\ell.}}\hat\sigma^2} t_{\alpha^*/2}(n-ab), \forall i \neq \ell} \right)
\geq 1-\alpha
$$

### 교호작용효과의 유의성 검정 *(Test for interaction effect)*
balanced 이원분류정규분포모형에서 교호작용효과의 유의성에 대한 가설

$$
H_0^{AB}:\gamma_{ij}=0\ \forall i,j \quad\text{vs}\quad
H_1^{AB}:\text{적어도 하나는 0이 아님}
$$

의 검정을 교호작용효과 유의성 검정이라 한다. 이런 가설은

$$
H_0^{AB}:\mu_{ij}\text{가 $i$의 함수와 $j$의 함수의 합으로 표현된다}
\quad\text{vs}\quad H_1^{AB}:H_0^{AB}\text{가 아니다}
$$

라는 가설과 동치이며, 최대가능도비 검정은 아래와 같다.
### 정리 10.2.4 균형된 이원분류정규분포모형에서 교호작용효과의 유의성 검정 *(F-test for interaction effect)*
> 참고: 이런 최대가능도비 검정은 $n_{ij}=n_{i.}n_{.j}/n$를 만족시키는 이원분류정규분포모형의 경우에도 성립한다.

$n_{ij}=r$인 균형된 이원분류정규분포모형에서,

$$
SS_{AB} = \sum_{i=1}^a\sum_{j=1}^b
r(\bar X_{ij\cdot}-\bar X_{i\cdot\cdot}-\bar X_{\cdot j\cdot}+\bar X_{\cdots})^2, \quad
SSE=\sum_{i=1}^a\sum_{j=1}^b\sum_{k=1}^r (X_{ijk}-\bar X_{ij\cdot})^2
$$

이라고 하면, 검정통계량과은

$$
F_n = \frac{SS_{AB}/((a-1)(b-1))}
{SSE/(n-ab)}
\sim F((a-1)(b-1),n-ab)
$$

크기 $\alpha (0 < \alpha < 1)$의 유의수준에서의 기각역은

$$
F_n\ge F_\alpha((a-1)(b-1),n-ab)
$$

#### 증명  
**(1) 최대가능도비 검정(LRT)와 제곱합 분해**  
정규모형에서 로그가능도와 각 모수공간에서의 최대가능도추정값을 대입하면(상수항 제외)

$$
l(\theta)
= -\frac{1}{2\sigma^2}\sum_{i=1}^a\sum_{j=1}^b\sum_{k=1}^r (x_{ijk}-\mu_{ij})^2
-\frac{n}{2}\log(2\pi\sigma^2),
\quad n=rab
$$

교호작용을 포함한 모형공간 $\Omega$에서는 $(\mu_{ij})$가 자유모수이므로

$$
\hat\mu_{ij}^\Omega=\bar x_{ij\cdot}, \qquad
{\hat\sigma^2}^\Omega
=\frac{1}{n}\sum_{i=1}^a\sum_{j=1}^b\sum_{k=1}^r (x_{ijk}-\bar x_{ij\cdot})^2 =\frac{SSE}{n} \\
\therefore \hat\theta^\Omega=\big(\bar x_{11\cdot},\dots,\bar x_{ab\cdot},\ {\hat\sigma^2}^\Omega\big)
$$

한편 $H_0^{AB}$ 하의 가법모형(교호작용 없음)에서는 

$$
{\hat{\bar\mu}_{..}}^0 = \bar x_{\cdot\cdot\cdot}=\frac{1}{n}\sum_{i=1}^a\sum_{j=1}^b\sum_{k=1}^r x_{ijk} \\
{\hat\alpha_i}^0 = \bar x_{i\cdot\cdot}-\bar x_{\cdot\cdot\cdot}, \quad {\hat\beta_j}^0 = \bar x_{\cdot j\cdot}-\bar x_{\cdot\cdot\cdot}, \quad {\hat\gamma_{ij}}^0 = 0 \\
{\hat\sigma^2}^0 =\sum_{i=1}^a\sum_{j=1}^b\sum_{k=1}^r (x_{ijk}-{\hat{\bar\mu}_{..}}^0-{\hat\alpha_i}^0-{\hat\beta_j}^0)^2
$$

정규모형에서 MLE를 대입한 로그가능도는 $l(\hat\theta)= -\frac{n}{2}\Big(1+\log(2\pi)+\log(\hat\sigma^2)\Big)$ 꼴이므로, 위 식들을 대입하면 다음의 LRT 표현으로:

$$
2\{l(\hat\theta^\Omega)-l(\hat\theta^{0})\}
=n\log\!\left(\frac{{\hat\sigma^2}^0}{{\hat\sigma^2}^\Omega}\right)
$$

$\log$는 단조증가함수이므로 LRT의 기각역은 ${\hat\sigma^2}^0/{\hat\sigma^2}^\Omega$가 큰 경우와 동치이다. 이때,

$$
n{\hat\sigma^2}^0 = \sum_{i=1}^a\sum_{j=1}^b\sum_{k=1}^r (x_{ijk}-\bar{x_{i..}}-\bar{x_{.j.}} + \bar{x_{...}})^2 =\sum_{i=1}^a\sum_{j=1}^b\sum_{k=1}^r (x_{ijk}-{\hat{\bar\mu}_{..}}^0-{\hat\alpha_i}^0-{\hat\beta_j}^0)^2 \\ = SS_{AB}+SSE\\
n{\hat\sigma^2}^\Omega = \sum_{i=1}^a\sum_{j=1}^b\sum_{k=1}^r (x_{ijk}-\bar x_{ij\cdot})^2 = SSE
$$

이므로, 검정통계량은 

$$F_n=\frac{SS_{AB}/((a-1)(b-1))}{SSE/(n-ab)}$$

이며 (남은 증명에서 보임), 기각역은 $F_n$이 큰 경우로 정리된다. 따라서 남은 것은 $H_0^{AB}$ 하에서 $F_n$이 $F$-분포를 따름을 보이는 일이다.

**(2) 직교투영(orthogonal projection)으로 $SS_{AB}$를 이차형식으로 표현**  

한편, $n\times 1$ 벡터를 $n \times 1$ 값으로 보내는 선형변환(투영)행렬 $\Pi$를 생각하자:

$$
X_{ijk}\ \mapsto\ 
\bar X_{ij\cdot}-\bar X_{i\cdot\cdot}-\bar X_{\cdot j\cdot}+\bar X_{\cdots} = \Pi(x_{ijk})
$$

그러면 정의에 의해

$$
SS_{AB}
=\sum_{i=1}^a\sum_{j=1}^b\sum_{k=1}^r \Big(\bar X_{ij\cdot}-\bar X_{i\cdot\cdot}-\bar X_{\cdot j\cdot}+\bar X_{\cdots}\Big)^2
=(\Pi x_{ijk})^\top(\Pi x_{ijk})
$$

이때 $z = \Pi(x)$로 두면,

$$
\bar{z_{i..}} = \frac{1}{b}\sum_{j=1}^b z_{ij\cdot} = \frac{1}{b}\sum_{j=1}^b (\bar x_{ij\cdot}-\bar x_{i\cdot\cdot}-\bar x_{\cdot j\cdot}+\bar x_{\cdots}) = 0 \\
\bar{z_{.j.}} = \frac{1}{a}\sum_{i=1}^a z_{ij\cdot} = \frac{1}{a}\sum_{i=1}^a (\bar x_{ij\cdot}-\bar x_{i\cdot\cdot}-\bar x_{\cdot j\cdot}+\bar x_{\cdots}) = 0 \\
\bar{z_{...}} = \frac{1}{ab}\sum_{i=1}^a\sum_{j=1}^b z_{ij\cdot} = \frac{1}{ab}\sum_{i=1}^a\sum_{j=1}^b (\bar x_{ij\cdot}-\bar x_{i\cdot\cdot}-\bar x_{\cdot j\cdot}+\bar x_{\cdots}) = 0 \\
\therefore \Pi(z) \mapsto (\bar{z_{ij\cdot}}-\bar{z_{i\cdot\cdot}}-\bar{z_{\cdot j\cdot}}+\bar{z_{\cdots}}) = \bar{z_{ij\cdot}} = 1/r\sum_{k=1}^r z_{ijk} = z_{ijk}
$$

그리고

$$
x^\top\Pi y = \sum_{i=1}^a\sum_{j=1}^b\sum_{k=1}^r x_{ijk}(\bar{y_{ij.}}-\bar{y_{i\cdot\cdot}}-\bar{y_{\cdot j\cdot}}+\bar{y_{\cdots}}) = \sum_{i=1}^a\sum_{j=1}^b r \cdot \bar x_{ij\cdot}(\bar{y_{ij.}}-\bar{y_{i\cdot\cdot}}-\bar{y_{\cdot j\cdot}}+\bar{y_{\cdots}}) \\
x^\top\Pi^\top y = \sum_{i=1}^a\sum_{j=1}^b\sum_{k=1}^r (\bar{x_{ij.}}-\bar{x_{i\cdot\cdot}}-\bar{x_{\cdot j\cdot}}+\bar{x_{\cdots}})y_{ijk} = \sum_{i=1}^a\sum_{j=1}^b r \cdot \bar y_{ij\cdot}(\bar{x_{ij.}}-\bar{x_{i\cdot\cdot}}-\bar{x_{\cdot j\cdot}}+\bar{x_{\cdots}})
$$

위 식은 x, y 대칭으로 동일하므로 $\Pi^\top=\Pi$.  

따라서 $\Pi$는 멱등 + 대칭: $SS_{AB} = X^\top \Pi X$ 로 이차형식(quadratic form) 표현이 된다.

**(3) $H_0^{AB}$ 하에서 $SS_{AB}/\sigma^2$의 카이제곱 분포와 자유도 계산**  
$H_0^{AB}$ (가법모형) 하에서

$$
Z_{ij} = \frac{X_{ijk} - (\bar\mu_{..}+\alpha_i+\beta_j)}{\sigma}
$$

라 하면, 

$$
Z_{ijk}=\frac{X_{ijk}-(\bar\mu_{..}+\alpha_i+\beta_j)}{\sigma} \sim N(0,1) \\
SS_{AB} = \sum_{i=1}^a\sum_{j=1}^b\sum_{k=1}^r (\bar X_{ij\cdot}-\bar X_{i\cdot\cdot}-\bar X_{\cdot j\cdot}+\bar X_{\cdots})^2 = \sigma^2\sum_{i=1}^a\sum_{j=1}^b\sum_{k=1}^r \left(\bar Z_{ij\cdot}-\bar Z_{i\cdot\cdot}-\bar Z_{\cdot j\cdot}+\bar Z_{\cdots}\right)^2 \\
= \sigma^2 Z^\top \Pi Z
$$

이제 정리 4.4.5(정규벡터 이차형식의 분포: $A^2=A$이면 $Z^\top A Z\sim\chi^2(\mathrm{trace}(A))$)를 적용하면

$$
\frac{SS_{AB}}{\sigma^2}\sim \chi^2(m),\qquad m=\mathrm{trace}(\Pi)
$$

이때 $m=\mathrm{trace}(\Pi)$는 투영공간(교호작용 부분공간)의 차원이고, 균형 이원배치에서 교호작용 자유도 $m=(a-1)(b-1)$ 이다(동일하게 $\mathrm{trace}(\Pi)$를 직접 계산해도 $(a-1)(b-1)$이 된다).  

$$ m = trace(\Pi) = \sum_{i=1}^a\sum_{j=1}^b \sum_{k} (\frac1r-\frac1{rb}-\frac1{ra}+\frac1{rab}) = (a-1)(b-1) $$

**(4) $SSE/\sigma^2$의 분포 및 $SS_{AB}$와의 독립성**  
또한

$$
SSE=\sum_{i=1}^a\sum_{j=1}^b\sum_{k=1}^r (X_{ijk}-\bar X_{ij\cdot})^2
$$

이고, 이는 셀 내부(within-cell) 편차만으로 구성되어

$$
\frac{SSE}{\sigma^2}\sim \chi^2(n-ab) = (n-ab)\frac{\hat\sigma^2}{\sigma^2}
$$

가 성립한다(각 셀에서 자유도 $r-1$짜리 카이제곱이 합쳐져 $ab(r-1)=n-ab$).

마지막으로 $SS_{AB}$는 오직 셀 평균들 $(\bar X_{ij\cdot})$의 함수이고, $SSE$는 각 셀의 "평균으로부터의 편차 제곱합"의 합이므로 정리 10.2.1의 독립성(셀 평균과 셀 내 제곱합의 독립성, 그리고 서로 다른 셀의 독립성)으로부터

$$SS_{AB}\ \perp\!\!\!\perp\ SSE$$

가 성립한다.

**(5) 결론: $F$-분포**  
따라서 $H_0^{AB}$ 하에서 서로 독립인

$$\frac{SS_{AB}}{\sigma^2}\sim \chi^2((a-1)(b-1)),\qquad\frac{SSE}{\sigma^2}\sim \chi^2(n-ab)
$$

를 얻으므로 $F$-분포의 정의에 의해

$$
F_n=\frac{SS_{AB}/((a-1)(b-1))}{SSE/(n-ab)}
=\frac{(SS_{AB}/\sigma^2)/((a-1)(b-1))}{(SSE/\sigma^2)/(n-ab)}
\sim F((a-1)(b-1),\,n-ab)
$$

따라서 유의수준 $\alpha$에서 기각역은 $F_n\ge F_\alpha((a-1)(b-1),n-ab)$로 주어진다.  

같은 증명방법으로 주효과의 유의성 검정도 증명할 수 있다.

### 정리 10.2.5 균형된 이원분류정규분포모형에서 주효과의 유의성 검정 *(Tests for main effects)*
$n_{ij} = r$인 균형된 이원분류정규분포모형에서  
**(a) 요인 $A$의 주효과**  

$$
H_0^A:\alpha_1=\cdots=\alpha_a=0,\quad H^A_1: not \ H_0^A \\
F_n=\frac{SS_A/(a-1)}{SSE/(n-ab)}\sim F(a-1,n-ab)
$$

이때 $SS_A=\sum_{i=1}^a br(\bar X_{i\cdot\cdot}-\bar X_{\cdots})^2$ 이고 기각역은

$$F_n \ge F_\alpha((a-1), n-ab)$$

**(b) 요인 $B$의 주효과**  

$$
H_0^B:\beta_1=\cdots=\beta_b=0,\quad H^B_1: not \ H_0^B \\
F_n=\frac{SS_B/(b-1)}{SSE/(n-ab)}\sim F(b-1,n-ab)
$$

이때 $SS_B=\sum_{j=1}^b ar(\bar X_{\cdot j\cdot}-\bar X_{\cdots})^2$ 이고 기각역은

$$F_n \ge F_\alpha((b-1), n-ab)$$


## 분산분석에서의 검정력 함수 *(Power Functions in Analysis of Variance)*
분산분석에서 유의성 검정에 사용되는 F-통계량은 **귀무가설 하에서는 중심 F 분포**, **대립가설 하에서는 비중심 F 분포**를 따른다. 검정력 함수는 이 비중심성모수(noncentrality parameter)를 통해 표현된다.

서로 독립인 $X_i \sim N(\mu_i,1),\quad i=1,\dots,r$에 대해 $Y=\sum_{i=1}^r X_i^2$ 의 분포를 **자유도 $r$**, **비중심성모수** $\delta=\sum_{i=1}^r \mu_i^2$
를 갖는 비중심 카이제곱분포라 하고

$$Y\sim \chi^2(r;\delta)$$

로 표기한다.  
### 정리 10.3.1 비중심 카이제곱분포의 성질

**(a) 누적생성함수(cgf)**  
$\delta = \mu_1^2 + \dots + \mu_r^2$ 라 하면, $Y$의 누율생성함수는 다음과 같다:

$$\mathrm{cgf}_Y(t) =\sum_{k=1}^\infty \frac{t^k}{k!}\,2^{k-1}(k-1)!(r+k\delta) \quad t < 1/2$$

**(b) 혼합분포 표현 (Poisson mixture)**  
$Poisson(\delta/2)$분포화 $\chi^2(r+2k)$분포의 확률밀도함수를 각각 $pdf_{P(\delta/2)}(k), pdf_{\chi^2(r+2k)}(y)$라 하면 $Y$의 확률밀도함수는 다음과 같다:

$$
pdf_Y(y) = \sum_{k=0}^\infty pdf_{P(\delta/2)}(k) \cdot pdf_{\chi^2(r+2k)}(y),\qquad y>0
$$

즉, 비중심 카이제곱분포는 **자유도가 증가하는 중심 카이제곱분포의 Poisson 혼합**으로 표현된다.

정리하면, 

$$Y\sim \chi^2(r;\delta) \Leftrightarrow Y \overset{d}{\equiv} X_1^2 + \cdots + X_r^2, \quad X_i \sim N(\mu_i,1)\\
\Leftrightarrow cfg_Y(t) = \sum_{k=1}^\infty \frac{t^k}{k!}\,2^{k-1}(k-1)!(r+k\delta) \\
\Leftrightarrow pdf_Y(y) = \sum_{k=0}^\infty pdf_{P(\delta/2)}(k) \cdot pdf_{\chi^2(r+2k)}(y)
$$

#### 증명
증명은 적률생성함수(mgf)와 그 로그(cgf)를 계산한 뒤, 급수 전개 및 mgf의 유일성으로 결론을 얻는다.

**1) $X\sim N(\mu,1)$일 때 $X^2$의 mgf 계산**  

$$
E(e^{tX^2}) =\int_{-\infty}^{\infty} \frac{1}{\sqrt{2\pi}} \exp\!\left(t x^2-\frac{(x-\mu)^2}{2}\right) dx \\
= \int_{-\infty}^{\infty} \frac{1}{\sqrt{2\pi}} \exp\!\left(-\frac{1-2t}{2}\left(x-\frac{\mu}{1-2t}\right)^2+\frac{\mu^2 t}{1-2t}\right) dx \\
= \frac{1}{\sqrt{2\pi}} \exp\!\left(\frac{\mu^2 t}{1-2t}\right) \int_{-\infty}^{\infty} \exp\!\left(-\frac{1-2t}{2}\left(x-\frac{\mu}{1-2t}\right)^2\right) dx
$$

가우스 적분 공식

$$
\int_{-\infty}^{\infty} e^{-a(x-b)^2} dx=\sqrt{\frac{\pi}{a}}\qquad(a>0)
$$

을 적용하면

$$
=(1-2t)^{-1/2}\exp\!\left(\frac{\mu^2 t}{1-2t}\right), \quad t<\frac12
$$

**a) $Y=\sum_{i=1}^r X_i^2$의 mgf 및 cgf**  

$$
mgf_Y(t)=E(e^{tY})
=\prod_{i=1}^r E(e^{tX_i^2}) \\
=\prod_{i=1}^r (1-2t)^{-1/2}\exp\!\left(\frac{\mu_i^2 t}{1-2t}\right)
=(1-2t)^{-r/2}\exp\!\left(\frac{t}{1-2t}\sum_{i=1}^r \mu_i^2\right)
$$

$\delta=\sum_{i=1}^r\mu_i^2$이므로

$$
=(1-2t)^{-r/2}\exp\!\left(\frac{\delta t}{1-2t}\right),\qquad t<\frac12
$$

따라서 $cgf_Y(t)$는

$$
cgf_Y(t)= -\frac{r}{2}\log(1-2t)+ \frac{\delta t}{1-2t},\qquad t<\frac12
$$

>이때 $-\log(1-x)=\sum_{k=1}^\infty \frac{x^k}{k} \quad(|x|<1)$ 이므로, $x=2t$를 대입하면
>
>$$
>-\log(1-2t)=\sum_{k=1}^\infty \frac{(2t)^k}{k}
>$$
>
>또한 $\frac{1}{1-2t}=\sum_{m=0}^\infty (2t)^m \quad(|2t|<1)$ 이므로
>
>$$
>\frac{t}{1-2t} =t\sum_{m=0}^\infty (2t)^m =\frac12 \sum_{k=1}^\infty (2t)^k
>$$
>

$$
=\frac{r}{2}\sum_{k=1}^\infty \frac{(2t)^k}{k}
+\frac{\delta}{2}\sum_{k=1}^\infty (2t)^{k}
=\sum_{k=1}^\infty\left(\frac{r}{2}\cdot\frac{(2t)^k}{k}+\frac{\delta}{2}\cdot (2t)^{k}\right) \\
=\sum_{k=1}^\infty \frac{t^k}{k!} 2^{k-1}(k-1)!(r+k\delta)
$$

**b) Poisson, $\chi^2$ 혼합 표현**  
(a)로부터 $Y$의 분포는 $\delta = \sum_{i=1}^r \mu_i^2$을 통해 $\mu$에 의존함을 알 수 있다.  
따라서 $V \sim \chi^2(r-2), \quad Z \sim N(\theta, 1), \quad \theta^2 = \delta$ 이고 서로 독립인 $V$와 $Z$에 대해 $Y \overset{d}\equiv V + Z^2$임을 알 수 있다.

>**$V + Z^2 \equiv Y$ 임을 보이기**  
>$V \sim \chi^2(r-2), \quad Z \sim N(\theta, 1)$이고 서로 독립이라 하자.
>
>**(1) mgf를 이용한 증명**  
>$V$와 $Z$의 적률생성함수:
>
>$$M_V(t) = (1-2t)^{-(r-2)/2}, \quad t < \frac{1}{2} \\ M_Z(t) = E(e^{tZ}) = E(e^{t \cdot (\theta + \sigma N(0,1))}) = e^{t\theta} \cdot (1-2t)^{-1/2}, \quad t < \frac{1}{2}$$
>
>따라서 $Z^2$의 mgf는:
>
>$$M_{Z^2}(t) = E(e^{tZ^2}) = (1-2t)^{-1/2} e^{\frac{\theta^2 t}{1-2t}}$$
>
>$V + Z^2$의 mgf (독립성):
>
>$$M_{V+Z^2}(t) = M_V(t) \cdot M_{Z^2}(t) = (1-2t)^{-(r-2)/2} \cdot (1-2t)^{-1/2} e^{\frac{\theta^2 t}{1-2t}}\\ = (1-2t)^{-r/2} e^{\frac{\theta^2 t}{1-2t}}$$
>
>이는 정리 10.3.1의 cgf에서 $\delta = \theta^2$인 경우의 비중심 카이제곱 $\chi^2(r;\delta)$의 mgf와 **정확히 일치**한다.  
>적률생성함수는 분포를 유일하게 결정하므로, $V + Z^2 \equiv Y$ 
>

$$ \therefore pdf_Y(y) = \int_0^{\infin}pdf_V(y-x) pdf_{Z^2}(x) dx = \int_0^{\infty} pdf_{\chi^2(r-1)}(y-x) \cdot pdf_{Z^2}(x) dx \\
$$

한편, 표준정규분포의 누적분포함수를 $\Phi(z)$라 하면, 

$$
pdf_{Z^2}(x) = \frac d{dx} \{ \Phi(\sqrt{x} -\theta) - \Phi(-\sqrt{x} -\theta) \} \\
= \frac 1{2\sqrt{x}} \frac1{\sqrt{2\pi}} \left( e^{-(\sqrt{x}-\theta)^2/2} + e^{-(\sqrt{x}+\theta)^2/2} \right) = \frac 1{2\sqrt{x}} \frac1{\sqrt{2\pi}} e^{-(x+\theta^2)/2} \left( e^{\sqrt{x}\theta} + e^{-\sqrt{x}\theta} \right) \\
= \sum_{k=0}^\infty pdf_{P(\delta/2)}(k) \cdot pdf_{\chi^2(1+2k)}(x)c_k, \quad c_k = \frac{\Gamma(k+1/2)\Gamma(k+1)2^{2k}}{\Gamma(2k+1)\Gamma(1/2)}
$$

그런데 

$$
\Gamma(1/2)\Gamma(2k+1) = \Gamma(1/2)(1 \cdot 3 \cdot \dots (2k -1))(2 \cdot 4 \cdot \dots (2k)) \\
= \Gamma(1/2)((1/2) \cdot \dots (k-1/2))2^k \cdot 2^k(k!) \\
= \Gamma(k+1/2)\Gamma(k+1)2^{2k}
$$

이므로 $pdf_{Z^2}(x) = \sum_0^{\infty}pdf_{P(\delta/2)}(k) \cdot pdf_{\chi^2(1+2k)}(x)$ 이고, 따라서

$$
pdf_Y(y) = \int_0^{\infty} pdf_{\chi^2(r-1)}(y-x) \left( \sum_{k=0}^\infty pdf_{P(\delta/2)}(k) \cdot pdf_{\chi^2(1+2k)}(x) \right) dx \\
= \sum_{k=0}^\infty pdf_{P(\delta/2)}(k) \cdot \int_0^{\infty} pdf_{\chi^2(r-1)}(y-x) \cdot pdf_{\chi^2(1+2k)}(x) dx \\
= \sum_{k=0}^\infty pdf_{P(\delta/2)}(k) \cdot pdf_{\chi^2(r+2k)}(y)
$$

### 정리 10.3.2 이차형식의 분포
아래 정리는 정리4.4.5를 일반화한 것으로, 서로 독립인 정규분포를 따르는 확률변수들의 이차형식의 분포가 비중심 카이제곱분포일 조건을 주고 있다.  
$X\sim N(\mu,I)$이고 $A^2=A$이면 다음이 성립한다:

$$
X^\top A X \sim \chi^2(r;\delta),\quad
r=\mathrm{trace}(A),\quad
\delta=\mu^\top A\mu
$$

* 분산분석의 제곱합(SS)은 **정규벡터의 이차형식**이다.
* 귀무가설 하에서는 $(\delta=0)$이 되어 중심 카이제곱분포를 따른다.
* 대립가설 하에서는 $(\delta>0)$이 되어 비중심 분포를 따른다.

>비중심 F 분포 *(Noncentral F Distribution)* 의 정의
>
>$$F\sim F(r_1,r_2;\delta) \Leftrightarrow Y=\frac{V_1/r_1}{V_2/r_2},\quad V_1\sim \chi^2(r_1;\delta),\ V_2\sim \chi^2(r_2), \quad V_1 \perp V_2$$
>
> 이런 비중심 카이제곱분포나 비중심 F분포의 정의에서 $\delta$를 비중심도모수(noncentrality parameter)라 한다.

#### 증명
$X\sim N(\mu,I_n)$이고 $A$가 멱등행렬(idempotent)라 하자.  
대칭 멱등행렬 $A$는 고유값이 $0$ 또는 $1$만 가지며, 어떤 직교행렬 $P$가 존재하여

$$
A=P
\begin{pmatrix}
I_r & 0\\
0 & 0
\end{pmatrix}
P^\top, \quad P^\top P=PP^\top=I_n
$$

이다. $r=\mathrm{rank}(A)$이다.

>**정리 (직교변환의 불변성 / orthogonal invariance of MVN)**  
>$X\sim N_n(\mu,\Sigma)$이고 $P$가 직교행렬($P^\top P=I_n$)이면
>
>$$Z=P^\top X \sim N_n(P^\top\mu,\ P^\top\Sigma P)$$
>
>특히 $\Sigma=I_n$이면
>
>$$Z\sim N_n(P^\top\mu,\ I_n)$$
>
>즉 공분산이 항등행렬인 정규벡터는 직교변환(회전/반사)을 해도 공분산이 변하지 않는다. 이를 (공분산의) 불변성이라 한다.

$Z=P^\top X, \quad \eta=P^\top\mu$ 라 두면, 직교변환의 불변성으로 $Z\sim N(P^\top\mu,\ P^\top I P)=N(\eta,I_n)$
이때 $Z$를

$$
Z=\begin{pmatrix}Z_1\\ Z_2\end{pmatrix},\qquad
\eta=\begin{pmatrix}\eta_1\\ \eta_2\end{pmatrix}
\quad (Z_1,\eta_1\in\mathbb R^r)
$$

처럼 분할하면

$$Z_1\sim N(\eta_1,I_r)$$

이고, $Z_1$의 성분들은 서로 독립이다.  

이제

$$
X^\top A X
= X^\top P
\begin{pmatrix}
I_r & 0\\
0 & 0
\end{pmatrix}
P^\top X
= Z^\top
\begin{pmatrix}
I_r & 0\\
0 & 0
\end{pmatrix}
Z = Z_1^\top Z_1
=\sum_{i=1}^r Z_{1i}^2
$$

각 $Z_{1i}\sim N(\eta_{1i},1)$가 독립이므로 정의에 의해

$$
\sum_{i=1}^r Z_{1i}^2 \sim \chi^2\!\left(r;\ \delta\right),
\quad \delta=\sum_{i=1}^r \eta_{1i}^2=\|\eta_1\|^2
$$

마지막으로, 

$$
\mu^\top A\mu
=\mu^\top P
\begin{pmatrix}
I_r & 0\\
0 & 0
\end{pmatrix}
P^\top \mu
=\eta^\top
\begin{pmatrix}
I_r & 0\\
0 & 0
\end{pmatrix}
\eta
=\eta_1^\top\eta_1
=\|\eta_1\|^2
=\delta
$$

따라서

$$
X^\top A X \sim \chi^2(r;\delta),\quad r=\mathrm{trace}(A),\quad \delta=\mu^\top A\mu
$$

#### 예 10.3.1 일원분류 분산분석에서의 검정력 함수
모형과 가설, 일원분산분석의 검정통계량의 분포는 어떻게 되어있을까?

$$
X_{ij}=\mu+\alpha_i+e_{ij},\quad \sum_{i=1}^k n_i\alpha_i=0,\quad
e_{ij}\overset{iid}\sim N(0,\sigma^2),\\
H_0:\alpha_1=\cdots=\alpha_k=0
\quad\text{vs}\quad
H_1:\text{적어도 하나의 }\alpha_i\neq 0. \\
F_n=\frac{SSB/(k-1)}{SSW/(n-k)},
\qquad
SSB=\sum_{i=1}^k n_i(\bar X_i-\bar X)^2,
\quad
SSW=\sum_{i=1}^k\sum_{j=1}^{n_i}(X_{ij}-\bar X_i)^2
$$

**1) $SSB$를 이차형식으로 표현**  
정리 10.1.5의 증명에서처럼 

$$
P_1=\mathrm{blockdiag}\!\left(\frac1{n_1}\mathbf 1_{n_1}\mathbf 1_{n_1}^\top,\ \dots,\ \frac1{n_k}\mathbf 1_{n_k}\mathbf 1_{n_k}^\top\right) \\
P_0=\frac1n\mathbf 1\mathbf 1^\top$$

여기서 $P_1Y$는 각 관측치를 "자기 집단 평균" $\bar X_i$로 치환한 벡터이고,  
$P_0Y$는 모든 관측치를 "전체평균" $\bar X$로 치환한 벡터이다.  
따라서 $Y-P_1Y$는 집단 내 편차(잔차)를 모은 벡터, $P_1Y-P_0Y$는 집단 평균들의 전체평균으로부터의 편차를 모은 벡터가 되어, 각각의 제곱노름이 제곱합(SS)과 일치한다:  
(정리 10.1.5의 증명에서는 $P_1 - P_0$을 하나의 행렬러 묶어 활용했었다.)

$$
\begin{aligned}
SSB
&=\|P_1Y-P_0Y\|^2 \\
&=(P_1Y-P_0Y)^\top(P_1Y-P_0Y) \\
&=Y^\top(P_1-P_0)^\top(P_1-P_0)Y \\
&=Y^\top(P_1-P_0)(P_1-P_0)Y \qquad (\because P_1^\top=P_1,\ P_0^\top=P_0)\\
&=Y^\top\big(P_1^2-P_1P_0-P_0P_1+P_0^2\big)Y \\
&=Y^\top\big(P_1-P_0-P_0+P_0\big)Y 
\qquad (\because P_1^2=P_1,\ P_0^2=P_0,\ P_1P_0=P_0P_1=P_0)\\
&=Y^\top(P_1-P_0)Y. \\
SSW
&=\|Y-P_1Y\|^2 \\
&=(Y-P_1Y)^\top(Y-P_1Y) \\
&=Y^\top(I-P_1)^\top(I-P_1)Y \\
&=Y^\top(I-P_1)(I-P_1)Y \qquad (\because (I-P_1)^\top=I-P_1)\\
&=Y^\top\big(I-2P_1+P_1^2\big)Y \\
&=Y^\top(I-P_1)Y \qquad (\because P_1^2=P_1).
\end{aligned}
$$

이고 $A:=P_1-P_0 \quad\Rightarrow\quad A^2=A,\ A^\top=A,\ \mathrm{trace}(A)=k-1$

**2) $H_1$ 하에서 $SSB/\sigma^2$의 분포 (비중심 카이제곱)**  
$H_1$ 하에서 평균벡터를 $m=E(Y)$라 하면 $Y\sim N_n(m,\sigma^2 I)$이고,

$$
Z=\frac{Y-m}{\sigma}\sim N_n(0,I_n),\quad \frac{Y}{\sigma}\sim N_n\!\left(\frac{m}{\sigma}, I_n\right)
$$

정리 10.3.2(정규벡터 이차형식)로부터

$$
\frac{SSB}{\sigma^2}
=\frac{1}{\sigma^2}Y^\top A Y
\sim \chi^2\!\left(k-1;\ \delta\right),
\quad \delta=\frac{1}{\sigma^2}m^\top A m = \frac{1}{\sigma^2}\sum_{i=1}^k n_i\alpha_i^2
$$

또한, 정리 10.1.5의 (Cochran 정리) 논리와 동일하게

$$
\frac{SSW}{\sigma^2}\sim \chi^2(n-k),
\qquad
SSB\ \perp\!\!\!\perp\ SSW
$$

이므로, 비중심 $F$ 분포의 정의로부터

$$ F_n = \frac{SSB/(k-1)}{SSW/(n-k)} \sim F(k-1, n-k; \delta), \quad \delta = \frac{1}{\sigma^2}\sum_{i=1}^k n_i\alpha_i^2 $$

유의수준 $\alpha$에서 기각역은 $F_n\ge F_\alpha(k-1,n-k)$이고, 검정력 함수는

$$
\boxed{\ \mathrm{Power}(\delta) = P_\delta\!\left(F_n \ge F_\alpha(k-1,n-k)\right)\ }
$$

한편 비중심 카이제곱분포의 대의적 정의로부터

$$
F_n \overset{d}\equiv \frac{(V + Z^2)/(k-1)}{W/(n-k)}, \quad V \sim \chi^2(k-2), \quad Z \sim N(\sqrt{\delta}, 1), \quad W \sim \chi^2(n-k) \\ V \perp Z, \quad V \perp W, \quad Z \perp W
$$

이로부터 

$$
f_{\alpha} = F_{\alpha}(k-1, n-k),\quad Y = (k-1)f_{\alpha}W/(n-k) - V,\quad \theta = \sqrt{\delta}
$$

라고 하면 검정력 함수를 다음과 같이 나타낼 수 있다:  

$$
P_\delta(F_n \ge f_{\alpha}) = P_\delta\!\left(\frac{V + Z^2}{k-1} \ge \frac{W}{n-k}f_{\alpha}\right) = P_\delta\!\left(Z^2 \ge Y\right), \quad Z\sim N(\theta, 1), \quad Z \perp Y
$$

이때 표준정규분포의 확률밀도함수 $\phi(z)$에서 

$$
\frac d{d\theta}P_\theta(Z^2 \geq y) = e^{-\frac12 y}\phi(\theta)(e^{\sqrt y \theta}-e^{-\sqrt y \theta}) > 0 \quad \forall \theta > 0
$$

임을 알 수 있다. 따라서 F검정의 검정력 함수인 $P_\delta(F_n \ge f_{\alpha}) = P_\delta\!\left(Z^2 \ge Y\right)$는 $\delta = \sum_i n_i\alpha_i^2/\sigma^2$의 **증가함수**이다.

성질(해석)
* $\mathrm{Power}(\delta)$는 $\delta$의 **증가함수**이다. (효과가 커질수록/표본이 커질수록 검정력 증가)
* $\delta=0$ (즉 $H_0$)이면 $F_n$은 중심 $F$이므로 $\mathrm{Power}(0)=\alpha$.
* $\delta=\dfrac{1}{\sigma^2}\sum_i n_i\alpha_i^2$이므로  
    처리효과 크기($\alpha_i$)가 커지거나 표본크기($n_i$)가 커지거나, 오차분산($\sigma^2$)이 작아질수록 검정력이 커진다.

#### 예 10.3.2 이원분류 분산분석에서의 검정력 함수 *(power function in two-way ANOVA)*
반복횟수가 $r$인 균형 이원분류모형에서

$$
X_{ijk}=\mu+\alpha_i+\beta_j+\gamma_{ij}+e_{ijk},\quad
e_{ijk}\overset{iid}\sim N(0,\sigma^2),
\quad i=1,\dots,a,\ j=1,\dots,b,\ k=1,\dots,r
$$

(식별을 위해 통상 $\sum_i\alpha_i=0,\ \sum_j\beta_j=0,\ \sum_i\gamma_{ij}=0,\ \sum_j\gamma_{ij}=0$ 제약을 둔다.)
전체 표본수는 $n=rab$이다.

**1) 교호작용효과 $A\times B$ 유의성 검정의 검정력**  
가설은

$$
H_0^{AB}:\gamma_{ij}=0\ \forall i,j
\quad\text{vs}\quad
H_1^{AB}:\text{적어도 하나의 }\gamma_{ij}\neq 0
$$

정리 10.2.4에서 교호작용 제곱합은

$$
SS_{AB}
=\sum_{i=1}^a\sum_{j=1}^b
r\big(\bar X_{ij\cdot}-\bar X_{i\cdot\cdot}-\bar X_{\cdot j\cdot}+\bar X_{\cdots}\big)^2
$$

이고, $SS_{AB}=Y^\top \Pi_{AB}Y$ 꼴의 **이차형식**으로 쓸 수 있으며

$$
\Pi_{AB}^2=\Pi_{AB},\quad \Pi_{AB}^\top=\Pi_{AB},\quad
\mathrm{trace}(\Pi_{AB})=(a-1)(b-1)
$$

이다(교호작용 부분공간으로의 직교투영).  
따라서 정리 10.3.2(이차형식의 분포)로부터 대립가설 하에서

$$
\frac{SS_{AB}}{\sigma^2}\sim \chi^2\!\big((a-1)(b-1);\ \delta_{AB}\big), \quad \delta_{AB}=\frac{1}{\sigma^2}m^\top \Pi_{AB}m
$$

여기서 $m=E(Y)$는 평균벡터이다. 균형설계에서는 교호작용 성분만이 $\Pi_{AB}$에 의해 남으며(주효과/전체평균은 소거), 또한 (정리 10.2.4의 독립성 논리와 동일하게)

$$
\frac{SSE}{\sigma^2}\sim \chi^2(n-ab), \quad SS_{AB}\ \perp\!\!\!\perp\ SSE
$$

이므로, 

$$ F_n=\frac{SS_{AB}/((a-1)(b-1))}{SSE/(n-ab)} \sim F\big((a-1)(b-1),\ n-ab;\ \delta_{AB}\big), \quad \delta_{AB} = \frac{r}{\sigma^2}\sum_{i=1}^a\sum_{j=1}^b \gamma_{ij}^2 $$

유의수준 $\alpha$에서 기각역은 $F_n\ge F_\alpha((a-1)(b-1),n-ab)$이고, 검정력 함수는

$$
\boxed{\ \mathrm{Power}_{AB}(\delta_{AB})
= P_{\delta_{AB}}\!\left(
F_n\ge F_\alpha((a-1)(b-1),n-ab)
\right)\ }
$$

비중심 $F$ 분포의 성질로 $\mathrm{Power}_{AB}(\delta_{AB})$는 $\delta_{AB}$의 **증가함수**이다.

**2) 주효과 $A$, $B$ 유의성 검정의 비중심도모수**  
정리 10.2.5의 주효과 검정도 같은 방식(투영에 대한 이차형식 $\Rightarrow$ 비중심 카이제곱 $\Rightarrow$ 비중심 $F$)으로 정리된다.

* **요인 $A$ 주효과**

$$
H_0^{A}:\alpha_1=\cdots=\alpha_a=0, \qquad
SS_A=\sum_{i=1}^a br(\bar X_{i\cdot\cdot}-\bar X_{\cdots})^2
$$

대립가설 하에서

$$
\frac{SS_A}{\sigma^2}\sim \chi^2(a-1;\delta_A), \qquad
\boxed{\ \delta_A=\frac{br}{\sigma^2}\sum_{i=1}^a \alpha_i^2\ } \\
\therefore F_n^{A}=\frac{SS_A/(a-1)}{SSE/(n-ab)}
\sim F(a-1,n-ab;\delta_A)
$$

* **요인 $B$ 주효과**

$$
H_0^{B}:\beta_1=\cdots=\beta_b=0,
\qquad
SS_B=\sum_{j=1}^b ar(\bar X_{\cdot j\cdot}-\bar X_{\cdots})^2
$$

대립가설 하에서

$$
\frac{SS_B}{\sigma^2}\sim \chi^2(b-1;\delta_B),
\qquad
\boxed{\ \delta_B=\frac{ar}{\sigma^2}\sum_{j=1}^b \beta_j^2\ } \\
\therefore F_n^{B}=\frac{SS_B/(b-1)}{SSE/(n-ab)}
\sim F(b-1,n-ab;\delta_B)
$$

각 검정력 함수 $\mathrm{Power}_A(\delta_A)$, $\mathrm{Power}_B(\delta_B)$ 역시 해당 $\delta$의 **증가함수**이다.

**3) (요약) 균형 이원분산분석표와 비중심도모수**  
| Source | SS | degree of freedom | MS | F-statistic ($=MS/MSE$) | noncentrality $\delta$ (under $H_1$) |
|---|---:|---:|---:|---:|---:|
| A | $SS_A=\sum_i br(\bar X_{i\cdot\cdot}-\bar X_{\cdots})^2$ | $a-1$ | $MS_A=SS_A/(a-1)$ | $F^A=MS_A/MSE$ | $\displaystyle \delta_A=\frac{br}{\sigma^2}\sum_i \alpha_i^2$ |
| B | $SS_B=\sum_j ar(\bar X_{\cdot j\cdot}-\bar X_{\cdots})^2$ | $b-1$ | $MS_B=SS_B/(b-1)$ | $F^B=MS_B/MSE$ | $\displaystyle \delta_B=\frac{ar}{\sigma^2}\sum_j \beta_j^2$ |
| $A\times B$ | $\displaystyle SS_{AB}=\sum_{i,j} r(\bar X_{ij\cdot}-\bar X_{i\cdot\cdot}-\bar X_{\cdot j\cdot}+\bar X_{\cdots})^2$ | $(a-1)(b-1)$ | $MS_{AB}=SS_{AB}/((a-1)(b-1))$ | $F^{AB}=MS_{AB}/MSE$ | $\displaystyle \delta_{AB}=\frac{r}{\sigma^2}\sum_{i,j}\gamma_{ij}^2$ |
| Error | $\displaystyle SSE=\sum_{i,j}\sum_{k=1}^r (X_{ijk}-\bar X_{ij\cdot})^2$ | $n-ab=ab(r-1)$ | $MSE=SSE/(n-ab)$ |  |  |
| Total | $SST$ | $n-1$ |  |  |  |
- 여기서 $n=rab$.
- 각 $F$-통계량은 귀무가설 하에서는 중심 $F$, 대립가설 하에서는 해당 $\delta$를 갖는 **비중심 $F$** 를 따른다.
- 따라서 검정력은 모두 $\delta$의 증가함수로 표현된다.


## 10.4 회귀분석 *(Regression Analysis)*
변수 사이의 함수관계 조사에 가장 기본적으로 사용되는 모형. 

### 선형회귀 정규분포모형 *(Normal Linear Regression Model)*
* 관측치 $Y_i$와 설명변수 $x_{i1},\dots,x_{ip}$에 대해

$$Y_i = x_{i0}\beta_0 + x_{i1}\beta_1 + \cdots + x_{ip}\beta_p + e_i,\quad e_i\overset{iid}\sim N(0,\sigma^2)$$

* 벡터/행렬로 쓰면

$$Y = \mathbf{X}\boldsymbol\beta + e,\quad e\sim N_n(0,\sigma^2 I),\quad \mathrm{rank}(\mathbf{X})=p+1,\quad \sigma^2>0$$

* $Y\in\mathbb{R}^n$, $\mathbf{X}\in\mathbb{R}^{n\times(p+1)}$, $\boldsymbol\beta\in\mathbb{R}^{p+1}$.

### 정리 10.4.1 선형회귀정규분포모형에서의 추정량과 표본분포
선형회귀 정규분포모형에서는 회귀계수의 **최대가능도추정량(MLE)** 과 **최소제곱추정량(OLS)** 이 일치한다 (4장, 6장 참고).  

**(a) $c^\top\beta$와 $\sigma^2$의 전역최소분산불편추정량 *(UMVU estimators)***  
* 임의의 벡터 $c\in\mathbb{R}^{p+1}$에 대해

    $$\widehat{c^\top\beta}=c^\top \boldsymbol{\hat\beta},\quad \boldsymbol{\hat\beta}=(\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top Y$$

* 분산의 불편추정량

    $$\hat\sigma^2=\frac{\|Y-\mathbf{X}\boldsymbol{\hat\beta}\|^2}{n-p-1} =\frac{(Y-\mathbf{X}\boldsymbol{\hat\beta})^\top(Y-\mathbf{X}\boldsymbol{\hat\beta})}{n-p-1}$$

**(b) $\hat\beta$의 분포**  

$$\boldsymbol{\hat\beta} = (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top Y \sim N_{p+1}\!\big(\boldsymbol\beta,\ \sigma^2(\mathbf{X}^\top \mathbf{X})^{-1}\big)$$

**(c) $\hat\beta$와 $\hat\sigma^2$의 독립성**  

$$\boldsymbol{\hat\beta} \perp \hat\sigma^2$$

**(d) $\hat\sigma^2$의 카이제곱 분포**  

$$\frac{(n-p-1)\hat\sigma^2}{\sigma^2}\sim \chi^2(n-p-1)$$

#### 증명
이 경우에 $Y$의 확률밀도함수는 $\theta=(\boldsymbol\beta^\top,\sigma^2)^\top\in \mathbb{R}^{p+1}\times(0,\infty)$를 모수로 하여

$$
pdf(y;\theta)=\exp\!\left\{-\frac{1}{2\sigma^2}(y-\mathbf{X}\boldsymbol\beta)^\top(y-\mathbf{X}\boldsymbol\beta)-\frac{n}{2}\log(2\pi\sigma^2)\right\} \\
=\exp\!\left\{-\frac{y^\top y}{2\sigma^2}+\frac{y^\top \mathbf{X}\boldsymbol\beta}{\sigma^2}-\frac{\boldsymbol\beta^\top \mathbf{X}^\top \mathbf{X}\boldsymbol\beta}{2\sigma^2}-\frac{n}{2}\log(2\pi\sigma^2)\right\}
$$

와 같이 나타내어지고 정리 8.3.2의 조건을 만족시키는 지수족의 경우이다. 따라서 $(\mathbf{X}^\top Y, Y^\top Y)$가 $\theta=(\boldsymbol\beta^\top,\sigma^2)^\top\in\mathbb{R}^{p+1}\times(0,\infty)$에 관한 완비충분통계량이다. 그런데

$$
\mathbf{X}^\top Y=(\mathbf{X}^\top \mathbf{X})\hat\beta, \qquad Y^\top Y=(n-p-1)\hat\sigma^2+\hat\beta^\top(\mathbf{X}^\top \mathbf{X})\hat\beta
$$

이므로 $(\hat\beta,\hat\sigma^2)$은 $(\mathbf{X}^\top Y, Y^\top Y)$의 일대일 함수로서 이 역시 $\theta$에 관한 완비충분통계량이다.

한편 정리 6.5.2로부터 $c^\top\hat\beta$과 $\hat\sigma^2$은 각각 $c^\top\beta$와 $\sigma^2$의 불편추정량이다. 따라서 완비충분통계량의 함수인 $c^\top\hat\beta$과 $\hat\sigma^2$은 각각 $c^\top\beta$와 $\sigma^2$의 전역최소분산불편 추정량이다.

표본분포에 관한 (b), (c), (d)의 증명은 정리 4.4.6에 주어져 있다.

### 정리 10.4.2 선형회귀정규분포모형에서의 신뢰집합과 동시신뢰구간
계수가 $r$인 $(p+1)\times r$ 행렬 $C$에 대해, $C$의 열공간을 $\mathrm{col}(C) = \{Ca: a \in \mathbb{R}^r\}$라 하자.

**(a) $C^\top\boldsymbol\beta$에 대한 신뢰집합 *(confidence set)***  

$$
P_{\boldsymbol\beta,\sigma^2}\!\left(
(C^\top\boldsymbol\beta-C^\top \boldsymbol{\hat\beta})^\top\Big(C^\top(\mathbf{X}^\top \mathbf{X})^{-1}C\Big)^{-1}(C^\top\boldsymbol\beta-C^\top\boldsymbol{\hat\beta})
\le r\hat\sigma^2\,F_\alpha(r,n-p-1)
\right)=1-\alpha
$$

**(b) $\mathrm{col}(C)$ 상의 모든 선형결합에 대한 동시신뢰구간 *(simultaneous CI)***  

$$
P_{\boldsymbol\beta,\sigma^2}\!\left(
|c^\top\boldsymbol\beta-c^\top\boldsymbol{\hat\beta}|
\le \sqrt{c^\top(\mathbf{X}^\top \mathbf{X})^{-1}c\ \hat\sigma^2}\ \sqrt{rF_\alpha(r,n-p-1)},
\ \forall c\in \mathrm{col}(C)
\right)=1-\alpha
$$

* 해석: $c$를 $C$의 열공간 안에서 움직이는 "많은" 선형결합에 대해 한 번에 커버리지를 보장하는 구간이다.

TODO:
### 정리 10.4.3 선형회귀정규분포모형에서 회귀계수의 유의성 검정
회귀계수 전체의 유의성보단 일부의 유의성에 대한 판단이 필요한 경우가 많다. 예6.5.1처럼 절편$\beta_0$을 제외 한 회귀계수가 관심의 대상인 것 처럼. 이런 경우 가설 검정을 아래와 같이 할 수 있다.  

**일부 회귀계수 블록 검정 *(partial/regression block test)***   

$$
Y=\mathbf{X}_0\boldsymbol\beta_0 + \mathbf{X}_1\boldsymbol\beta_1 + e, \quad
\boldsymbol\beta_0\in\mathbb{R}^{p_0},\quad \boldsymbol\beta_1\in\mathbb{R}^{p_1},\quad
\mathbf{X}_0\in\mathbb{R}^{n\times p_0},\quad \mathbf{X}_1\in\mathbb{R}^{n\times p_1}, \\
e\sim N_n(0,\sigma^2 I_n),\quad \sigma^2>0, \quad
\mathrm{rank}(\mathbf{X}_0,\mathbf{X}_1)=p_0+p_1,\quad \mathrm{rank}(\mathbf{X}_0)=p_0,\quad \mathrm{rank}(\mathbf{X}_1)=p_1
$$

회귀계수 유의성에 대한 가설: $H_0:\boldsymbol\beta_1=0\quad \text{vs}\quad H_1:\boldsymbol\beta_1\neq 0$

위 가설의 최대가능도비 검정에 대해 아래가 성립한다.  
**(a) 투영행렬과 분해**  
* $\mathbf{X}_0$로의 정사영행렬: $\Pi_0 = \mathbf{X}_0(\mathbf{X}_0^\top \mathbf{X}_0)^{-1}\mathbf{X}_0^\top$
* $\mathbf{X}_0$를 제거한 $\mathbf{X}_1$: $\mathbf{X}_{1|0}=(I-\Pi_0)\mathbf{X}_1$
* 그 열공간으로의 정사영행렬: $\Pi_{1|0}=\mathbf{X}_{1|0}(\mathbf{X}_{1|0}^\top \mathbf{X}_{1|0})^{-1}\mathbf{X}_{1|0}^\top$

이면,     

$$
\boxed{\ \min_{\boldsymbol\beta_0}\|Y-\mathbf{X}_0\boldsymbol\beta_0\|^2
= \min_{\boldsymbol\beta_0,\boldsymbol\beta_1}\|Y-\mathbf{X}_0\boldsymbol\beta_0-\mathbf{X}_1\boldsymbol\beta_1\|^2
+Y^\top\Pi_{1|0}Y\ }
$$

**(b) 검정통계량 (최대가능도비 검정과 동치 형태)**  
* 회귀로 설명되는 제곱합: $R(1|0)=Y^\top\Pi_{1|0}Y$
  * 즉, 추가 설명력 
  * (R: regression sum of squares)
* 오차제곱합: $SSE = Y^\top(I-\Pi_0-\Pi_{1|0})Y$

검정통계량 $F$-통계량은

$$F_n=\frac{R(1|0)/p_1}{SSE/(n-p_0-p_1)}$$

이고, 수준 $\alpha$의 기각역은 $F_n \ge F_\alpha(p_1,\ n-p_0-p_1)$

**(c) 대립가설 하 분포와 검정력 *(power)***  
* 대립가설 하에서 검정통계량에 대해
    
    $$F_n = \frac{R(1|0)/p_1}{SSE/(n-p_0-p_1)} \sim F(p_1,\ n-p_0-p_1;\ \delta)$$

    이 성립하고, 여기서 $\delta$는 비중심도모수 *(noncentrality parameter)*:
    
    $$
    \delta = \frac{\beta_1^\top X_1^\top\Pi_{1|0}X_1\beta_1}{\sigma^2}
    $$

* 검정력 함수는 $\delta$의 증가함수다: $\pi(\delta)=P_\delta\!\left(F_n\ge F_\alpha(p_1,n-p_0-p_1)\right)$
    
    * 해석: 효과크기($\beta_1$)가 커지거나, 잡음($\sigma^2$)이 작아지거나, 설계가 좋아져 $\Pi_{1|0}$ 방향으로 신호가 커질수록 검정력이 증가한다.

#### 증명
$Y=X_0\boldsymbol\beta_0 + X_1\boldsymbol\beta_1 + e$, $e\sim N_n(0,\sigma^2 I)$를 가정한다.  
이 경우에 $X=(X_0,X_1),\ \Pi_{0,1}=X(X^tX)^{-1}X^t,\ \Pi_0=X_0(X_0^tX_0)^{-1}X_0^t$라고 하면 $\Pi_{0,1},\ \Pi_0,\ \Pi_{1|0}$는 모두 정사영행렬이고 다음이 성립하는 것을 정리 6.5.3에서 알고 있다.

$$
\Pi_{0,1}=\Pi_0+\Pi_{1|0},\qquad \Pi_0^t\Pi_{1|0}=0 \\
\therefore I-\Pi_0=(I-\Pi_{0,1})+\Pi_{1|0}
$$

아래 두 식의 증명은 정리 6.5.1을 따른다.

$$
\min_{\beta_0\in R^{p_0},\beta_1\in R^{p_1}}|Y-X_0\beta_0|^2 =Y^t(I-\Pi_0)Y \\
\min_{\beta_0\in R^{p_0},\beta_1\in R^{p_1}}|Y-X_0\beta_0-X_1\beta_1|^2 =Y^t(I-\Pi_{0,1})Y
$$

따라서 두 식의 관계에 따라 (a)가 성립한다.

한편, 이 경우에 $Y \sim N_n(X_0\beta_0+X_1\beta_1, \sigma^2 I_n)$이므로 로그가능도는

$$
l(\theta)=-\frac{1}{2\sigma^2}|y-X_0\beta_0-X_1\beta_1|^2-\frac{n}{2}\log(2\pi\sigma^2) \\
\dot l(\theta) = \begin{pmatrix}\frac{\partial l}{\partial \beta_0} \\ \frac{\partial l}{\partial \beta_1} \\ \frac{\partial l}{\partial \sigma^2}\end{pmatrix}
= \begin{pmatrix} \frac{1}{\sigma^2}X_0^t(y-X_0\beta_0-X_1\beta_1) \\ \frac{1}{\sigma^2}X_1^t(y-X_0\beta_0-X_1\beta_1) \\ -\frac{n}{2\sigma^2}+\frac{1}{2\sigma^4}|y-X_0\beta_0-X_1\beta_1|^2 \end{pmatrix}
$$

이므로, $\sigma^2$에 관해 미분하여 얻은 전체 모수공간과 귀무가설하에서의 최대가능도 추정량:

$$
\hat\sigma^2_\Omega=\min_{\beta_0\in R^{p_0},\beta_1\in R^{p_1}}|Y-X_0\beta_0-X_1\beta_1|^2/n=SSE/n \\
\hat\sigma^2_0=\min_{\beta_0\in R^{p_0}}|Y-X_0\beta_0|^2/n=(SSE+Y^t\Pi_{1|0}Y)/n \\
\therefore\ 2\bigl(l(\hat\theta_\Omega)-l(\hat\theta_0)\bigr)
=n\log(\hat\sigma^2_0/\hat\sigma^2_\Omega)
=n\log\left(1+\frac{Y^t\Pi_{1|0}Y}{SSE}\right)
$$

이는 $Y^t\Pi_{1|0}Y/SSE$의 증가함수이므로, 최대가능도비 검정과 $F_n$ 검정은 동치이다.  
따라서 검정통계량은 $F_n$으로 주어지고 기각역은 $F_n$의 큰 값으로 주어진다.

한편 $\Pi_{1|0}$가 멱등행렬이고  
$\mathrm{trace}(\Pi_{1|0}) =\mathrm{trace}{X_{1|0}(X_{1|0}^tX_{1|0})^{-1}X_{1|0}^t} =\mathrm{trace}{(X_{1|0}^tX_{1|0})^{-1}X_{1|0}^tX_{1|0}} =p_1$ 이므로,  

정리 10.3.2로부터  
$R(1|0)/\sigma^2=Y^t\Pi_{1|0}Y/\sigma^2\sim\chi^2(p_1;\delta),\quad
\delta=(X_0\beta_0+X_1\beta_1)^t\Pi_{1|0}(X_0\beta_0+X_1\beta_1)/\sigma^2$  
그런데 $\Pi_{1|0}X_0=0$이므로 $\delta=\beta_1^tX_1^t\Pi_{1|0}X_1\beta_1/\sigma^2$  
또한 같은 방법으로 $SSE/\sigma^2=Y^t(I-\Pi_{0,1})Y/\sigma^2\sim\chi^2(n-p_0-p_1)$  

이때 $\Pi_{1|0}(I-\Pi_{0,1}) =\Pi_{1|0}(I-\Pi_0-\Pi_{1|0}) = \Pi_{1|0}-0-\Pi_{1|0}^2= \Pi_{1|0} - \Pi_{1|0} = 0$ 이므로 $\Pi_{1|0}Y$와 $(I-\Pi_{0,1})Y$는 공분산이 0이다.  
또한 $Y$가 정규분포를 따르므로 두 벡터는 결합정규분포(jointly normal)이고, 따라서 서로 독립이다: $\Pi_{1|0}Y \perp\!\!\!\perp (I-\Pi_{0,1})Y$  
그러므로 각각의 제곱노름인 $R(1|0)=Y^\top\Pi_{1|0}Y$와 $SSE=Y^\top(I-\Pi_{0,1})^\top(I-\Pi_{0,1})Y$도 독립이다: 

$$
R(1|0)=Y^t\Pi_{1|0}Y \perp SSE=Y^t(I-\Pi_0-\Pi_{1|0})Y
$$

따라서 일반적으로

$$
F_n=\frac{R(1|0)/p_1}{SSE/(n-p_0-p_1)} \sim F(p_1,n-p_0-p_1;\delta)
$$

한편 귀무가설 $H_0:\beta_1=0$하에서는 $\Pi_{1|0}X_0=0$이므로 $\delta=\beta_1^tX_0^t\Pi_{1|0}X_0\beta_1/\sigma^2=0$  
따라서 크기 $\alpha(0<\alpha<1)$인 기각역은 (b)에서와 같이 $F_n\ge F_\alpha(p_1,n-p_0-p_1)$ 로 주어진다.

또한 예 10.3.1과 같은 방법으로 이 검정의 검정력 함수가 $\delta$의 증가함수인 것을 밝힐 수 있다.

### 정리 10.4.4 회귀계수의 선형결합에 대한 유의성 검정 *(general linear hypothesis test)*
일반 선형가설 $H_0: C^\top\beta=0\quad \text{vs}\quad H_1:C^\top\beta\neq 0$  
$C$는 $(p+1)\times r$, $\mathrm{rank}(C)=r$  
위 가설의 최대가능도비 검정에 대해 아래가 성립한다.  

**(a) 해당 정사영행렬**  

$$
\Pi_{1|0}
= X(X^\top X)^{-1}C\Big(C^\top(X^\top X)^{-1}C\Big)^{-1}C^\top(X^\top X)^{-1}X^\top \\
\hat\beta = (X^\top X)^{-1}X^\top Y
$$

라 하면, 

$$
\boxed{\ 
\min_{\beta,\ C^\top\beta=0}\ \|Y-X\beta\|^2
= \min_{\beta}\ \|Y-X\beta\|^2
+Y^\top\Pi_{1|0}Y\ }
$$

또한 $\hat\beta=(X^\top X)^{-1}X^\top Y$이고 $\mathrm{Var}(C^\top\hat\beta)=\sigma^2\,C^\top(X^\top X)^{-1}C$이므로,

$$
\boxed{\ 
\frac{Y^\top\Pi_{1|0}Y}{\sigma^2}
= (C^\top\hat\beta)^\top\Big(\sigma^2\,C^\top(X^\top X)^{-1}C\Big)^{-1}(C^\top\hat\beta)
= (C^\top\hat\beta)^\top\big[\mathrm{Var}(C^\top\hat\beta)\big]^{-1}(C^\top\hat\beta)\ }
$$

**(b) 검정통계량**  
* 전체 모형의 정사영행렬 $\Pi_{1,0}=X(X^\top X)^{-1}X^\top,\quad R(1|0)=Y^\top\Pi_{1|0}Y,\quad SSE=Y^\top(I-\Pi_{1,0})Y$ 이라 하면, 검정통계량과 기각역은 다음과 같다:

$$
F_n=\frac{R(1|0)/r}{SSE/(n-p-1)}, \quad F_n \ge F_\alpha(r,\ n-p-1)
$$

**(c) 대립가설 하 비중심 $F$와 비중심도모수**  
검정통계량에 대해 아래가 성립하고, 검정력함수는 $\delta$의 증가함수다

$$
F_n\sim F(r,\ n-p-1;\ \delta), \quad \delta = \frac{\beta^\top C\Big(C^\top(X^\top X)^{-1}C\Big)^{-1}C^\top\beta}{\sigma^2}
$$

#### 증명
증명의 핵심은 이 정리에서의 가설 검정을 <정리 10.4.3>의 가설 검정으로 대응하여 인지하는 것이다. 이러한 대응을 이해하기 위하여, 행렬 $C$의 열벡터공간의 정규직교기저(orthonormal basis) 벡터를 열로 갖는 $(p+1)\times r$ 행렬을 $C_1$이라고 하고 이를 확장하여 $\mathbb{R}^{p+1}$의 정규직교기저 벡터를 열로 갖는 $(p+1)\times(p+1)$ 행렬을 $(C_0\ C_1)$이라고 하자. $C_0$는 $C$의 열벡터공간의 정규직교기저 벡터를 열로 갖는 $(p+1)\times(p+1-r)$ 행렬이다.
즉

$$
C_0^t C_0 = I_{p+1-r},\quad C_1^t C_1 = I_r,\quad C_0^t C_1 = 0
$$

이고 $C = C_1 B$인 정칙행렬 $B$가 존재한다. 이로부터

$$
\begin{cases}
X\beta = X(C_0\ C_1)(C_0\ C_1)^t \beta = D_0 \gamma_0 + D_1 \gamma_1 \\
D_0 = X C_0,\quad D_1 = X C_1,\quad \gamma_0 = C_0^t \beta,\quad \gamma_1 = C_1^t \beta \\
\operatorname{rank}(D_0) = p+1-r,\quad \operatorname{rank}(D_1) = r,\quad \operatorname{rank}(D_0, D_1) = p+1
\end{cases}
$$

와 같이 <정리 10.4.3>의 선형회귀모형에 대응하도록 모형을 나타낼 수 있다. 또한

$$
C^t \beta = 0 \iff B^t C_1^t \beta = 0 \iff C_1^t \beta = 0 \iff \gamma_1 = 0 \\
\min_{\beta \in \mathbb{R}^{p+1},\ C^t \beta = 0} |Y - X\beta|^2 = \min_{\gamma_0 \in \mathbb{R}^{p+1-r}} |Y - D_0 \gamma_0|^2
$$

이므로, <정리 10.4.3>의 (a)로부터 행렬 $(D_0, D_1)$의 열벡터공간

$$
\operatorname{col}((D_0, D_1)) = {D_0 \gamma_0 + D_1 \gamma_1 : \gamma_0 \in \mathbb{R}^{p+1-r},\ \gamma_1 \in \mathbb{R}^r}
$$

에서 행렬 $D_0$의 열벡터공간의 직교여공간(orthogonal complement)

$$
\operatorname{col}(D_{1|0}) = {a \in \operatorname{col}((D_0, D_1)) : a^t D_0 = 0}
$$

으로의 정사영행렬을 $\Pi_{1|0}$라고 하면 다음이 성립하는 것을 알고 있다.

$$
\min_{\beta \in \mathbb{R}^{p+1},\ C^t \beta = 0} |Y - X\beta|^2 = \min_{\gamma_0 \in \mathbb{R}^{p+1-r},\ \gamma_1 \in \mathbb{R}^r} |Y - D_0 \gamma_0 - D_1 \gamma_1|^2 + Y^t \Pi_{1|0} Y
$$

한편 $(D_0, D_1) = X(C_0\ C_1)$이므로

$$
\operatorname{col}((D_0, D_1)) = {X(C_0 \gamma_0 + C_1 \gamma_1) : \gamma_0 \in \mathbb{R}^{p+1-r},\ \gamma_1 \in \mathbb{R}^r} \\
= {X\beta : \beta \in \mathbb{R}^{p+1}} = \operatorname{col}(X)
$$

임을 알 수 있고, 행렬 $D_0$의 열벡터공간의 직교여공간 $\operatorname{col}(D_{1|0})$이 행렬

$$
S = X(X^t X)^{-1} C
$$

의 열벡터공간임을 다음으로부터 알 수 있다.

$$
\operatorname{col}(D_{1|0}) = \{a \in \operatorname{col}((D_0, D_1)) : a^t D_0 = 0\} \\
= \{a \in \operatorname{col}(X) : a^t X C_0 = 0\} \\
= \{X\beta : \beta^t X^t X C_0 = 0\} \\
= \{X\beta : X^t X \beta \in \operatorname{col}(C)\} \\
= \{X(X^t X)^{-1} C \xi : \xi \in \mathbb{R}^r\} \\
= \operatorname{col}(X(X^t X)^{-1} C)
$$

따라서 행렬 $D_0$의 열벡터공간의 직교여공간 $\operatorname{col}(D_{1|0})$으로의 정사영행렬이

$$
\Pi_{1|0} = S(S^t S)^{-1} S^t = X(X^t X)^{-1} C (C^t (X^t X)^{-1} C)^{-1} C^t (X^t X)^{-1} X^t
$$

로 주어지고, (a)가 성립하는 것을 알 수 있다. 또한 <정리 10.4.3>에서와 같은 방법으로 (b), (c)가 성립하는 것은 명백하다.

### 선형모형 *(General Linear Model)* 로의 확장
지금까지는 $\mathrm{rank}(X)=p+1$로 $X^\top X$가 가역인 경우를 다루었다. 그러나 분산분석(ANOVA) 같은 모형은 설계행렬 열들이 선형종속이어서 $\mathrm{rank}(X)$가 열 개수보다 작은 경우가 많다. 이를 제약조건$C^\top\beta=0$을 추가한 선형모형으로 표현한다:

$$
Y=X\beta+e,\quad C^\top\beta=0,\quad e\sim N_n(0,\sigma^2 I) \\
\mathrm{rank}(X)=p+1-r<p+1,\quad \mathrm{rank}(C^\top)=r,\quad X\text{와 }C^\top\text{의 행들은 선형독립}
$$

이런 모형을 선형모형(linear model)이라 하며, 이는 일원분류모형, 이원분류모형 등 분산분석 모형이나 선형회귀모형을 모두 포괄하는 모형이다.

#### 예 10.4.1 일원분류 정규분포모형을 선형모형으로 쓰기 
일원분류모형을 $X_{ij}=\mu+\alpha_i+e_{ij},\quad \sum_{i=1}^k n_i\alpha_i=0,\quad e_{ij}\stackrel{iid}{\sim}N(0,\sigma^2)$ 로 두자. 관측치를 한 벡터로 쌓아

$$
Y=(X_{11},\dots,X_{1n_1},\ X_{21},\dots,X_{kn_k})^\top\in\mathbb R^n,\quad n=\sum_{i=1}^k n_i
$$

라 하면 다음과 같이 선형모형 $Y = X\beta + e$로 쓸 수 있다.

**1) 설계행렬 $X$ (상수항 + 집단 더미)**  
모수벡터를

$$
\beta=(\mu,\alpha_1,\dots,\alpha_k)^\top\in\mathbb R^{k+1}
$$

로 잡고, 설계행렬을 "상수항 1개 + 집단 더미 $k$개로 잡으면

$$
X=\begin{pmatrix}
\mathbf 1_{n_1} & \mathbf 1_{n_1} & 0 & \cdots & 0\\
\mathbf 1_{n_2} & 0 & \mathbf 1_{n_2} & \cdots & 0\\
\vdots & \vdots & \vdots & \ddots & \vdots\\
\mathbf 1_{n_k} & 0 & 0 & \cdots & \mathbf 1_{n_k}
\end{pmatrix}\in\mathbb R^{n\times(k+1)}
\qquad
(\mathbf 1_{m}:\ m\text{-벡터, 전부 }1)
$$

가 된다. 즉, 집단 $i$의 관측치 행들은 반복해서

$$
(1,\ 0,\dots,0,\ \underbrace{1}_{(i+1)\text{번째 열}},\ 0,\dots,0)
$$

꼴이며, 이때

$$Y=X\beta+e$$

는 각 관측치에 대해 $X_{ij}=\mu+\alpha_i+e_{ij}$를 정확히 재현한다.

**2) $\mathrm{rank}(X)=k<k+1$ (왜 $X^\top X$가 특이해지는가?)**  

위 $X$는 열이 $k+1$개지만 다음 선형관계가 항상 성립한다:

$$
\text{(상수항 열)}=\sum_{i=1}^k \text{(집단 }i\text{ 더미 열)}
$$

따라서 $\mathrm{rank}(X)=k<k+1$이고 $X^\top X$ 는 정칙이 아니며 $(X^\top X)^{-1}$가 존재하지 않는다.

**3) 제약조건 $c^\top\beta=0$를 추가한 "선형모형 관점 (식별성)**  

이 경우 책에서처럼 모수에 제약을 추가하여 모형을 식별한다.  
일원분류모형의 제약 $\sum_i n_i\alpha_i=0$는 $c^\top\beta=0,\quad c^\top=(0,\ n_1,\dots,n_k)$ 로 쓸 수 있다. 즉,

$$
\boxed{Y=X\beta+e,\quad c^\top\beta=0,\quad e\sim N_n(0,\sigma^2I)}
$$

가 "제약조건이 있는 선형모형(linear model with constraints)의 형태다.

* 핵심: **설계행렬 $X$ 자체는 랭크가 부족하지만**, 모수공간을 $\{\beta:\ c^\top\beta=0\}$로 제한하면 (즉 불필요한 자유도를 제거하면) **모형이 식별 가능**해진다.

**4) $X^\top X$의 역행렬이 없을 때의 추정/검정(개념)**  
$X^\top X$가 특이하므로 정리 10.4.4에서처럼 $(X^\top X)^{-1}$로 정사영행렬을 쓰는 표현은 그대로 사용할 수 없다. 대신

* 제약 최소제곱(constrained LS)로 $\hat\beta$를 정의하거나,
* 일반화 역행렬(generalized inverse) 등을 사용하여 정사영행렬(또는 그에 준하는 투영)을 표현한다.

이때 정사영행렬의 "구체적 공식은 표현 선택(제약식/기저 선택)에 따라 달라질 수 있으나,
**투영(Projection)으로 제곱합(SS)을 분해하고, 그 비로 $F$-검정을 만들며, 대립가설 하에서 비중심 $F$가 된다**는 큰 구조는 정리 10.4.4와 같은 방식으로 전개될 수 있다.

> 참고: 같은 자료를 "상수항 없이 집단 더미 $k$개만으로 두면 설계행렬이 $n\times k$가 되어 $\mathrm{rank}(X)=k$ (보통)로 만들 수 있지만, 이 경우 $\mu,\alpha_i$의 해석(모수화)이 위 표현과 달라진다.

#### (참고) 10.3과의 연결(검정력)
* 위의 회귀 $F$-검정에서 대립가설 하 분포가 비중심 $F$이고, 검정력이 $\delta$의 증가함수라는 결론은 10.3에서 정리한 비중심 카이제곱/비중심 $F$의 성질을 사용해 얻는 구성이다.
