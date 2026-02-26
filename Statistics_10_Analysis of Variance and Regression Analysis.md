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
    $$
    X_{ij}=\mu_i+e_{ij},\qquad e_{ij}\stackrel{iid}{\sim}N(0,\sigma^2) \\
    -\infty<\mu_i<\infty,\quad \sigma^2>0.
    $$
* 주된 목적: 가설 $H_0:\mu_1=\cdots=\mu_k$ (또는 $\alpha_1=\cdots=\alpha_k=0$)를 **ANOVA의 $F$-검정**으로 검정하고, 필요한 경우 대비(contrast) 및 동시신뢰구간으로 수준 간 차이를 정량화한다.

### 모수의 재표현 *(overall mean and treatment effects)*
전반적인 처리 평균(가중 평균)과 처리 효과를 다음과 같이 정의할 수 있다.
$$
\bar\mu=\frac{1}{n}\sum_{i=1}^k n_i\mu_i,\qquad \alpha_i=\mu_i-\bar\mu.
$$
그러면 모형은
$$
X_{ij}=\bar\mu+\alpha_i+e_{ij},
\qquad \sum_{i=1}^k n_i\alpha_i=0
$$
로도 표현된다. 여기서 $\bar\mu$는 전체 수준을 합친 전반 평균, $\alpha_i$는 수준 $i$의 처리 효과로 해석한다.

### 정리 10.1.1  일원분류 정규분포모형에서의 전역최소분산불편 추정량 *(UMVU estimators in one-way normal model)*
일원분류 정규분포모형에서 각 모수의 전역최소분산불편(UMVU, Uniformly Minimum Variance Unbiased) 추정량은 다음과 같다.
$$
\hat\mu_i=\bar X_i=\frac{1}{n_i}\sum_{j=1}^{n_i}X_{ij}\qquad(i=1,\dots,k) \\
\hat{\bar\mu}=\frac{1}{n}\sum_{i=1}^k n_i\bar X_i \\
\hat\alpha_i=\hat\mu_i-\hat{\bar\mu}\qquad(i=1,\dots,k) \\
\hat\sigma^2=\frac{1}{n-k}\sum_{i=1}^k\sum_{j=1}^{n_i}(X_{ij}-\bar X_i)^2
$$

#### 증명
결합밀도는
$$
pdf(x;\theta)=\exp\!\left[-\frac{1}{2\sigma^2}\sum_{i=1}^k\sum_{j=1}^{n_i}(x_{ij}-\mu_i)^2-\frac{n}{2}\log(2\pi\sigma^2)\right],
\quad \theta=(\mu_1,\dots,\mu_k,\sigma^2)^t
$$
로 쓸 수 있으며, 이는 적절한 조건을 만족하는 지수족이다. 따라서 $\theta$에 대한 완비충분통계량을
$$
\Big(\bar X_1,\dots,\bar X_k,\ \sum_{i=1}^k\sum_{j=1}^{n_i}(X_{ij}-\bar X_i)^2\Big)^t
$$
로 잡을 수 있고, 이의 함수로 표현되는 불편추정량은 UMVU가 된다. 위에 제시된 $\hat\mu_i,\hat{\bar\mu},\hat\alpha_i,\hat\sigma^2$는 모두 불편이며 위 완비충분통계량의 함수이므로 UMVU이다.  
또한 모분산의 최대가능도 추정량은
$$
\hat\sigma^2_{MLE}=\frac{1}{n}\sum_{i=1}^k\sum_{j=1}^{n_i}(X_{ij}-\bar X_i)^2
$$
이며, $\sigma^2$만 분모가 $n$과 $n-k$로 달라짐을 확인할 수 있다.

### 정리 10.1.2  일원분류 정규분포모형에서의 표본분포에 관한 기본 정리 *(basic sampling distributions)*
(a) 집단별 표본평균 $\bar X_i$들은 서로 독립이며
$$
\bar X_i\sim N\!\left(\mu_i,\frac{\sigma^2}{n_i}\right)\qquad(i=1,\dots,k)
$$
(b) $\hat\sigma^2=\dfrac{1}{n-k}\sum_{i=1}^k\sum_{j=1}^{n_i}(X_{ij}-\bar X_i)^2$와 $(\bar X_1,\dots,\bar X_k)$는 서로 독립이다.

(c)
$$
\frac{(n-k)\hat\sigma^2}{\sigma^2}\sim\chi^2(n-k)
$$

#### 증명
정리 4.2.2 적용:  
각 집단 $i$에서 $(X_{i1},\dots,X_{in_i})$는 $N(\mu_i,\sigma^2)$의 랜덤표본이므로 정규표본의 성질에 의해 $\bar X_i$와 $SS_i=\sum_{j=1}^{n_i}(X_{ij}-\bar X_i)^2$
는 독립이고 $$\bar X_i\sim N(\mu_i,\sigma^2/n_i), (SS_i/\sigma^2\sim\chi^2(n_i-1))$$
또한 서로 다른 집단의 표본은 독립이므로 $(\bar X_1,\dots,\bar X_k)$는 서로 독립이다. 한편
$$
(n-k)\hat\sigma^2=\sum_{i=1}^k\sum_{j=1}^{n_i}(X_{ij}-\bar X_i)^2=\sum_{i=1}^k SS_i
$$
이므로 카이제곱분포의 가법성과 독립성으로부터 (b), (c)가 성립한다.

### 정리 10.1.3  일원분류 정규분포모형에서의 신뢰집합과 동시신뢰구간 *(confidence region & simultaneous confidence intervals)*
정리10.1.2로부터 모평균 $\mu_i$들의 선형결합에 대한 신뢰집합을 구할 수 있고, 이런 신뢰집합을 아래와 같이 **동시신뢰구간(simultaneous confidence interval)** 로 나타낼 수 있다.  

>참고: 동시신뢰구간  
>개별 신뢰구간을 $m$개 만들고 각각
>$$P(L_j\in CI_j)=1-\alpha\qquad (j=1,\dots,m)$$
>로 해석하더라도, 동시에 모두 포함될 확률(동시 포함확률, coverage)은 일반적으로 $1-\alpha$보다 작아진다. 예를 들어 서로 독립이라고 가정하면
>$$P(L_1\in CI_1,\dots,L_m\in CI_m)=(1-\alpha)^m<1-\alpha\qquad (m\ge 2)$$
>반면 **동시신뢰구간(simultaneous confidence intervals)** 은 구간을 더 넓혀
>$$P(L_1\in CI_1,\dots,L_m\in CI_m)=1-\alpha$$
>(또는 $\ge 1-\alpha$)를 보장하도록 만든 것이다.  
>대표적 구성 방법:
>* **Bonferroni**: 유한 개 $m$개의 구간을 동시에 보장(보통 보수적).
>* **Scheffé**: "모든 대비(contrast)"에 대해 동시 보장 가능.
>* **Tukey**: 쌍비교(pairwise) 전용으로 자주 사용.

계수가 r인 $k\times r$ 행렬 $C$에 대해 열공간을 $\mathrm{col}(C)=\{Ca:a\in\mathbb{R}^r\}$로 두고,
$$
D=\mathrm{Var}_{\mu,\sigma^2}(\hat\mu)/\sigma^2=\mathrm{diag}(1/n_i)
\quad(\hat\mu=(\bar X_1,\dots,\bar X_k)^t)
$$
라 하자. 그러면 일원분류정규분포모형에서 다음이 성립한다.

(a) (**신뢰집합 / confidence region**)  
$$
P_{\mu,\sigma^2}\!\left(
(C^t\mu-C^t\hat\mu)^t(C^tDC)^{-1}(C^t\mu-C^t\hat\mu)
\le r\hat\sigma^2\, F_\alpha(r,n-k)
\right)=1-\alpha.
$$
- 의미(중요한 것만 끊어서)
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
        $$P_{\mu,\sigma^2}(C^\top\mu\in R)=1-\alpha.$$

(b) 임의의 $c\in\mathrm{col}(C)$에 대해 동시로
$$
P_{\mu,\sigma^2}\!\left(
|c^t\mu-c^t\hat\mu|
\le
\sqrt{c^\top D c\ \hat\sigma^2}\,\sqrt{rF_\alpha(r,n-k)},
\ \forall c\in\mathrm{col}(C)
\right)=1-\alpha.
$$
- 의미(중요한 것만 끊어서)
    - **무엇에 대한 구간?**  
        $c^\top\mu$ (즉 $\mu$의 **선형결합 1개**)에 대한 신뢰구간이지만, $c$를 $\mathrm{col}(C)$ 안에서 움직여도 **동시에(∀)** 성립하게 만든 **동시신뢰구간**.
    - **중심(center)**: $c^\top\hat\mu$  
        관심 모수 $c^\top\mu$의 추정치.
    - **표준오차(SE) 역할**:  
        $\sqrt{\sum_i \frac{c_i^2\hat\sigma^2}{n_i}}=\sqrt{c^\top D c\ \hat\sigma^2}$  
        (여기서 $D=\mathrm{diag}(1/n_i)$) — 선형결합 $c^\top\hat\mu$의 변동성을 반영.
    - **임계값(동시 보정)**: $\sqrt{rF_\alpha(r,n-k)}$  
        보통의 $t$-임계값 대신, **여러 방향($c$들)을 한꺼번에 보장**하기 위해 $F$-기반 반지름을 사용.
    - **확률 해석(coverage)**:  
        반복 표본추출 시, **모든** $c\in\mathrm{col}(C)$에 대해 위 부등식이 동시에 성립할 확률이 $1-\alpha$.

#### 증명
정리10.1.2 (a)와 다변량 정규분포 성질로부터
$$
C^t\hat\mu\sim N\!\left(C^t\mu,\ (C^tDC)\sigma^2\right) \\
\therefore (C^t\hat\mu-C^t\mu)^t(C^tDC\sigma^2)^{-1}(C^t\hat\mu-C^t\mu)\sim\chi^2(r)
$$
또한 정리 10.1.2로부터 $((n-k)\hat\sigma^2/\sigma^2\sim\chi^2(n-k))$이고 위 $\chi^2(r)$와 서로 독립이므로 F-분포의 정의로 (a)가 성립한다.  

(b)는 (a)다르게 표현하여 증명할 수 있다.
$$
\max_{c: col(C), c\neq0}\frac{(c^t(\mu-\hat\mu))^2}{c^tDc}= \max_{a\in R^r, a\neq 0}\frac{(a^tC^t(\mu-\hat\mu))^2}{a^tC^tDCa} \\ 
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
  - 의의: 모든 μ_i에 동일한 상수 c를 더하더라도, (예: 분산분석에서) 관심 대상이 되는 값/통계량은 변하지 않는다. 즉, 전체 평균 수준(location) 에 의존하지 않고 상대적 차이만 남는다.
  
이런 대비에 대한 동시신뢰구간은 아래와 같다
### 정리 10.1.4  일원분류 정규분포모형에서의 대비에 대한 동시신뢰구간 *(simultaneous CIs for contrasts)*
(a) 모든 대비 $c$ $(c_1+\cdots+c_k=0)$에 대해 동시로
$$
P_{\mu,\sigma^2}\!\left(
|c^t\alpha-c^t\hat\alpha|
\le
\sqrt{\sum_{i=1}^k \frac{c_i^2\hat\sigma^2}{n_i}}\,
\sqrt{(k-1)F_\alpha(k-1,n-k)},
\ \forall c:\sum c_i=0
\right)=1-\alpha.
$$

(b) 모든 $i\neq j$에 대해 동시로
$$
P_{\mu,\sigma^2}\!\left(
|(\alpha_i-\alpha_j)-(\hat\alpha_i-\hat\alpha_j)|
\le
\sqrt{\left(\frac1{n_i}+\frac1{n_j}\right)\hat\sigma^2}\,
\sqrt{(k-1)F_\alpha(k-1,n-k)},
\ \forall i\neq j
\right)\ge 1-\alpha.
$$

(c) $m=k(k-1)/2$, $\alpha^*=\alpha/m$라 두면 (본페로니 방식)
$$
P_{\mu,\sigma^2}\!\left(
|(\alpha_i-\alpha_j)-(\hat\alpha_i-\hat\alpha_j)|
\le
\sqrt{\left(\frac1{n_i}+\frac1{n_j}\right)\hat\sigma^2}\,
t_{\alpha^*/2}(n-k),
\ \forall i\neq j
\right)\ge 1-\alpha.
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
c^{(ij)\top}\alpha=\alpha_i-\alpha_j,\qquad
c^{(ij)\top}\hat\alpha=\hat\alpha_i-\hat\alpha_j
$$
이고
$$
\sum_{\ell=1}^k\frac{(c_\ell^{(ij)})^2}{n_\ell}
=\frac1{n_i}+\frac1{n_j}.
$$
또한 정리 10.1.3 (b)를 $r=1$인 경우로 적용하면(즉 한 개의 선형결합에 대한 신뢰구간),
$$
P_{\mu,\sigma^2}\!\left(
\big|(\alpha_i-\alpha_j)-(\hat\alpha_i-\hat\alpha_j)\big|
\le
\sqrt{\left(\frac1{n_i}+\frac1{n_j}\right)\hat\sigma^2}\,
t_{\alpha^*/2}(n-k)
\right)=1-\alpha^*,
\qquad \forall\, i\ne j
$$
가 성립한다. 이제 $m=k(k-1)/2$, $\alpha^*=\alpha/m$로 두고 각 사건을 $A_{ij}$라 하면,
본페로니 부등식으로부터
$$
P\Big(\bigcap_{i<j}A_{ij}\Big)\ge 1-\sum_{i<j}\big(1-P(A_{ij})\big)
=1-m\alpha^*=1-\alpha
$$
를 얻어 (c)가 따른다.

* (c)의 동시신뢰구간을 **본페로니(Bonferroni) 동시신뢰구간**이라 한다.
* (b)의 동시신뢰구간을 **셰페(Scheffé) 동시신뢰구간**이라 한다.
* 처리 개수가 많을수록 "쌍비교만" 할 때는 본페로니가 더 짧아지는 경향이 있으며, 셰페는 쌍비교보다 더 넓은 대비류에도 적용 가능하다는 장점이 있다.

#### 표10.1.1 (생략)

### 처리 평균의 유의성 검정 *(overall significance test / ANOVA F-test)*
처리 유의성 검정: 수준 간 평균 차이가 있는지 판단하는 대표 가설.
$$
H_0:\mu_1=\cdots=\mu_k
\quad\text{vs}\quad
H_1:\mu_1,\dots,\mu_k\text{가 모두 같지는 않다}
$$
이며, $\alpha_i$ 표현으로는
$$
H_0:\alpha_1=\cdots=\alpha_k=0
\quad\text{vs}\quad
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
SSB=\sum_{i=1}^k n_i(\bar X_i-\bar X)^2,\qquad
SSW=\sum_{i=1}^k\sum_{j=1}^{n_i}(X_{ij}-\bar X_i)^2=(n-k)\hat\sigma^2, \\
\bar X=\frac{1}{n}\sum_{i=1}^k\sum_{j=1}^{n_i}X_{ij}.
$$
그러면 유의성 검정(최대가능도비 검정)의 검정통계량은
$$
F_n=\frac{SSB/(k-1)}{SSW/(n-k)}
$$
이고, 크기 $\alpha$인 기각역은
$$
F_n\ge F_\alpha(k-1,n-k)
$$

#### 증명
로그가능도는
$$
l(\theta)= -\frac{1}{2\sigma^2}\sum_{i=1}^k\sum_{j=1}^{n_i}(x_{ij}-\mu_i)^2-\frac{n}{2}\log(2\pi\sigma^2),
\qquad
\theta=(\mu_1,\dots,\mu_k,\sigma^2)^\top.
$$
**(1) 최대가능도비 검정(LRT)와 제곱합 분해**  

정규모형에서 (제약 없음) 전체 모수공간 $\Omega$에서의 MLE는
$$
\hat\mu_i=\bar x_i,\qquad
\hat\sigma_\Omega^2=\frac{1}{n}\sum_{i=1}^k\sum_{j=1}^{n_i}(x_{ij}-\bar x_i)^2
=\frac{SSW}{n}.
$$
귀무가설 $H_0:\mu_1=\cdots=\mu_k(=\mu)$ 하에서의 MLE는
$$
\hat\mu_0=\bar x,\qquad
\hat\sigma_0^2=\frac{1}{n}\sum_{i=1}^k\sum_{j=1}^{n_i}(x_{ij}-\bar x)^2.
$$

한편 다음 제곱합 분해 항등식
$$
\sum_{i=1}^k\sum_{j=1}^{n_i}(x_{ij}-\bar x)^2
=\sum_{i=1}^k\sum_{j=1}^{n_i}(x_{ij}-\bar x_i)^2
+\sum_{i=1}^k n_i(\bar x_i-\bar x)^2
$$
로부터
$$
SSW=\sum_{i=1}^k\sum_{j=1}^{n_i}(x_{ij}-\bar x_i)^2,\qquad
SSB=\sum_{i=1}^k n_i(\bar x_i-\bar x)^2, \\
n\hat\sigma_0^2=SSW+SSB,\qquad n\hat\sigma_\Omega^2=SSW
$$
가 되어 $\hat\sigma_0^2=(SSW+SSB)/n$임을 알 수 있다.

정규모형에서 MLE를 대입한 로그가능도는
$$
l(\hat\theta)=-\frac{n}{2}\Big(1+\log(2\pi)+\log(\hat\sigma^2)\Big) \\
\therefore 2\{l(\hat\theta_\Omega)-l(\hat\theta_0)\}
=
-n\Big(\log(SSW/n)-\log((SSW+SSB)/n)\Big)
=
n\log\!\left(1+\frac{SSB}{SSW}\right).
$$
$\log$는 단조증가함수이므로 LRT의 기각역은 $\frac{SSB}{SSW}$가 큰 경우와 동치이고, 이는 (상수배/자유도 조정까지 포함해) 통상적인 ANOVA의
$$
F_n=\frac{SSB/(k-1)}{SSW/(n-k)}
$$
가 큰 경우로 정리된다.

**2) $H_0$ 하에서의 분포 및 독립성**  
관측치를 집단별로 쌓아
$$
Y=(X_{11},\dots,X_{1n_1},\ X_{21},\dots,X_{kn_k})^\top\in\mathbb R^n
$$
라 두자. $H_0:\mu_1=\cdots=\mu_k(=\mu)$ 하에서는
$$
Y\sim N_n(\mu\mathbf 1,\ \sigma^2 I_n).
$$
이제 다음 두 개의 투영행렬(멱등행렬)을 정의한다.
* 집단별 평균으로의 투영:
$$
P_1=\mathrm{blockdiag}\!\left(\frac1{n_1}\mathbf 1_{n_1}\mathbf 1_{n_1}^\top,\ \dots,\ \frac1{n_k}\mathbf 1_{n_k}\mathbf 1_{n_k}^\top\right)
$$
* 전체 평균(상수항)으로의 투영:
$$
P_0=\frac1n\mathbf 1\mathbf 1^\top.
$$

그러면 제곱합들이 다음과 같이 이차형식으로 표현된다.
$$
SSW=Y^\top(I-P_1)Y,\qquad
SSB=Y^\top(P_1-P_0)Y.
$$
(실제로 $P_1Y$는 각 관측치를 "해당 집단의 평균"으로 바꾼 벡터이므로 $Y-P_1Y$가 집단 내 편차, $P_1Y-P_0Y$가 집단 평균의 전체평균으로부터의 편차를 모아준다.)

이제 표준화하여 $Z=(Y-\mu\mathbf 1)/\sigma\sim N_n(0,I_n)$로 두면
$$
\frac{SSW}{\sigma^2}=Z^\top(I-P_1)Z,\qquad
\frac{SSB}{\sigma^2}=Z^\top(P_1-P_0)Z.
$$

여기서
* $I-P_1$, $P_1-P_0$는 모두 멱등행렬이고,
* $(I-P_1)(P_1-P_0)=0$ (서로 직교하는 부분공간으로의 투영),
* 랭크는
$$
\mathrm{rank}(I-P_1)=n-k,\qquad \mathrm{rank}(P_1-P_0)=k-1
$$

따라서 **Cochran의 정리(정규벡터의 직교투영에 대한 카이제곱 분해)** 로부터
$$
\frac{SSB}{\sigma^2}\sim \chi^2(k-1),\qquad
\frac{SSW}{\sigma^2}\sim \chi^2(n-k),
\qquad
SSB\ \perp\!\!\!\perp\ SSW
$$

**3) 결론: $F$-통계량**  
따라서 $F$-분포의 정의로부터
$$
F_n=\frac{SSB/(k-1)}{SSW/(n-k)}
=
\frac{(SSB/\sigma^2)/(k-1)}{(SSW/\sigma^2)/(n-k)}
\sim F(k-1,n-k)
$$
이고, 기각역은 $F_n$의 큰 값으로 주어진다.  
즉, 유의수준 $\alpha$에서 $F_n\ge F_\alpha(k-1,n-k)$이면 $H_0$를 기각한다.

### 처리 효과 모수의 다른 정의 *(alternative parametrization)*
처리 효과를
$$
\tilde\alpha_i=\mu_i-\tilde\mu,\qquad \tilde\mu=\frac{1}{k}\sum_{i=1}^k\mu_i
$$
로 두고 일원분류정규분포모형을
$$
X_{ij}=\tilde\mu+\tilde\alpha_i+e_{ij},\qquad \sum_{i=1}^k\tilde\alpha_i=0
$$
처럼 쓰기도 한다(가중치 $n_i$가 아닌 단순합 제약). 이때의 유의성 검정 귀무가설도 결국 $H_0:\mu_1=\cdots=\mu_k$와 동치이므로 동일한 $F$-검정으로 처리한다.

#### 예 10.1.1  $\tilde\alpha_i$의 UMVU 및 동시신뢰구간 *(UMVU & simultaneous CIs for $\tilde\alpha_i$)*
* $\tilde\alpha_i=\mu_i-\tilde\mu$의 UMVU:
$$
\widehat{\tilde\alpha}_i=\hat \mu_i-\hat{\tilde \mu}=\bar X_i-\bar{\tilde X},
\qquad \bar{\tilde X}=\frac{1}{k}\sum_{i=1}^k\bar X_i
$$
* $\widehat{\tilde\alpha}_i$에 대한 동시신뢰구간(정리 10.1.3 (b) 적용 형태):
$$
P_{\mu,\sigma^2}\!\left(
|\widehat{\tilde\alpha}_i|
\le
\sqrt{\frac{\big((k-1)^2n_i^{-1}+\sum_{j\ne i}n_j^{-1}\big)\hat\sigma^2}{k^2}}\,
\sqrt{(k-1)F_\alpha(k-1,n-k)},
\ \forall i
\right)\ge 1-\alpha.
$$
* 한편 $\alpha_i=\mu_i-\bar\mu$에 대해서는(가중 평균 기준)
$$
P_{\mu,\sigma^2}\!\left(
|\hat\alpha_i|
\le
\sqrt{(n_i^{-1}-n^{-1})\hat\sigma^2}\,
\sqrt{(k-1)F_\alpha(k-1,n-k)},
\ \forall i
\right)\ge 1-\alpha.
$$


## 이원분류모형의 분산분석 *(Two-Way Analysis of Variance)*
두 개의 요인(*factor*) $A, B$가 각각 $a, b$개의 수준(*level*)을 가질 때, 각 수준 조합을 **처리(treatment)** 라 하고, 처리 효과를 분석하기 위해 **이원분류 정규분포 모형(two-way normal model)** 을 설정한다.

### 모형 설정 *(Two-way normal model)*
* 요인 $A$: 수준 $i=1,\dots,a$
* 요인 $B$: 수준 $j=1,\dots,b$
* $n_{ij}$: $(i,j)$칸에서 처리가 반복되는 횟수 

모형은
$$
X_{ijk} = \mu_{ij} + e_{ijk},
\qquad
e_{ijk}\stackrel{iid}{\sim} N(0,\sigma^2),
\qquad
-\infty<\mu_{ij}<\infty,\quad \sigma^2>0.
$$
여기서 $n_{ij}$는 셀(cell) $(i,j)$에서의 반복 횟수이다.

TODO: 
### 정리 10.2.1 이원분류정규분포모형에서의 추정량과 표본분포 *(Estimators and sampling distributions in two-way normal model)*
**(a) 전역최소분산불편 추정량 *(UMVU estimators)***  
* 모평균 추정량:
    $$
    \hat\mu_{ij} = \bar X_{ij\cdot}
    = \frac{1}{n_{ij}}\sum_{k=1}^{n_{ij}} X_{ijk}
    $$
* 모분산 추정량:
    $$
    \hat\sigma^2
    =
    \frac{1}{n-ab}
    \sum_{i=1}^a\sum_{j=1}^b\sum_{k=1}^{n_{ij}}
    (X_{ijk}-\bar X_{ij\cdot})^2,
    \qquad n=\sum_{i=1}^a\sum_{j=1}^b n_{ij}
    $$
    - ab가 빠지는 이유:
        - 모형을 가법모형(additive model, 상호작용 없음)으로 가정하면 A×B 상호작용항(ab)은 포함하지 않는다.
        - 또는 각 처리조합(셀)당 반복이 없어 상호작용을 따로 추정할 수 없는 경우, ab 성분이 오차(잔차)로 흡수되어 분해식/자유도에서 별도 항으로 나타나지 않는다

**(b) 표본분포**  
* $\bar X_{ij\cdot}$들은 서로 독립이고
    $$
    \bar X_{ij\cdot}\sim N\!\left(\mu_{ij},\frac{\sigma^2}{n_{ij}}\right)
    $$

**(c) 독립성**  
$$
\hat\sigma^2 \ \perp\!\!\!\perp\ (\bar X_{11\cdot},\dots,\bar X_{ab\cdot})
$$

**(d) 분산의 분포**  
$$
\frac{(n-ab)\hat\sigma^2}{\sigma^2}\sim\chi^2(n-ab)
$$

> 참고(최대가능도추정량)  
>이원분류 정규분포모형에서 모평균 $\mu_{ij}$의 최대가능도추정량은 각 셀의 표본평균
>$$
>\hat\mu_{ij,\mathrm{MLE}}=\bar X_{ij\cdot}
>$$
>임이 명백하다. 또한 모분산의 최대가능도추정량은
>$$ \hat\sigma^2_{\mathrm{MLE}}
>=\frac{1}{n}\sum_{i=1}^a\sum_{j=1}^b\sum_{k=1}^{n_{ij}}(X_{ijk}-\bar X_{ij\cdot})^2$$
>로 주어진다. 따라서 정리 10.2.1의 불편추정량 $\hat\sigma^2$와는 분모만 $n$ vs. $n-ab$로 다르다.
#### 증명
각 셀 $(i,j)$에서 $(X_{ij1},\dots,X_{ijn_{ij}})$는 정규표본이므로, 정규표본의 기본 성질을 각 셀에 적용하면 바로 성립한다. 서로 다른 셀의 표본이 독립이라는 점을 이용하면 전체 결과가 따른다.

### 정리 10.2.2 이원분류정규분포모형에서의 신뢰집합과 동시신뢰구간 *(Confidence regions and simultaneous confidence intervals)*
계수가 $p$인 $(ab)\times p$ 행렬 $C$에 대해, $C$의 열벡터공간(column space)을
$$
\mathrm{col}(C)=\{Cx:\ x\in\mathbb{R}^p\}\subset\mathbb{R}^{ab}
$$
로 둔다. 이때 $c\in\mathrm{col}(C)$는 $\mu=(\mu_{11},\dots,\mu_{ab})^\top$의 선형결합 $c^\top\mu$를 지정하는 계수벡터다.
$$
let \quad \hat\mu=(\hat\mu_{11},\dots,\hat\mu_{ab})^t,
\quad
D=\mathrm{Var}(\hat\mu)/\sigma^2=\mathrm{diag}(1/n_{ij})
$$
**(a) 신뢰집합**  
$$
P_{\mu,\sigma^2}\!\left(
(C^t\mu-C^t\hat\mu)^t
(C^tDC)^{-1}
(C^t\mu-C^t\hat\mu)
\le
p\hat\sigma^2 F_\alpha(p,n-ab)
\right)
=1-\alpha.
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
=1-\alpha.
$$
  - 이는 $\mathrm{col}(C)$에 속하는 **모든** 선형결합 $c^\top\mu$에 대해 동시에 커버리지를 보장하는 동시신뢰구간을 준다.
#### 증명
다변량 정규분포의 이차형식과 정리 10.2.1의 독립성, 그리고 F-분포의 정의를 그대로 적용하면 일원분류의 경우와 동일한 논리로 얻어진다.

### 균형된 이원분류모형 *(Balanced two-way ANOVA)*
균형된 경우($n_{ij}$가 일정)에는 각 처리조합 평균 $\mu_{ij}$를 **전반 평균 + A의 처리효과 + B의 처리효과 + 교호작용효과**로 분해해 해석하는 것이 표준적이다.

* **전반적인 처리효과(전반 평균, overall mean)**  
    모든 처리조합을 평균낸 기준 수준:
    $$
    \bar\mu=\frac{1}{ab}\sum_{i=1}^a\sum_{j=1}^b\mu_{ij}.
    $$

* **A 인자 수준 $i$의 처리효과 (A-level effect)**  
    수준 $i$의 평균이 전반 평균에서 얼마나 벗어나는지:
    $$
    \mu_{i\cdot}=\frac{1}{b}\sum_{j=1}^b\mu_{ij},\qquad
    \alpha_i=\mu_{i\cdot}-\bar\mu.
    $$

* **B 인자 수준 $j$의 처리효과 (B-level effect)**  
    수준 $j$의 평균이 전반 평균에서 얼마나 벗어나는지:
    $$
    \mu_{\cdot j}=\frac{1}{a}\sum_{i=1}^a\mu_{ij},\qquad
    \beta_j=\mu_{\cdot j}-\bar\mu.
    $$

* **주효과(main effect)**  
    각 요인이 단독으로 평균에 미치는 영향으로, 위의 $\alpha_i$, $\beta_j$를 각각 요인 $A$, $B$의 주효과라 부른다.

* **교호작용효과(interaction effect)**  
    가법모형(주효과의 합)로 설명되지 않는 비가법적(non-additive) 처리효과:
    $$
    \gamma_{ij}=\mu_{ij}-\mu_{i\cdot}-\mu_{\cdot j}+\bar\mu.
    $$
    특히 **가법모형(additive model, 상호작용 없음)** 은
    $$
    \gamma_{ij}=0\ \ \forall i,j
    $$
    인 경우를 의미한다.

반복 횟수가 모든 셀에서 같을 때,
$$
n_{ij}=r,\quad n=rab
$$
라 하면 **균형된 이원분류모형**이라 한다. 이 경우 해석과 분해가 아래와 같이 단순해진다.
$$
\bar\mu=\frac{1}{ab}\sum_{i=1}^a\sum_{j=1}^b\mu_{ij},
\quad
\mu_{i\cdot}=\frac{1}{b}\sum_{j=1}^b\mu_{ij},
\quad
\mu_{\cdot j}=\frac{1}{a}\sum_{i=1}^a\mu_{ij} \\
\alpha_i=\mu_{i\cdot}-\bar\mu,
\quad
\beta_j=\mu_{\cdot j}-\bar\mu,
\quad
\gamma_{ij}=\mu_{ij}-\mu_{i\cdot}-\mu_{\cdot j}+\bar\mu.
$$
* $\alpha_i$: 요인 $A$의 **주효과(main effect)**
* $\beta_j$: 요인 $B$의 **주효과(main effect)**
* $\gamma_{ij}$: **교호작용효과(interaction effect)**

이때 모형과 제약조건은
$$
X_{ijk}=\bar\mu+\alpha_i+\beta_j+\gamma_{ij}+e_{ijk} \\
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
\sqrt{\left(\frac1{br}+\frac1{br}\right)\hat\sigma^2}\,
\sqrt{(a-1)F_\alpha(a-1,n-ab)},
\ \forall i\neq\ell
\right)
\ge 1-\alpha.
$$

(b) $a^*=\alpha/m, m = a(a-1)/2$라 하면 
$$
P_{\mu, \sigma^2} \left( {|(\alpha_i-\alpha_\ell)-(\hat\alpha_i-\hat\alpha_\ell)| \leq \sqrt{\frac1{n_{i.}}+\frac1{n_{l.}}\hat\sigma^2} t_{\alpha^*/2}(n-ab), \forall i \neq l} \right)
\geq 1-\alpha
$$


### 교호작용효과의 유의성 검정 *(Test for interaction effect)*
balanced 이원분류정규분포모형에서 교호작용효과의 유의성에 대한 가설
$$
H_0^{AB}:\gamma_{ij}=0\ \forall i,j
\quad\text{vs}\quad
H_1^{AB}:\text{적어도 하나는 0이 아님}
$$
의 검정을 교호작용효과 유의성 검정이라 한다. 이런 가설은
$$
H_0^{AB}:\mu_{ij}\text{가 $i$의 함수와 $j$의 함수의 합으로 표현된다}
\quad\text{vs}\quad
H_1^{AB}:H_0^{AB}\text{가 아니다}
$$
라는 가설과 동치이며, 최대가능도비 검정은 아래와 같다.
### 정리 10.2.4 균형된 이원분류정규분포모형에서 교호작용효과의 유의성 검정 *(F-test for interaction effect)*
> 참고: 이런 최대가능도비 검정은 $n_{ij}=n_{i.}n_{.j}/n$를 만족시키는 이원분류정규분포모형의 경우에도 성립한다.
$$
SS_{AB}
=
\sum_{i=1}^a\sum_{j=1}^b
r(\bar X_{ij\cdot}-\bar X_{i\cdot\cdot}-\bar X_{\cdot j\cdot}+\bar X_{\cdots})^2
$$
$$
SSE=\sum_{i=1}^a\sum_{j=1}^b\sum_{k=1}^r (X_{ijk}-\bar X_{ij\cdot})^2
$$
이라고 하면, 검정통계량과 기각역은
$$
F_n
=
\frac{SS_{AB}/((a-1)(b-1))}
{SSE/(n-ab)}
\sim F((a-1)(b-1),n-ab) \\
F_n\ge F_\alpha((a-1)(b-1),n-ab).
$$

#### 증명  
**(1) 최대가능도비 검정(LRT)와 제곱합 분해**  
정규모형에서 로그가능도와 각 모수공간에서의 최대가능도추정값을 대입하면(상수항 제외)
$$
l(\theta)
=
-\frac{1}{2\sigma^2}\sum_{i=1}^a\sum_{j=1}^b\sum_{k=1}^r (x_{ijk}-\mu_{ij})^2
-\frac{n}{2}\log(2\pi\sigma^2),
\qquad n=rab.
$$

교호작용을 포함한 모형공간 $\Omega$에서는 $(\mu_{ij})$가 자유모수이므로
$$
\hat\mu_{ij}^\Omega=\bar x_{ij\cdot},
\qquad
\hat\sigma_\Omega^2
=\frac{1}{n}\sum_{i=1}^a\sum_{j=1}^b\sum_{k=1}^r (x_{ijk}-\bar x_{ij\cdot})^2
=\frac{SSE}{n},
$$
따라서
$$
\hat\theta^\Omega=\big(\bar x_{11\cdot},\dots,\bar x_{ab\cdot},\ \hat\sigma_\Omega^2\big).
$$

한편 $H_0^{AB}$ 하의 가법모형(교호작용 없음)에서는
$$
\mu_{ij}=\bar\mu+\alpha_i+\beta_j
$$
이고, 해당 평균의 MLE(최소제곱 적합값)는
$$
\hat\mu_{ij}^{\Omega_0}
=
\bar x_{i\cdot\cdot}+\bar x_{\cdot j\cdot}-\bar x_{\cdots},
$$
이므로 잔차제곱합은
$$
\sum_{i,j,k}\big(x_{ijk}-\hat\mu_{ij}^{\Omega_0}\big)^2
=
SSE+SS_{AB}.
$$
따라서
$$
\hat\sigma_0^2=\frac{SSE+SS_{AB}}{n},
\qquad
\hat\theta^{\Omega_0}=\big(\hat\mu_{11}^{\Omega_0},\dots,\hat\mu_{ab}^{\Omega_0},\ \hat\sigma_0^2\big).
$$

결국 정규모형에서 MLE를 대입한 로그가능도는
$$
l(\hat\theta)= -\frac{n}{2}\Big(1+\log(2\pi)+\log(\hat\sigma^2)\Big)
$$
꼴이므로, 위 식들을 대입하면 다음의 LRT 표현으로 이어진다.

$$
2\{l(\hat\theta^\Omega)-l(\hat\theta^{\Omega_0})\}
=n\log\!\left(\frac{\hat\sigma_0^2}{\hat\sigma_\Omega^2}\right).
$$
여기서 $\Omega$는 "교호작용 포함 모형", $\Omega_0$는 $H_0^{AB}:\gamma_{ij}=0$ (즉 가법모형) 하의 모형공간이다. $\log$는 단조증가함수이므로 LRT의 기각역은 $\hat\sigma_0^2/\hat\sigma_\Omega^2$가 큰 경우와 동치이고, 이는 (자유도 보정까지 포함하면) 통상적인
$$
F_n=\frac{SS_{AB}/((a-1)(b-1))}{SSE/(n-ab)}
$$
가 큰 경우로 정리된다. 따라서 남은 것은 $H_0^{AB}$ 하에서 $F_n$이 $F$-분포를 따름을 보이는 일이다.

**(2) 직교투영(orthogonal projection)으로 $SS_{AB}$를 이차형식으로 표현**  
관측치들을 한 벡터로 쌓아
$$
Y=(X_{111},\dots,X_{11r},\ X_{121},\dots,\ X_{abr})^\top\in\mathbb R^{n},\qquad n=rab
$$
라 두자. 또한 각 관측치 $X_{ijk}$를 다음 값으로 보내는 선형변환(투영)행렬 $\Pi$를 생각하자:
$$
X_{ijk}\ \mapsto\ 
\bar X_{ij\cdot}-\bar X_{i\cdot\cdot}-\bar X_{\cdot j\cdot}+\bar X_{\cdots},
$$
즉 벡터로 쓰면
$$
\Pi(Y)=\Big(\bar X_{ij\cdot}-\bar X_{i\cdot\cdot}-\bar X_{\cdot j\cdot}+\bar X_{\cdots}\Big)_{i,j,k}\in\mathbb R^n.
$$
(균형설계이므로 같은 셀 $(i,j)$의 $k=1,\dots,r$에 대해 위 값이 동일하게 반복된다.)

그러면 정의에 의해
$$
SS_{AB}
=\sum_{i=1}^a\sum_{j=1}^b\sum_{k=1}^r
\Big(\bar X_{ij\cdot}-\bar X_{i\cdot\cdot}-\bar X_{\cdot j\cdot}+\bar X_{\cdots}\Big)^2
=\|\Pi(Y)\|^2
=(\Pi Y)^\top(\Pi Y).
$$
또한 $\Pi$는 "교호작용(가법으로 설명되지 않는 부분)"으로의 **직교투영**이므로
$$
\Pi^2=\Pi,\qquad \Pi^\top=\Pi
$$
가 성립한다(멱등 + 대칭). 따라서
$$
SS_{AB} = Y^\top \Pi Y
$$
로 이차형식(quadratic form) 표현이 된다.

**(3) $H_0^{AB}$ 하에서 $SS_{AB}/\sigma^2$의 카이제곱 분포와 자유도 계산**  
$H_0^{AB}$ (가법모형) 하에서는
$$
X_{ijk}=\bar\mu+\alpha_i+\beta_j+e_{ijk},\qquad e_{ijk}\overset{iid}\sim N(0,\sigma^2).
$$
표준화하여
$$
Z_{ijk}=\frac{X_{ijk}-(\bar\mu+\alpha_i+\beta_j)}{\sigma}
$$
라 두면 $Z=(Z_{111},\dots,Z_{abr})^\top\sim N_n(0,I_n)$이다. 위의 $SS_{AB}$는 평균항이 투영에서 소거되므로
$$
\frac{SS_{AB}}{\sigma^2}
=
Z^\top \Pi Z.
$$
이제 정리 4.4.5(정규벡터 이차형식의 분포: $A^2=A$이면 $Z^\top A Z\sim\chi^2(\mathrm{trace}(A))$)를 적용하면
$$
\frac{SS_{AB}}{\sigma^2}\sim \chi^2(m),\qquad m=\mathrm{trace}(\Pi).
$$

이때 $m=\mathrm{trace}(\Pi)$는 투영공간(교호작용 부분공간)의 차원이고, 균형 이원배치에서 교호작용 자유도는
$$
m=(a-1)(b-1)
$$
이다(동일하게 $\mathrm{trace}(\Pi)$를 직접 계산해도 $(a-1)(b-1)$이 된다).

따라서
$$
\frac{SS_{AB}}{\sigma^2}\sim \chi^2((a-1)(b-1)).
$$

**(4) $SSE/\sigma^2$의 분포 및 $SS_{AB}$와의 독립성**  
또한
$$
SSE=\sum_{i=1}^a\sum_{j=1}^b\sum_{k=1}^r (X_{ijk}-\bar X_{ij\cdot})^2
$$
이고, 이는 셀 내부(within-cell) 편차만으로 구성되어
$$
\frac{SSE}{\sigma^2}\sim \chi^2(n-ab)
$$
가 성립한다(각 셀에서 자유도 $r-1$짜리 카이제곱이 합쳐져 $ab(r-1)=n-ab$).

마지막으로 $SS_{AB}$는 오직 셀 평균들 $(\bar X_{ij\cdot})$의 함수이고, $SSE$는 각 셀의 "평균으로부터의 편차 제곱합"의 합이므로 정리 10.2.1의 독립성(셀 평균과 셀 내 제곱합의 독립성, 그리고 서로 다른 셀의 독립성)으로부터
$$
SS_{AB}\ \perp\!\!\!\perp\ SSE
$$
가 성립한다.

**(5) 결론: $F$-분포**  
따라서 $H_0^{AB}$ 하에서 서로 독립인
$$
\frac{SS_{AB}}{\sigma^2}\sim \chi^2((a-1)(b-1)),\qquad
\frac{SSE}{\sigma^2}\sim \chi^2(n-ab)
$$
를 얻으므로 $F$-분포의 정의에 의해
$$
F_n=\frac{SS_{AB}/((a-1)(b-1))}{SSE/(n-ab)}
=\frac{(SS_{AB}/\sigma^2)/((a-1)(b-1))}{(SSE/\sigma^2)/(n-ab)}
\sim F((a-1)(b-1),\,n-ab).
$$
따라서 유의수준 $\alpha$에서 기각역은 $F_n\ge F_\alpha((a-1)(b-1),n-ab)$로 주어진다.  


같은 증명방법으로 주효과의 유의성 검정도 아래와 같이 주어지는 것을 밝힐 수 있다.
### 정리 10.2.5 균형된 이원분류정규분포모형에서 주효과의 유의성 검정 *(Tests for main effects)*
균형된 이원분류정규분포모형에서  
**(a) 요인 $A$의 주효과**  
$$
H_0^A:\alpha_1=\cdots=\alpha_a=0, H^A_1: not \  H_0^A \\
let \quad SSA=\sum_{i=1}^a br(\bar X_{i\cdot\cdot}-\bar X_{\cdots})^2 \\
F_n=\frac{SSA/(a-1)}{SSE/(n-ab)}\sim F(a-1,n-ab) \\
$$
이고 기각역은
$$
F_n \ge F_\alpha((a-1), n-ab)
$$

**(b) 요인 $B$의 주효과**  
$$
H_0^B:\beta_1=\cdots=\beta_b=0, H^B_1: not \  H_0^B \\
let \quad SSB=\sum_{j=1}^b ar(\bar X_{\cdot j\cdot}-\bar X_{\cdots})^2 \\
F_n=\frac{SSB/(b-1)}{SSE/(n-ab)}\sim F(b-1,n-ab)
$$
이고 기각역은
$$
F_n \ge F_\alpha((b-1), n-ab)
$$


## 분산분석에서의 검정력 함수 *(Power Functions in Analysis of Variance)*
분산분석에서 유의성 검정에 사용되는 F-통계량은 **귀무가설 하에서는 중심 F 분포**, **대립가설 하에서는 비중심 F 분포**를 따른다. 검정력 함수는 이 비중심성모수(noncentrality parameter)를 통해 표현된다.

### 비중심 카이제곱분포 *(Noncentral Chi-Square Distribution)*
서로 독립인 $X_i \sim N(\mu_i,1),\quad i=1,\dots,r$에 대해
$$
Y=\sum_{i=1}^r X_i^2
$$
의 분포를 **자유도 $r$**, **비중심성모수** $\delta=\sum_{i=1}^r \mu_i^2$
를 갖는 비중심 카이제곱분포라 하고
$$
Y\sim \chi^2(r;\delta)
$$
로 표기한다.  
이와 아래 정리의 (a), (b) 셋은 모두 동치다.
### 정리 10.3.1 비중심 카이제곱분포의 성질
서로 독립인 $X_i\sim N(\mu_i,1)\ (i=1,\dots,r)$에 대해
$$
Y=\sum_{i=1}^r X_i^2,\qquad \delta=\sum_{i=1}^r \mu_i^2
$$
이면 $Y\sim \chi^2(r;\delta)$라 한다. 이때 다음이 성립한다.

**(a) 누적생성함수(cgf)**  
$t<\tfrac12$에서 $Y$의 cgf는
$$
\mathrm{cgf}_Y(t)
=\sum_{k=1}^\infty \frac{t^k}{k!}\,2^{k-1}(k-1)!(r+k\delta).
$$

**(b) 혼합분포 표현 (Poisson mixture)**  
$$
f_Y(y)=\sum_{k=0}^\infty \Pr(K=k)\, f_{\chi^2(r+2k)}(y),
\qquad K\sim \mathrm{Poisson}(\delta/2).
$$
즉, 비중심 카이제곱분포는 **자유도가 증가하는 중심 카이제곱분포의 Poisson 혼합**으로 표현된다.

#### 증명
증명은 적률생성함수(mgf)와 그 로그(cgf)를 계산한 뒤, 급수 전개 및 mgf의 유일성으로 결론을 얻는다.

**1) $X\sim N(\mu,1)$일 때 $X^2$의 mgf 계산**  
$X\sim N(\mu,1)$이고 $t<\tfrac12$라 하자. 
$$
E\!\left(e^{tX^2}\right)
=\int_{-\infty}^{\infty} e^{tx^2}\frac{1}{\sqrt{2\pi}}
\exp\!\left(-\frac{(x-\mu)^2}{2}\right)\,dx.
$$
여기서 $t<\tfrac12$이므로 $a=\frac12-t>0$로 두면 $t-\frac12=-a$이고,
이를 적분에 대입하면
$$
E\!\left(e^{tX^2}\right)
=\frac{1}{\sqrt{2\pi}}
\exp\!\left(\frac{\mu^2}{4a}-\frac{\mu^2}{2}\right)
\int_{-\infty}^{\infty}\exp\!\left(-a\left(x-\frac{\mu}{2a}\right)^2\right)\,dx.
$$
가우스 적분 공식
$$
\int_{-\infty}^{\infty} e^{-a(x-b)^2}\,dx=\sqrt{\frac{\pi}{a}}\qquad(a>0)
$$
을 적용하면
$$
E\!\left(e^{tX^2}\right)
=\frac{1}{\sqrt{2\pi}}
\exp\!\left(\frac{\mu^2}{4a}-\frac{\mu^2}{2}\right)\sqrt{\frac{\pi}{a}}
=\frac{1}{\sqrt{2a}}\exp\!\left(\frac{\mu^2}{4a}-\frac{\mu^2}{2}\right).
$$
$$
=(1-2t)^{-1/2}\exp\!\left(\frac{\mu^2 t}{1-2t}\right),
\qquad t<\frac12.
$$

**2) $Y=\sum_{i=1}^r X_i^2$의 mgf 및 cgf**  
이제 $X_1,\dots,X_r$가 서로 독립이고 $X_i\sim N(\mu_i,1)$라 하자. 위 결과를 각 $X_i$에 적용하면
$$
E\!\left(e^{tX_i^2}\right)=(1-2t)^{-1/2}\exp\!\left(\frac{\mu_i^2 t}{1-2t}\right).
$$
독립성이므로
$$
M_Y(t)=E(e^{tY})
=E\!\left(\exp\!\left(t\sum_{i=1}^r X_i^2\right)\right)
=\prod_{i=1}^r E(e^{tX_i^2}) \\
=\prod_{i=1}^r (1-2t)^{-1/2}\exp\!\left(\frac{\mu_i^2 t}{1-2t}\right)
=(1-2t)^{-r/2}\exp\!\left(\frac{t}{1-2t}\sum_{i=1}^r \mu_i^2\right).
$$
$\delta=\sum_{i=1}^r\mu_i^2$이므로
$$
M_Y(t)=(1-2t)^{-r/2}\exp\!\left(\frac{\delta t}{1-2t}\right),\qquad t<\frac12.
$$
따라서 cgf $K_Y(t)=\log M_Y(t)$는
$$
K_Y(t)= -\frac{r}{2}\log(1-2t)+\frac{\delta t}{1-2t},\qquad t<\frac12.
$$

**3) (a) cgf의 거듭제곱급수 전개**  
$t<\tfrac12$에서 $|2t|<1$이므로 다음 급수 전개를 쓸 수 있다.
$$
-\log(1-x)=\sum_{k=1}^\infty \frac{x^k}{k}\qquad(|x|<1).
$$
여기서 $x=2t$를 대입하면
$$
-\log(1-2t)=\sum_{k=1}^\infty \frac{(2t)^k}{k}
=\sum_{k=1}^\infty \frac{2^k}{k}t^k.
$$
또한
$$
\frac{1}{1-2t}=\sum_{m=0}^\infty (2t)^m\qquad(|2t|<1)
$$
이므로
$$
\frac{t}{1-2t}
=t\sum_{m=0}^\infty (2t)^m
=\sum_{m=0}^\infty 2^m t^{m+1}
=\sum_{k=1}^\infty 2^{k-1} t^k.
$$
이제
$$
K_Y(t)= -\frac{r}{2}\log(1-2t)+\frac{\delta t}{1-2t}
$$
에 위 전개를 대입하면
$$
K_Y(t)
=\frac{r}{2}\sum_{k=1}^\infty \frac{2^k}{k}t^k
+\delta\sum_{k=1}^\infty 2^{k-1}t^k
=\sum_{k=1}^\infty\left(\frac{r}{2}\cdot\frac{2^k}{k}+\delta\cdot 2^{k-1}\right)t^k.
$$
계수를 정리하면
$$
\frac{r}{2}\cdot\frac{2^k}{k}+\delta\cdot 2^{k-1}
=2^{k-1}\left(\frac{r}{k}+\delta\right)
=2^{k-1}\frac{r+k\delta}{k}.
$$
따라서
$$
K_Y(t)=\sum_{k=1}^\infty 2^{k-1}\frac{r+k\delta}{k}\,t^k.
$$
이를 (문제의 표기처럼) $\frac{t^k}{k!}$ 형태로 쓰기 위해 $k=(k!)/( (k-1)!)$를 이용하면
$$
2^{k-1}\frac{r+k\delta}{k}\,t^k
=2^{k-1}(r+k\delta)\,t^k\cdot\frac{(k-1)!}{k!}
=\frac{t^k}{k!}\,2^{k-1}(k-1)!(r+k\delta).
$$
결국
$$
\mathrm{cgf}_Y(t)=K_Y(t)=\sum_{k=1}^\infty \frac{t^k}{k!}\,2^{k-1}(k-1)!(r+k\delta),
\qquad t<\frac12,
$$
이 되어 (a)가 증명된다.

**4) (b) Poisson 혼합 표현**  
위에서 얻은 mgf를 다시 쓴다:
$$
M_Y(t)=(1-2t)^{-r/2}\exp\!\left(\frac{\delta t}{1-2t}\right).
$$
지수항을 Poisson 형태가 나오도록 변형한다.
$$
\exp\!\left(\frac{\delta t}{1-2t}\right)
=\exp\!\left(\frac{\delta}{2}\left(\frac{1}{1-2t}-1\right)\right)
=e^{-\delta/2}\exp\!\left(\frac{\delta}{2}\cdot\frac{1}{1-2t}\right).
$$
또한 $e^u=\sum_{k=0}^\infty \frac{u^k}{k!}$이므로
$$
\exp\!\left(\frac{\delta}{2}\cdot\frac{1}{1-2t}\right)
=\sum_{k=0}^\infty \frac{1}{k!}\left(\frac{\delta}{2}\cdot\frac{1}{1-2t}\right)^k
=\sum_{k=0}^\infty \frac{(\delta/2)^k}{k!}(1-2t)^{-k}.
$$
따라서
$$
\exp\!\left(\frac{\delta t}{1-2t}\right)
=e^{-\delta/2}\sum_{k=0}^\infty \frac{(\delta/2)^k}{k!}(1-2t)^{-k}.
$$
이를 $M_Y(t)$에 대입하면
$$
M_Y(t)
=(1-2t)^{-r/2}\cdot e^{-\delta/2}\sum_{k=0}^\infty \frac{(\delta/2)^k}{k!}(1-2t)^{-k}
=e^{-\delta/2}\sum_{k=0}^\infty \frac{(\delta/2)^k}{k!}(1-2t)^{-(r/2+k)}.
$$
여기서
$$
(r/2+k)=\frac{r+2k}{2}
$$
이므로
$$
M_Y(t)=\sum_{k=0}^\infty \left(e^{-\delta/2}\frac{(\delta/2)^k}{k!}\right)\,(1-2t)^{-(r+2k)/2}.
$$
그런데 중심 카이제곱분포 $W\sim \chi^2(\nu)$의 mgf는
$$
M_W(t)=E(e^{tW})=(1-2t)^{-\nu/2},\qquad t<\frac12.
$$
따라서 위 식은
$$
M_Y(t)=\sum_{k=0}^\infty \Pr(K=k)\,M_{\chi^2(r+2k)}(t)
$$
와 같고, 여기서
$$
\Pr(K=k)=e^{-\delta/2}\frac{(\delta/2)^k}{k!},\qquad k=0,1,2,\dots
$$
이므로 $K\sim \mathrm{Poisson}(\delta/2)$이다.

즉, $Y$는
$$
Y\ \stackrel{d}{=}\ \chi^2(r+2K)\quad\text{(단, }K\sim\mathrm{Poisson}(\delta/2)\text{)}
$$
의 혼합으로 표현되며, 밀도로 쓰면
$$
f_Y(y)=\sum_{k=0}^\infty \Pr(K=k)\,f_{\chi^2(r+2k)}(y).
$$
여기서 중심 카이제곱분포의 밀도는 $y>0$에서
$$
f_{\chi^2(\nu)}(y)=\frac{1}{2^{\nu/2}\Gamma(\nu/2)}y^{\nu/2-1}e^{-y/2}
$$
이므로 (b)가 성립한다. $\square$

### 정리 10.3.2 이차형식의 분포
아래 정리는 정리4.4.5를 일반화한 것으로, 서로 독립인 정규분포를 따르는 확률변수들의 이차형식의 분포가 비중심 카이제곱분포일 조건을 주고 있다.  
$X\sim N(\mu,I)$이고 $A^2=A$이면
$$
X^\top A X \sim \chi^2(r;\delta),\quad
r=\mathrm{trace}(A),\quad
\delta=\mu^\top A\mu
$$
* 분산분석의 제곱합(SS)은 **정규벡터의 이차형식**이다.
* 귀무가설 하에서는 $(\delta=0)$이 되어 중심 카이제곱분포를 따른다.
* 대립가설 하에서는 $(\delta>0)$이 되어 비중심 분포를 따른다.

>비중심 F 분포 *(Noncentral F Distribution)*
>$$
>F=\frac{V_1/r_1}{V_2/r_2},\quad
>V_1\sim \chi^2(r_1;\delta),\ V_2\sim \chi^2(r_2) \\
>\Leftrightarrow F\sim F(r_1,r_2;\delta).
>$$
>여기서 V1, V2는 독립이고, $\delta$를 **비중심성모수**라 한다.

#### 증명  ( $X^\top A X\sim \chi^2(r;\delta)$ )
$X\sim N(\mu,I_n)$이고 $A$가 **멱등행렬(idempotent)**, 즉 $A^2=A$라고 하자. (또한 $X^\top A X$가 항상 실수/비음이 되도록 보통 $A$는 대칭 멱등으로 둔다.)

**1) 멱등행렬의 직교대각화**  
대칭 멱등행렬 $A$는 고유값이 $0$ 또는 $1$만 가지며, 어떤 직교행렬 $P$가 존재하여
$$
A=P
\begin{pmatrix}
I_r & 0\\
0 & 0
\end{pmatrix}
P^\top,
\qquad P^\top P=PP^\top=I_n
$$
로 쓸 수 있다. 여기서 $r=\mathrm{rank}(A)$이고, 대칭 멱등행렬에서는 $r=\mathrm{rank}(A)=\mathrm{trace}(A)$가 성립한다 (대각화에서 $1$의 개수가 곧 trace이자 rank).

**2) 정규벡터의 직교변환**  
>**정리 (직교변환의 불변성 / orthogonal invariance of MVN)**  
>$X\sim N_n(\mu,\Sigma)$이고 $P$가 직교행렬($P^\top P=I_n$)이면
>$$
>Z=P^\top X \sim N_n(P^\top\mu,\ P^\top\Sigma P).
>$$
>특히 $\Sigma=I_n$이면
>$$
>Z\sim N_n(P^\top\mu,\ I_n).
>$$
>즉 공분산이 항등행렬인 정규벡터는 직교변환(회전/반사)을 해도 공분산이 변하지 않는다. 이를 (공분산의) 불변성이라 한다.

$Z=P^\top X$라 두면, 직교변환의 불변성으로 $Z\sim N(P^\top\mu,\ P^\top I P)=N(\eta,I_n), \quad \eta=P^\top\mu$  
이때 $Z$를
$$
Z=\begin{pmatrix}Z_1\\ Z_2\end{pmatrix},\qquad
\eta=\begin{pmatrix}\eta_1\\ \eta_2\end{pmatrix}
\quad (Z_1,\eta_1\in\mathbb R^r)
$$
처럼 분할하면
$$
Z_1\sim N(\eta_1,I_r)
$$
이고, $Z_1$의 성분들은 서로 독립이다.

**3) 이차형식의 단순화 및 분포**  
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
=\sum_{i=1}^r Z_{1i}^2.
$$
각 $Z_{1i}\sim N(\eta_{1i},1)$가 독립이므로 정의에 의해
$$
\sum_{i=1}^r Z_{1i}^2 \sim \chi^2\!\left(r;\ \delta\right),
\qquad
\delta=\sum_{i=1}^r \eta_{1i}^2=\|\eta_1\|^2.
$$

**4) 비중심성 모수의 표현 $\delta=\mu^\top A\mu$**  
대각화 표현을 이용하면
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
=\delta.
$$

따라서
$$
X^\top A X \sim \chi^2(r;\delta),\quad
r=\mathrm{trace}(A),\quad
\delta=\mu^\top A\mu.
\qquad\square
$$

#### 예 10.3.1 일원분류 분산분석에서의 검정력 함수
모형과 가설, 일원분산분석의 검정통계량
$$
X_{ij}=\mu+\alpha_i+e_{ij},\quad
\sum_{i=1}^k n_i\alpha_i=0,\quad
e_{ij}\overset{iid}\sim N(0,\sigma^2),\\
H_0:\alpha_1=\cdots=\alpha_k=0
\quad\text{vs}\quad
H_1:\text{적어도 하나의 }\alpha_i\neq 0. \\
F_n=\frac{SSB/(k-1)}{SSW/(n-k)},
\qquad
SSB=\sum_{i=1}^k n_i(\bar X_i-\bar X)^2,
\quad
SSW=\sum_{i=1}^k\sum_{j=1}^{n_i}(X_{ij}-\bar X_i)^2.
$$

**1) $SSB$를 이차형식으로 표현**  
정리 10.1.5의 증명에서처럼 관측치를 한 벡터로 쌓아
$$
Y=(X_{11},\dots,X_{1n_1},\ X_{21},\dots,X_{kn_k})^\top\in\mathbb R^n
$$
라 두고, 다음 두 투영행렬을 정의한다:
* 집단별 평균으로의 투영
$$
P_1=\mathrm{blockdiag}\!\left(\frac1{n_1}\mathbf 1_{n_1}\mathbf 1_{n_1}^\top,\ \dots,\ \frac1{n_k}\mathbf 1_{n_k}\mathbf 1_{n_k}^\top\right),
$$
* 전체평균(상수)으로의 투영: $P_0=\frac1n\mathbf 1\mathbf 1^\top$ 이라 하면

여기서 $P_1Y$는 각 관측치를 "자기 집단 평균" $\bar X_i$로 치환한 벡터이고, $P_0Y$는 모든 관측치를 "전체평균" $\bar X$로 치환한 벡터이다. 따라서 $Y-P_1Y$는 집단 내 편차(잔차)를 모은 벡터, $P_1Y-P_0Y$는 집단 평균들의 전체평균으로부터의 편차를 모은 벡터가 되어, 각각의 제곱노름이 제곱합(SS)과 일치한다:
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
Z=\frac{Y-m}{\sigma}\sim N_n(0,I_n),\qquad \frac{Y}{\sigma}\sim N_n\!\left(\frac{m}{\sigma}, I_n\right).
$$
정리 10.3.2(정규벡터 이차형식)로부터
$$
\frac{SSB}{\sigma^2}
=\frac{1}{\sigma^2}Y^\top A Y
\sim \chi^2\!\left(k-1;\ \delta\right),
\qquad
\delta=\frac{1}{\sigma^2}m^\top A m.
$$

여기서 $m$의 구조를 이용해 $\delta$를 간단히 만들 수 있다. 모형에서 집단 $i$의 평균은 $\mu+\alpha_i$이므로
$$
m=(\mu+\alpha_1)\mathbf 1_{n_1}\oplus \cdots \oplus (\mu+\alpha_k)\mathbf 1_{n_k}.
$$
이때 $A=P_1-P_0$는 "상수항(전체평균)" 성분을 제거하므로 $\mu$는 소거되고, 결과적으로
$$
m^\top A m=\sum_{i=1}^k n_i\alpha_i^2
\quad(\text{그리고 }\sum_i n_i\alpha_i=0\text{ 제약과 일치})
$$
가 되어
$$
\boxed{\ \delta=\frac{1}{\sigma^2}\sum_{i=1}^k n_i\alpha_i^2\ }.
$$

**3) $SSW/\sigma^2$의 분포 및 독립성**  
정리 10.1.5의 (Cochran 정리) 논리와 동일하게
$$
\frac{SSW}{\sigma^2}\sim \chi^2(n-k),
\qquad
SSB\ \perp\!\!\!\perp\ SSW.
$$

**4) 결론: $F_n$의 비중심 $F$ 분포와 검정력**  
따라서 비중심 $F$ 분포의 정의로부터
$$
\boxed{\ F_n\sim F(k-1,\ n-k;\ \delta)\ },
\qquad
\delta=\frac{1}{\sigma^2}\sum_{i=1}^k n_i\alpha_i^2.
$$

유의수준 $\alpha$에서 기각역은 $F_n\ge F_\alpha(k-1,n-k)$이고, 검정력 함수는
$$
\boxed{\ \mathrm{Power}(\delta)
=
P_\delta\!\left(
F_n \ge F_\alpha(k-1,n-k)
\right)\ }.
$$
성질(해석)
* $\mathrm{Power}(\delta)$는 $\delta$의 **증가함수**이다. (효과가 커질수록/표본이 커질수록 검정력 증가)
* $\delta=0$ (즉 $H_0$)이면 $F_n$은 중심 $F$이므로 $\mathrm{Power}(0)=\alpha$.
* $\delta=\dfrac{1}{\sigma^2}\sum_i n_i\alpha_i^2$이므로  
    처리효과 크기($\alpha_i$)가 커지거나 표본크기($n_i$)가 커지거나, 오차분산($\sigma^2$)이 작아질수록 검정력이 커진다.

#### 예 10.3.2 이원분류 분산분석에서의 검정력 함수 *(power function in two-way ANOVA)*
반복횟수가 $r$인 **균형** 이원분류모형에서
$$
X_{ijk}=\mu+\alpha_i+\beta_j+\gamma_{ij}+e_{ijk},\qquad
e_{ijk}\overset{iid}\sim N(0,\sigma^2),
\qquad i=1,\dots,a,\ j=1,\dots,b,\ k=1,\dots,r,
$$
(식별을 위해 통상 $\sum_i\alpha_i=0,\ \sum_j\beta_j=0,\ \sum_i\gamma_{ij}=0,\ \sum_j\gamma_{ij}=0$ 제약을 둔다.)
전체 표본수는 $n=rab$이다.

**1) 교호작용효과 $A\times B$ 유의성 검정의 검정력**  
가설은
$$
H_0^{AB}:\gamma_{ij}=0\ \forall i,j
\quad\text{vs}\quad
H_1^{AB}:\text{적어도 하나의 }\gamma_{ij}\neq 0.
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
\frac{SS_{AB}}{\sigma^2}\sim \chi^2\!\big((a-1)(b-1);\ \delta_{AB}\big),
\qquad
\delta_{AB}=\frac{1}{\sigma^2}m^\top \Pi_{AB}m,
$$
여기서 $m=E(Y)$는 평균벡터이다. 균형설계에서는 교호작용 성분만이 $\Pi_{AB}$에 의해 남으며(주효과/전체평균은 소거),
$$
\boxed{\ \delta_{AB}=\frac{r}{\sigma^2}\sum_{i=1}^a\sum_{j=1}^b \gamma_{ij}^2\ }.
$$

또한 (정리 10.2.4의 독립성 논리와 동일하게)
$$
\frac{SSE}{\sigma^2}\sim \chi^2(n-ab),
\qquad SS_{AB}\ \perp\!\!\!\perp\ SSE \\
\therefore F_n=\frac{SS_{AB}/((a-1)(b-1))}{SSE/(n-ab)} \sim F\big((a-1)(b-1),\ n-ab;\ \delta_{AB}\big)\
$$

유의수준 $\alpha$에서 기각역은 $F_n\ge F_\alpha((a-1)(b-1),n-ab)$이고, 검정력 함수는
$$
\boxed{\ \mathrm{Power}_{AB}(\delta_{AB})
=
P_{\delta_{AB}}\!\left(
F_n\ge F_\alpha((a-1)(b-1),n-ab)
\right)\ }.
$$
비중심 $F$ 분포의 성질로 $\mathrm{Power}_{AB}(\delta_{AB})$는 $\delta_{AB}$의 **증가함수**이다.

**2) 주효과 $A$, $B$ 유의성 검정의 비중심도모수**  
정리 10.2.5의 주효과 검정도 같은 방식(투영에 대한 이차형식 $\Rightarrow$ 비중심 카이제곱 $\Rightarrow$ 비중심 $F$)으로 정리된다.

* **요인 $A$ 주효과**
$$
H_0^{A}:\alpha_1=\cdots=\alpha_a=0,
\qquad
SSA=\sum_{i=1}^a br(\bar X_{i\cdot\cdot}-\bar X_{\cdots})^2.
$$
대립가설 하에서
$$
\frac{SSA}{\sigma^2}\sim \chi^2(a-1;\delta_A),
\qquad
\boxed{\ \delta_A=\frac{br}{\sigma^2}\sum_{i=1}^a \alpha_i^2\ } \\
\therefore F_n^{A}=\frac{SSA/(a-1)}{SSE/(n-ab)}
\sim F(a-1,n-ab;\delta_A).
$$

* **요인 $B$ 주효과**
$$
H_0^{B}:\beta_1=\cdots=\beta_b=0,
\qquad
SSB=\sum_{j=1}^b ar(\bar X_{\cdot j\cdot}-\bar X_{\cdots})^2.
$$
대립가설 하에서
$$
\frac{SSB}{\sigma^2}\sim \chi^2(b-1;\delta_B),
\qquad
\boxed{\ \delta_B=\frac{ar}{\sigma^2}\sum_{j=1}^b \beta_j^2\ } \\
\therefore F_n^{B}=\frac{SSB/(b-1)}{SSE/(n-ab)}
\sim F(b-1,n-ab;\delta_B).
$$

각 검정력 함수 $\mathrm{Power}_A(\delta_A)$, $\mathrm{Power}_B(\delta_B)$ 역시 해당 $\delta$의 **증가함수**이다.

**3) (요약) 균형 이원분산분석표와 비중심도모수**  
| Source | SS | degree of freedom | MS | F-statistic | noncentrality $\delta$ (under $H_1$) |
|---|---:|---:|---:|---:|---:|
| A | $SSA=\sum_i br(\bar X_{i\cdot\cdot}-\bar X_{\cdots})^2$ | $a-1$ | $MSA=SSA/(a-1)$ | $F^A=MSA/MSE$ | $\displaystyle \delta_A=\frac{br}{\sigma^2}\sum_i \alpha_i^2$ |
| B | $SSB=\sum_j ar(\bar X_{\cdot j\cdot}-\bar X_{\cdots})^2$ | $b-1$ | $MSB=SSB/(b-1)$ | $F^B=MSB/MSE$ | $\displaystyle \delta_B=\frac{ar}{\sigma^2}\sum_j \beta_j^2$ |
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
    $$
    Y_i = x_{i0}\beta_0 + x_{i1}\beta_1 + \cdots + x_{ip}\beta_p + e_i,\qquad
    e_i\overset{iid}\sim N(0,\sigma^2).
    $$
* 벡터/행렬로 쓰면
    $$
    Y = X\beta + e,\qquad
    e\sim N_n(0,\sigma^2 I),\qquad
    \mathrm{rank}(X)=p+1,\quad \sigma^2>0.
    $$

    * $Y\in\mathbb{R}^n$, $X\in\mathbb{R}^{n\times(p+1)}$, $\beta\in\mathbb{R}^{p+1}$.

### 정리 10.4.1 선형회귀정규분포모형에서의 추정량과 표본분포
선형회귀 정규분포모형에서는 회귀계수의 **최대가능도추정량(MLE)** 과 **최소제곱추정량(OLS)** 이 일치한다(4장, 6장 참고).  

**(a) $c^\top\beta$와 $\sigma^2$의 전역최소분산불편추정량 *(UMVU estimators)***  
* 임의의 벡터 $c\in\mathbb{R}^{p+1}$에 대해
    $$
    \widehat{c^\top\beta}=c^\top\hat\beta,\qquad
    \hat\beta=(X^\top X)^{-1}X^\top Y
    $$
* 오차분산의 불편추정량
    $$
    \hat\sigma^2=\frac{\|Y-X\hat\beta\|^2}{n-p-1}
    =\frac{(Y-X\hat\beta)^\top(Y-X\hat\beta)}{n-p-1}
    $$
**(b) $\hat\beta$의 분포**  
$$
\hat\beta \sim N_{p+1}\!\big(\beta,\ \sigma^2(X^\top X)^{-1}\big)
$$
**(c) $\hat\beta$와 $\hat\sigma^2$의 독립성**  
$$
\hat\beta \perp \hat\sigma^2
$$
**(d) $\hat\sigma^2$의 카이제곱 분포**  
$$
\frac{(n-p-1)\hat\sigma^2}{\sigma^2}\sim \chi^2(n-p-1)
$$

#### 증명
정규선형회귀모형에서
$$
Y\sim N_n(X\beta,\ \sigma^2 I_n),\qquad \mathrm{rank}(X)=p+1
$$
라 하자. 또한 정사영행렬
$$
P:=X(X^\top X)^{-1}X^\top
$$

**(a) $\hat\beta$의 형태(OLS = MLE), 그리고 UMVU(완비충분통계량 + Lehmann–Scheffé)**  

**1) $\hat\beta$ (OLS = MLE) 및 불편성**  
정규분포의 로그가능도는 상수항을 제외하면
$$
\ell(\beta,\sigma^2)
= -\frac{1}{2\sigma^2}(Y-X\beta)^\top(Y-X\beta)-\frac{n}{2}\log\sigma^2.
$$
$\beta$에 대해 최대화하는 것은 $(Y-X\beta)^\top(Y-X\beta)$를 최소화하는 것과 같고, 정상방정식에서
$$
X^\top X\,\hat\beta=X^\top Y
\quad\Rightarrow\quad
\hat\beta=(X^\top X)^{-1}X^\top Y
$$
(유일성은 $\mathrm{rank}(X)=p+1$이므로 $X^\top X$ 가역). 따라서 OLS와 MLE가 일치한다.  
또한 $E(Y)=X\beta$이므로
$$
E(\hat\beta)
=(X^\top X)^{-1}X^\top E(Y)
=(X^\top X)^{-1}X^\top X\beta
=\beta,
$$
즉 $\hat\beta$는 불편이고, 임의의 $c$에 대해 $c^\top\hat\beta$는 $c^\top\beta$의 불편추정량이다.

**2) 완비충분통계량 (complete sufficient statistic)**  
$Y\sim N_n(X\beta,\sigma^2 I_n)$의 밀도는
$$
f(y;\beta,\sigma^2)
=(2\pi\sigma^2)^{-n/2}
\exp\!\left(-\frac{1}{2\sigma^2}(y-X\beta)^\top(y-X\beta)\right).
$$
전개하면
$$
(y-X\beta)^\top(y-X\beta)=y^\top y-2\beta^\top X^\top y+\beta^\top X^\top X\,\beta
$$
이므로
$$
f(y;\beta,\sigma^2)
=(2\pi\sigma^2)^{-n/2}
\exp\!\left(
\frac{1}{\sigma^2}\beta^\top X^\top y
-\frac{1}{2\sigma^2}y^\top y
-\frac{1}{2\sigma^2}\beta^\top X^\top X\,\beta
\right).
$$
따라서 이 모형은 자연통계량
$$
T(Y)=\big(X^\top Y,\ Y^\top Y\big)
$$
를 갖는 (정칙) 지수족이며, **분해정리(Factorization theorem)** 로부터 $T(Y)$는 $(\beta,\sigma^2)$에 대한 **충분통계량**이다.

또한 $\mathrm{rank}(X)=p+1$이면 자연모수
$$
\eta=\frac{\beta}{\sigma^2}\in\mathbb R^{p+1},\qquad
\tau=-\frac{1}{2\sigma^2}<0
$$
를 사용해 위 밀도를
$$
f(y;\eta,\tau)=h(y)\exp\!\left(\eta^\top X^\top y+\tau\,y^\top y-A(\eta,\tau)\right)
$$
꼴로 쓸 수 있고, 자연모수공간 $\{(\eta,\tau):\eta\in\mathbb R^{p+1},\ \tau<0\}$는 **열린 집합**(상대적 의미에서)이다.  
따라서 “자연모수공간이 열린 집합을 포함하는 full(정칙) 지수족 $\Rightarrow$ 자연통계량은 완비”라는 표준정리로부터 $T(Y)$는 **완비(complete)** 이다. 즉
$$
E_{\beta,\sigma^2}\{g(T(Y))\}=0\ \ \forall(\beta,\sigma^2)
\quad\Rightarrow\quad
P_{\beta,\sigma^2}(g(T(Y))=0)=1\ \ \forall(\beta,\sigma^2).
$$

마지막으로 $(\hat\beta,SSE)$는 $T(Y)$의 일대일 변환이다. 실제로
$$
\hat\beta=(X^\top X)^{-1}X^\top Y
$$
는 $X^\top Y$의 함수이고, 또
$$
SSE=\|Y-X\hat\beta\|^2
=Y^\top Y-\hat\beta^\top X^\top Y
=Y^\top Y-Y^\top X(X^\top X)^{-1}X^\top Y
$$
이므로 $(X^\top Y,\ Y^\top Y)$와 $(\hat\beta,\ SSE)$는 서로 일대일로 결정된다.  
따라서 $(\hat\beta,SSE)$ 역시 $(\beta,\sigma^2)$에 대한 **완비충분통계량**이다.

**3) UMVU (Lehmann–Scheffé 정리 적용)**  
이제
* $c^\top\hat\beta$는 $c^\top\beta$의 **불편추정량**이고,
* $c^\top\hat\beta$는 완비충분통계량 $(\hat\beta,SSE)$의 함수이므로,

**Lehmann–Scheffé 정리**에 의해
$$
\boxed{\ c^\top\hat\beta\ \text{는}\ c^\top\beta\ \text{의 UMVU 추정량이다.}\ }
$$

마찬가지로 (아래 (d)에서 보이듯) $E(SSE)=(n-p-1)\sigma^2$이므로
$$
\hat\sigma^2=\frac{SSE}{n-p-1}
$$
는 $\sigma^2$의 불편추정량이며 $(\hat\beta,SSE)$의 함수이므로,
$$
\boxed{\ \hat\sigma^2=SSE/(n-p-1)\ \text{는}\ \sigma^2\ \text{의 UMVU 추정량이다.}\ }
$$

(표본분포에 관한 b,c,d증명은 4.4.6에도 주어져 있다.)  
**(b) $\hat\beta$의 분포**  
$Y$가 정규이고 $\hat\beta$가 $Y$의 선형변환이므로 $\hat\beta$도 정규이며,
$$
\mathrm{Var}(\hat\beta)
=(X^\top X)^{-1}X^\top \mathrm{Var}(Y)X(X^\top X)^{-1}
=(X^\top X)^{-1}X^\top(\sigma^2 I)X(X^\top X)^{-1}
=\sigma^2(X^\top X)^{-1} \\
\therefore \hat\beta \sim N_{p+1}\!\big(\beta,\ \sigma^2(X^\top X)^{-1}\big).
$$

**(c) $\hat\beta$와 $\hat\sigma^2$의 독립성**  
적합값과 잔차를 $\hat Y=PY,\qquad \hat e=Y-\hat Y=(I-P)Y$ 로 두면,
$$
\hat Y \sim N_n(X\beta,\ \sigma^2 P),\qquad
\hat e \sim N_n(0,\ \sigma^2(I-P)) \\
\mathrm{Cov}(\hat Y,\hat e)=\mathrm{Cov}(PY,(I-P)Y)
=P\,\mathrm{Var}(Y)\,(I-P)
=\sigma^2 P(I-P)=0
$$
(정사영행렬 성질 $P^2=P$ 사용). $(\hat Y,\hat e)$는 $Y$의 선형변환이므로 공동정규이고, 공동정규에서 공분산 0이면 독립이므로
$$
\hat Y\ \perp\!\!\!\perp\ \hat e.
$$
한편 $\hat\beta$는 $\hat Y= X\hat\beta$의 함수이고, $\hat\sigma^2$는 $\|\hat e\|^2$의 함수이므로
$$
\hat\beta \ \perp\!\!\!\perp\ \hat\sigma^2.
$$

**(d) $(n-p-1)\hat\sigma^2/\sigma^2$의 카이제곱 분포**  
잔차제곱합을
$$
SSE=\|Y-X\hat\beta\|^2=\hat e^\top \hat e=Y^\top(I-P)Y
$$
라 하면 $I-P$는 대칭 멱등행렬이며
$$
\mathrm{rank}(I-P)=n-\mathrm{rank}(P)=n-(p+1)=n-p-1.
$$
또한 평균을 제거한 정규벡터 $Z:=\hat e/\sigma \sim N_n(0,\ I-P)$에 대해, $I-P$의 직교대각화(고유값이 1인 축이 $n-p-1$개)를 쓰면
$$
\frac{SSE}{\sigma^2}=\frac{\hat e^\top\hat e}{\sigma^2}=Z^\top Z
\sim \chi^2(n-p-1).
$$
따라서
$$
\frac{(n-p-1)\hat\sigma^2}{\sigma^2}
=\frac{SSE}{\sigma^2}\sim \chi^2(n-p-1).
$$

**(UMVU 언급)**  
이 모형은 완비 지수족이며 $(\hat\beta, SSE)$ (동치로 $(X^\top Y, SSE)$)는 $(\beta,\sigma^2)$에 대한 완비충분통계량이 된다. 위에서 $c^\top\hat\beta$와 $\hat\sigma^2=SSE/(n-p-1)$가 각각 불편이고 완비충분통계량의 함수이므로(Lehmann–Scheffé 정리) $c^\top\beta$와 $\sigma^2$의 UMVU 추정량이다.  
$\square$

### 정리 10.4.2 선형회귀정규분포모형에서의 신뢰집합과 동시신뢰구간
계수가 $r$인 $(p+1)\times r$ 행렬 $C$에 대해, $C$의 열공간을 $\mathrm{col}(C)$라 하자.

**(a) $C^\top\beta$에 대한 신뢰집합 *(confidence set)***  
$$
P_{\beta,\sigma^2}\!\left(
(C^\top\beta-C^\top\hat\beta)^\top\Big(C^\top(X^\top X)^{-1}C\Big)^{-1}(C^\top\beta-C^\top\hat\beta)
\le r\hat\sigma^2\,F_\alpha(r,n-p-1)
\right)=1-\alpha.
$$

**(b) $\mathrm{col}(C)$ 상의 모든 선형결합에 대한 동시신뢰구간 *(simultaneous CI)***  
$$
P_{\beta,\sigma^2}\!\left(
|c^\top\beta-c^\top\hat\beta|
\le \sqrt{c^\top(X^\top X)^{-1}c\ \hat\sigma^2}\ \sqrt{rF_\alpha(r,n-p-1)},
\ \forall c\in \mathrm{col}(C)
\right)=1-\alpha.
$$

* 해석: $c$를 $C$의 열공간 안에서 움직이는 "많은" 선형결합에 대해 한 번에 커버리지를 보장하는 구간이다.

### 정리 10.4.3 선형회귀정규분포모형에서 회귀계수의 유의성 검정
회귀계수 전체의 유의성보단 일부의 유의성에 대한 판단이 필요한 경우가 많다. 예6.5.1처럼 절편$\beta_0$을 제외 한 회귀계수가 관심의 대상인 것 처럼. 이런 경우 가설 검정을 아래와 같이 할 수 있다.  

일부 회귀계수 블록 검정 *(partial/regression block test)*  
* 설계행렬을 블록으로 쪼갠다:
    $$
    Y=X_0\beta_0 + X_1\beta_1 + e \\
    \beta_0\in\mathbb{R}^{p_0},\quad \beta_1\in\mathbb{R}^{p_1},\quad
    X_0\in\mathbb{R}^{n\times p_0},\quad X_1\in\mathbb{R}^{n\times p_1}, \\
    e\sim N_n(0,\sigma^2 I_n),\quad \sigma^2>0 \\
    \mathrm{rank}(X_0,X_1)=p_0+p_1,\quad \mathrm{rank}(X_0)=p_0,\quad \mathrm{rank}(X_1)=p_1.
    $$
* 회귀계수 유의성에 대한 가설:
    $$
    H_0:\beta_1=0\quad \text{vs}\quad H_1:\beta_1\neq 0.
    $$
가설의 최대가능도비 검정에 대해 아래가 성립한다.  
**(a) 투영행렬과 분해**  
* let $X_0$로의 정사영행렬
    $$
    \Pi_0 = X_0(X_0^\top X_0)^{-1}X_0^\top.
    $$
* let $X_0$를 제거한 $X_1$
    $$
    X_{1|0}=(I-\Pi_0)X_1.
    $$
* let 그 열공간으로의 정사영행렬
    $$
    \Pi_{1|0}=X_{1|0}(X_{1|0}^\top X_{1|0})^{-1}X_{1|0}^\top.
    $$
이면,     $$
    \boxed{\ \min_{\beta_0}\|Y-X_0\beta_0\|^2
    =
    \min_{\beta_0,\beta_1}\|Y-X_0\beta_0-X_1\beta_1\|^2
    +Y^\top\Pi_{1|0}Y\ }.
    $$
**(b) 검정통계량 (최대가능도비 검정과 동치 형태)**  
* let 회귀로 설명되는 제곱합(추가 설명력)
    $$
    R(10)=Y^\top\Pi_{1|0}Y.
    $$
* let 오차제곱합
    $$
    SSE = Y^\top(I-\Pi_0-\Pi_{1|0})Y.
    $$
이면 검정통계량 $F$-통계량은
$$
F_n=\frac{R(10)/p_1}{SSE/(n-p_0-p_1)}.
$$
이고, 수준 $\alpha$의 기각역은
$$
F_n \ge F_\alpha(p_1,\ n-p_0-p_1).
$$

**(c) 대립가설 하 분포와 검정력 *(power)***  
* 대립가설 하에서 검정통계량에 대해
    $$
    F_n = \frac{R(1|0)/p_1}{SSE/(n-p_0-p_1)} \sim F(p_1,\ n-p_0-p_1;\ \delta),
    $$
    이 성립하고, 여기서 $\delta$는 비중심도모수 *(noncentrality parameter)*:
    $$
    \delta = \frac{\beta_1^\top X_1^\top\Pi_{1|0}X_1\beta_1}{\sigma^2}.
    $$
* 검정력 함수는 $\delta$의 증가함수다.
    $$
    \pi(\delta)=P_\delta\!\left(F_n\ge F_\alpha(p_1,n-p_0-p_1)\right)
    $$
    * 해석: 효과크기($\beta_1$)가 커지거나, 잡음($\sigma^2$)이 작아지거나, 설계가 좋아져 $\Pi_{1|0}$ 방향으로 신호가 커질수록 검정력이 증가한다.

#### 증명 (TODO: 일단 pass 교재랑 비교 공부)
**(a) 투영행렬과 분해**

$X_0$의 열공간으로의 정사영을 $\Pi_0:=X_0(X_0^\top X_0)^{-1}X_0^\top$라 두고 $M_0:=I-\Pi_0$라 하자. 그러면
$$
\min_{\beta_0}\|Y-X_0\beta_0\|^2=\|M_0Y\|^2=Y^\top M_0Y.
$$

또한 $X_{1|0}:=M_0X_1$로 두면 $\mathrm{col}(X_{1|0})\subset \mathrm{col}(X_0)^\perp$ 이고, 그 열공간으로의 정사영은
$$
\Pi_{1|0}:=X_{1|0}(X_{1|0}^\top X_{1|0})^{-1}X_{1|0}^\top
$$
이다. (가정 $\mathrm{rank}(X_0,X_1)=p_0+p_1$로부터 $\mathrm{rank}(X_{1|0})=p_1$이므로 역행렬 존재.)

$\mathrm{col}(X_0)\perp \mathrm{col}(X_{1|0})$ 이므로
$$
\Pi_0\Pi_{1|0}=0,\qquad \Pi_{1|0}\Pi_0=0,\qquad \Pi_{1|0}M_0=\Pi_{1|0}.
$$
따라서 $M_0Y$를 $\mathrm{col}(X_{1|0})\oplus \mathrm{col}(X_{1|0})^\perp$로 직교분해하면
$$
M_0Y=\Pi_{1|0}Y+(M_0-\Pi_{1|0})Y=\Pi_{1|0}Y+(I-\Pi_0-\Pi_{1|0})Y
$$
이고 직교성이므로 피타고라스 정리에 의해
$$
\|M_0Y\|^2=\|\Pi_{1|0}Y\|^2+\|(I-\Pi_0-\Pi_{1|0})Y\|^2.
$$
즉
$$
\min_{\beta_0}\|Y-X_0\beta_0\|^2
=
Y^\top\Pi_{1|0}Y
+
Y^\top(I-\Pi_0-\Pi_{1|0})Y.
$$
한편 전체모형 $Y=X_0\beta_0+X_1\beta_1+e$에서의 최소제곱 잔차제곱합은
$$
\min_{\beta_0,\beta_1}\|Y-X_0\beta_0-X_1\beta_1\|^2=\|(I-\Pi_0-\Pi_{1|0})Y\|^2
$$
이므로 결론적으로
$$
\min_{\beta_0}\|Y-X_0\beta_0\|^2
=
\min_{\beta_0,\beta_1}\|Y-X_0\beta_0-X_1\beta_1\|^2
+Y^\top\Pi_{1|0}Y
$$
가 성립한다.

**(b) $H_0$ 하에서의 분포, 독립성, 그리고 $F$-통계량**

귀무가설 $H_0:\beta_1=0$ 하에서는 $Y\sim N_n(X_0\beta_0,\sigma^2I)$이고 $X_0\beta_0\in\mathrm{col}(X_0)$이므로
$$
\Pi_{1|0}X_0\beta_0=0,\qquad (I-\Pi_0-\Pi_{1|0})X_0\beta_0=0.
$$
따라서
$$
\frac{R(1|0)}{\sigma^2}=\frac{Y^\top\Pi_{1|0}Y}{\sigma^2}\sim \chi^2(p_1),\qquad
\frac{SSE}{\sigma^2}=\frac{Y^\top(I-\Pi_0-\Pi_{1|0})Y}{\sigma^2}\sim \chi^2(n-p_0-p_1),
$$
여기서 자유도는 각각
$$
\mathrm{rank}(\Pi_{1|0})=p_1,\qquad \mathrm{rank}(I-\Pi_0-\Pi_{1|0})=n-p_0-p_1.
$$
또한 $\Pi_{1|0}$와 $I-\Pi_0-\Pi_{1|0}$는 서로 직교하는 부분공간으로의 대칭 멱등투영이므로(Cochran 정리),
$$
R(1|0)\ \perp\!\!\!\perp\ SSE.
$$
따라서
$$
F_n=\frac{R(1|0)/p_1}{SSE/(n-p_0-p_1)}\sim F(p_1,\ n-p_0-p_1),
$$
기각역은 $F_n\ge F_\alpha(p_1,n-p_0-p_1)$로 주어진다.

**(c) 대립가설 하 비중심 $F$ 및 비중심도모수**

$H_1$ 하에서는 $Y\sim N_n(m,\sigma^2I)$, $m=X_0\beta_0+X_1\beta_1$이다. 그러면 (정규벡터의 멱등행렬 이차형식 분포)
$$
\frac{R(1|0)}{\sigma^2}=\frac{Y^\top\Pi_{1|0}Y}{\sigma^2}\sim \chi^2\big(p_1;\delta\big),
\qquad
\delta=\frac{1}{\sigma^2}m^\top\Pi_{1|0}m.
$$
여기서 $\Pi_{1|0}X_0=0$이므로
$$
\delta=\frac{1}{\sigma^2}(X_1\beta_1)^\top\Pi_{1|0}(X_1\beta_1)
=\frac{1}{\sigma^2}\beta_1^\top X_1^\top\Pi_{1|0}X_1\beta_1.
$$
또한 $SSE/\sigma^2\sim \chi^2(n-p_0-p_1)$는 (평균이 제거되는 투영이므로) 중심 카이제곱이고, 위와 같은 직교투영에 의해 $R(1|0)\perp\!\!\!\perp SSE$가 유지된다. 따라서
$$
F_n=\frac{R(1|0)/p_1}{SSE/(n-p_0-p_1)}\sim F(p_1,\ n-p_0-p_1;\delta),
$$
검정력은
$$
\pi(\delta)=P_\delta\!\left(F_n\ge F_\alpha(p_1,n-p_0-p_1)\right)
$$
로 주어지며 $\delta$의 증가함수이다.  $\square$

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
\min_{\beta:\,C^\top\beta=0}\ \|Y-X\beta\|^2
=
\min_{\beta}\ \|Y-X\beta\|^2
+
Y^\top\Pi_{1|0}Y\ }.
$$

또한 $\hat\beta=(X^\top X)^{-1}X^\top Y$이고 $\mathrm{Var}(C^\top\hat\beta)=\sigma^2\,C^\top(X^\top X)^{-1}C$이므로,
$$
\boxed{\ 
\frac{Y^\top\Pi_{1|0}Y}{\sigma^2}
=
(C^\top\hat\beta)^\top\Big(\sigma^2\,C^\top(X^\top X)^{-1}C\Big)^{-1}(C^\top\hat\beta)
=
(C^\top\hat\beta)^\top\big[\mathrm{Var}(C^\top\hat\beta)\big]^{-1}(C^\top\hat\beta)\ }.
$$
**(b) 검정통계량**  
* 전체 모형의 정사영행렬 $\Pi_{1,0}=X(X^\top X)^{-1}X^\top,\quad R(1|0)=Y^\top\Pi_{1|0}Y,\quad, SSE=Y^\top(I-\Pi_{1,0})Y$ 이라 하면, 검정통계량과 기각역은 다음과 같다:
$$
F_n=\frac{R(1|0)/r}{SSE/(n-p-1)} \\
F_n \ge F_\alpha(r,\ n-p-1)
$$

**(c) 대립가설 하 비중심 $F$와 비중심도모수**  
검정통계량에 대해 아래가 성립하고, 검정력함수는 $\delta$의 증가함수다
$$
F_n\sim F(r,\ n-p-1;\ \delta) \\
\delta = \frac{\beta^\top C\Big(C^\top(X^\top X)^{-1}C\Big)^{-1}C^\top\beta}{\sigma^2}.
$$

#### 증명
정규선형회귀모형
$$
Y\sim N_n(X\beta,\ \sigma^2 I_n),\qquad \mathrm{rank}(X)=p+1
$$
을 가정한다. 또한
$$
\hat\beta=(X^\top X)^{-1}X^\top Y,\qquad
P:=X(X^\top X)^{-1}X^\top,\qquad
M:=I-P
$$
로 둔다.

**(a) 제약 최소제곱과 제곱합 분해**  
**1) 제약 최소제곱해의 형태**  
제약문제
$$
\min_{\beta:\ C^\top\beta=0}\ \|Y-X\beta\|^2
$$
를 라그랑주 승수로 풀면, 어떤 $\lambda\in\mathbb R^r$에 대해 정상조건
$$
X^\top X\,\tilde\beta + C\lambda = X^\top Y,\qquad C^\top\tilde\beta=0
$$
을 만족하는 $\tilde\beta$가 해가 된다. 첫 식에서
$$
\tilde\beta=\hat\beta-(X^\top X)^{-1}C\lambda
$$
이고 이를 제약식에 대입하면
$$
0=C^\top\tilde\beta
= C^\top\hat\beta - C^\top(X^\top X)^{-1}C\,\lambda
$$
이므로 (가정 $\mathrm{rank}(C)=r$로 $C^\top(X^\top X)^{-1}C$ 가역)
$$
\lambda=\Big(C^\top(X^\top X)^{-1}C\Big)^{-1}C^\top\hat\beta.
$$
따라서
$$
\tilde\beta
=
\hat\beta
-(X^\top X)^{-1}C\Big(C^\top(X^\top X)^{-1}C\Big)^{-1}C^\top\hat\beta.
$$

**2) SSE의 증가량(추가 제약의 비용)**  
전체모형의 잔차제곱합은
$$
SSE=\min_\beta\|Y-X\beta\|^2=\|MY\|^2=Y^\top MY.
$$
제약모형의 잔차제곱합을 $SSE_0:=\|Y-X\tilde\beta\|^2$라 하면, 표준적인 최소제곱의 직교분해 성질로
$$
SSE_0 - SSE
= (\hat\beta-\tilde\beta)^\top X^\top X (\hat\beta-\tilde\beta).
$$
위에서 구한 $\hat\beta-\tilde\beta=(X^\top X)^{-1}C\lambda$를 대입하면
$$
SSE_0-SSE
=\lambda^\top C^\top(X^\top X)^{-1}C\,\lambda
=(C^\top\hat\beta)^\top\Big(C^\top(X^\top X)^{-1}C\Big)^{-1}(C^\top\hat\beta).
$$
또한 $\hat\beta=(X^\top X)^{-1}X^\top Y$이므로
$$
(C^\top\hat\beta)^\top\Big(C^\top(X^\top X)^{-1}C\Big)^{-1}(C^\top\hat\beta)
=
Y^\top \Pi_{1|0}Y
$$
가 되며, 여기서 $\Pi_{1|0}$는 정리에 제시된 행렬이다. 따라서
$$
\min_{\beta:\,C^\top\beta=0}\|Y-X\beta\|^2
=
\min_\beta\|Y-X\beta\|^2 + Y^\top\Pi_{1|0}Y
$$
가 성립한다.

**(b) $H_0$ 하에서의 분포와 $F$-검정**  
귀무가설 $H_0:C^\top\beta=0$ 하에서는
$$
C^\top\hat\beta \sim N_r\!\Big(0,\ \sigma^2\,C^\top(X^\top X)^{-1}C\Big).
$$
따라서 정규벡터의 이차형식 성질로
$$
\frac{R(1|0)}{\sigma^2}
=
\frac{(C^\top\hat\beta)^\top\big(C^\top(X^\top X)^{-1}C\big)^{-1}(C^\top\hat\beta)}{\sigma^2}
\sim \chi^2(r).
$$
또한
$$
\frac{SSE}{\sigma^2}=\frac{Y^\top MY}{\sigma^2}\sim \chi^2(n-p-1).
$$
여기서 $Y^\top\Pi_{1|0}Y$와 $Y^\top MY$는 서로 직교하는 부분공간으로의 투영에 대응하는 이차형식이므로(대칭 멱등행렬의 곱이 0인 경우) Cochran 정리로 독립이다. 따라서
$$
F_n=\frac{R(1|0)/r}{SSE/(n-p-1)}\sim F(r,\ n-p-1)
$$
이고, 기각역은 $F_n\ge F_\alpha(r,n-p-1)$로 주어진다.

**(c) $H_1$ 하에서의 비중심 $F$ 및 비중심도모수**  
대립가설 하에서는 $C^\top\beta\neq 0$이고
$$
C^\top\hat\beta \sim N_r\!\Big(C^\top\beta,\ \sigma^2\,C^\top(X^\top X)^{-1}C\Big).
$$
따라서
$$
\frac{R(1|0)}{\sigma^2}\sim \chi^2\!\big(r;\delta\big),
\qquad
\delta
=
\frac{(C^\top\beta)^\top\big(C^\top(X^\top X)^{-1}C\big)^{-1}(C^\top\beta)}{\sigma^2}
=
\frac{\beta^\top C\big(C^\top(X^\top X)^{-1}C\big)^{-1}C^\top\beta}{\sigma^2}.
$$
한편 $SSE/\sigma^2\sim\chi^2(n-p-1)$는 여전히 중심 카이제곱이고, 위와 같은 직교투영 구조로 독립성이 유지된다. 따라서 비중심 $F$의 정의로
$$
F_n\sim F(r,\ n-p-1;\delta)
$$
가 성립한다. $\square$

### 선형모형 *(General Linear Model)* 로의 확장
지금까지는 $\mathrm{rank}(X)=p+1$로 $X^\top X$가 가역인 경우를 다루었다. 그러나 분산분석(ANOVA) 같은 모형은 설계행렬 열들이 선형종속이어서 $\mathrm{rank}(X)$가 열 개수보다 작은 경우가 많다. 이를 제약조건$C^\top\beta=0$을 추가한 선형모형으로 표현한다:
$$
Y=X\beta+e,\quad C^\top\beta=0,\quad e\sim N_n(0,\sigma^2 I).
$$
$$
\mathrm{rank}(X)=p+1-r<p+1,\quad \mathrm{rank}(C^\top)=r,\quad X\text{와 }C^\top\text{의 행들은 선형독립}.
$$
이런 모형을 선형모형(linear model)이라 하며, 이는 일원분류모형, 이원분류모형 등 분산분석 모형이나 선형회귀모형을 모두 포괄하는 모형이다.

#### 예 10.4.1 일원분류 정규분포모형을 선형모형으로 쓰기 (설계행렬을 실제로 적기)
일원분류모형을
$$
X_{ij}=\mu+\alpha_i+e_{ij},\qquad 
\sum_{i=1}^k n_i\alpha_i=0,\qquad 
e_{ij}\stackrel{iid}{\sim}N(0,\sigma^2)
$$
로 두자. 관측치를 한 벡터로 쌓아
$$
Y=(X_{11},\dots,X_{1n_1},\ X_{21},\dots,X_{kn_k})^\top\in\mathbb R^n,\qquad n=\sum_{i=1}^k n_i
$$
라 하면 다음과 같이 선형모형 $Y=X\beta+e$로 쓸 수 있다.

**1) 설계행렬 $X$ (상수항 + 집단 더미)**  
모수벡터를
$$
\beta=(\mu,\alpha_1,\dots,\alpha_k)^\top\in\mathbb R^{k+1}
$$
로 잡고, 설계행렬을 “상수항 1개 + 집단 더미 $k$개”로 잡으면
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
$$
Y=X\beta+e
$$
는 각 관측치에 대해 $X_{ij}=\mu+\alpha_i+e_{ij}$를 정확히 재현한다.

**2) $\mathrm{rank}(X)=k<k+1$ (왜 $X^\top X$가 특이해지는가?)**  

위 $X$는 열이 $k+1$개지만 다음 선형관계가 항상 성립한다:
$$
\text{(상수항 열)}=\sum_{i=1}^k \text{(집단 }i\text{ 더미 열)}.
$$
따라서
$$
\mathrm{rank}(X)=k<k+1,\qquad X^\top X\ \text{는 정칙이 아니며 }(X^\top X)^{-1}\text{가 존재하지 않는다.}
$$

**3) 제약조건 $c^\top\beta=0$를 추가한 “선형모형” 관점 (식별성)**  

이 경우 책에서처럼 모수에 제약을 추가하여 모형을 식별한다.  
일원분류모형의 제약 $\sum_i n_i\alpha_i=0$는
$$
c^\top\beta=0,\qquad c^\top=(0,\ n_1,\dots,n_k)
$$
로 쓸 수 있다. 즉,
$$
\boxed{
Y=X\beta+e,\quad c^\top\beta=0,\quad e\sim N_n(0,\sigma^2I)
}
$$
가 “제약조건이 있는 선형모형(linear model with constraints)”의 형태다.

* 핵심: **설계행렬 $X$ 자체는 랭크가 부족하지만**, 모수공간을 $\{\beta:\ c^\top\beta=0\}$로 제한하면 (즉 불필요한 자유도를 제거하면) **모형이 식별 가능**해진다.

**4) $X^\top X$의 역행렬이 없을 때의 추정/검정(개념)**  
$X^\top X$가 특이하므로 정리 10.4.4에서처럼 $(X^\top X)^{-1}$로 정사영행렬을 쓰는 표현은 그대로 사용할 수 없다. 대신

* 제약 최소제곱(constrained LS)로 $\hat\beta$를 정의하거나,
* 일반화 역행렬(generalized inverse) 등을 사용하여 정사영행렬(또는 그에 준하는 투영)을 표현한다.

이때 정사영행렬의 “구체적 공식”은 표현 선택(제약식/기저 선택)에 따라 달라질 수 있으나,
**투영(Projection)으로 제곱합(SS)을 분해하고, 그 비로 $F$-검정을 만들며, 대립가설 하에서 비중심 $F$가 된다**는 큰 구조는 정리 10.4.4와 같은 방식으로 전개될 수 있다.

> 참고: 같은 자료를 “상수항 없이 집단 더미 $k$개만”으로 두면 설계행렬이 $n\times k$가 되어 $\mathrm{rank}(X)=k$ (보통)로 만들 수 있지만, 이 경우 $\mu,\alpha_i$의 해석(모수화)이 위 표현과 달라진다.

#### (참고) 10.3과의 연결(검정력)
* 위의 회귀 $F$-검정에서 대립가설 하 분포가 비중심 $F$이고, 검정력이 $\delta$의 증가함수라는 결론은 10.3에서 정리한 비중심 카이제곱/비중심 $F$의 성질을 사용해 얻는 구성이다.
