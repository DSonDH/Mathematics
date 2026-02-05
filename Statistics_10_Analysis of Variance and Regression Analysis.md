# 제10장 분산분석과 회귀분석 *(Analysis of Variance and Regression Analysis)*

## 일원분류모형의 분산분석 *(One-Way Analysis of Variance)*
여러 모집단의 평균을 비교할 때 가장 흔히 사용하는 모형으로 **일원분류 정규분포 모형(one-way normal model)** 을 둔다.

> 참고: 4장 일원분류모형(one-way classification model)

* 요인(인자, factor): 실험 조건을 구분하는 범주형 변수
    * $k$개의 수준(*level*)을 가지며, 각 수준의 적용을 처리(*treatment*)로 해석한다.
    * 수준(*level*):요인이 가질 수 있는 값이고, 인덱스 $i=1,\dots,k$로 색인
* 집합(group, 집단): 레벨 $i$에서의 관측치 $j=1,\dots,n_i$
    * 총 표본수 $n=\sum_{i=1}^k n_i$
* 주의: design matrix(설계행렬)이랑 다름. 설계행렬은 0아니면 1이고 $n\times k$행렬
    * $X_{ij}$는 스칼라이고, $X_i$ $i$번째 수준의 관측값들로, $n_i$개가 있다
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
(실제로 $P_1Y$는 각 관측치를 “해당 집단의 평균”으로 바꾼 벡터이므로 $Y-P_1Y$가 집단 내 편차, $P_1Y-P_0Y$가 집단 평균의 전체평균으로부터의 편차를 모아준다.)

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
* 모평균:
    $$
    \hat\mu_{ij} = \bar X_{ij\cdot}
    = \frac{1}{n_{ij}}\sum_{k=1}^{n_{ij}} X_{ijk}
    $$
* 모분산:
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
\mu_{\cdot j}=\frac{1}{a}\sum_{i=1}^a\mu_{ij}
$$
$$
\alpha_i=\mu_{i\cdot}-\bar\mu,
\quad
\beta_j=\mu_{\cdot j}-\bar\mu,
\quad
\gamma_{ij}=\mu_{ij}-\mu_{i\cdot}-\mu_{\cdot j}+\bar\mu.
$$

* $\alpha_i$: 요인 $A$의 **주효과(main effect)**
* $\beta_j$: 요인 $B$의 **주효과(main effect)**
* $\gamma_{ij}$: **교호작용효과(interaction effect)**

이때 모형은
$$
X_{ijk}=\bar\mu+\alpha_i+\beta_j+\gamma_{ij}+e_{ijk}
$$
이며 제약조건은
$$
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

특히 $n_{ij}=r$인 균형 설계에서 요인 $A$의 주효과는
$$
\alpha_i=\mu_{i\cdot}-\bar\mu,\qquad 
\mu_{i\cdot}=\frac1b\sum_{j=1}^b\mu_{ij},\qquad 
\bar\mu=\frac1{ab}\sum_{i=1}^a\sum_{j=1}^b\mu_{ij}
$$
로 정의되며, 두 수준 $i,\ell$의 주효과 차이는
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
$n_{ij}=r$일 때,
$$
\hat\alpha_i=\bar X_{i\cdot\cdot}-\bar X_{\cdots}
$$
에 대해
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

### 교호작용효과의 유의성 검정 *(Test for interaction effect)*
가설
$$
H_0^{AB}:\gamma_{ij}=0\ \forall i,j
\quad\text{vs}\quad
H_1^{AB}:\text{적어도 하나는 0이 아님}
$$

이는
> "$\mu_{ij}$가 $i$의 함수와 $j$의 함수의 합으로 표현된다"  
라는 가설과 동치이다.

### 정리 10.2.4 균형된 이원분류정규분포모형에서 교호작용효과의 유의성 검정 *(F-test for interaction effect)*
$$
SS_{AB}
=
\sum_{i=1}^a\sum_{j=1}^b
r(\bar X_{ij\cdot}-\bar X_{i\cdot\cdot}-\bar X_{\cdot j\cdot}+\bar X_{\cdots})^2
$$
$$
SSE=\sum_{i=1}^a\sum_{j=1}^b\sum_{k=1}^r (X_{ijk}-\bar X_{ij\cdot})^2
$$

검정통계량:
$$
F_n
=
\frac{SS_{AB}/((a-1)(b-1))}
{SSE/(n-ab)}
\sim F((a-1)(b-1),n-ab)
$$

기각역:
$$
F_n\ge F_\alpha((a-1)(b-1),n-ab).
$$

#### 증명
정규모형의 직교투영(orthogonal projection)과 카이제곱 분해 정리를 이용하여
$$
SS_{AB}/\sigma^2\sim\chi^2((a-1)(b-1)),
\quad
SSE/\sigma^2\sim\chi^2(n-ab)
$$
이며 두 통계량이 독립임을 보이면 F-분포가 따른다.

### 정리 10.2.5 균형된 이원분류정규분포모형에서 주효과의 유의성 검정 *(Tests for main effects)*
**(a) 요인 $A$의 주효과**  
$$
H_0^A:\alpha_1=\cdots=\alpha_a=0
$$
$$
SSA=\sum_{i=1}^a br(\bar X_{i\cdot\cdot}-\bar X_{\cdots})^2
$$
$$
F_n=\frac{SSA/(a-1)}{SSE/(n-ab)}\sim F(a-1,n-ab)
$$

**(b) 요인 $B$의 주효과**  
$$
H_0^B:\beta_1=\cdots=\beta_b=0
$$
$$
SSB=\sum_{j=1}^b ar(\bar X_{\cdot j\cdot}-\bar X_{\cdots})^2
$$
$$
F_n=\frac{SSB/(b-1)}{SSE/(n-ab)}\sim F(b-1,n-ab)
$$

## 분산분석에서의 검정력 함수 *(Power Functions in Analysis of Variance)*
분산분석에서 유의성 검정에 사용되는 F-통계량은 **귀무가설 하에서는 중심 F 분포**, **대립가설 하에서는 비중심 F 분포**를 따른다. 검정력 함수는 이 비중심성모수(noncentrality parameter)를 통해 표현된다.

### 비중심 카이제곱분포 *(Noncentral Chi-Square Distribution)*
정의  
서로 독립이고
$$
X_i \sim N(\mu_i,1),\quad i=1,\dots,r
$$
일 때
$$
Y=\sum_{i=1}^r X_i^2
$$
의 분포를 **자유도 $r$**, **비중심성모수**
$$
\delta=\sum_{i=1}^r \mu_i^2
$$
를 갖는 비중심 카이제곱분포라 하고
$$
Y\sim \chi^2(r;\delta)
$$
로 표기한다.

### 정리 10.3.1 비중심 카이제곱분포의 성질
**(a) 누적생성함수 (cgf)**  
$$
\mathrm{cgf}_Y(t)
=\sum_{k=1}^\infty \frac{t^k}{k!}\,2^{k-1}(k-1)!(r+k\delta),
\quad t<\tfrac12
$$

**(b) 혼합분포 표현**  
$$
f_Y(y)=\sum_{k=0}^\infty
\Pr(K=k)\, f_{\chi^2(r+2k)}(y),
\quad K\sim \text{Poisson}(\delta/2)
$$
즉, 비중심 카이제곱분포는 **자유도가 증가하는 중심 카이제곱분포의 Poisson 혼합**으로 표현된다.

#### 증명 개요
* $X\sim N(\mu,1)$에 대해
    $$
    E(e^{tX^2})=(1-2t)^{-1/2}\exp\!\Big(\frac{\mu^2 t}{1-2t}\Big)
    $$
* 독립성을 이용해 곱으로 결합한다.
* 로그를 취해 급수 전개하면 (a)를 얻는다.
* (b)는 분해 및 convolution을 이용해 도출한다.
### 정리 10.3.2 이차형식의 분포
$X\sim N(\mu,I)$이고 $A^2=A$이면
$$
X^\top A X \sim \chi^2(r;\delta),\quad
r=\mathrm{trace}(A),\quad
\delta=\mu^\top A\mu
$$

의미  
* 분산분석의 제곱합(SS)은 **정규벡터의 이차형식**이다.
* 귀무가설 하에서는 $(\delta=0)$이 되어 중심 카이제곱분포를 따른다.
* 대립가설 하에서는 $(\delta>0)$이 되어 비중심 분포를 따른다.

### 비중심 F 분포 *(Noncentral F Distribution)*
정의  
$$
F=\frac{V_1/r_1}{V_2/r_2},\quad
V_1\sim \chi^2(r_1;\delta),\ V_2\sim \chi^2(r_2)
$$
이면
$$
F\sim F(r_1,r_2;\delta).
$$

여기서 $\delta$를 **비중심성모수**라 한다.

### 일원분류 분산분석에서의 검정력 함수
모형  
$$
X_{ij}=\mu+\alpha_i+e_{ij},\quad
\sum_{i=1}^k n_i\alpha_i=0,\quad
e_{ij}\sim N(0,\sigma^2).
$$

가설  
$$
H_0:\alpha_1=\cdots=\alpha_k=0
\quad\text{vs}\quad
H_1:\text{적어도 하나는 0이 아님}.
$$

검정통계량  
$$
F_n=\frac{SSB/(k-1)}{SSW/(n-k)}.
$$

대립가설 하의 분포  
정리 10.3.2를 이용하면
$$
\frac{SSB}{\sigma^2}\sim \chi^2(k-1;\delta),
\quad
\delta=\frac{1}{\sigma^2}\sum_{i=1}^k n_i\alpha_i^2,
$$
이고
$$
\frac{SSW}{\sigma^2}\sim \chi^2(n-k)
$$
이며 서로 독립이다.

따라서
$$
F_n\sim F(k-1,n-k;\delta).
$$

검정력 함수  
유의수준 $\alpha$에서
$$
\text{Power}(\delta)
=
P_\delta\!\left(
F_n \ge F_\alpha(k-1,n-k)
\right).
$$

성질  
* 검정력 함수는 $\delta$의 **증가함수**다.
* $\delta=0$이면 검정력은 $\alpha$다.
* $\delta$는 처리효과의 크기와 표본크기 $(n_i)$에 비례한다.

### 이원분류 분산분석에서의 검정력 함수
균형 이원분류모형  
$$
X_{ijk}=\mu+\alpha_i+\beta_j+\gamma_{ij}+e_{ijk},
\quad e_{ijk}\sim N(0,\sigma^2).
$$

교호작용 효과 검정

가설  
$$
H_0^{AB}:\gamma_{ij}=0,\ \forall i,j.
$$

검정통계량  
$$
F_n=\frac{SS_{AB}/((a-1)(b-1))}{SSE/(n-ab)}.
$$

대립가설 하 분포  
$$
F_n\sim F((a-1)(b-1),n-ab;\delta_{AB}),
\quad
\delta_{AB}=\frac{1}{\sigma^2}\sum_{i,j} r\gamma_{ij}^2.
$$

주효과 검정  
* $A$ 요인:
    $$
    \delta_A=\frac{1}{\sigma^2}\sum_{i=1}^a br\,\alpha_i^2
    $$
* $B$ 요인:
    $$
    \delta_B=\frac{1}{\sigma^2}\sum_{j=1}^b ar\,\beta_j^2
    $$

각각 대응하는 비중심 $F$ 분포를 따른다.


## 10.4 회귀분석 *(Regression Analysis)*

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
**(a) $c^\top\beta$와 $\sigma^2$의 전역최소분산불편추정량 *(UMVUE)***  
* 임의의 벡터 $c\in\mathbb{R}^{p+1}$에 대해
    $$
    \widehat{c^\top\beta}=c^\top\hat\beta,\qquad
    \hat\beta=(X^\top X)^{-1}X^\top Y.
    $$
* 오차분산 추정량(불편)
    $$
    \hat\sigma^2=\frac{\|Y-X\hat\beta\|^2}{n-p-1}
    =\frac{(Y-X\hat\beta)^\top(Y-X\hat\beta)}{n-p-1}.
    $$

**(b) $\hat\beta$의 분포**  
$$
\hat\beta \sim N_{p+1}\!\big(\beta,\ \sigma^2(X^\top X)^{-1}\big).
$$

**(c) $\hat\beta$와 $\hat\sigma^2$의 독립성**  
$$
\hat\beta\ \perp\!\!\!\perp\ \hat\sigma^2.
$$

**(d) $\hat\sigma^2$의 카이제곱 분포**  
$$
\frac{(n-p-1)\hat\sigma^2}{\sigma^2}\sim \chi^2(n-p-1).
$$

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
일부 회귀계수 블록 검정 *(partial/regression block test)*  

* 설계행렬을 블록으로 쪼갠다:
    $$
    Y=X_0\beta_0 + X_1\beta_1 + e.
    $$
    $$
    \mathrm{rank}(X_0,X_1)=p_0+p_1,\quad \mathrm{rank}(X_0)=p_0,\quad \mathrm{rank}(X_1)=p_1.
    $$
* 가설:
    $$
    H_0:\beta_1=0\quad \text{vs}\quad H_1:\beta_1\neq 0.
    $$

**(a) 투영행렬과 분해**  
* $X_0$로의 정사영행렬
    $$
    \Pi_0 = X_0(X_0^\top X_0)^{-1}X_0^\top.
    $$
* $X_0$를 제거한 $X_1$
    $$
    X_{1|0}=(I-\Pi_0)X_1.
    $$
* 그 열공간으로의 정사영행렬
    $$
    \Pi_{1|0}=X_{1|0}(X_{1|0}^\top X_{1|0})^{-1}X_{1|0}^\top.
    $$

**(b) 검정통계량 (최대가능도비 검정과 동치 형태)**  
* 회귀로 설명되는 제곱합(추가 설명력)
    $$
    R(10)=Y^\top\Pi_{1|0}Y.
    $$
* 오차제곱합
    $$
    SSE = Y^\top(I-\Pi_0-\Pi_{1|0})Y.
    $$
* $F$-통계량
    $$
    F_n=\frac{R(10)/p_1}{SSE/(n-p_0-p_1)}.
    $$
* 유의수준 $\alpha$의 기각역
    $$
    F_n \ge F_\alpha(p_1,\ n-p_0-p_1).
    $$

**(c) 대립가설 하 분포와 검정력 *(power)***  
* 대립가설 하에서
    $$
    F_n \sim F(p_1,\ n-p_0-p_1;\ \delta),
    $$
    여기서 $\delta$는 비중심도모수 *(noncentrality parameter)*:
    $$
    \delta = \frac{\beta_1^\top X_1^\top\Pi_{1|0}X_1\beta_1}{\sigma^2}.
    $$
* 검정력 함수는
    $$
    \pi(\delta)=P_\delta\!\left(F_n\ge F_\alpha(p_1,n-p_0-p_1)\right)
    $$
    이며 $\delta$의 증가함수로 정리되어 있다.

    * 해석: 효과크기($\beta_1$)가 커지거나, 잡음($\sigma^2$)이 작아지거나, 설계가 좋아져 $\Pi_{1|0}$ 방향으로 신호가 커질수록 검정력이 증가한다.

### 정리 10.4.4 회귀계수의 선형결합에 대한 유의성 검정 *(general linear hypothesis test)*
* 일반 선형가설:
    $$
    H_0: C^\top\beta=0\quad \text{vs}\quad H_1:C^\top\beta\neq 0.
    $$

    * $C$는 $(p+1)\times r$, $\mathrm{rank}(C)=r$.

**(a) 해당 정사영행렬**  
$$
\Pi_{1|0}
= X(X^\top X)^{-1}C\Big(C^\top(X^\top X)^{-1}C\Big)^{-1}C^\top(X^\top X)^{-1}X^\top.
$$

* 전체 모형의 정사영행렬
    $$
    \Pi_{1,0}=X(X^\top X)^{-1}X^\top.
    $$

**(b) 검정통계량**  
$$
R(10)=Y^\top\Pi_{1|0}Y,\qquad
SSE=Y^\top(I-\Pi_{1,0})Y.
$$
$$
F_n=\frac{R(10)/r}{SSE/(n-p-1)}.
$$

* 기각역:
    $$
    F_n \ge F_\alpha(r,\ n-p-1).
    $$

**(c) 대립가설 하 비중심 $F$와 비중심도모수**  
$$
F_n\sim F(r,\ n-p-1;\ \delta),
$$
$$
\delta = \frac{\beta^\top C\Big(C^\top(X^\top X)^{-1}C\Big)^{-1}C^\top\beta}{\sigma^2}.
$$

따라서 검정력은
$$
\pi(\delta)=P_\delta\!\left(F_n\ge F_\alpha(r,n-p-1)\right)
$$
이고 $\delta$ 증가함수로 정리되어 있다.

### 10.4.1 선형모형 *(General Linear Model)*로의 확장
지금까지는 $\mathrm{rank}(X)=p+1$로 $X^\top X$가 가역인 경우를 다루었다. 그러나 분산분석(ANOVA) 같은 모형은 설계행렬 열들이 선형종속이어서 $\mathrm{rank}(X)$가 열 개수보다 작은 경우가 많다. 이를 제약조건을 추가한 선형모형으로 표현한다:
$$
Y=X\beta+e,\quad C^\top\beta=0,\quad e\sim N_n(0,\sigma^2 I).
$$
$$
\mathrm{rank}(X)=p+1-r<p+1,\quad \mathrm{rank}(C^\top)=r,\quad X\text{와 }C^\top\text{의 행들은 선형독립}.
$$
이 틀 안에서 ANOVA 모형도 포함된다고 설명한다.

#### 예 10.4.1 일원분류 정규분포모형을 선형모형으로 쓰기
* 일원분류모형:
    $$
    X_{ij}=\mu+\alpha_i+e_{ij},\quad
    \sum_{i=1}^k n_i\alpha_i=0,\quad
    e_{ij}\sim N(0,\sigma^2).
    $$
* 벡터화하면 $Y=X\beta+e$ 꼴이 되며, 책에서는 설계행렬 $X$를 "집단별 더미 + 전체항(상수항)" 형태로 쓴다.
* 이때 $\mathrm{rank}(X)=k<k+1$이어서 $X^\top X$가 정칙이 아니지만(역행렬 불가),
    제약 $c^\top\beta=0$ (여기서 $c^\top=(0,n_1,\dots,n_k)$)를 넣으면 모형이 식별 가능하다고 설명한다.
* 결론: 정사영행렬의 구체적 표현은 달라질 수 있으나, **"투영(Projection)으로 SS를 나누고, 그 비로 F-검정을 만들며, 대립가설 하에서는 비중심 F가 된다"** 는 구조가 유지된다는 방향으로 이어진다.

#### (참고) 10.3과의 연결(검정력)
* 위의 회귀 $F$-검정에서 대립가설 하 분포가 비중심 $F$이고, 검정력이 $\delta$의 증가함수라는 결론은 10.3에서 정리한 비중심 카이제곱/비중심 $F$의 성질을 사용해 얻는 구성이다.


#TODO: 이거 끝나면 부록 내용도 쭉 같이 정리
