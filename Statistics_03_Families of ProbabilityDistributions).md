# 제3장 여러 가지 확률분포 (Families of Probability Distributions)

## 3.1 초기하분포 (Hypergeometric Distribution)

### 모집단과 표본, 단순랜덤추출
통계 조사에서 관심의 대상이 되는 전체를 **모집단** (population)이라 하고, 모집단의 개체 수를 $N$이라 하자.
모집단에서 $n$개의 개체를 **랜덤하게** 선택하여 조사한 뒤, 이를 바탕으로 모집단 전체에 대한 추론을 수행한다.

모집단 전체가 가지고 있는 분포를 **모집단분포** (population distribution)라 한다.

모집단에서 "랜덤하게 $n$개를 선택한다"는 것은 다음을 의미한다.

* 각 추출 단계에서 모든 개체가 동일한 확률로 선택된다.
* 한 번 선택된 개체는 다시 넣지 않는다.

이러한 추출을 **비복원추출** (sampling without replacement)이라 하며,
이 방법을 **단순랜덤추출** (simple random sampling)이라 한다.
이때 추출된 $n$개의 개체를 **랜덤표본** (random sample), 또는 간단히 **표본** (sample)이라 한다.

### 두 가지 분류의 모집단
각 개체의 특성이 두 가지 값만을 갖는 경우를 고려한다. 예를 들면
* 양호 / 불량
* 성공 / 실패
* $1$ / $0$

모집단에서 값이 $1$인 개체가 $D$개, 값이 $0$인 개체가 $N-D$개 있다고 하자.
이때 모집단에서 $1$의 비율
$$
p = \frac{D}{N}
$$
을 **모비율** (population proportion)이라 한다.

이 모집단의 분포는 값 $1$에 확률 $p$, 값 $0$에 확률 $1-p$를 대응시키는 분포다.

### 초기하분포의 정의
위와 같은 모집단에서 $n$개를 단순랜덤추출할 때,
표본에 포함된 $1$의 개수를 확률변수 $X$라 하자.

그러면 $X$의 확률질량함수는 다음과 같다.

$$
P(X=x)
= \frac{\binom{D}{x}\binom{N-D}{n-x}}{\binom{N}{n}},
\quad
0 \le x \le D,; 0 \le n-x \le N-D
$$

이와 같은 분포를 **초기하분포** (hypergeometric distribution)라 하며,
기호로

$$
X \sim H(n; N, D)
$$

와 같이 나타낸다.
### 정리 3.1.1 (초기하분포의 평균과 분산)
$X \sim H(n; N, D)$이고 $p=D/N$이면

$$
E(X) = np, \qquad \mathrm{Var}(X) = \frac{N-n}{N-1} \cdot np(1-p)
$$

#### 증명
**(평균)** 조합 항등식 $x\binom{D}{x} = D\binom{D-1}{x-1}$과 $\sum_x \binom{D-1}{x-1}\binom{N-D}{n-x} = \binom{N-1}{n-1}$을 이용하면
$$
E(X) = D \frac{\binom{N-1}{n-1}}{\binom{N}{n}} = \frac{nD}{N} = np
$$

**(분산)** 항등식 $x(x-1)\binom{D}{x} = D(D-1)\binom{D-2}{x-2}$로부터
$$
E[X(X-1)] = \frac{n(n-1)D(D-1)}{N(N-1)}
$$

이고, $\mathrm{Var}(X) = E[X(X-1)] + E[X] - (E[X])^2$에 대입하면 결과를 얻는다. □

#### 예제 3.1.1
$N=60$개의 제품 중 불량품이 $D=6$개(불량률 $10%$)인 로트에서
$n=5$개를 단순랜덤추출하여 검사한다고 하자.

불량품 개수가 미리 정한 허용 한계를 초과하면 로트를 불합격시킨다.
불량 로트를 불합격시킬 확률을 $1%$ 이하로 하려면,
허용 불량 개수 $c$는 얼마로 해야 하는가?

합격 조건을 "불량품 개수 $\le c$"라 하면
$$
P(X \le c) \ge 0.99
$$
여야 한다. 즉
$$
\sum_{x=0}^c
\frac{\binom{6}{x}\binom{54}{5-x}}{\binom{60}{5}}
\ge 0.99
$$

을 만족하는 최소의 $c$를 찾는다.

계산 결과
* $c=1$: $P(X\le 1)=0.92648$
* $c=2$: $P(X\le 2)=0.99461$
이므로

$$
c=2
$$

### 초기하분포와 이항분포의 근사
모집단 크기 $N$이 표본 크기 $n$에 비해 충분히 크면,
$$
\frac{\binom{D}{x}\binom{N-D}{n-x}}{\binom{N}{n}}
\approx \binom{n}{x}\left(\frac{D}{N}\right)^x
\left(1-\frac{D}{N}\right)^{n-x}
$$
이 된다.

이는 **복원추출** (sampling with replacement)에서의 확률과 동일하며,
이 경우 초기하분포는 이항분포로 근사된다.

## 3.2 이항분포와 다항분포 (Binomial and Multinomial Distributions)

### 이항분포 (Binomial Distribution)
모집단에서 한 개체를 복원추출할 때, 결과가
* $1$일 확률 $p$
* $0$일 확률 $1-p$

두가지 인 경우, 이를 **베르누이 시행** (Bernoulli trial)이라 한다.

서로 독립인 $n$번의 베르누이 시행에서 성공($1$)의 개수를 $X$라 하면,
$$
P(X=x)=\binom{n}{x}p^x(1-p)^{n-x},\quad x=0,1,\dots,n
$$

이고, 이를 **이항분포** (binomial distribution)라 하며
$$
X \sim B(n,p)
$$
로 표기한다.

### 이항분포의 평균과 분산
$$
E(X)=np,\qquad \mathrm{Var}(X)=np(1-p)
$$

#### 증명
$X$를 $n$개의 독립인 베르누이 확률변수 $Z_1, \ldots, Z_n$의 합으로 나타내면
$$
X = Z_1 + \cdots + Z_n, \quad Z_i \sim \text{Bernoulli}(p)
$$

각 $Z_i$에 대해 $E(Z_i) = p$, $\text{Var}(Z_i) = p(1-p)$이므로,
독립합의 성질에 의해

$$
E(X) = \sum_{i=1}^n E(Z_i) = np
$$

$$
\text{Var}(X) = \sum_{i=1}^n \text{Var}(Z_i) = np(1-p)
$$
□

### 이항분포의 **대의적 정의** (representational definition)
$$
B(n,p) \overset{d}{\equiv} \mathrm{Bernoulli}_1(p) \oplus \cdots \oplus \mathrm{Bernoulli}_n(p)
$$
$$
X \sim B(n,p) \Leftrightarrow X \overset{d}{\equiv} Z_1+\cdots+Z_n,
\quad Z_i \overset{iid}\sim \mathrm{Bernoulli}(p)
$$
  - $\overset{d}{\equiv}$: 분포상 동치, 두 확률변수가 동일한 확률분포를 가짐
  - iid: 서로 독립이며 동일한 확률분포를 따르는 확률변수들
  - $\oplus$ (직합 / 독립이거나 상관계수 0인 확률변수의 합)

### 정리 3.2.1 (이항분포의 성질)
$X \sim B(n,p)$이면 다음이 성립한다.

**(a) 적률생성함수**  
$$
M_X(t) = (pe^t + q)^n,\quad q=1-p
$$

**(b) 재생성 (Reproductive Property)**  
$X_1 \sim B(n_1, p)$, $X_2 \sim B(n_2, p)$이고 $X_1$, $X_2$가 독립이면
$$
X_1 + X_2 \sim B(n_1 + n_2, p)
$$

#### 증명
**(a)** 베르누이분포의 적률생성함수는
$$
M_{Z_i}(t) = E(e^{tZ_i}) = pe^t + q
$$
이므로, $X = Z_1 + \cdots + Z_n$이고 $Z_i$들이 독립이므로
$$
M_X(t) = \prod_{i=1}^n M_{Z_i}(t) = (pe^t + q)^n
$$

**(b)** $X_1 = \sum_{i=1}^{n_1} Z_i$, $X_2 = \sum_{j=1}^{n_2} W_j$이고  
모든 $Z_i, W_j \overset{iid}{\sim} \text{Bernoulli}(p)$이며 서로 독립이므로
$$
X_1 + X_2 = \sum_{i=1}^{n_1+n_2} Y_i, \quad Y_i \overset{iid}{\sim} \text{Bernoulli}(p)
$$
따라서 $X_1 + X_2 \sim B(n_1 + n_2, p)$이다.

또는 적률생성함수를 이용하면
$$
M_{X_1+X_2}(t) = M_{X_1}(t) \cdot M_{X_2}(t) = (pe^t+q)^{n_1}(pe^t+q)^{n_2} = (pe^t+q)^{n_1+n_2}
$$
이므로 $X_1+X_2 \sim B(n_1+n_2, p)$이다. □

#### 예 3.2.1
서로 독립이고 $X_1,\dots,X_N \sim \mathrm{Bernoulli}(p)$일 때,
$n = \sum_{i=1}^N X_i$, $D \leq N$라 하면
$$
\sum_{i=1}^D X_i \,\bigg|\, \sum_{i=1}^N X_i = n \sim H(n; N, D)
$$

임을 보이자.

**증명**
$Y = \sum_{i=1}^D X_i$라 하고, $S = \sum_{i=1}^N X_i = n$이라 하자.

조건부 확률을 계산하면
$$
P(Y = y \mid S = n) = \frac{P(Y = y, S = n)}{P(S = n)}
$$

**분자:** $Y = y$이고 $S = n$이면 $\sum_{i=D+1}^N X_i = n - y$이어야 한다.
$X_i$들이 독립이므로
$$
P(Y = y, S = n) = P(Y = y) \cdot P\left(\sum_{i=D+1}^N X_i = n-y\right)
$$

$Y \sim B(D, p)$이고 $\sum_{i=D+1}^N X_i \sim B(N-D, p)$이므로
$$
P(Y = y, S = n) = \binom{D}{y}p^y(1-p)^{D-y} \cdot \binom{N-D}{n-y}p^{n-y}(1-p)^{N-D-n+y}
$$
$$
= \binom{D}{y}\binom{N-D}{n-y} p^n(1-p)^{N-n}
$$

**분모:** $S = \sum_{i=1}^N X_i \sim B(N, p)$이므로
$$
P(S = n) = \binom{N}{n}p^n(1-p)^{N-n}
$$

따라서
$$
P(Y = y \mid S = n) = \frac{\binom{D}{y}\binom{N-D}{n-y} p^n(1-p)^{N-n}}{\binom{N}{n}p^n(1-p)^{N-n}} = \frac{\binom{D}{y}\binom{N-D}{n-y}}{\binom{N}{n}}
$$

이는 $H(n; N, D)$의 확률질량함수이다. □

### 다항분포 (Multinomial Distribution)
각 시행의 결과가 $k$개의 범주 중 하나로 나타나고,
각 범주의 확률이 $p_1,\dots,p_k$ ($\sum p_i=1$)일 때,
$n$번의 독립 시행에서 각 범주의 발생 횟수를

$$
X=(X_1,\dots,X_k)^t
$$
라 하면,

$$
P(X_1=x_1,\dots,X_k=x_k)
=\frac{n!}{x_1!\cdots x_k!}p_1^{x_1}\cdots p_k^{x_k},
\quad \sum_{i=1}^k x_i=n
$$
이다.
  - $\frac{n!}{x_1!\cdots x_k!} = \binom{D}{x_1x_2 \dots x_k}$: multinomial coefficient  
  - 여러 유형으로 분류되는 모집단에서 한 개씩 추출하여 관측하는것을 다항시행 (multinomial trial)이라 한다

이를 **다항분포** (multinomial distribution)라 하며
$$
X \sim \mathrm{Multi}(n; p_1,\dots,p_k)
$$
로 나타낸다.
### 다항분포의 **대의적 정의** (representational definition)
$$
\mathrm{Multi}(n, p) \overset{d}{\equiv} \mathrm{Multi}_1(1,p) \oplus \cdots \oplus \mathrm{Multi}_n(1,p) \\
X \sim \mathrm{Multi}(n; p_1,\dots,p_k) \Leftrightarrow X \overset{d}{\equiv} Z_1+\cdots+Z_n,
\quad Z_i \overset{iid}\sim \mathrm{Multi}(1; p_1,\dots,p_k)
$$
- $Z_i = (Z_{i1}, \dots, Z_{ik})^t$는 하나의 다항시행 결과를 나타내는 벡터
- 각 $Z_i$는 정확히 하나의 성분만 1이고 나머지는 0

### 정리 3.2.2 다항분포의 성질
$X\sim \mathrm{Multi}(n;p_1,\dots,p_k)$이면

* $E(X_i)=np_i$
* $\mathrm{Var}(X_i)=np_i(1-p_i)$
* $\mathrm{Cov}(X_i,X_j)=-np_ip_j\quad(i\ne j)$
* $M_X(t) = \left(\sum_{i=1}^k p_i e^{t_i}\right)^n$ (적률생성함수, $t=(t_1,\dots,t_k)^t$)
이다.

#### 증명
다항분포 역시 $X=Z_1+\cdots+Z_n $으로 표현되며, 각 $Z_i$는 $\mathrm{Multi}(1;p_1,\dots,p_k)$를 따른다.
독립합의 성질과 베르누이 벡터의 공분산 계산으로부터 결과가 따른다. □


## 기하분포와 음이항분포 (Geometric and Negative Binomial Distributions)
서로 독립이고 성공확률이 $p$인 베르누이 시행
$$
X_1,X_2,\dots
$$
을 관측한다고 하자. 여기서 각 $X_i$는 성공이면 1, 실패이면 0을 취한다.

### 기하분포 (Geometric distribution)
첫 번째 성공이 관측될 때까지의 **시행 횟수**를 $W_1$이라 하면,
연속된 실패 후에 처음으로 성공이 발생해야 하므로
$$
P(W_1=x)=(1-p)^{x-1}p,\quad x=1,2,\dots
$$
이러한 확률분포를 **기하분포**라 하며,
$$
W_1\sim \mathrm{Geo}(p)
$$
로 나타낸다.

> 각주
> 무한급수 $\sum_{x=1}^\infty q^{x-1}$을 기하급수라고 부르며,
> 이로부터 기하분포라는 명칭이 유래하였다. 여기서 $q=1-p$이다.

### 정리 3.3.1 기하분포의 성질
$W_1\sim \mathrm{Geo}(p)$이면 다음이 성립한다.

**(a) 적률생성함수**  
$$
M_{W_1}(t) = \frac{p e^t}{1-(1-p)e^t}, \quad t<-\log(1-p)
$$

**(b) 평균과 분산**  
$$
E(W_1) = \frac{1}{p}, \qquad \mathrm{Var}(W_1) = \frac{1-p}{p^2}
$$

#### 증명
**(a) 적률생성함수**  
정의에 의해
$$
\mathrm{mgf}_{W_1}(t)=E(e^{tW_1})
=\sum_{x=1}^\infty e^{tx}(1-p)^{x-1}p \\
p e^t \sum_{x=1}^\infty \big((1-p)e^t\big)^{x-1}
$$

기하급수의 합 공식을 이용하면
$$
\mathrm{mgf}_{W_1}(t)
=\frac{p e^t}{1-(1-p)e^t},
\quad t<-\log(1-p)
$$

**(b) 평균과 분산**  
기하분포의 누율생성함수는
$$
\mathrm{cgf}_{W_1}(t)
=\log \mathrm{mgf}_{W_1}(t)
= -\log\{1-(1-p)e^t\}+t+\log p
$$

로그함수의 멱급수 전개
$$
-\log(1-A)=A+\frac{A^2}{2}+\frac{A^3}{3}+\cdots \quad(|A|<1)
$$
를 이용하여 전개하고 $t$의 멱차수별로 정리하면

$$
\mathrm{cgf}_{W_1}(t)
=\frac{1}{p}t+\frac{1-p}{2p^2}t^2+\cdots
$$

따라서
$$
E(W_1)=\mathrm{cgf}'_{W_1}(0)=\frac{1}{p},
\qquad
\mathrm{Var}(W_1)=\mathrm{cgf}''_{W_1}(0)=\frac{1-p}{p^2}
$$
이다. □

### 음이항분포 (Negative binomial distribution)
이번에는 서로 독립이고 성공확률이 $p$인 베르누이 시행에서, $r$번째 성공이 관측될 때까지의 **시행 횟수**를 $W_r$라 하자.
$W_r=x$라는 사건은 다음을 의미한다.

* 앞의 $x-1$번 시행 중에서 정확히 $(r-1)$번 성공
* $x$번째 시행에서 성공

따라서
$$
P(W_r=x)
=\binom{x-1}{r-1}p^{r}(1-p)^{x-r},
\quad x=r,r+1,\dots
$$

이며, 이 분포를 **음이항분포**라 하고
$$
W_r\sim \mathrm{Negbin}(r,p)
$$
로 쓴다.

### 참고: 음이항분포의 명칭 유래
음이항분포라는 명칭은 **음의 지수를 갖는 이항전개식**으로부터 유래하였다.

이항정리의 일반화로부터
$$
(1+t)^{-r} = \sum_{k=0}^{\infty} \binom{-r}{k} t^k
= \sum_{k=0}^{\infty} \binom{r+k-1}{k} (-1)^k t^k
$$
가 성립한다. 여기서 일반화된 이항계수는
$$
\binom{-r}{k} = \frac{(-r)(-r-1)\cdots(-r-k+1)}{k!} = (-1)^k \binom{r+k-1}{k}
$$
이다.

이제 우변의 무한합에서 $x = r+k$로 치환하면 $k = x-r$이고,
$$
(1+t)^{-r} = \sum_{x=r}^{\infty} \binom{x-1}{r-1} (-t)^{x-r}
$$

여기서 $-t = 1-p$를 대입하면
$$
p^{-r} = (1-(1-p))^{-r} = \sum_{x=r}^{\infty} \binom{x-1}{r-1} (1-p)^{x-r}
$$

양변에 $p^r$을 곱하면
$$
1 = \sum_{x=r}^{\infty} \binom{x-1}{r-1} p^r (1-p)^{x-r}
$$

이 식의 우변이 바로 음이항분포의 확률질량함수의 총합이 된다.

### 음이항분포의 대의적 정의
$$
X \sim Negbin(r,p) \overset{d}{\equiv} Geo_1(p) \oplus \dots \oplus Geo_r(p) 
$$
즉
$$
X \sim Negbin(r,p) \Leftrightarrow X \overset{d}{\equiv} Z_1,Z_2,\dots,Z_r \stackrel{\text{iid}}{\sim}\mathrm{Geo}(p)
$$
> 음이항분포는 **서로 독립인 기하분포 확률변수의 합**이다.  
이 정의는 이후 평균, 분산, 적률생성함수의 계산에서 핵심적인 역할을 한다.

### 정리 3.3.2 음이항분포의 성질
**(a) 평균과 분산**  
대의적 정의와 기댓값, 분산의 가법성으로부터
$$
E(W_r)=E(Z_1+\cdots+Z_r)=rE(Z_1)=\frac{r}{p}
$$

$$
\mathrm{Var}(W_r)
=\mathrm{Var}(Z_1)+\cdots+\mathrm{Var}(Z_r)
=r\frac{1-p}{p^2}
$$

**(b) 적률생성함수**  
서로 독립이므로
$$
\mathrm{mgf}_{W_r}(t)
=\prod_{i=1}^r \mathrm{mgf}_{Z_i}(t)
=\left(\frac{p e^t}{1-(1-p)e^t}\right)^r,
\quad t<-\log(1-p)
$$

**(c) 닫힘성 (가법성)**  
$X_1\sim \mathrm{Negbin}(r_1,p)$,
$X_2\sim \mathrm{Negbin}(r_2,p)$가 서로 독립이면

각각을 기하분포의 합으로 표현할 수 있으므로
$$
X_1+X_2\sim \mathrm{Negbin}(r_1+r_2,p)
$$


## 포아송분포 (Poisson Distribution)
이항분포 $X_n \sim \mathrm{Bin}(n,p_n)$에서 시행횟수 $n$이 매우 크고 성공확률 $p_n$이 매우 작아
$$
n p_n \to \lambda \quad(\lambda>0)
$$
일 때, 특정 사건의 발생 횟수는 다음 분포로 근사된다.
> 이 극한 계산에서 사용되는 기본 결과는
> $\lim_{n\to\infty}(1+a_n/n)^n=e^a$ 이다.
$$
P(X=x)=e^{-\lambda}\frac{\lambda^x}{x!},
\quad x=0,1,2,\dots
$$

이 분포를 **포아송분포**라 하며
$$
X\sim \mathrm{Poisson}(\lambda)
$$
로 나타낸다.

>포아송분포는 다음과 같은 상황을 모델링한다.
>* 단위 시간 또는 단위 공간에서
>* 개별 사건은 드물게 발생하며
>* 서로 간섭하지 않고
>* 평균 발생률만이 중요할 때
>
>예를 들면 단위 시간 동안의 전화 요청 수, 단위 길이당 결함 개수 등이 이에 해당한다.

### 이항분포로부터의 유도 (포아송 근사)
이항분포의 확률질량함수는
$$
P(X_n=x)
=\binom{n}{x}p_n^x(1-p_n)^{n-x}
$$
이다.

여기서 $p_n=\lambda/n$으로 두면
$$
\binom{n}{x}
=\frac{n(n-1)\cdots(n-x+1)}{x!}
$$
이므로
$$
P(X_n=x)
=\frac{n(n-1)\cdots(n-x+1)}{x!}
\left(\frac{\lambda}{n}\right)^x
\left(1-\frac{\lambda}{n}\right)^{n-x}
$$

$n\to\infty$로 보내면
$$
\frac{n(n-1)\cdots(n-x+1)}{n^x}\to1,
\qquad
\left(1-\frac{\lambda}{n}\right)^n\to e^{-\lambda}
$$
이므로
$$
P(X_n=x)\to e^{-\lambda}\frac{\lambda^x}{x!}
$$
가 된다.

### 정리 3.4.1 포아송분포의 성질
$X \sim \mathrm{Poisson}(\lambda)$이면 다음이 성립한다.

**(a) 적률생성함수**  
$$
M_X(t) = \exp\{\lambda(e^t-1)\}, \quad -\infty < t < \infty
$$

**(b) 평균과 분산**  
$$
E(X) = \lambda, \qquad \mathrm{Var}(X) = \lambda
$$

**(c) 재생성 (독립합의 닫힘성)**  
$X_1 \sim \mathrm{Poisson}(\lambda_1)$, $X_2 \sim \mathrm{Poisson}(\lambda_2)$가 서로 독립이면
$$
X_1 + X_2 \sim \mathrm{Poisson}(\lambda_1 + \lambda_2)
$$

#### 증명
**(a) 적률생성함수**  
$$
M_X(t) = E(e^{tX})
= \sum_{x=0}^\infty e^{tx} e^{-\lambda}\frac{\lambda^x}{x!} 
= e^{-\lambda}\sum_{x=0}^\infty \frac{(\lambda e^t)^x}{x!}\\
= e^{-\lambda}e^{\lambda e^t}
= \exp\{\lambda(e^t-1)\}
$$
이며 $-\infty < t < \infty$에서 정의된다.

**(b) 평균과 분산**  
누율생성함수는 $\mathrm{cgf}_X(t) = \log M_X(t) = \lambda(e^t-1)$
$$
\therefore E(X) = \mathrm{cgf}'_X(0) = \lambda, \quad
\mathrm{Var}(X) = \mathrm{cgf}''_X(0) = \lambda
$$

**(c) 재생성**  
$X_1 \sim \mathrm{Poisson}(\lambda_1)$,
$X_2 \sim \mathrm{Poisson}(\lambda_2)$가 서로 독립이면

적률생성함수의 곱셈 성질로부터
$$
M_{X_1+X_2}(t)
= M_{X_1}(t) \cdot M_{X_2}(t)
= \exp\{(\lambda_1+\lambda_2)(e^t-1)\} \\
\therefore X_1 + X_2 \sim \mathrm{Poisson}(\lambda_1+\lambda_2)
$$
□

### 포아송과정
**포아송과정** (Poisson process)은 시간 또는 공간에서 사건이 발생하는 현상을 모델링하는 확률과정이다.
시간 $t \geq 0$에서 발생한 사건의 누적 개수를 $N_t$라 할 때, 다음 성질을 만족하면 $\{N_t, t \geq 0\}$를 **강도** (intensity) $\lambda > 0$인 포아송과정이라 한다.

**(1) 정상성 (Stationarity)**  
임의의 시간 구간 $(s, s+t]$에서 발생하는 사건의 개수 $N_{s+t} - N_s$의 분포는 구간의 길이 $t$에만 의존하고 시작 시점 $s$와는 무관하다.

**(2) 독립증분성 (Independent Increments)**  
서로 겹치지 않는 시간 구간들에서의 사건 발생 횟수는 서로 독립이다. 즉, $0 \leq t_1 < t_2 < \cdots < t_n$일 때
$$
N_{t_1}, \quad N_{t_2} - N_{t_1}, \quad \ldots, \quad N_{t_n} - N_{t_{n-1}}
$$
는 서로 독립이다.

**(3) 비례성 (Proportionality)**  
충분히 짧은 시간 구간 $(t, t+h]$에서 사건이 발생할 확률은 구간의 길이에 비례한다.
$$
P(N_{t+h} - N_t = 1) = \lambda h + o(h)
$$
여기서 $o(h)$는 $h \to 0$일 때 $h$보다 빠르게 0으로 수렴하는 항이다.

**(4) 희귀성 (Rarity)**  
충분히 짧은 시간 구간에서 두 개 이상의 사건이 동시에 발생할 확률은 무시할 수 있다.
$$
P(N_{t+h} - N_t \geq 2) = o(h)
$$

이러한 성질로부터 구간 $(0,t]$에서 발생하는 사건의 개수 $N_t$는 모수 $\lambda t$인 포아송분포를 따르게 된다.

### 정리 3.4.2 포아송과정에서 발생횟수의 분포
$$
N_t \sim \mathrm{Poisson}(\lambda t)
$$

#### 증명
미소 시간구간 $(t, t+h]$를 고려하자. 포아송과정의 성질로부터
$$
P(N_{t+h} - N_t = 0) = 1 - \lambda h + o(h) \\
P(N_{t+h} - N_t = 1) = \lambda h + o(h) \\
P(N_{t+h} - N_t \geq 2) = o(h)
$$

$P_n(t) = P(N_t = n)$이라 하면, 정상성과 독립증분성으로부터
$$
P_n(t+h) = P(N_{t+h} = n) \\
= P(N_t = n, N_{t+h} - N_t = 0) + P(N_t = n-1, N_{t+h} - N_t = 1) + \cdots \\
= P_n(t) \cdot (1-\lambda h + o(h)) + P_{n-1}(t) \cdot (\lambda h + o(h)) + o(h)
$$

정리하면
$$
P_n(t+h) - P_n(t) = -\lambda h P_n(t) + \lambda h P_{n-1}(t) + o(h)
$$

양변을 $h$로 나누고 $h \to 0$으로 극한을 취하면
$$
P_n'(t) = -\lambda P_n(t) + \lambda P_{n-1}(t)
$$

초기조건 $P_0(0) = 1$, $P_n(0) = 0$ ($n \geq 1$)과 함께 이 미분방정식을 풀면

$n=0$일 때: $P_0'(t) = -\lambda P_0(t)$이므로 $P_0(t) = e^{-\lambda t}$

귀납적으로 $P_n(t) = e^{-\lambda t} \frac{(\lambda t)^n}{n!}$을 얻는다.

따라서 $N_t \sim \mathrm{Poisson}(\lambda t)$이다. □

#### 예제 3.4.1 (포아송과정에서 결점 수의 확률 계산)
단위 길이당 평균 결점 수가 0.05로 알려진 전선을 생산하는 공정에서 나타나는 결점 수 모형으로서 포아송과정이 타당하다고 할 때, 100단위 길이에 해당하는 전선에서 10개 이상의 결점이 나타날 확률을 구하자.

**풀이**  
포아송과정의 강도 $\lambda = 0.05$이고, 구간의 길이 $t = 100$이므로 정리 3.4.2에 의해 100단위 길이에서 나타나는 결점 수 $X$는
$$
X \sim \mathrm{Poisson}(\lambda t) = \mathrm{Poisson}(0.05 \times 100) = \mathrm{Poisson}(5)
$$
를 따른다.

따라서 구하고자 하는 확률은
$$
P(X \geq 10) = 1 - P(X \leq 9) = 1 - \sum_{x=0}^{9} e^{-5}\frac{5^x}{x!}
$$

포아송 누적확률표나 계산을 통해
$$
P(X \leq 9) \approx 0.9682
$$
이므로
$$
P(X \geq 10) \approx 1 - 0.9682 = 0.0318
$$

즉, 100단위 길이의 전선에서 10개 이상의 결점이 나타날 확률은 약 3.18%이다.


## 지수분포와 감마분포 (Exponential and Gamma Distributions)

### 포아송과정 (Poisson process)에서의 도입
발생률(occurrence rate)이 $\lambda$인 포아송과정 $\{N_t:t\ge 0\}$에서 "첫 번째 사건이 시각 $t$ 이후에 발생"한다는 것은 "시각 $t$까지 사건이 0번 발생"과 동치이므로
$$
(W_1>t)\iff (N_t=0)
$$
이다. 따라서
$$
P(W_1>t)=P(N_t=0)=e^{-\lambda t},\quad t\ge 0
$$
이고 누적분포함수(cdf)는
$$
P(W_1\le t)=
\begin{cases}
1-e^{-\lambda t},& t\ge 0\\
0,& t<0
\end{cases}
$$
이다. 이를 미분하면 확률밀도함수(pdf)는
$$
f_{W_1}(t)=\lambda e^{-\lambda t}\mathbf{1}(t\ge 0)
$$
가 된다. 즉 $W_1$은 지수분포(exponential distribution)를 따르며 
$$
W_1\sim \mathrm{Exp}(1/\lambda)\quad (\lambda>0)
$$
로 나타낸다.

### 정리 3.5.1 지수분포의 성질
**(a) 적률생성함수(mgf)**  
$W_1\sim \mathrm{Exp}(1/\lambda)$이면
$$
\mathrm{mgf}_{W_1}(t)=E(e^{tW_1})=(1-t/\lambda)^{-1},\quad t<\lambda
$$
이다.

**증명** 
$$
E(e^{tW_1})
=\int_{-\infty}^{+\infty} e^{tx}\lambda e^{-\lambda x}\mathbf{1}(x\ge 0)\,dx
=\lambda\int_{0}^{\infty} e^{-(\lambda-t)x}\,dx
$$
이며 $t<\lambda$일 때 $\int_0^\infty e^{-ax}dx=1/a$를 써서
$$
E(e^{tW_1})=\frac{\lambda}{\lambda-t}=\left(1-\frac{t}{\lambda}\right)^{-1}
$$
이다. □

**(b) 평균과 분산**  
$W_1\sim \mathrm{Exp}(1/\lambda)$이면
$$
E(W_1)=\frac{1}{\lambda},\qquad \mathrm{Var}(W_1)=\frac{1}{\lambda^2}
$$
이다.

**증명** 
$$
\mathrm{cgf}_{W_1}(t)=\log \mathrm{mgf}_{W_1}(t)=-\log\left(1-\frac{t}{\lambda}\right),\quad t<\lambda
$$
로그의 멱급수 전개
$$
-\log(1-A)=A+\frac{A^2}{2}+\frac{A^3}{3}+\cdots\quad (|A|<1)
$$
를 $A=t/\lambda$에 적용하면
$$
\mathrm{cgf}_{W_1}(t)=\frac{t}{\lambda}+\frac{1}{2}\left(\frac{t}{\lambda}\right)^2+\cdots
$$
따라서
$$
E(W_1)=\mathrm{cgf}'_{W_1}(0)=\frac{1}{\lambda},\qquad
\mathrm{Var}(W_1)=\mathrm{cgf}''_{W_1}(0)=\frac{1}{\lambda^2}
$$
□

### 감마분포 (Gamma distribution)의 도입: $r$번째 사건까지의 대기시간
포아송과정에서 $r$번째 사건이 시각 $t$ 이후에 발생한다는 것은, 시각 $t$까지 사건이 $r-1$번 이하로 발생했다는 것과 동치이므로
$$
(W_r>t)\iff (N_t\le r-1)
$$
이다. 따라서
$$
P(W_r\le t)=1-P(W_r>t)=1-P(N_t\le r-1)
=1-\sum_{k=0}^{r-1} e^{-\lambda t}\frac{(\lambda t)^k}{k!},\quad t\ge 0
$$
이고, 이를 미분하여 pdf를 구하면
$$
f_{W_r}(t)=\frac{\lambda^r t^{r-1}e^{-\lambda t}}{(r-1)!},\quad t>0
$$
가 되어
$$
W_r\sim \mathrm{Gamma}(r,1/\lambda)
$$
로 나타낸다.

감마분포의 모양은 모수 $r$에 따라 달라진다. 여기서 $r$을 형상모수, shape parameter라고 하고, $\beta=1/\lambda$를 척도모수, scale parameter로 둔다. 일반적으로는 shape가 자연수로 제한되지 않고 양수(real positive)일 수 있으므로 $r$ 대신 $\alpha$를 쓰기도 한다.

### 감마함수 (Gamma function)와 일반형 감마분포

$\alpha>0$에 대해 감마함수는
$$
\Gamma(\alpha)=\int_{0}^{\infty} x^{\alpha-1}e^{-x}\,dx
$$
로 정의된다. 이를 이용하면 $\alpha>0,\ \beta>0$인 일반형 감마분포 $\mathrm{Gamma}(\alpha,\beta)$의 pdf는
$$
f(x)=\frac{1}{\Gamma(\alpha)\beta^{\alpha}}x^{\alpha-1}e^{-x/\beta}\mathbf{1}(x>0)
$$
로 쓸 수 있다.

감마함수의 성질로
* $\Gamma(\alpha)=(\alpha-1)\Gamma(\alpha-1)$ ($\alpha>1$)
* 특히 자연수 $n$에 대해 $\Gamma(n)=(n-1)!$
* $\Gamma(1/2)=\sqrt{\pi}$

가 주어진다.

### 정리 3.5.2 감마분포의 성질
**(a) 평균과 분산**  
$X\sim \mathrm{Gamma}(\alpha,\beta)$이면
$$
E(X)=\alpha\beta,\qquad \mathrm{Var}(X)=\alpha\beta^2
$$

**(b) 적률생성함수(mgf)**  
$X\sim \mathrm{Gamma}(\alpha,\beta)$이면
$$
\mathrm{mgf}_X(t)=(1-\beta t)^{-\alpha},\quad t<1/\beta
$$

**(c) 같은 $\beta$를 갖는 감마분포의 합**  
$X_1\sim \mathrm{Gamma}(\alpha_1,\beta)$, $X_2\sim \mathrm{Gamma}(\alpha_2,\beta)$이고 서로 독립이면
$$
X_1+X_2\sim \mathrm{Gamma}(\alpha_1+\alpha_2,\beta)
$$

**증명**  
**(a) 평균과 분산**  
먼저 $E(X)$는
$$
E(X)=\int_0^\infty x\cdot \frac{1}{\Gamma(\alpha)\beta^\alpha}x^{\alpha-1}e^{-x/\beta}\,dx
=\frac{1}{\Gamma(\alpha)\beta^\alpha}\int_0^\infty x^{\alpha}e^{-x/\beta}\,dx
$$
여기서 $x/\beta=y$ (즉 $x=\beta y,\,dx=\beta dy$)로 치환하면
$$
E(X)=\frac{1}{\Gamma(\alpha)\beta^\alpha}\int_0^\infty (\beta y)^{\alpha}e^{-y}\beta\,dy
=\frac{\beta}{\Gamma(\alpha)}\int_0^\infty y^{\alpha}e^{-y}\,dy
=\frac{\beta\Gamma(\alpha+1)}{\Gamma(\alpha)}=\alpha\beta
$$
이다(감마함수 점화식 사용).

또한
$$
E(X^2)=\int_0^\infty x^2 f(x)\,dx
=\frac{1}{\Gamma(\alpha)\beta^\alpha}\int_0^\infty x^{\alpha+1}e^{-x/\beta}\,dx
$$
에 같은 치환 $x=\beta y$를 쓰면
$$
E(X^2)=\frac{\beta^2}{\Gamma(\alpha)}\int_0^\infty y^{\alpha+1}e^{-y}\,dy
=\frac{\beta^2\Gamma(\alpha+2)}{\Gamma(\alpha)}
=\alpha(\alpha+1)\beta^2
$$
따라서
$$
\mathrm{Var}(X)=E(X^2)-\{E(X)\}^2=\alpha(\alpha+1)\beta^2-(\alpha\beta)^2=\alpha\beta^2
$$
이다. □

**(b) 적률생성함수(mgf)**  
$$
\mathrm{mgf}_X(t)=E(e^{tX})
=\int_0^\infty e^{tx}\frac{1}{\Gamma(\alpha)\beta^\alpha}x^{\alpha-1}e^{-x/\beta}\,dx
=\frac{1}{\Gamma(\alpha)\beta^\alpha}\int_0^\infty x^{\alpha-1}e^{-(1/\beta-t)x}\,dx
$$
여기서 $1/\beta-t>0$ (즉 $t<1/\beta$)일 때 $y=(1/\beta-t)x$로 치환하면
$$
\mathrm{mgf}_X(t)=\frac{1}{\Gamma(\alpha)\beta^\alpha}\cdot \frac{1}{(1/\beta-t)^\alpha}\int_0^\infty y^{\alpha-1}e^{-y}\,dy
=\frac{1}{\beta^\alpha(1/\beta-t)^\alpha}
=(1-\beta t)^{-\alpha}
$$
이다. □

**(c) 같은 $\beta$를 갖는 감마분포의 합**  
$$
\mathrm{mgf}_{X_1+X_2}(t)=\mathrm{mgf}_{X_1}(t)\mathrm{mgf}_{X_2}(t)
=(1-\beta t)^{-\alpha_1}(1-\beta t)^{-\alpha_2}=(1-\beta t)^{-(\alpha_1+\alpha_2)}
$$
이므로 mgf의 분포결정성으로 결론이 성립한다. □

### 형상모수가 자연수인 감마분포의 대의적 정의
shape 모수 $r$이 자연수이면
$$
\mathrm{Gamma}(r,\beta)\ \overset{d}{\equiv}\ \mathrm{Exp}_1(\beta)\oplus\cdots\oplus \mathrm{Exp}_r(\beta) \\
X\sim \mathrm{Gamma}(r,\beta)\iff X\overset{d}{\equiv}Z_1+\cdots+Z_r,\quad Z_i\stackrel{\text{iid}}{\sim}\mathrm{Exp}(\beta)
$$
포아송과정에서는 "사건 사이 대기시간"들이 서로 독립이고 동일한 지수분포를 따르므로, $r$번째 사건까지의 총 대기시간 $W_r$가 감마분포를 따른다는 직관과도 일치한다.

## 정규분포 (Normal Distribution)
**이항분포의 정규근사(De Moivre–Laplace approximation)**  
이항분포 누적확률을 적분으로 근사하는 식:
$$
\sum_{x:\ a\le \frac{x-np}{\sqrt{np(1-p)}}\le b}\binom{n}{x}p^x(1-p)^{n-x}
\ \simeq\
\int_a^b \frac{1}{\sqrt{2\pi}}e^{-z^2/2}\,dz,\quad n\to\infty
$$
여기서
$$
\phi(z)=\frac{1}{\sqrt{2\pi}}e^{-z^2/2},\quad -\infty<z<\infty
$$
는 적분값이 1이 되는 함수로서 표준정규분포(standard normal distribution) $N(0,1)$의 pdf로 정의된다.

**De Moivre–Laplace 정리 증명**  
$X_n \sim B(n,p)$이고 $\mu=np$, $\sigma^2=np(1-p)$라 하자. 표준화된 확률변수
$$
Z_n = \frac{X_n - np}{\sqrt{np(1-p)}}
$$
의 적률생성함수를 고려하면
$$
M_{Z_n}(t) = E(e^{tZ_n}) = e^{-\mu t/\sigma} M_{X_n}(t/\sigma)
$$
이고, 이항분포의 mgf $M_{X_n}(t) = (pe^t + 1-p)^n$을 이용하면
$$
M_{Z_n}(t) = e^{-\mu t/\sigma} \left(pe^{t/\sigma} + 1-p\right)^n
$$

$pe^{t/\sigma} + 1-p$를 테일러 전개하면
$$
pe^{t/\sigma} + 1-p = 1 + p\left(\frac{t}{\sigma} + \frac{t^2}{2\sigma^2} + O(t^3/\sigma^3)\right)
$$
$$
= 1 + \frac{pt}{\sigma} + \frac{pt^2}{2\sigma^2} + O(t^3/\sigma^3)
$$

$\mu/\sigma = np/\sigma$, $\sigma^2 = np(1-p)$를 이용하여 정리하면
$$
\log M_{Z_n}(t) = -\frac{\mu t}{\sigma} + n\log\left(1 + \frac{pt}{\sigma} + \frac{pt^2}{2\sigma^2} + O(t^3/\sigma^3)\right)
$$

$\log(1+A) = A - A^2/2 + O(A^3)$를 적용하고 $n\to\infty$일 때 주도항만 남기면
$$
\log M_{Z_n}(t) \to \frac{t^2}{2}
$$

따라서 $M_{Z_n}(t) \to e^{t^2/2}$이며, 이는 표준정규분포 $N(0,1)$의 mgf이다. 적률생성함수의 연속성 정리에 의해 $Z_n$은 분포수렴하여 $N(0,1)$로 근사된다. □

**$\phi$의 적분값이 1 증명**  
$$
I = \int_{-\infty}^{\infty} \phi(z)\,dz = \int_{-\infty}^{\infty} \frac{1}{\sqrt{2\pi}}e^{-z^2/2}\,dz
$$
를 계산한다.

양변을 제곱하면
$$
I^2 = \left(\int_{-\infty}^{\infty} \frac{1}{\sqrt{2\pi}}e^{-x^2/2}\,dx\right)\left(\int_{-\infty}^{\infty} \frac{1}{\sqrt{2\pi}}e^{-y^2/2}\,dy\right)
$$
$$
= \frac{1}{2\pi}\int_{-\infty}^{\infty}\int_{-\infty}^{\infty} e^{-(x^2+y^2)/2}\,dx\,dy
$$

극좌표 $(x,y) = (r\cos\theta, r\sin\theta)$로 변환하면 $dx\,dy = r\,dr\,d\theta$이고 $x^2+y^2=r^2$이므로
$$
I^2 = \frac{1}{2\pi}\int_{0}^{2\pi}\int_{0}^{\infty} e^{-r^2/2} r\,dr\,d\theta
$$

$u = r^2/2$로 치환하면 $du = r\,dr$이므로
$$
I^2 = \frac{1}{2\pi}\int_{0}^{2\pi}\,d\theta \int_{0}^{\infty} e^{-u}\,du = \frac{1}{2\pi} \cdot 2\pi \cdot 1 = 1
$$

따라서 $I = 1$이다. □

### 정규분포의 정의
일반적인 정규분포 $N(\mu,\sigma^2)$의 pdf는
$$
f(x)=\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right),\quad -\infty<x<\infty
$$
이며 $X\sim N(\mu,\sigma^2)$로 나타낸다.  

**정규분포의 형태**  
정규분포의 확률밀도함수는 $x=\mu$에 대칭인 종 모양(bell-shaped curve)의 곡선이다.
* $\mu$는 분포의 중심 위치를 결정하는 **위치모수**(location parameter)
* $\sigma$는 분포의 흩어진 정도를 나타내는 **척도모수**(scale parameter)
  - $\sigma$가 클수록 곡선이 완만하게 퍼지고, $\sigma$가 작을수록 뾰족하게 모인다.

### 정리 3.6.1 정규분포의 성질
**(a) 평균과 분산**  
$X\sim N(\mu,\sigma^2)$이면
$$
E(X)=\mu,\qquad \mathrm{Var}(X)=\sigma^2
$$
이다.

**(b) 적률생성함수(mgf)**  
$X\sim N(\mu,\sigma^2)$이면
$$
\mathrm{mgf}_X(t)=\exp\left(\mu t+\frac{1}{2}\sigma^2 t^2\right),\quad -\infty<t<\infty
$$
이다.

**(c) 독립 정규의 합**  
$X_1\sim N(\mu_1,\sigma_1^2)$, $X_2\sim N(\mu_2,\sigma_2^2)$이고 서로 독립이면
$$
X_1+X_2\sim N(\mu_1+\mu_2,\ \sigma_1^2+\sigma_2^2)
$$
이다.

#### 증명
**(a) 평균과 분산**  
$z=(x-\mu)/\sigma$로 치환하면 $\phi(z)$를 이용해
$$
E(X)=\int_{-\infty}^{\infty} x\cdot \frac{1}{\sigma}\phi\left(\frac{x-\mu}{\sigma}\right)\,dx
=\sigma\int_{-\infty}^{\infty} z\phi(z)\,dz+\mu
$$
인데, $z\phi(z)$는 홀함수이므로 적분이 0이 되어 $E(X)=\mu$이다.

분산은
$$
\mathrm{Var}(X)=\int_{-\infty}^{\infty}(x-\mu)^2\cdot \frac{1}{\sigma}\phi\left(\frac{x-\mu}{\sigma}\right)\,dx
=\sigma^2\int_{-\infty}^{\infty} z^2\phi(z)\,dz
$$
로 된다. 감마함수 성질을 이용해
$$
\int_{-\infty}^{\infty} z^2\phi(z)\,dz=1
$$
을 보이고, 따라서 $\mathrm{Var}(X)=\sigma^2$를 얻는다($\Gamma(3/2)$와 $\Gamma(1/2)=\sqrt{\pi}$ 사용). □

**(b) 적률생성함수(mgf)**  
$$
\mathrm{mgf}_X(t)=E(e^{tX})
=\int_{-\infty}^{\infty} e^{tx}\frac{1}{\sigma}\phi\left(\frac{x-\mu}{\sigma}\right)\,dx
=\int_{-\infty}^{\infty} e^{t(\sigma z+\mu)}\phi(z)dz \\
=e^{\mu t}\int_{-\infty}^{\infty} e^{\sigma tz}\phi(z)\,dz
$$
여기서 지수항을 $\exp\{-(z^2/2)+\sigma tz\}$로 합쳐 제곱완성을 하면
$$
-\frac{z^2}{2}+\sigma tz=-\frac{(z-\sigma t)^2}{2}+\frac{\sigma^2 t^2}{2}
$$
이므로
$$
\int_{-\infty}^{\infty} e^{\sigma tz}\phi(z)\,dz
=e^{\sigma^2 t^2/2}\int_{-\infty}^{\infty}\frac{1}{\sqrt{2\pi}}e^{-(z-\sigma t)^2/2}\,dz
=e^{\sigma^2 t^2/2}
$$
마지막 적분은 평균이 $\sigma t$인 정규 pdf의 적분이므로 1이다□

**(c) 독립 정규의 합**  
$$
\mathrm{mgf}_{X_1+X_2}(t)=\mathrm{mgf}_{X_1}(t)\mathrm{mgf}_{X_2}(t)
=\exp\left((\mu_1+\mu_2)t+\frac{1}{2}(\sigma_1^2+\sigma_2^2)t^2\right)
$$
이므로 분포결정성으로 결론이 나온다. □

### 정리 3.6.2 정규분포의 대의적 정의
**(a) 선형변환**  
$X\sim N(\mu,\sigma^2)$이면 상수 $a,b$에 대해
$$
aX+b\sim N(a\mu+b,\ a^2\sigma^2)
$$

**(b) 표준화**  
$$
X\sim N(\mu,\sigma^2)\iff \frac{X-\mu}{\sigma}\sim N(0,1)\iff X\overset{d}{\equiv}\sigma Z+\mu,\ Z\sim N(0,1)
$$

#### 증명
**(a) 선형변환**  
$$
\mathrm{mgf}_{aX+b}(t)=E(e^{t(aX+b)})=e^{bt}\mathrm{mgf}_X(at)
=e^{bt}\exp\left(\mu(at)+\frac{1}{2}\sigma^2(at)^2\right)
$$
$$
=\exp\left((a\mu+b)t+\frac{1}{2}(a^2\sigma^2)t^2\right)
$$
이 mgf는 $N(a\mu+b,a^2\sigma^2)$의 mgf이므로 결론. □

**(b) 표준화**  
이는 (a)의 특별한 경우로 $a=1/\sigma$, $b=-\mu/\sigma$를 대입하면 된다. □

### 누적분포함수(cdf)와 표준정규표
표준정규분포의 cdf를
$$
\Phi(x)=\int_{-\infty}^{x}\frac{1}{\sqrt{2\pi}}e^{-z^2/2}\,dz
$$
로 두면, $X\sim N(\mu,\sigma^2)$에 대해
$$
P(X\le x)=P\left(\frac{X-\mu}{\sigma}\le \frac{x-\mu}{\sigma}\right)=\Phi\left(\frac{x-\mu}{\sigma}\right)
$$
이다. 또한 표준정규는 0에 대해 대칭이므로
$$
\Phi(-z)=1-\Phi(z)
$$
이다. 표준정규표로 예시값 $\Phi(1.64)=0.9495$, $\Phi(1.65)=0.9505$, $\Phi(1.96)=0.9750$ 등을 확인할 수 있다. **two tailed**  

### 예 3.6.1
$X\sim N(3,4)$에서 $Z=(X-3)/\sqrt{4}=(X-3)/2\sim N(0,1)$로 표준화하여 계산.

**(a)** $P(5<X\le 7)$
$$
P(5<X\le 7)=\Phi\left(\frac{7-3}{2}\right)-\Phi\left(\frac{5-3}{2}\right)=\Phi(2.0)-\Phi(1.0)
$$
$$
=0.9772-0.8413=0.1359
$$

**(b)** $P(1<X\le 4)$
$$
P(1<X\le 4)=\Phi\left(\frac{4-3}{2}\right)-\Phi\left(\frac{1-3}{2}\right)=\Phi(0.5)-\Phi(-1.0)
$$
$$
=\Phi(0.5)-\{1-\Phi(1.0)\}=0.6915-(1-0.8413)=0.5328
$$

**(c)** $P(X>0)$
$$
P(X>0)=1-P(X\le 0)=1-\Phi\left(\frac{0-3}{2}\right)=1-\Phi(-1.5)=\Phi(1.5)\approx 0.9332
$$

### 상방 $\alpha$ 분위수(upper $\alpha$ quantile)
**one tailed**  
표준정규 $Z\sim N(0,1)$에서
$$
P(Z>z_\alpha)=\alpha\quad (0<\alpha<1)
$$
를 만족하는 $z_\alpha$를 상방 $\alpha$ 분위수라고 한다. 예로 $z_{0.025}=1.96$, $z_{0.05}=1.645$가 제시된다.

정리 3.6.2로부터 $X\sim N(\mu,\sigma^2)$이면
$$
P(X>\mu+\sigma z_\alpha)=\alpha
$$
가 성립하므로 분위수 계산에 사용한다.

### 예 3.6.2
$X\sim N(3,4)$에서 $\mu=3,\sigma=2$이다.

**(a)** $q_{0.95}$ (95 분위수): $P(X\le q_{0.95})=0.95$는 $P(X>q_{0.95})=0.05$와 같으므로
$$
q_{0.95}=3+2z_{0.05}=3+2(1.645)=6.290
$$

**(b)** $q_{0.025}$ (2.5 분위수): $P(X\le q_{0.025})=0.025$는 $P(X>q_{0.025})=0.975$와 같고,
$$
q_{0.025}=3+2z_{0.975}=3-2z_{0.025}=3-2(1.96)=-0.92
$$

### 정규분포의 주요 성질 요약
$X \sim N(\mu, \sigma^2)$일 때 다음이 성립한다.

**(1) 표준화 및 스케일-위치 변환**
$$
X \overset{d}{\equiv} \sigma Z + \mu, \quad Z \sim N(0,1)
$$

**(2) 선형변환의 닫힘성**
상수 $a, b$에 대해
$$
aX + b \sim N(a\mu + b, a^2\sigma^2)
$$

**(3) 독립합의 닫힘성**
$X_1 \sim N(\mu_1, \sigma_1^2)$, $X_2 \sim N(\mu_2, \sigma_2^2)$가 서로 독립이면
$$
X_1 \oplus X_2 \sim N(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)
$$


## 여러 분포의 정의 정리
### 표 3.1 이산확률분포 요약

| 분포 | 확률질량함수 (pmf) | 대의적 정의 | 적률생성함수 (mgf) | 누율생성함수 (cgf) |
|------|-------------------|-------------|-------------------|-------------------|
| 베르누이<br>$\text{Bernoulli}(p)$ | $P(X=x)=p^x(1-p)^{1-x}$<br>$x=0,1$ | - | $M_X(t)=pe^t+(1-p)$ | $K_X(t)=\log(pe^t+1-p)$ |
| 이항분포<br>$B(n,p)$ | $P(X=x)=\binom{n}{x}p^x(1-p)^{n-x}$<br>$x=0,1,\ldots,n$ | $X\overset{d}{\equiv}\sum_{i=1}^n Z_i$<br>$Z_i\stackrel{\text{iid}}{\sim}\text{Bernoulli}(p)$ | $M_X(t)=(pe^t+1-p)^n$ | $K_X(t)=n\log(pe^t+1-p)$ |
| 기하분포<br>$\text{Geo}(p)$ | $P(W_1=x)=(1-p)^{x-1}p$<br>$x=1,2,\ldots$ | - | $M_{W_1}(t)=\frac{pe^t}{1-(1-p)e^t}$<br>$t<-\log(1-p)$ | $K_{W_1}(t)=-\log\{1-(1-p)e^t\}+t+\log p$ |
| 음이항분포<br>$\text{Negbin}(r,p)$ | $P(W_r=x)=\binom{x-1}{r-1}p^r(1-p)^{x-r}$<br>$x=r,r+1,\ldots$ | $W_r\overset{d}{\equiv}\sum_{i=1}^r Z_i$<br>$Z_i\stackrel{\text{iid}}{\sim}\text{Geo}(p)$ | $M_{W_r}(t)=\left(\frac{pe^t}{1-(1-p)e^t}\right)^r$<br>$t<-\log(1-p)$ | $K_{W_r}(t)=r[-\log\{1-(1-p)e^t\}+t+\log p]$ |
| 포아송분포<br>$\text{Poisson}(\lambda)$ | $P(X=x)=e^{-\lambda}\frac{\lambda^x}{x!}$<br>$x=0,1,2,\ldots$ | - | $M_X(t)=\exp\{\lambda(e^t-1)\}$<br>$-\infty<t<\infty$ | $K_X(t)=\lambda(e^t-1)$ |
| 다항분포<br>$\text{Multi}(n;p_1,\ldots,p_k)$ | $P(X_1=x_1,\ldots,X_k=x_k)$<br>$=\frac{n!}{x_1!\cdots x_k!}p_1^{x_1}\cdots p_k^{x_k}$<br>$\sum x_i=n$ | $X\overset{d}{\equiv}\sum_{i=1}^n Z_i$<br>$Z_i\stackrel{\text{iid}}{\sim}\text{Multi}(1;p_1,\ldots,p_k)$ | $M_X(t)=\left(\sum_{i=1}^k p_ie^{t_i}\right)^n$<br>$t=(t_1,\ldots,t_k)^t$ | $K_X(t)=n\log\left(\sum_{i=1}^k p_ie^{t_i}\right)$ |

### 표 3.2 연속확률분포 요약

| 분포 | 확률밀도함수 (pdf) | 대의적 정의 | 적률생성함수 (mgf) | 누율생성함수 (cgf) |
|------|-------------------|-------------|-------------------|-------------------|
| 지수분포<br>$\text{Exp}(\beta)$ | $f(x)=\frac{1}{\beta}e^{-x/\beta}\mathbf{1}(x\ge 0)$ | - | $M_X(t)=(1-\beta t)^{-1}$<br>$t<1/\beta$ | $K_X(t)=-\log(1-\beta t)$ |
| 감마분포<br>$\text{Gamma}(\alpha,\beta)$ | $f(x)=\frac{1}{\Gamma(\alpha)\beta^\alpha}x^{\alpha-1}e^{-x/\beta}\mathbf{1}(x>0)$ | $X\overset{d}{\equiv}\sum_{i=1}^r Z_i$<br>$Z_i\stackrel{\text{iid}}{\sim}\text{Exp}(\beta)$<br>(when $\alpha=r\in\mathbb{N}$) | $M_X(t)=(1-\beta t)^{-\alpha}$<br>$t<1/\beta$ | $K_X(t)=-\alpha\log(1-\beta t)$ |
| 정규분포<br>$N(\mu,\sigma^2)$ | $f(x)=\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$<br>$-\infty<x<\infty$ | $X\overset{d}{\equiv}\sigma Z+\mu$<br>$Z\sim N(0,1)$ | $M_X(t)=\exp\left(\mu t+\frac{1}{2}\sigma^2 t^2\right)$<br>$-\infty<t<\infty$ | $K_X(t)=\mu t+\frac{1}{2}\sigma^2 t^2$ |

### 표 3.3 주요 분포의 평균과 분산

| 분포 | 평균 $E(X)$ | 분산 $\text{Var}(X)$ | 비고 |
|------|-------------|---------------------|------|
| $\text{Bernoulli}(p)$ | $p$ | $p(1-p)$ | |
| $B(n,p)$ | $np$ | $np(1-p)$ | |
| $H(n;N,D)$ | $np$ ($p=D/N$) | $\frac{N-n}{N-1}\cdot np(1-p)$ | 초기하분포 |
| $\text{Geo}(p)$ | $\frac{1}{p}$ | $\frac{1-p}{p^2}$ | |
| $\text{Negbin}(r,p)$ | $\frac{r}{p}$ | $\frac{r(1-p)}{p^2}$ | |
| $\text{Poisson}(\lambda)$ | $\lambda$ | $\lambda$ | |
| $\text{Multi}(n;p_1,\ldots,p_k)$ | $E(X_i)=np_i$ | $\text{Var}(X_i)=np_i(1-p_i)$<br>$\text{Cov}(X_i,X_j)=-np_ip_j$ | 다항분포 |
| $\text{Exp}(\beta)$ | $\beta$ | $\beta^2$ | $\beta=1/\lambda$ |
| $\text{Gamma}(\alpha,\beta)$ | $\alpha\beta$ | $\alpha\beta^2$ | |
| $N(\mu,\sigma^2)$ | $\mu$ | $\sigma^2$ | |