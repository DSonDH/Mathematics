# 제4장 표본분포 *(Sampling Distributions)*

## 4.1 통계량의 분포 *(Distributions of Statistics)*
모집단의 개수가 유한개일 경우에 비복원추출에 의한 표본을 이용하여 모집단에 대한 추측을 하는 방법의 타당성과 효율성을 연구하려면, 비복원추출로 인한 추출 결과의 종속성(dependence)에서 야기되는 어려움이 많다.

한편, 복원추출(with replacement)에 의한 표본의 경우에는 추출 결과 사이의 독립성(independence)이 있기 때문에 추측 방법의 성질을 연구하는 것이 용이하다. 또한 초기하분포의 이항분포 근사에서 알 수 있듯이, 모집단 크기가 커짐에 따라 비복원추출에 따른 추측 방법의 성질이 복원추출에 의한 방법의 성질과 거의 같게 된다.

이러한 용이성과 근사성 때문에 통계적 추측 방법의 성질을 연구할 때에는 흔히 유한모집단에서의 복원추출을 개념화하여 동일한 모집단을 독립적으로 관측하는 것을 전제로 한다.

이러한 이유에서 앞으로는 **랜덤표본(random sample)** 이란 서로 독립이고 동일한 분포를 따르는 확률변수들을 뜻한다. 또한 이러한 연구에서 가능한 모집단의 분포들을 모형으로 설정할 때, 모형의 설정에 사용되는 매개변수를 **모수(parameter)** 라고 하며, 가능한 모수 전체의 집합을 **모수공간(parameter space)** 이라고 한다.

#### 예 4.1.1
(a) 공정의 불량률에 대한 통계 조사의 경우에 모집단분포는 베르누이분포 $Bernoulli(p)$로 나타낼 수 있고, 불량률 $p$가 모수이고 미지인 불량률의 집합
$$
\Omega=\{p:0\le p\le 1\}
$$
가 모수공간이다.

(b) 하천의 오염도를 조사할 때 하천 물의 단위 부피당 부유생물의 수에 대한 모형으로 포아송분포 $Poisson(\lambda)$ 를 설정한다면, 평균 부유생물의 수 $\lambda$가 모수이고 미지인 $\lambda$의 집합
$$
\Omega=\{\lambda:\lambda\ge 0\}
$$
가 모수공간이다.

(c) 많은 과학 실험에서 오차의 분포에 대한 모형으로 정규분포 Normal $N(\mu,\sigma^2)$를 모형으로 설정한다. 이 경우에는 평균과 분산을 나타내는
$$
\theta=(\mu,\sigma^2)
$$
가 모수이고, 미지의 평균과 분산의 집합
$$
\Omega=\{(\mu,\sigma^2):-\infty<\mu<+\infty,\ \sigma^2>0\}
$$
가 모수공간이다.

이들 예에서 알 수 있듯이, 모수는 모집단 분포를 결정짓는 미지의 특성치로서 흔히 $\theta$로 나타내고, 모수공간은 $\Omega$로 나타내며 모집단 분포는 확률밀도함수 $f(x;\theta)$로 나타낸다. 이러한 미지의 모집단 분포 또는 모수에 대한 추측에 사용하는 '랜덤표본의 관측 가능한 함수'를 **통계량(statistic)** 이라고 한다.

### 랜덤표본과 통계량
모집단 분포가 확률밀도함수 $f(x;\theta)$, $\theta\in\Omega$인 모집단에서의 랜덤표본 $X_1,X_2,\dots,X_n$이란
$$
X_1,X_2,\dots,X_n \overset{iid}{\sim} f(x;\theta),\quad \theta\in\Omega
$$
를 뜻하고, 통계량 $u(X_1,X_2,\dots,X_n)$이란 랜덤표본의 함수로서 랜덤표본의 값이 주어지면 그 값이 정해지는 함수를 뜻한다.

#### 예 4.1.2
(a) 불량률 $p$를 모수로 하는 베르누이분포 $Bernoulli(p)$에서의 랜덤표본을 $X_1,X_2,\dots,X_n$이라고 할 때, 표본 중 불량개수의 합계 $X_1+X_2+\cdots+X_n$과 표본에서의 불량률을 나타내는 표본비율(sample proportion)
$$
\hat p=\frac{X_1+X_2+\cdots+X_n}{n}
$$
은 모두 통계량이다.

(b) 랜덤표본 $X_1,X_2,\dots,X_n$의 함수로서
$$
\bar X=\frac1n(X_1+X_2+\cdots+X_n),\qquad
S^2=\frac1{n-1}\sum_{i=1}^n(X_i-\bar X)^2
$$
를 각각 표본평균(sample mean), 표본분산(sample variance)이라고 하며, 모평균과 모분산의 추측에 사용되는 통계량이다.

(c) 모집단 분포가 연속형인 경우에 이 모집단에서의 랜덤표본 $X_1,X_2,\dots,X_n$을 크기 순서로 늘어놓은 것을
$$
X_{(1)}<X_{(2)}<\cdots<X_{(n)}
$$
이라고 할 때 이를 **순서통계량(order statistics)** 이라고 한다. 또한 이들 크기의 가운데를 나타내는 통계량
$$
\text{med},X_i=
\begin{cases}
X_{(m+1)}, & n=2m+1 \\
\frac{X_{(m)}+X_{(m+1)}}{2}, & n=2m
\end{cases}
$$
을 표본중앙값(sample median)이라고 하며, 모집단 분포의 중심부에 대한 추측에 사용되는 통계량이다.

랜덤표본 $X_1,X_2,\dots,X_n$의 함수인 통계량 $u(X_1,X_2,\dots,X_n)$은 확률변수로서 분포를 가지며, 이는 모집단 분포에 따라 결정된다. 이러한 통계량의 분포를 일반적으로 **표본분포(sampling distribution)** 라고 한다.

#### 예 4.1.3
(a) 불량률 $p$를 모수로 하는 베르누이분포 Bernoulli$(p)$에서의 랜덤표본을 $X_1,X_2,\dots,X_n$이라고 할 때, 표본 중 불량품의 개수인 $X_1+\cdots+X_n$은 이항분포 $Bin(n,p)$를 갖는다. 따라서 표본비율 $\hat p=\frac{X_1+\cdots+X_n}{n}$은
$$
P(\hat p=k/n)=\binom nk p^k(1-p)^{n-k},\quad k=0,1,\dots,n
$$
으로 주어지는 분포를 가지며 그 평균과 분산은 각각
$$
E(\hat p)=p,\qquad \text{Var}(\hat p)=\frac{p(1-p)}n
$$ 즉 표본비율 $\hat p$는 표본크기가 커짐에 따라 모비율 $p$ 주위에 집중되는 분포를 갖는다.

(b) 단위 부피당 평균 부유생물의 수가 $\lambda$인 포아송분포 $Poisson(\lambda)$에서의 랜덤표본을 $X_1,X_2,\dots,X_n$이라고 할 때, 정리 3.4.1로부터 표본 중 부유생물의 합계 $X_1+\cdots+X_n$은 포아송분포 $Poisson(n\lambda)$를 갖는다. 따라서 표본평균 $\bar X$는
$$
P(\bar X=k/n)=e^{-n\lambda}\frac{(n\lambda)^k}{k!},\quad k=0,1,\dots
$$
으로 주어지는 분포를 가지며
$$
E(\bar X)=\lambda,\qquad \text{Var}(\bar X)=\frac{\lambda}{n}
$$ 즉 표본평균 $\bar X$는 표본크기가 커짐에 따라 모평균 $\lambda$ 주위에 집중되는 분포를 갖는다.

이제 랜덤표본의 함수인 통계량의 분포를 일반적인 경우에 구하는 방법을 알아보자. 일반적으로 확률변수
$$
X=(X_1,\dots,X_n)^T
$$
의 분포로부터 $X$의 함수인
$$
Y=u(X)
$$
의 분포를 구하려면, $Y$에 관한 확률을 대응하는 $X$에 관한 확률로 바꾸어 계산해야 한다.

### (a) 이산형 경우
$$
P(Y=y)=P(u(X)=y)=\sum_{x:u(x)=y}P(X=x) \\
\therefore \text{pdf}_Y(y)=\sum_{x:u(x)=y}\text{pdf}_X(x)
$$

### (b) 연속형 경우
$$
P(Y\in y\pm|\Delta y|)=P(u(X)\in y\pm|\Delta y|)
\approx\sum_{x:u(x)=y}P(X\in x\pm|\Delta x|) \\
\therefore \text{pdf}_Y(y)|\Delta y|
\approx\sum_{x:u(x)=y}\text{pdf}_X(x)|\Delta x|
$$

#### 예 4.1.4
(a) 서로 독립이고 각각 이항분포 $Bin(n_1,p)$, $Bin(n_2,p)$를 따르는 확률변수 $X_1,X_2$에 대하여 $Y=X_1+X_2$의 확률밀도함수는
$$
Y\sim \text{Bin}(n_1+n_2,p)
$$

(b) 서로 독립이고 각각 포아송분포 $Poisson(\lambda_1)$, $Poisson(\lambda_2)$를 따르는 확률변수 $X_1,X_2$에 대하여
$$
Y=X_1+X_2\sim \text{Poisson}(\lambda_1+\lambda_2)
$$

(c) 표준정규분포를 따르는 확률변수 $X$에 대하여 $Y=X^2$이면
$$
Y\sim \text{Gamma}\left(\frac12,2\right)
$$
#### (c) 증명
$Y = X^2$이므로 $y > 0$에 대하여
$$
F_Y(y) = P(Y \le y) = P(X^2 \le y) = P(-\sqrt{y} \le X \le \sqrt{y})
$$

확률밀도함수를 구하기 위해 미분하면
$$
f_Y(y) = \frac{d}{dy}F_Y(y) = f_X(\sqrt{y})\cdot\frac{1}{2\sqrt{y}} + f_X(-\sqrt{y})\cdot\frac{1}{2\sqrt{y}}
$$

$X \sim N(0,1)$이므로 $f_X(x) = \frac{1}{\sqrt{2\pi}}e^{-x^2/2}$이고, 대칭성에 의해 $f_X(\sqrt{y}) = f_X(-\sqrt{y})$이므로
$$
f_Y(y) = 2 \cdot \frac{1}{\sqrt{2\pi}}e^{-y/2} \cdot \frac{1}{2\sqrt{y}} = \frac{1}{\sqrt{2\pi y}}e^{-y/2}, \quad y > 0
$$

이는 $Gamma\left(\frac{1}{2}, 2\right)$의 확률밀도함수
$$
f(y) = \frac{1}{\Gamma(1/2) \cdot 2^{1/2}}y^{-1/2}e^{-y/2} = \frac{1}{\sqrt{2\pi y}}e^{-y/2}
$$
와 일치한다. (단, $\Gamma(1/2) = \sqrt{\pi}$)

### 정리 4.1.1 (연속형 변수의 일대일 변환과 확률밀도함수)
연속형 $k$차원 확률변수
$$
X=(X_1,\dots,X_k)^T
$$
와 함수
$$
u=(u_1,\dots,u_k)^T:\mathcal X\to\mathcal Y
$$
에 대하여 다음이 성립한다고 하자.

(a) $P(X\in\mathcal X)=1$  
(b) $u$는 정의역 $\mathcal X$, 치역 $\mathcal Y$인 일대일 함수  
(c) $\mathcal X$는 열린집합이고 $u$는 미분 가능하며 야코비안
$$
J_u(x)=\det\left(\frac{\partial y}{\partial x}\right)\neq0
$$

이때 $Y=u(X)$의 확률밀도함수는
$$
\text{pdf}_Y(y)=\text{pdf}_X(x)\left|\det\left(\frac{\partial y}{\partial x}\right)\right|^{-1}
$$

#### 참고: 치환적분
정의역 $\mathcal Y$가 $n$차원 공간 $\mathbb{R}^n$에서의 열린 집합인 함수 $w:\mathcal Y\to\mathcal X$가 미분가능하고, 1차 편도함수가 연속함수로서 야코비안 행렬식
$$
J_w(y)=\det\left(\frac{\partial w(y)}{\partial y}\right)
$$
이 정의역에서 0이 아닐 때, 함수 $w$의 치역에서 정의된 적분가능한 함수 $f$에 대해 다음이 성립한다.
$$
\int_A f(x)dx = \int_{w^{-1}(A)}f(w(y))\left|J_w(y)\right|dy
$$

위 치환적분 공식을 간략하게
$$
f(x)dx = f(w(y))\left|\det\left(\frac{\partial w(y)}{\partial y}\right)\right|dy
$$
로 나타내기도 하며, 실제 적용할 때는 함수 $y=u(x)$의 역함수인 $w(y) = u^{-1}(y)$로 치환하는 경우가 많다. 이런 경우에는 역함수 정리로부터
$$
\frac{\partial}{\partial y}u^{-1}(y) = \left(\frac{\partial}{\partial x}u(x)\right)^{-1},\quad y=u(x)
$$
임을 이용하여 야코비안 행렬식을 계산할 수도 있다.

#### 증명
치환적분법으로부터
$$
P(Y\in B)=\int_{x\in u^{-1}(B)}\text{pdf}_X(x)dx
=\int_{y\in B}\text{pdf}_X(u^{-1}(y))|J_{u^{-1}}(y)|dy
$$

#### 예 4.1.5  위치모수와 척도모수를 이용한 확률분포
연속형 확률변수 $Z$의 확률밀도함수가 $f(z)$일 때, 양수 $\sigma$와 실수 $\mu$에 대하여
$$
X=\sigma Z+\mu
$$
로 정의된 확률변수 $X$의 확률밀도함수는
$$
\text{pdf}_X(x)
= f(z)\left|\frac{dx}{dz}\right|^{-1},\quad x=\sigma z+\mu
$$
$$
=\frac{1}{\sigma}f\left(\frac{x-\mu}{\sigma}\right)
$$
로 주어진다. 이러한 꼴의 확률밀도함수에서 $\mu$와 $\sigma$를 각각 **위치모수**(location parameter)와 **척도모수**(scale parameter)라고 한다.

(a) **정규분포** $N(\mu,\sigma^2)$
$$
N(\mu,\sigma^2)\ \overset{d}{\equiv}\ \sigma N(0,1)+\mu
$$

$$
\text{pdf}_X(x)
=\frac{1}{\sigma}\phi\left(\frac{x-\mu}{\sigma}\right),
\qquad
\phi(z)=\frac{1}{\sqrt{2\pi}}e^{-\frac12 z^2},
\quad -\infty<z<+\infty \\
X\sim N(\mu,\sigma^2)
\Leftrightarrow
X\overset{d}{\equiv}\sigma Z+\mu,\quad Z\sim N(0,1)
$$

(b) **로지스틱분포** $L(\mu,\sigma)$
$$
L(\mu,\sigma)\ \overset{d}{\equiv}\ \sigma L(0,1)+\mu
$$

$$
\text{pdf}_X(x)
=\frac{1}{\sigma}f\left(\frac{x-\mu}{\sigma}\right),
\qquad
f(z)=\frac{e^{z}}{(1+e^{z})^2},
\quad -\infty<z<+\infty \\
X\sim L(\mu,\sigma)
\Leftrightarrow
X\overset{d}{\equiv}\sigma Z+\mu,\quad Z\sim L(0,1)
$$

(c) **이중지수분포**(double exponential distribution) $DE(\mu,\sigma)$
$$
DE(\mu,\sigma)\ \overset{d}{\equiv}\ \sigma DE(0,1)+\mu
$$

$$
\text{pdf}_X(x)
=\frac{1}{\sigma}f\left(\frac{x-\mu}{\sigma}\right),
\qquad
f(z)=\frac12 e^{-|z|},
\quad -\infty<z<+\infty \\
X\sim DE(\mu,\sigma)
\Leftrightarrow
X\overset{d}{\equiv}\sigma Z+\mu,\quad Z\sim DE(0,1)
$$

(d) **코시(Cauchy)분포** $C(\mu,\sigma)$
$$
C(\mu,\sigma)\ \overset{d}{\equiv}\ \sigma C(0,1)+\mu
$$

$$
\text{pdf}_X(x)
=\frac{1}{\sigma}f\left(\frac{x-\mu}{\sigma}\right),
\qquad
f(z)=\frac{1}{\pi(1+z^2)},
\quad -\infty<z<+\infty \\
X\sim C(\mu,\sigma)
\Leftrightarrow
X\overset{d}{\equiv}\sigma Z+\mu,\quad Z\sim C(0,1)
$$

(e) **지수분포** $\mathrm{Exp}(\sigma)$
$$
\mathrm{Exp}(\sigma)\ \overset{d}{\equiv}\ \sigma \mathrm{Exp}(1)
$$

$$
\text{pdf}_X(x)
=\frac{1}{\sigma}f\left(\frac{x}{\sigma}\right),
\qquad
f(z)=e^{-z}I_{(0,\infty)}(z) \\
X\sim \mathrm{Exp}(\sigma)
\Leftrightarrow
X\overset{d}{\equiv}\sigma Z,\quad Z\sim \mathrm{Exp}(1)
$$

(f) **감마분포** $\mathrm{Gamma}(\alpha,\beta)$
$$
\mathrm{Gamma}(\alpha,\beta)\ \overset{d}{\equiv}\ \beta,\mathrm{Gamma}(\alpha,1)
$$

$$
\text{pdf}_X(x)
=\frac{1}{\beta}f\left(\frac{x}{\beta}\right),
\qquad
f(z)=\frac{1}{\Gamma(\alpha)}z^{\alpha-1}e^{-z}I_{(0,\infty)}(z) \\
X\sim \mathrm{Gamma}(\alpha,\beta)
\Leftrightarrow
X\overset{d}{\equiv}\beta Z,\quad Z\sim \mathrm{Gamma}(\alpha,1)
$$

#### 예 4.1.6
서로 독립이고 각각 감마분포 $Gamma(\alpha_1,\beta)$, $Gamma(\alpha_2,\beta)$를 따르는 확률변수 $X_1,X_2$에 대하여
$$
Y_1=\frac{X_1}{X_1+X_2},\quad Y_2=X_1+X_2
$$
일 때 $Y_1,Y_2$의 결합확률밀도함수와 주변확률밀도함수는?

**풀이**
$y_1 = \frac{x_1}{x_1+x_2}, \quad y_2 = x_1+x_2$ 역변환을 구하면
$$
x_1 = y_1 y_2, \quad x_2 = (1-y_1)y_2
$$

야코비안을 계산하면
$$
J = \det\begin{pmatrix}
\frac{\partial x_1}{\partial y_1} & \frac{\partial x_1}{\partial y_2} \\
\frac{\partial x_2}{\partial y_1} & \frac{\partial x_2}{\partial y_2}
\end{pmatrix}
= \det\begin{pmatrix}
y_2 & y_1 \\
-y_2 & 1-y_1
\end{pmatrix}
= y_2
$$

$X_1, X_2$가 독립이므로 결합확률밀도함수는
$$
f_{X_1,X_2}(x_1,x_2) = \frac{1}{\Gamma(\alpha_1)\beta^{\alpha_1}}x_1^{\alpha_1-1}e^{-x_1/\beta} \cdot \frac{1}{\Gamma(\alpha_2)\beta^{\alpha_2}}x_2^{\alpha_2-1}e^{-x_2/\beta}
$$

정리 4.1.1에 의해
$$
f_{Y_1,Y_2}(y_1,y_2) = f_{X_1,X_2}(x_1,x_2)|J|
$$
$$
= \frac{1}{\Gamma(\alpha_1)\Gamma(\alpha_2)\beta^{\alpha_1+\alpha_2}}(y_1 y_2)^{\alpha_1-1}[(1-y_1)y_2]^{\alpha_2-1}e^{-y_2/\beta} \cdot y_2
$$
$$
= \frac{\Gamma(\alpha_1+\alpha_2)}{\Gamma(\alpha_1)\Gamma(\alpha_2)}y_1^{\alpha_1-1}(1-y_1)^{\alpha_2-1} \cdot \frac{1}{\Gamma(\alpha_1+\alpha_2)\beta^{\alpha_1+\alpha_2}}y_2^{\alpha_1+\alpha_2-1}e^{-y_2/\beta}
$$

따라서 $Y_1$과 $Y_2$는 독립이고
$$
Y_1 \sim \text{Beta}(\alpha_1, \alpha_2), \quad Y_2 \sim \text{Gamma}(\alpha_1+\alpha_2, \beta)
$$

### 베타분포의 정의 *(Beta Distribution)*
두 모수 $\alpha_1 > 0$, $\alpha_2 > 0$에 대하여 확률밀도함수가
$$
f(x) = \frac{\Gamma(\alpha_1+\alpha_2)}{\Gamma(\alpha_1)\Gamma(\alpha_2)} x^{\alpha_1-1}(1-x)^{\alpha_2-1}, \quad 0 < x < 1
$$
로 주어지는 분포를 **베타분포(Beta distribution)** $\text{Beta}(\alpha_1, \alpha_2)$라고 한다.

베타분포는 감마분포와 다음과 같은 관계가 있다:
$$
\text{Beta}(\alpha_1, \alpha_2) \ \overset{d}{\equiv} \ \frac{\text{Gamma}(\alpha_1, \beta)}{\text{Gamma}(\alpha_1, \beta) \oplus \text{Gamma}(\alpha_2, \beta)}
$$

즉, 서로 독립인 $Y_1 \sim \text{Gamma}(\alpha_1, \beta)$, $Y_2 \sim \text{Gamma}(\alpha_2, \beta)$에 대하여
$$
X \sim \text{Beta}(\alpha_1, \alpha_2) \Leftrightarrow X \overset{d}{\equiv} \frac{Y_1}{Y_1 + Y_2}
$$
$$
f(x)=\frac{\Gamma(\alpha_1+\alpha_2)}{\Gamma(\alpha_1)\Gamma(\alpha_2)}
x^{\alpha_1-1}(1-x)^{\alpha_2-1},\quad 0<x<1
$$

- 베타함수는 다음과 같은 적분으로도 정의된다:
$$
B(\alpha_1, \alpha_2) = \int_0^1 x^{\alpha_1-1}(1-x)^{\alpha_2-1}dx, \quad \alpha_1, \alpha_2 > 0 \\
= \frac{\Gamma(\alpha_1)\Gamma(\alpha_2)}{\Gamma(\alpha_1+\alpha_2)}
$$

#### 예 4.1.7
확률변수 $X_1, X_2$이 서로 독립이고 동일한 확률밀도함수 $f(x)=I_{(0,1)}(x)$ 를 가질 때, $Y=\frac{X_1+X_2}{2}$
의 확률밀도함수는?

이차원 확률변수 사이의 일대일 변환을 생각하기 위하여
$Y=\frac{X_1+X_2}{2}, Z=\frac{X_1-X_2}{2}$
라고 하면, 다음과 같이 함수 $u$와 그 정의역 $\mathcal X$, 치역 $\mathcal Y$, 역함수 $u^{-1}$를 생각할 수 있다.
$$
u:\begin{cases}
y=(x_1+x_2)/2 \\
z=(x_1-x_2)/2
\end{cases},
\qquad
u^{-1}:\begin{cases}
x_1=y+z \\
x_2=y-z
\end{cases}
$$

$$
\mathcal X={(x_1,x_2)^T:\text{pdf}_{X_1,X_2}(x_1,x_2)>0}
={(x_1,x_2)^T:0<x_1<1,\ 0<x_2<1} \\
\mathcal Y={(y,z)^T:0<y+z<1,\ 0<y-z<1}
$$

이로부터 함수 $u$는 $\mathcal X$에서 $\mathcal Y$로의 일대일 함수로서 <정리 4.1.1>의 조건을 만족시키고, 그 역함수 $u^{-1}$의 야코비안 행렬식이
$$
J_{u^{-1}}
=\det\begin{pmatrix}
1 & 1 \\
1 & -1
\end{pmatrix}
=-2
$$

따라서 <정리 4.1.1>로부터 $Y,Z$의 결합확률밀도함수는
$$
\text{pdf}_{Y,Z}(y,z)
=I_{(0<y+z<1,\ 0<y-z<1)}|-2|
$$

이로부터 $Y$의 주변확률밀도함수를 구하기 위하여, 지표함수의 연립부등식을 $z$에 대한 부등식으로 나타내면
$$
-y<z<1-y,\qquad y-1<z<y
$$
즉 $a\vee b=\max(a,b)$, $a\wedge b=\min(a,b)$라고 하면
$$
(-y)\vee(y-1)<z<(1-y)\wedge y
$$
이다.

그러므로 $y\le\frac12$인 경우와 $y>\frac12$인 경우로 나누어 pdf를 구하면 다음과 같다.

(i) $y\le\frac12$인 경우  
$(-y)\vee(y-1)=-y,\quad (1-y)\wedge y=y$
이고 $-y<y\Leftrightarrow y>0$이므로
$$
\text{pdf}_Y(y)
=\int_{-\infty}^{+\infty}\text{pdf}_{Y,Z}(y,z),dz
=\int_{-y}^{y}2,dz,I_{(0,1/2]}(y)
=4yI_{(0,1/2]}(y)
$$

(ii) $y>\frac12$인 경우  
$(-y)\vee(y-1)=y-1,\quad (1-y)\wedge y=1-y$
이고 $y-1<1-y\Leftrightarrow y<1$이므로
$$
\text{pdf}_Y(y)
=\int_{y-1}^{1-y}2,dz,I_{(1/2,1)}(y)
=4(1-y)I_{(1/2,1)}(y)
$$

따라서 $Y$의 주변확률밀도함수를 하나의 식으로 나타내면 <예 4.1.3>에서와 같이
$$
\text{pdf}_Y(y)=(2-4|y-1/2|)I_{(0,1)}(y)
$$

><예 4.1.7>에서 서로 독립인 $X_1,X_2$의 공통 분포를 구간 $(0,1)$에서의 균등분포(uniform distribution)라고 하며, 이들의 평균
>$$
>Y=\frac{X_1+X_2}{2}
>$$
>의 분포를 구간 $(0,1)$에서의 삼각분포(triangular distribution)라고 한다.
>
>한편, 일반적인 구간 $(a,b)$에서의 균등분포는 다음과 같이 정의한다.

### 균등분포의 정의
$$
U(a,b)\ \overset{d}{\equiv}\ (b-a)U(0,1)+a \\
X\sim U(a,b)
\Leftrightarrow
\text{pdf}_X(x)=\frac{1}{b-a}I_{(a,b)}(x)\\
\Leftrightarrow
X\overset{d}{\equiv}(b-a)Z+a,\quad Z\sim U(0,1)
$$

#### 예 4.1.8
서로 독립이고 표준정규분포 $N(0,1)$를 따르는 확률변수 $X,Y$에 대하여, 극좌표 변환
$$
\begin{cases}
X=R\cos\Theta \\
Y=R\sin\Theta
\end{cases},
\quad
0\le R<+\infty,\ 0\le\Theta<2\pi
$$
를 만족시키는 $R,\Theta$의 결합확률밀도함수를 구하여라.

$X,Y$의 결합확률밀도함수는
$$
\text{pdf}_{X,Y}(x,y)
=\frac{1}{2\pi}e^{-\frac12(x^2+y^2)}
I_{(-\infty<x<+\infty,\ -\infty<y<+\infty)}
$$
이고, 주어진 극좌표 변환
$$
w:\begin{cases}
x=r\cos\theta \\
y=r\sin\theta
\end{cases},
\quad
0\le r<+\infty,\ 0\le\theta<2\pi
$$
는 정의역
$$
\mathcal Y={(r,\theta):0\le r<+\infty,\ 0\le\theta<2\pi}
$$
와 치역
$$
\mathcal X={(x,y):-\infty<x<+\infty,\ -\infty<y<+\infty}
$$
인 일대일 함수로서 미분가능하고, 그 야코비안 행렬식은
$$
J_w(y)
=\det\begin{pmatrix}
\frac{\partial x}{\partial r} & \frac{\partial y}{\partial r} \\
\frac{\partial x}{\partial\theta} & \frac{\partial y}{\partial\theta}
\end{pmatrix}
=\det\begin{pmatrix}
\cos\theta & \sin\theta \\
-r\sin\theta & r\cos\theta
\end{pmatrix}
=r
$$
임을 알 수 있다.

따라서 극좌표 변환 $w$의 역함수를 $u=w^{-1}$라고 하면 역함수 정리로부터 함수 $u(x,y)$가 <정리 4.1.1>의 조건을 만족시키는 것을 알 수 있다.

그러므로 <정리 4.1.1>로부터 $R,\Theta$의 결합확률밀도함수는
$$
\text{pdf}_{R,\Theta}(r,\theta)
=\frac{1}{2\pi}e^{-\frac12 r^2(\cos^2\theta+\sin^2\theta)}
I_{[0,+\infty)}(r)I_{[0,2\pi)}(\theta)|r|
$$
$$
= r e^{-\frac12 r^2}I_{[0,+\infty)}(r)\frac{1}{2\pi}I_{[0,2\pi)}(\theta)
$$
로 주어진다.

즉 $R,\Theta$는 서로 독립이고
$$
\frac{R^2}{2}\sim\text{Exp}(1),\qquad \Theta\sim U[0,2\pi)
$$
이다.

#### 예 4.1.9
서로 독립이고 각각 감마분포 $\text{Gamma}(\alpha_i,\beta)$를 따르는 확률변수
$$
X_i\ (i=1,\dots,k+1)
$$
들에 대하여
$$
Y_1=\frac{X_1}{X_1+\cdots+X_{k+1}},\ \dots,
Y_k=\frac{X_k}{X_1+\cdots+X_{k+1}}
$$
라고 할 때, $Y_1,\dots,Y_k$의 결합확률밀도함수를 구하여라.


**풀이**  
$k+1$차원 변환을 사용한다. $Y_{k+1} = X_1 + \cdots + X_{k+1}$로 정의하면
$$
\begin{cases}
y_1 = \frac{x_1}{x_1+\cdots+x_{k+1}} \\
\vdots \\
y_k = \frac{x_k}{x_1+\cdots+x_{k+1}} \\
y_{k+1} = x_1+\cdots+x_{k+1}
\end{cases}
$$

역변환은
$$
\begin{cases}
x_1 = y_1 y_{k+1} \\
\vdots \\
x_k = y_k y_{k+1} \\
x_{k+1} = (1-y_1-\cdots-y_k)y_{k+1}
\end{cases}
$$

야코비안을 계산하면
$$
J = \det\begin{pmatrix}
y_{k+1} & 0 & \cdots & 0 & y_1 \\
0 & y_{k+1} & \cdots & 0 & y_2 \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & \cdots & y_{k+1} & y_k \\
-y_{k+1} & -y_{k+1} & \cdots & -y_{k+1} & 1-y_1-\cdots-y_k
\end{pmatrix}
= y_{k+1}^k
$$

$X_1, \ldots, X_{k+1}$이 독립이므로 결합확률밀도함수는
$$
f_{X_1,\ldots,X_{k+1}}(x_1,\ldots,x_{k+1}) = \prod_{i=1}^{k+1} \frac{1}{\Gamma(\alpha_i)\beta^{\alpha_i}}x_i^{\alpha_i-1}e^{-x_i/\beta}
$$

정리 4.1.1에 의해
$$
f_{Y_1,\ldots,Y_k,Y_{k+1}}(y_1,\ldots,y_k,y_{k+1}) = f_{X_1,\ldots,X_{k+1}}(x_1,\ldots,x_{k+1}) \cdot y_{k+1}^k
$$
$$
= \prod_{i=1}^{k+1} \frac{1}{\Gamma(\alpha_i)\beta^{\alpha_i}}(y_i y_{k+1})^{\alpha_i-1}e^{-y_i y_{k+1}/\beta} \cdot y_{k+1}^k
$$

여기서 $y_{k+1} = 1-y_1-\cdots-y_k$를 $x_{k+1}$로 사용하고 정리하면
$$
= \frac{1}{\prod_{i=1}^{k+1}\Gamma(\alpha_i)\beta^{\alpha_i}} \prod_{i=1}^{k} y_i^{\alpha_i-1} (1-y_1-\cdots-y_k)^{\alpha_{k+1}-1} y_{k+1}^{\alpha_1+\cdots+\alpha_{k+1}-1} e^{-y_{k+1}/\beta} \cdot y_{k+1}^k
$$
$$
= \frac{\Gamma(\alpha_1+\cdots+\alpha_{k+1})}{\Gamma(\alpha_1)\cdots\Gamma(\alpha_{k+1})} \prod_{i=1}^{k} y_i^{\alpha_i-1} (1-y_1-\cdots-y_k)^{\alpha_{k+1}-1}
$$
$$
\times \frac{1}{\Gamma(\alpha_1+\cdots+\alpha_{k+1})\beta^{\alpha_1+\cdots+\alpha_{k+1}}} y_{k+1}^{\alpha_1+\cdots+\alpha_{k+1}-1} e^{-y_{k+1}/\beta}
$$

따라서 $(Y_1,\ldots,Y_k)$와 $Y_{k+1}$은 독립이고
$$
(Y_1,\ldots,Y_k) \sim \text{Dirichlet}(\alpha_1,\ldots,\alpha_k,\alpha_{k+1}), \quad Y_{k+1} \sim \text{Gamma}(\alpha_1+\cdots+\alpha_{k+1}, \beta)
$$

그러므로 $(Y_1,\ldots,Y_k)$의 결합확률밀도함수는
$$
\text{pdf}_{Y_1,\ldots,Y_k}(y_1,\ldots,y_k) = \frac{\Gamma(\alpha_1+\cdots+\alpha_{k+1})}{\Gamma(\alpha_1)\cdots\Gamma(\alpha_{k+1})} \prod_{i=1}^{k} y_i^{\alpha_i-1} (1-y_1-\cdots-y_k)^{\alpha_{k+1}-1}
$$

이 예에서 구한 $Y_1,\dots,Y_k$의 결합분포는 베타분포를 다차원으로 일반화 한 분포로서, Dirichlet분포라 하며 정의는 아래와 같다.  
### 디리클레분포의 정의 *(Dirichlet Distribution)*
모수 $\alpha_1,\ldots,\alpha_{k+1}>0$에 대하여 다음과 같이 정의된다.
  - 참고: 알파의 의미
    - **감마분포** $\text{Gamma}(\alpha, \beta)$: 알파: 누적된 사건 수에 대응하는 shape 파라미터, 베타: 단위당 사건 발생률
    - **디리클레분포** $\text{Dirichlet}(\alpha_1,\ldots,\alpha_{k+1})$: 정규화되기 전 각 범주에 대응하는 감마분포의 shape 파라미터이며,
통계적으로는 해당 범주의 가상 관측 횟수를 의미
    - (**베타분포** $\text{Beta}(\alpha_1, \alpha_2)$: 성공 횟수 관련 모수)


서로 독립인 $X_1\sim\text{Gamma}(\alpha_1,\beta),\ldots,X_{k+1}\sim\text{Gamma}(\alpha_{k+1},\beta)$에 대하여
$$
(Y_1,\ldots,Y_k)\sim\text{Dirichlet}(\alpha_1,\ldots,\alpha_k,\alpha_{k+1})
$$
$$
\Leftrightarrow
(Y_1,\ldots,Y_k)\overset{d}{\equiv}\left(\frac{X_1}{X_1\oplus\cdots\oplus X_{k+1}},\ldots,\frac{X_k}{X_1\oplus\cdots\oplus X_{k+1}}\right)
$$

즉, 디리클레분포는 감마분포들의 합으로 정규화된 비율들의 결합분포이다.
$$
(Y_1,\dots,Y_k)^T\sim\text{Dirichlet}(\alpha_1,\dots,\alpha_k,\alpha_{k+1})
$$
$$
\Leftrightarrow
\text{pdf}_{Y_1,\dots,Y_k}(y_1,\dots,y_k)
=\\ 
\frac{\Gamma(\alpha_1+\cdots+\alpha_{k+1})}
{\Gamma(\alpha_1)\cdots\Gamma(\alpha_{k+1})}
\prod_{i=1}^k y_i^{\alpha_i-1}
(1-y_1-\cdots-y_k)^{\alpha_{k+1}-1}
$$

>일대일 변환 뿐만 아니라, 다대일 함수에도 변환을 적용할 수 있다.  
>정의역을 분할하고 각각의 정의역에서의 일대일 변환에 대해 치환적분의 원리를 이용한다.  
### 정리 4.1.2 (연속형 변수의 다대일 변환과 확률밀도함수)
연속형의 $k$차원 확률변수
$$
X=(X_1,\dots,X_k)^T
$$
와 함수
$$
u=(u_1,\dots,u_k)^T:\mathcal X\to\mathcal Y
$$
에 대하여 다음이 성립한다고 하자.

(a) $P(X\in\mathcal X)=1$

(b) 벡터값 함수 $u=(u_1,\dots,u_k)^T:\mathcal X\to\mathcal Y$는 정의역이 $\mathcal X$이고 치역이 $\mathcal Y$인 다대일 함수이다.

(c) 함수 $u=(u_1,\dots,u_k)^T$의 정의역 $\mathcal X$는 서로 공통 부분이 없는 열린집합 $\mathcal X_1,\dots,\mathcal X_m$의 합집합으로 나타낼 수 있고, 함수 $u$에서 정의역을 $\mathcal X_r\ (r=1,\dots,m)$로 제한한 함수
$$
u^{(r)}(x)=u(x),\quad x\in\mathcal X_r
$$
는 각각 $\mathcal X_r$에서 $\mathcal Y$로의 일대일 함수로서 미분가능하며, 1차 편도함수가 연속함수로서 0이 아닌 야코비안(Jacobian) 행렬식을 갖는다. 즉
$$
u^{(r)}(x)=(u_1^{(r)}(x),\dots,u_k^{(r)}(x))^T
$$
라고 할 때
$$
J_{u^{(r)}}(x)=\det\left(\frac{\partial u_j^{(r)}(x)}{\partial x_i}\right)\neq0,\quad \forall x\in\mathcal X_r
$$
이다.

이러한 조건하에서 확률변수 $X=(X_1,\dots,X_k)^T$와 함수 $u$를 이용하여 정의된 $k$차원 확률변수
$$
Y=u(X),\quad \text{즉 } Y=(Y_1,\dots,Y_k)^T=(u_1(X),\dots,u_k(X))^T
$$
의 확률밀도함수는 다음과 같이 주어진다.
$$
\text{pdf}_Y(y)=\sum_{r=1}^m\text{pdf}_X\left((u^{(r)})^{-1}(y)\right)\left|J_{(u^{(r)})^{-1}}(y)\right|,\quad y\in\mathcal Y
$$
즉
$$
\text{pdf}_Y(y)=\sum_{x:\ u(x)=y}\text{pdf}_X(x)\left|\det\left(\frac{\partial u(x)}{\partial x}\right)\right|^{-1},\quad y\in\mathcal Y
$$
이다.

#### 증명
<정리 4.1.1>의 증명에서와 같이 분할된 각각의 정의역 $\mathcal X_r\ (r=1,\dots,m)$에서의 일대일 함수
$$
y=u^{(r)}(x)
$$
의 역함수
$$
x=(u^{(r)})^{-1}(y)
$$
를 이용한 치환적분법을 적용하면, $\mathcal Y$의 임의의 부분집합 $B$에 대하여 다음이 성립함을 알 수 있다.
$$
P(Y\in B)=P(u(X)\in B)
$$
$$
=\sum_{r=1}^m P(u^{(r)}(X)\in B,\ X\in\mathcal X_r)
$$
$$
=\sum_{r=1}^m\int_{x\in(u^{(r)})^{-1}(B)}\text{pdf}_X(x)\,dx
$$
$$
=\int_{y\in B}\sum_{r=1}^m\text{pdf}_X\left((u^{(r)})^{-1}(y)\right)\left|J_{(u^{(r)})^{-1}}(y)\right|\,dy\quad (\because x=(u^{(r)})^{-1}(y))
$$

따라서
$$
\text{pdf}_Y(y)=\sum_{r=1}^m\text{pdf}_X\left((u^{(r)})^{-1}(y)\right)\left|J_{(u^{(r)})^{-1}}(y)\right|,\quad y\in\mathcal Y
$$
이고, 이는
$$
\text{pdf}_Y(y)=\sum_{x:\ u(x)=y}\text{pdf}_X(x)\left|\det\left(\frac{\partial u(x)}{\partial x}\right)\right|^{-1}
$$
과 같다.

#### 예 4.1.10
$(-1,1)$에서 균등분포를 따르는 확률변수 $X$에 대하여
$Y=X^2$의 확률밀도함수를 구하여라.

**풀이**  
$X \sim U(-1,1)$이므로 $\text{pdf}_X(x) = \frac{1}{2}I_{(-1,1)}(x)$

$Y = X^2$이고, $y > 0$일 때 $u(x) = x^2 = y$를 만족하는 $x$는 두 개 존재한다:
$$
x_1 = \sqrt{y}, \quad x_2 = -\sqrt{y}
$$

정의역을 분할하면:
- $\mathcal{X}_1 = (0,1)$에서 $u^{(1)}(x) = x^2$, 역함수는 $x = \sqrt{y}$
- $\mathcal{X}_2 = (-1,0)$에서 $u^{(2)}(x) = x^2$, 역함수는 $x = -\sqrt{y}$

각각의 야코비안은:
$$
J_{u^{(1)}}(x) = 2x, \quad J_{u^{(2)}}(x) = 2x
$$

따라서
$$
\left|J_{(u^{(1)})^{-1}}(y)\right| = \left|\frac{1}{2\sqrt{y}}\right| = \frac{1}{2\sqrt{y}}, \quad
\left|J_{(u^{(2)})^{-1}}(y)\right| = \left|\frac{1}{-2\sqrt{y}}\right| = \frac{1}{2\sqrt{y}}
$$

정리 4.1.2에 의해
$$
\text{pdf}_Y(y) = \text{pdf}_X(\sqrt{y}) \cdot \frac{1}{2\sqrt{y}} + \text{pdf}_X(-\sqrt{y}) \cdot \frac{1}{2\sqrt{y}}
$$
$$
= \frac{1}{2} \cdot \frac{1}{2\sqrt{y}} + \frac{1}{2} \cdot \frac{1}{2\sqrt{y}} = \frac{1}{2\sqrt{y}}I_{(0,1)}(y)
$$

#### 예 4.1.11
확률변수 $X=(X_1,X_2)^T$의 결합확률밀도함수가
$$
\text{pdf}_X(x_1,x_2)=\frac1\pi I_{(0<x_1^2+x_2^2<1)}
$$
일 때,
$$
Y_1=X_1^2+X_2^2,\qquad
Y_2=\frac{X_1^2}{X_1^2+X_2^2}
$$
로 정의된 $Y=(Y_1,Y_2)^T$의 결합확률밀도함수를 구하여라.

**풀이**  
변환 $y_1 = x_1^2 + x_2^2$, $y_2 = \frac{x_1^2}{x_1^2+x_2^2}$의 역변환을 구하면:
$$
x_1^2 = y_1 y_2, \quad x_2^2 = y_1(1-y_2)
$$

$X$의 정의역은 $\mathcal{X} = \{(x_1,x_2): x_1^2+x_2^2 < 1\}$이고, 이를 네 개의 사분면으로 분할한다:
- $\mathcal{X}_1: x_1>0, x_2>0$
- $\mathcal{X}_2: x_1<0, x_2>0$
- $\mathcal{X}_3: x_1<0, x_2<0$
- $\mathcal{X}_4: x_1>0, x_2<0$

각 사분면에서 역변환은:
$$
\begin{aligned}
\mathcal{X}_1: & \quad x_1 = \sqrt{y_1 y_2}, \quad x_2 = \sqrt{y_1(1-y_2)} \\
\mathcal{X}_2: & \quad x_1 = -\sqrt{y_1 y_2}, \quad x_2 = \sqrt{y_1(1-y_2)} \\
\mathcal{X}_3: & \quad x_1 = -\sqrt{y_1 y_2}, \quad x_2 = -\sqrt{y_1(1-y_2)} \\
\mathcal{X}_4: & \quad x_1 = \sqrt{y_1 y_2}, \quad x_2 = -\sqrt{y_1(1-y_2)}
\end{aligned}
$$

야코비안을 계산하면 ($\mathcal{X}_1$의 경우):
$$
J = \det\begin{pmatrix}
\frac{\partial x_1}{\partial y_1} & \frac{\partial x_1}{\partial y_2} \\
\frac{\partial x_2}{\partial y_1} & \frac{\partial x_2}{\partial y_2}
\end{pmatrix}
= \det\begin{pmatrix}
\frac{\sqrt{y_2}}{2\sqrt{y_1}} & \frac{\sqrt{y_1}}{2\sqrt{y_2}} \\
\frac{\sqrt{1-y_2}}{2\sqrt{y_1}} & -\frac{\sqrt{y_1}}{2\sqrt{1-y_2}}
\end{pmatrix}
= -\frac{1}{4\sqrt{y_2(1-y_2)}}
$$

네 사분면 모두에서 $|J| = \frac{1}{4\sqrt{y_2(1-y_2)}}$

정리 4.1.2에 의해
$$
\text{pdf}_{Y_1,Y_2}(y_1,y_2) = 4 \cdot \frac{1}{\pi} \cdot \frac{1}{4\sqrt{y_2(1-y_2)}}I_{(0<y_1<1, 0<y_2<1)}
$$
$$
= \frac{1}{\pi\sqrt{y_2(1-y_2)}}I_{(0,1)}(y_1)I_{(0,1)}(y_2)
$$
$$
= I_{(0,1)}(y_1) \cdot \frac{1}{\pi\sqrt{y_2(1-y_2)}}I_{(0,1)}(y_2)
$$

따라서 $Y_1$과 $Y_2$는 독립이고
$$
Y_1 \sim U(0,1), \quad Y_2 \sim \text{Beta}\left(\frac{1}{2},\frac{1}{2}\right)
$$


## 4.2 대표적인 표본분포 *(Common Sampling Distributions)*
모집단의 분포가 정규분포임을 가정하는 경우에 사용되는 통계량인 표본평균, 표본분산의 분포를 나타내는 대표적인 표본분포에 대하여 알아본다.

#### 예 4.2.1
서로 독립이고 표준정규분포 $N(0,1)$를 따르는 확률변수들 $X_1,\dots,X_r$에 대하여
$$
Y=X_1^2+\cdots+X_r^2
$$
의 확률밀도함수를 구하여라.

**풀이**  
<예 4.1.4>의 (c)로부터 $X_i^2\ (i=1,\dots,r)$는 각각 $\text{Gamma}(1/2,2)$ 분포를 따르고 이들이 서로 독립이므로, <정리 3.5.2>에 주어진 감마분포의 성질로부터
$$
Y=X_1^2+\cdots+X_r^2 \sim \text{Gamma}(r/2,2)
$$

따라서
$$
\text{pdf}_Y(y)
=\frac{1}{\Gamma(r/2)2^{r/2}}y^{r/2-1}e^{-y/2}I_{(0,\infty)}(y)
$$

<예 4.2.1>에서의 분포인 $\text{Gamma}(r/2,2)$ 분포를 자유도(degrees of freedom)가 $r$인 **카이제곱분포(chi-squared distribution)** 라고 하며 기호로는
$$
Y\sim\chi^2(r)
$$
로 나타낸다.

### 카이제곱분포의 정의 *(Chi-squared Distribution)*
$$
\chi^2(r)\overset{d}{\equiv}N_1(0,1)^2\oplus\cdots\oplus N_r(0,1)^2\overset{d}{\equiv}\text{Gamma}(r/2,2)
$$

$$
Y\sim\chi^2(r)\ (r>0)
\Leftrightarrow
Y\sim\text{Gamma}(r/2,2) \\
\Leftrightarrow
\text{pdf}_Y(y)
=\frac{1}{\Gamma(r/2)2^{r/2}}y^{r/2-1}e^{-y/2}I_{(0,\infty)}(y) \\
\Leftrightarrow
Y=X_1^2+\cdots+X_r^2,\quad X_i\overset{iid}{\sim} N(0,1)\ (i=1,\dots,r,\ r\text{이 자연수인 경우})
$$

표준정규분포의 경우와 같이, $Y\sim\chi^2(r)$일 때
$$
P(Y>\chi^2_\alpha(r))=\alpha\quad(0<\alpha<1)
$$
를 만족시키는 값 $\chi^2_\alpha(r)$를 자유도 $r$인 카이제곱분포 $\chi^2(r)$의 **상방 $\alpha$ 분위수**라고 한다.

또한 <예 4.1.5>의 (f)에서 알 수 있듯이
$$
X\sim\text{Gamma}(\nu,\beta)
\Leftrightarrow
X/\beta\sim\text{Gamma}(\nu,1)
\Leftrightarrow
2X/\beta\sim\text{Gamma}(\nu,2)=\chi^2(2\nu)
$$
이므로 감마분포의 상방 분위수를 카이제곱분포의 상방 분위수로 나타낼 수 있다. 즉
$$
P(X>\beta\chi^2_\alpha(2\nu)/2)
=P(2X/\beta>\chi^2_\alpha(2\nu))
=\alpha
$$
이므로 감마분포 $\text{Gamma}(\nu,\beta)$의 상방 $\alpha$ 분위수는 $\beta\chi^2_\alpha(2\nu)/2$로 주어진다.

#### 예 4.2.2
감마분포 $\text{Gamma}(2,3)$의 상방 $0.05$ 분위수는
$$
3\chi^2_{0.05}(4)/2
$$
로서, 부록 IV에 주어진 카이제곱분포의 누적확률 분포표로부터
$3\times9.488/2=14.232$

### 정리 4.2.1 카이제곱분포의 성질
(a) $Y\sim\chi^2(r)$이면
$$
E(Y)=r,\quad\mathrm{Var}(Y)=2r
$$

(b) $Y\sim\chi^2(r)$이면 그 적률생성함수는
$$
\mathrm{mgf}_Y(t)=(1-2t)^{-r/2},\quad t<1/2
$$

(c) $Y_1\sim\chi^2(r_1),\ Y_2\sim\chi^2(r_2)$이고 $Y_1,Y_2$가 서로 독립이면
$$
Y_1+Y_2\sim\chi^2(r_1+r_2)
$$

#### 증명
카이제곱분포 $\chi^2(r)$는 감마분포 $\text{Gamma}(r/2,2)$이므로 이 정리는 <정리 3.5.2>에 주어진 감마분포의 성질을 특별한 경우에 정리해 놓은 것이다.

구체적으로, 감마분포 $\text{Gamma}(\alpha,\beta)$의 적률생성함수는 $(1-\beta t)^{-\alpha}$이므로
$$
\mathrm{mgf}_Y(t) = (1-2t)^{-r/2}
$$
이고, 이로부터 평균과 분산을 구하면
$$
E(Y) = \mathrm{mgf}_Y'(0) = r, \quad \mathrm{Var}(Y) = \mathrm{mgf}_Y''(0) - [\mathrm{mgf}_Y'(0)]^2 = 2r
$$

또한 (c)는 독립인 감마분포의 합에 대한 성질을 이용한다.

#### 예 4.2.3
두 확률변수 $Z$와 $V$가 서로 독립이고 $Z\sim N(0,1)$, $V\sim\chi^2(r)$일 때
$$
X=\frac{Z}{\sqrt{V/r}}
$$
의 확률밀도함수를 구하여라.

**풀이**  
$Y=V$라고 하면, $(Z,V)$에서 $(X,Y)$로의 변환은 그 역변환이
$$
\begin{cases}
Z=X\sqrt{Y/r}\\
V=Y
\end{cases}
$$
로 주어지는 일대일 변환이다.

야코비안을 계산하면
$$
J = \det\begin{pmatrix}
\frac{\partial z}{\partial x} & \frac{\partial z}{\partial y} \\
\frac{\partial v}{\partial x} & \frac{\partial v}{\partial y}
\end{pmatrix}
= \det\begin{pmatrix}
\sqrt{y/r} & \frac{x}{2\sqrt{ry}} \\
0 & 1
\end{pmatrix}
= \sqrt{y/r}
$$

$Z$와 $V$가 독립이므로
$$
\text{pdf}_{Z,V}(z,v) = \frac{1}{\sqrt{2\pi}}e^{-z^2/2} \cdot \frac{1}{\Gamma(r/2)2^{r/2}}v^{r/2-1}e^{-v/2}
$$

정리 4.1.1에 의해
$$
\text{pdf}_{X,Y}(x,y) = \text{pdf}_{Z,V}(z,v)|J|
$$
$$
= \frac{1}{\sqrt{2\pi}}e^{-xy/(2r)} \cdot \frac{1}{\Gamma(r/2)2^{r/2}}y^{r/2-1}e^{-y/2} \cdot \sqrt{y/r}
$$

$X$의 주변확률밀도함수를 구하기 위해 $y$에 대해 적분하면
$$
\text{pdf}_X(x)
= \int_0^\infty \frac{1}{\sqrt{2\pi r}\Gamma(r/2)2^{r/2}} y^{r/2-1/2} e^{-y(1+x^2/r)/2} dy
$$

$u = y(1+x^2/r)/2$로 치환하면
$$
\text{pdf}_X(x)
= \frac{\Gamma((r+1)/2)}{\sqrt{\pi r}\Gamma(r/2)} \left(1+\frac{x^2}{r}\right)^{-(r+1)/2}
$$

여기서 $\Gamma(1/2) = \sqrt{\pi}$를 사용하였다.  
이 분포를 자유도가 $r$인 **t 분포(t distribution)** 라고 하며 기호로는
$$
X\sim t(r)
$$
로 나타낸다.

### t 분포의 정의 *(Student's t Distribution)*
$$
t(r)\overset{d}{\equiv}\frac{N(0,1)}{\sqrt{\chi^2(r)/r}},\quad N(0,1)\perp\chi^2(r)
$$

$$
X\sim t(r)\ (r>0)
\Leftrightarrow
X\overset{d}{\equiv}\frac{Z}{\sqrt{V/r}},\ Z\sim N(0,1),\ V\sim\chi^2(r),\ Z\perp V
$$

$$
\Leftrightarrow
\text{pdf}_X(x)
=\frac{\Gamma((r+1)/2)}{\sqrt{\pi r}\Gamma(r/2)}
\left(1+\frac{x^2}{r}\right)^{-(r+1)/2}
$$

t 분포는 대칭 분포이며, $r\to\infty$일 때 표준정규분포 $N(0,1)$에 수렴한다. t 분포의 상방 $\alpha$ 분위수를 $t_\alpha(r)$로 나타내며
$$
P(X > t_\alpha(r)) = \alpha
$$
를 만족한다. 대칭성에 의해 $t_{1-\alpha}(r) = -t_\alpha(r)$이다.

### 정리 4.2.2 정규모집단 경우의 표본분포에 관한 기본 정리
정규분포 $N(\mu,\sigma^2)$에서의 랜덤표본을 $X_1,\dots,X_n$이라 할 때, 다음이 성립한다.

(a) 표본평균
$\bar X=\frac{X_1+\cdots+X_n}{n}$의 분포는$\bar X\sim N(\mu,\sigma^2/n)$

(b) 표본분산 $S^2=\frac{1}{n-1}\sum_{i=1}^n(X_i-\bar X)^2$과 표본평균 $\bar X$는 서로 독립이다.

(c) $\frac{(n-1)S^2}{\sigma^2}\sim\chi^2(n-1)$는 자유도 n-1인 카이제곱분포를 따른다

#### 증명
(a) 표준화 변환  
$$
Z_i = \frac{X_i - \mu}{\sigma} \sim N(0,1), \quad i=1,\ldots,n \\
\bar Z = \frac{1}{n}\sum_{i=1}^n Z_i = \frac{\bar X - \mu}{\sigma}
$$
이고, $Z_i$들이 서로 독립이고 동일하게 $N(0,1)$를 따르므로 적률생성함수를 이용하면
$$
\text{mgf}_{\bar Z}(t) = \left[\text{mgf}_{Z_1}(t/n)\right]^n = \left[e^{t^2/(2n^2)}\right]^n = e^{t^2/(2n)}
$$
따라서 $\bar Z \sim N(0, 1/n)$이고, 이는 $\bar X \sim N(\mu, \sigma^2/n)$을 의미한다.  

(b) $\bar{X}$와 $S^2$의 독립성  
표본분산 $S^2$는 $(X_1-\bar{X}, \ldots, X_n-\bar{X})^T$의 함수이다. 독립인 확률변수들의 함수들도 독립이므로, $\bar{X}$와 $(X_1-\bar{X}, \ldots, X_n-\bar{X})^T$가 독립임을 밝히면 $\bar{X}$와 $S^2$도 독립임을 알 수 있다.

$Y = (X_1-\bar{X}, \ldots, X_n-\bar{X})^T$이고, $\bar{t} = \frac{1}{n}\sum_{j=1}^n t_j$라고 하자.  
$\bar{X}$와 $Y$의 결합적률생성함수를 구하면
$$
\text{mgf}_{\bar{X},Y}(s,t) = E\left[e^{s\bar{X} + t^T Y}\right] = E\left[\exp\left(s\bar{X} + \sum_{i=1}^n t_i(X_i-\bar{X})\right)\right] \\
= E\left[\exp\left(\left(s - \sum_{i=1}^n t_i\right)\bar{X} + \sum_{i=1}^n t_i X_i\right)\right] \\
= E\left[\exp\left(\sum_{i=1}^n \left(\frac{s}{n} - \frac{1}{n}\sum_{j=1}^n t_j + t_i\right) X_i\right)\right] \\
= E\left[\exp\left(\sum_{i=1}^n \left(\frac{s}{n} + t_i - \bar{t}\right) X_i\right)\right]
$$

$X_i$들이 독립이므로
$$
= \prod_{i=1}^n E\left[\exp\left(\left(\frac{s}{n} + t_i - \bar{t}\right) X_i\right)\right]
$$

$X_i \sim N(\mu, \sigma^2)$이므로
$$
= \prod_{i=1}^n \exp\left(\mu\left(\frac{s}{n} + t_i - \bar{t}\right) + \frac{\sigma^2}{2}\left(\frac{s}{n} + t_i - \bar{t}\right)^2\right)
$$

이제 $\sum_{i=1}^n (t_i - \bar{t}) = 0$임을 이용하여 정리하면

$$
= \exp\left(\mu s + \frac{\sigma^2 s^2}{2n}\right) \cdot \exp\left(\frac{\sigma^2}{2}\sum_{i=1}^n (t_i - \bar{t})^2\right)
$$

즉 $s$의 함수와 $t$의 함수로 분해됨을 보일 수 있다. 따라서 $\bar{X}$와 $Y$는 독립이고, 결과적으로 $\bar{X}$와 $S^2$도 독립이다.

(c) $(n-1)S^2/\sigma^2$의 분포  
(a)로부터 $\bar{X} \sim N(\mu, \sigma^2/n)$이므로
$$
\frac{\bar{X}-\mu}{\sigma/\sqrt{n}} \sim N(0,1)
$$

따라서
$$
\frac{n(\bar{X}-\mu)^2}{\sigma^2} \sim \chi^2(1)
$$

한편
$$
\sum_{i=1}^n \frac{(X_i-\mu)^2}{\sigma^2} = \sum_{i=1}^n \left(\frac{X_i-\mu}{\sigma}\right)^2 \sim \chi^2(n)
$$

그리고
$$
\sum_{i=1}^n (X_i-\mu)^2 = \sum_{i=1}^n (X_i-\bar X + \bar X-\mu)^2 =\sum_{i=1}^n (X_i-\bar{X})^2 + n(\bar{X}-\mu)^2
$$

이므로
$$
\frac{(n-1)S^2}{\sigma^2} = \sum_{i=1}^n \frac{(X_i-\bar{X})^2}{\sigma^2} = \sum_{i=1}^n \frac{(X_i-\mu)^2}{\sigma^2} - \frac{n(\bar{X}-\mu)^2}{\sigma^2}
$$

(b)로부터 $\bar{X}$와 $S^2$가 독립이므로, 좌항 $\chi^2(n)$과 우항 $\chi^2(1)$의 차이이며, 정리 4.2.1 (c)의 성질을 이용하면
$$
\frac{(n-1)S^2}{\sigma^2} \sim \chi^2(n-1)
$$

### 정리 4.2.3 정규모집단에서 모평균의 추론
정규분포 $N(\mu,\sigma^2)$에서의 랜덤표본 $X_1,\dots,X_n$에 대하여
$$
\frac{\bar X-\mu}{S/\sqrt n}\sim t(n-1) \\
P\left(
\bar X-t_{\alpha/2}(n-1)\frac{S}{\sqrt n}
\le\mu\le
\bar X+t_{\alpha/2}(n-1)\frac{S}{\sqrt n}
\right)=1-\alpha
$$

#### 증명
정리 4.2.2로부터
$$
\frac{\bar X-\mu}{\sigma/\sqrt{n}} \sim N(0,1), \quad \frac{(n-1)S^2}{\sigma^2} \sim \chi^2(n-1)
$$
이고, $\bar X$와 $S^2$는 독립이다. 따라서 t 분포의 정의에 의해
$$
\frac{\bar X-\mu}{S/\sqrt{n}} = \frac{(\bar X-\mu)/(\sigma/\sqrt{n})}{\sqrt{(n-1)S^2/\sigma^2/(n-1)}} \sim t(n-1)
$$

분위수 ($\alpha$ quantile)의 정의에 따라
$$
P\left(-t_{\alpha/2}(n-1) \le \frac{\bar X-\mu}{S/\sqrt{n}} \le t_{\alpha/2}(n-1)\right) = 1-\alpha
$$
이므로 $\mu$에 대해 정리하면 결과를 얻는다.

이 $\mu$에 관한 구간을 **신뢰수준(confidence level)** $(1-\alpha)$의 **신뢰구간(confidence interval)** 이라 하며, 신뢰수준이란 표본으로부터 계산되는 구간이 미지의 모수 $\mu$를 포함하게 되는 경우가 전체의 $100(1-\alpha)\%$일 것이라는 적중률을 의미한다.

모집단의 분포가 정규분포인 경우에 평균에 관한 신뢰구간과 마찬가지로 모집단의 분산에 관한 신뢰구간도 다음과 같이 주어진다.
### 정리 4.2.4 정규모집단에서 모분산의 추론
정규분포 $N(\mu,\sigma^2)$에서의 랜덤표본 $X_1,\dots,X_n$에 대하여
$$
P\left(
\frac{(n-1)S^2}{\chi^2_{\alpha/2}(n-1)}
\le\sigma^2\le
\frac{(n-1)S^2}{\chi^2_{1-\alpha/2}(n-1)}
\right)=1-\alpha
$$
이다.

#### 증명
정리 4.2.2 (c)로부터 $(n-1)S^2/\sigma^2 \sim \chi^2(n-1)$이므로
$$
P\left(\chi^2_{1-\alpha/2}(n-1) \le \frac{(n-1)S^2}{\sigma^2} \le \chi^2_{\alpha/2}(n-1)\right) = 1-\alpha
$$

$\sigma^2$에 대해 정리하면
$$
P\left(\frac{(n-1)S^2}{\chi^2_{\alpha/2}(n-1)} \le \sigma^2 \le \frac{(n-1)S^2}{\chi^2_{1-\alpha/2}(n-1)}\right) = 1-\alpha
$$

#### 예 4.2.4
서로 독립이고 $V_1\sim\chi^2(r_1)$, $V_2\sim\chi^2(r_2)$일 때
$$
X=\frac{V_1/r_1}{V_2/r_2}
$$
의 확률밀도함수를 구하여라.

**풀이**  
$Y=V_2$라고 하면 $(V_1,V_2)$에서 $(X,Y)$로의 변환은 역변환이
$$
\begin{cases}
V_1 = X \cdot \frac{r_1}{r_2} \cdot Y \\
V_2 = Y
\end{cases}
$$
로 주어지는 일대일 변환이다.

야코비안을 계산하면
$$
J = \det\begin{pmatrix}
\frac{r_1 y}{r_2} & \frac{r_1 x}{r_2} \\
0 & 1
\end{pmatrix}
= \frac{r_1 y}{r_2}
$$

$V_1$과 $V_2$가 독립이므로
$$
\text{pdf}_{V_1,V_2}(v_1,v_2) = \frac{1}{\Gamma(r_1/2)2^{r_1/2}}v_1^{r_1/2-1}e^{-v_1/2} \cdot \frac{1}{\Gamma(r_2/2)2^{r_2/2}}v_2^{r_2/2-1}e^{-v_2/2}
$$

정리 4.1.1에 의해
$$
\text{pdf}_{X,Y}(x,y) = \text{pdf}_{V_1,V_2}(v_1,v_2)|J|
$$

$X$의 주변확률밀도함수를 구하기 위해 $y$에 대해 적분하고 정리하면
$$
\text{pdf}_X(x) = \frac{\Gamma((r_1+r_2)/2)}{\Gamma(r_1/2)\Gamma(r_2/2)} \left(\frac{r_1}{r_2}\right)^{r_1/2} \frac{x^{r_1/2-1}}{(1+r_1x/r_2)^{(r_1+r_2)/2}}I_{(0,\infty)}(x)
$$

이 분포를 자유도가 $(r_1, r_2)$인 F distribution이라 하며, 정의는 아래와 같다.
### F 분포의 정의 *(F Distribution)*
$$
F(r_1,r_2)\overset{d}{\equiv}\frac{\chi^2(r_1)/r_1}{\chi^2(r_2)/r_2},\quad
\chi^2(r_1)\perp\chi^2(r_2)
$$

$$
X\sim F(r_1,r_2) \Leftrightarrow X \overset{d}{\equiv} \frac{V_1/r_1}{V_2/r_2}, \quad V_1\sim\chi^2(r_1), V_2\sim\chi^2(r_2), V_1\perp V_2
$$

$$
\Leftrightarrow
\text{pdf}_X(x)
=\frac{\Gamma((r_1+r_2)/2)}{\Gamma(r_1/2)\Gamma(r_2/2)}
\left(\frac{r_1}{r_2}\right)^{r_1/2}
\frac{x^{r_1/2-1}}{(1+r_1x/r_2)^{(r_1+r_2)/2}}
I_{(0,\infty)}(x)
$$

여기서 $r_1$을 분자의 자유도, $r_2$를 분모의 자유도라고 한다. F 분포의 상방 $\alpha$ 분위수를 $F_\alpha(r_1,r_2)$로 나타내며
$$
P(X > F_\alpha(r_1,r_2)) = \alpha
$$
를 만족한다. 

### 4.2.5 F 분포의 성질
(a) $X \sim F(r_1,r_2) \Rightarrow \frac{1}{X} \sim F(r_2,r_1)$. 따라서  
$$
F_{1-\alpha}(r_1,r_2) = \frac{1}{F_\alpha(r_2,r_1)}
$$

(b) $X \sim t(r)$이면 $X^2 \sim F(1,r)$. 따라서
$$
t^2_{\alpha/2}(r) = F_\alpha(1,r)
$$

#### 증명
**(a)**  
$X \sim F(r_1,r_2)$이면 정의에 의해
$$
X = \frac{V_1/r_1}{V_2/r_2}, \quad V_1 \sim \chi^2(r_1), \quad V_2 \sim \chi^2(r_2), \quad V_1 \perp V_2
$$

따라서
$$
\frac{1}{X} = \frac{V_2/r_2}{V_1/r_1} \sim F(r_2,r_1)
$$
분위수의 관계는 다음과 같이 유도된다:
$$
\alpha = P(X > F_\alpha(r_1,r_2)) = P\left(\frac{1}{X} < \frac{1}{F_\alpha(r_1,r_2)}\right)
$$

한편 $\frac{1}{X} \sim F(r_2,r_1)$이므로
$$
P\left(\frac{1}{X} < F_{1-\alpha}(r_2,r_1)\right) = 1-\alpha
$$

즉
$$
P\left(\frac{1}{X} > F_{1-\alpha}(r_2,r_1)\right) = \alpha
$$

따라서
$$
\frac{1}{F_\alpha(r_1,r_2)} = F_{1-\alpha}(r_2,r_1)
$$

**(b)**  
t 분포의 정의로부터 $X \sim t(r)$이면
$$
X = \frac{Z}{\sqrt{V/r}}, \quad Z \sim N(0,1), \quad V \sim \chi^2(r), \quad Z \perp V
$$

따라서
$$
X^2 = \frac{Z^2}{V/r}
$$

<예 4.1.4>의 (c)로부터 $Z^2 \sim \chi^2(1)$이고, $Z^2 \perp V$이므로 F 분포의 정의에 의해
$$
X^2 = \frac{Z^2/1}{V/r} \sim F(1,r)
$$

분위수의 관계는 다음과 같이 유도된다:
$$
\alpha = P(X^2 > t^2_{\alpha/2}(r)) = P(|X| > t_{\alpha/2}(r))
$$

한편 $X^2 \sim F(1,r)$이므로
$$
\alpha = P(X^2 > F_\alpha(1,r))
$$

따라서 $t^2_{\alpha/2}(r) = F_\alpha(1,r)$

### 정리 4.2.6 두 정규모집단 모분산 비교
두 정규모집단
$N(\mu_1,\sigma_1^2)$, $N(\mu_2,\sigma_2^2)$에서 서로 독립이고 각각 크기 $n_1$, $n_2$인 랜덤표본
$X_{11},\dots,X_{1n_1}$, $X_{21},\dots,X_{2n_2}$를 추출하자.

표본평균과 표본분산을
$$
\bar X_i=\frac{1}{n_i}\sum_{j=1}^{n_i}X_{ij},\qquad
S_i^2=\frac{1}{n_i-1}\sum_{j=1}^{n_i}(X_{ij}-\bar X_i)^2\quad(i=1,2)
$$
라 하면 다음이 성립한다.
$$
\frac{S_1^2/\sigma_1^2}{S_2^2/\sigma_2^2}\sim F(n_1-1,n_2-1)
$$

또한
$$
P\left(
\frac{S_1^2/S_2^2}{F_{\alpha/2}(n_1-1,n_2-1)}
\le
\frac{\sigma_1^2}{\sigma_2^2}
\le
\frac{S_1^2/S_2^2}{F_{\alpha/2}(n_2-1,n_1-1)}
\right)=1-\alpha
$$

#### 증명
정리 4.2.2로부터
$$
\frac{(n_1-1)S_1^2}{\sigma_1^2}\sim\chi^2(n_1-1),\qquad
\frac{(n_2-1)S_2^2}{\sigma_2^2}\sim\chi^2(n_2-1)
$$
이며, 두 랜덤표본이 서로 독립이므로 $S_1^2$와 $S_2^2$도 서로 독립이다.

따라서 F 분포의 정의에 의해
$$
\frac{S_1^2/\sigma_1^2}{S_2^2/\sigma_2^2}\sim F(n_1-1,n_2-1)
$$
확률식은 위 분포식에 양변을 적절히 변형하여 얻는다.


### 여러 정규모집단의 모평균 비교 *(Comparison of Means from Multiple Normal Populations)*

여러 모집단을 비교할 때 각 모집단의 분산이 동일하다는 전제 하에서 모평균 비교를 하는 경우가 많다. 이런 비교를 목적으로 하여 흔히 사용되는 모형이 **일원분류모형(one-way classification model)** 이다. 이는 분산이 동일한 여러 정규모집단에서 서로 독립인 랜덤표본 $k$개를 관측한다는 뜻이다.

구체적으로, $k$개의 정규모집단 $N(\mu_1,\sigma^2),\dots,N(\mu_k,\sigma^2)$에서 각각 크기 $n_1,\dots,n_k$인 랜덤표본을 추출하는 상황을 다음과 같은 모형으로 표현한다:
$$
X_{ij}=\mu_i+e_{ij},\qquad
e_{ij}\overset{iid}{\sim}N(0,\sigma^2),\quad
i=1,\dots,k,\ j=1,\dots,n_i
$$

여기서 $X_{ij}$는 $i$번째 모집단에서의 $j$번째 관측값이고, $\mu_i$는 $i$번째 모집단의 평균이며, $e_{ij}$는 오차항이다. 모든 모집단의 분산이 $\sigma^2$로 동일하다는 것이 이 모형의 핵심 가정이다.

이러한 일원분류모형에서는 다음과 같은 통계량들이 중요한 역할을 한다:

**집단 내 평균(group mean)**
$$
\bar X_i=\frac{1}{n_i}\sum_{j=1}^{n_i}X_{ij},\quad i=1,\dots,k
$$

**전체 평균(grand mean)**
$$
\bar X=\frac{1}{n}\sum_{i=1}^k\sum_{j=1}^{n_i}X_{ij}=\frac{1}{n}\sum_{i=1}^k n_i\bar X_i,\quad
n=\sum_{i=1}^k n_i
$$

**전체 모평균(overall population mean), 표본평균**
$$
\bar\mu=\frac{1}{n}\sum_{i=1}^k n_i\mu_i
$$

**집단 간 변동(between-group variation)**
$$
SS_B=\sum_{i=1}^k n_i(\bar X_i-\bar X)^2
$$

**집단 내 변동(within-group variation)**
$$
SS_W=\sum_{i=1}^k\sum_{j=1}^{n_i}(X_{ij}-\bar X_i)^2
$$

이들 변동은 다음과 같은 **분산분석 항등식(ANOVA identity)** 을 만족한다:
$$
\sum_{i=1}^k\sum_{j=1}^{n_i}(X_{ij}-\bar X)^2
=\sum_{i=1}^k n_i(\bar X_i-\bar X)^2
+\sum_{i=1}^k\sum_{j=1}^{n_i}(X_{ij}-\bar X_i)^2
$$
즉, 총변동 = 집단 간 변동 + 집단 내 변동

### 정리 4.2.7 여러 개의 정규모집단에서 모평균의 비교
다음의 **일원분류모형(one-way classification model)** 에서, 다음이 성립한다.

(a)
$$
\sum_{i=1}^k n_i(\bar X_i-\bar X-(\mu_i-\bar\mu))^2/\sigma^2
\sim \chi^2(k-1)
$$

(b)
$$
\sum_{i=1}^k\sum_{j=1}^{n_i}(X_{ij}-\bar X_i)^2/\sigma^2
\sim \chi^2(n-k)
$$

(c) (a)와 (b)는 서로 독립이다.

따라서
$$
\frac{
\frac{1}{k-1}\sum_{i=1}^k n_i(\bar X_i-\bar X-(\mu_i-\bar\mu))^2
}{
\frac{1}{n-k}\sum_{i=1}^k\sum_{j=1}^{n_i}(X_{ij}-\bar X_i)^2
}
\sim F(k-1,n-k)
$$

$$
P\left(
\sum_{i=1}^k n_i(\mu_i-\bar\mu-(\bar X_i-\bar X))^2
\le
(k-1)\hat{\sigma^2}F_{\alpha}(k-1,n-k)
\right)=1-\alpha
$$

#### 증명
정규분포 $N(\mu_i,\sigma^2)$에서의 랜덤표본 $X_{i1},\dots,X_{in_i}$에 대해
정리 4.2.2를 적용하면
$$
\bar X_i\sim N(\mu_i,\sigma^2/n_i),\qquad
\frac{(n_i-1)S_i^2}{\sigma^2}\sim\chi^2(n_i-1)
$$
이며, $\bar X_i$와 $S_i^2$는 서로 독립이다.

또한 서로 다른 집단의 표본들은 독립이므로
$\bar X_1,\dots,\bar X_k$와 $S_1^2,\dots,S_k^2$는 서로 독립이다.
독립인 변수들의 함수들도 독립이므로, (c)가 성립한다.  

정리 4.2.1카이제곱분포의 가법성으로
$$
\sum_{i=1}^k\sum_{j=1}^{n_i}(X_{ij}-\bar X_i)^2/\sigma^2
\sim\chi^2\left(\sum_{i=1}^k(n_i-1)\right)
=\chi^2(n-k)
$$
이므로 (b)가 성립한다.

(a)의 일반적인 증명은 4장 4절에서 다룬다.  
여기선, $n_1=\cdots=n_k=\bar n$인 경우만 고려하여 보자.  
$n_1=\cdots=n_k=\bar{n}$인 경우, 
$$
Z_i = \frac{\bar X_i - \mu_i}{\sigma/\sqrt{\bar{n}}} \sim N(0,1), \quad i=1,\ldots,k
$$
는 서로 독립이고 각각 $N(0,1)$을 따르는 확률변수다.

전체 표본평균과 전체 모평균은
$$
\bar X = \frac{1}{k}\sum_{i=1}^k \bar X_i, \qquad \bar\mu = \frac{1}{k}\sum_{i=1}^k \mu_i
$$

따라서
$$
\frac{\bar X - \bar\mu}{\sigma/\sqrt{k\bar{n}}} = \frac{1}{\sqrt{k}}\sum_{i=1}^k Z_i \sim N(0,1)
$$

정리 4.2.2를 랜덤표본 $Z_1,\ldots,Z_k$에 적용하면
$$
\sum_{i=1}^k (Z_i - \bar Z)^2 \sim \chi^2(k-1)
$$

여기서
$$
\sum_{i=1}^k (Z_i - \bar Z)^2 = \sum_{i=1}^k \left(\frac{\bar X_i - \mu_i}{\sigma/\sqrt{\bar{n}}} - \frac{\bar X - \bar\mu}{\sigma/\sqrt{k\bar{n}}\cdot\sqrt{k}}\right)^2 = \sum_{i=1}^k \frac{\bar{n}}{\sigma^2}(\bar X_i - \bar X - (\mu_i - \bar\mu))^2
$$
이므로 성립한다.  

한편, F 분포의 정의와 (a), (b), (c)로부터
$$
\frac{\sum_{i=1}^k n_i(\bar X_i-\bar X-(\mu_i-\bar\mu))^2/\sigma^2/(k-1)}{\sum_{i=1}^k\sum_{j=1}^{n_i}(X_{ij}-\bar X_i)^2/\sigma^2/(n-k)}
\sim F(k-1,n-k)
$$

분모를 $\hat\sigma^2 = \frac{1}{n-k}\sum_{i=1}^k\sum_{j=1}^{n_i}(X_{ij}-\bar X_i)^2$로 나타내면
$$
\frac{\sum_{i=1}^k n_i(\bar X_i-\bar X-(\mu_i-\bar\mu))^2}{(k-1)\hat\sigma^2}
\sim F(k-1,n-k)
$$

따라서 F 분포의 상방 $\alpha$ 분위수의 정의에 의해
$$
P\left(\frac{\sum_{i=1}^k n_i(\bar X_i-\bar X-(\mu_i-\bar\mu))^2}{(k-1)\hat\sigma^2} \le F_\alpha(k-1,n-k)\right) = 1-\alpha
$$

즉
$$
P\left(\sum_{i=1}^k n_i(\bar X_i-\bar X-(\mu_i-\bar\mu))^2 \le (k-1)\hat\sigma^2 F_\alpha(k-1,n-k)\right) = 1-\alpha
$$

양변에 음수를 곱하고 $\mu_i - \bar\mu$와 $\bar X_i - \bar X$의 순서를 바꾸면
$$
P\left(\sum_{i=1}^k n_i(\mu_i-\bar\mu-(\bar X_i-\bar X))^2 \le (k-1)\hat\sigma^2 F_\alpha(k-1,n-k)\right) = 1-\alpha
$$

**참고**  
여기서 주어진 $\mu_i-\bar\mu$에 관한 집합은 신뢰수준 $100(1-\alpha)$%의 신뢰집합(confidence set)이라고 하며 이는 신뢰구간을 일반화한 것이다. 이의 이해를 위하여 두 모평균 비교를 하는 경우인 $k=2$인 경우를 살펴보자.
$$
\mu_1-\bar\mu=\frac{n_2}{n_1+n_2}(\mu_1-\mu_2),\qquad
\mu_2-\bar\mu=\frac{n_1}{n_1+n_2}(\mu_2-\mu_1)
$$
이며,
$$
\bar X_1-\bar X=\frac{n_2}{n_1+n_2}(\bar X_1-\bar X_2),\qquad
\bar X_2-\bar X=\frac{n_1}{n_1+n_2}(\bar X_2-\bar X_1)
$$
이다.

따라서
$$
\frac{1}{k-1}\sum_{i=1}^k n_i(\bar X_i-\bar X-(\mu_i-\bar\mu))^2
=
\frac{n_1n_2}{n_1+n_2}(\bar X_2-\bar X_1-(\mu_2-\mu_1))^2
$$

또한 정리 4.2.5로부터
$$
F_\alpha(1,n_1+n_2-2)=t^2_{\alpha/2}(n_1+n_2-2)
$$
이므로, $k=2$인 경우의 신뢰집합은 모평균 차이 $\mu_2-\mu_1$에 대한
t 분포 기반 신뢰구간으로 환원된다.  
이런 신뢰구간 또는 정리 4.2.7의 신뢰집합이 원점을 포함하는가에 따라 여러개의 모평균 동일성 여부에 대한 추론을 할 수 있다.  

## 4.3 순서통계량의 분포 *(Distribution of Order Statistics)*
모집단 분포가 연속형인 경우, 모집단의 중심 부분에 대한 추론에는 표본평균과 더불어 **표본중앙값(sample median)** 이 많이 사용된다. 이러한 표본중앙값은 랜덤표본 $X_1,X_2,\dots,X_n$을 크기 순서로 늘어놓은 **순서통계량(order statistics)**
$$
X_{(1)}<X_{(2)}<\cdots<X_{(n)}
$$
중 하나이다. 이 절에서는 이러한 순서통계량의 분포에 대하여 알아본다.  

**순서통계량은 값의 순서로 정의되며 인덱스가 랜덤이므로,
예를들어, 두 개의 순서통계량(X1, X2)의 분포를 고려하더라도 나머지 n−2개의 상대적 위치를 함께 고려해야 한다.**

#### 예 4.3.1
표준지수분포 $\text{Exp}(1)$에서의 랜덤표본 $X_1,X_2,X_3$을 크기 순서로 늘어놓은 순서통계량을 $X_{(1)}<X_{(2)}<X_{(3)}$이라 하자. 이때
$$
Y=(X_{(1)},X_{(2)},X_{(3)})^T
$$
의 결합확률밀도함수를 구하여라.

**풀이**  
순서통계량을 나타내는 함수를
$$
u(X_1,X_2,X_3)=(X_{(1)},X_{(2)},X_{(3)})^T
$$
라고 하자.

함수 $u$는
$$
\mathcal{X}=\{(x_1,x_2,x_3)^T: x_i>0\ (i=1,2,3),\ x_1\neq x_2,\ x_2\neq x_3,\ x_3\neq x_1\}
$$
에서
$$
\mathcal{Y}=\{(y_1,y_2,y_3)^T: 0<y_1<y_2<y_3\}
$$
로의 $3!$ 대일 함수이며, 다대일 변환에 관한 정리(정리 4.1.2)의 조건을 만족한다.

집합 $\{1,2,3\}$에서의 치환(permutation)을 $\pi$로 나타내고
$$
\mathcal{X}_\pi=\{(x_1,x_2,x_3)^T: 0<x_{\pi_1}<x_{\pi_2}<x_{\pi_3}\}
$$
라고 하자.

함수 $u$를 $\mathcal{X}_\pi$로 제한한 함수
$$
u^\pi(x_1,x_2,x_3)=(x_{\pi_1},x_{\pi_2},x_{\pi_3})^T,\quad x\in\mathcal{X}_\pi
$$
는 $\mathcal{X}_\pi$에서 $\mathcal{Y}$로의 일대일 함수이며 미분가능하다.  
이때 야코비안 행렬은 단위행렬의 행과 열을 치환한 행렬이므로
$$
J_{u^\pi}(x)=\det\left(\frac{\partial x_{\pi_j}}{\partial x_i}\right)=\pm1,\quad \forall x\in\mathcal{X}_\pi
$$
이다.

예를 들어 $(\pi_1,\pi_2,\pi_3)=(3,2,1)$인 경우
$$
J_{u^\pi}(x)=\det\begin{pmatrix}
0 & 0 & 1 \\
0 & 1 & 0 \\
1 & 0 & 0
\end{pmatrix}=-1
$$

따라서 다대일 변환 정리(정리 4.1.2)에 의해
$$
\text{pdf}_Y(y)=\sum_\pi\text{pdf}_X\left((u^\pi)^{-1}(y)\right)\left|J_{(u^\pi)^{-1}}(y)\right|,\quad y\in\mathcal{Y}
$$

한편 $X=(X_1,X_2,X_3)^T$의 결합확률밀도함수는
$$
\text{pdf}_X(x_1,x_2,x_3)=e^{-(x_1+x_2+x_3)}I_{(x_1>0,x_2>0,x_3>0)}
$$
이다.

치환의 개수는 $3!=6$개이므로
$$
\text{pdf}_Y(y_1,y_2,y_3)
=\sum_\pi e^{-(y_{\pi_1}+y_{\pi_2}+y_{\pi_3})}I_{(0<y_1<y_2<y_3)}
=6e^{-(y_1+y_2+y_3)}I_{(0<y_1<y_2<y_3)}
$$

### 정리 4.3.1 (순서통계량의 결합확률밀도함수)
모집단 분포가 연속형이고 그 확률밀도함수가 $f(x)$일 때, 랜덤표본 $X_1,X_2,\dots,X_n$을 크기 순서로 늘어놓은 순서통계량을
$X_{(1)} < X_{(2)}< \dots< X_{(n)}$라고 하면, $Y=(X_{(1)},X_{(2)},\dots,X_{(n)})^T$의 결합확률밀도함수는 다음과 같이 주어진다.
$$
\text{pdf}_Y(y_1,\dots,y_n)
=n!f(y_1)\cdots f(y_n)I_{(y_1<\cdots<y_n)}
$$

#### 증명
순서통계량을 나타내는 함수를 $u(X_1,\dots,X_n)=(X_{(1)},\dots,X_{(n)})^T$라고 하자.  
함수 $u$는 $\mathcal{X}=\{(x_1,\dots,x_n)^T:f(x_i)>0,\ x_1,\dots,x_n\ \text{서로 다른 실수}\}$
에서
$$
\mathcal{Y}=\{(y_1,\dots,y_n)^T:f(y_i)>0,\ y_1<\cdots<y_n\}
$$
로의 $n!$대일 함수로서 정리4.1.2를 만족한다.  

집합 $\{1,\dots,n\}$에서의 치환(permutation)을 $\pi$로 나타내고
$$
\mathcal{X}_\pi=\{(x_1,\dots,x_n)^T:f(x_i)>0,\ x_{\pi_1}<\cdots<x_{\pi_n}\}
$$
라 하면, 함수 $u$에서 정의역을 $\mathcal{X}_\pi$로 제한한 함수
$$
u^\pi(x_1,\dots,x_n)=(x_{\pi_1},\dots,x_{\pi_n})^T
$$
는 $\mathcal{X}_\pi$에서 $\mathcal{Y}$로의 일대일 함수이며 미분가능하다.

이때 야코비안 행렬은 단위행렬의 행과 열을 치환한 행렬이므로
$$
J_{u^\pi}(x)=\det\left(\frac{\partial x_{\pi_j}}{\partial x_i}\right)=\pm1
$$

따라서 다대일 변환에 관한 정리(정리 4.1.2)에 의해
$$
\text{pdf}_Y(y)
=\sum_\pi\text{pdf}_X\left((u^\pi)^{-1}(y)\right)\left|J_{(u^\pi)^{-1}}(y)\right|
$$

한편
$$
\text{pdf}_X(x_1,\dots,x_n)=f(x_1)\cdots f(x_n)
$$
이고 치환의 개수는 $n!$개이므로
$$
\text{pdf}_Y(y_1,\dots,y_n)
=n!f(y_1)\cdots f(y_n)I_{(y_1<\cdots<y_n)}
$$

**정리 4.3.2 아이디어**  
순서통계량 $X_{(r)}$이 $x$ 근방에 있을 사건 $(x < X_{(r)} \le x + |\Delta x|)$는 $X_1, \ldots, X_n$들 중에서 
- $(r-1)$개는 $x$ 이하
- $1$개는 $x$와 $x+|\Delta x|$ 사이
- 나머지 $(n-r)$개는 $x+|\Delta x|$를 초과

하는 사건이다. 

$X_1,\ldots,X_n$은 iid이므로 이런 사건의 확률은
$$
c_r[F(x)]^{r-1}[F(x+|\Delta x|)-F(x)][1-F(x+|\Delta x|)]^{n-r}
$$
여기서 $c_r$은 $n$개 중에서 $(r-1)$개, $1$개, $(n-r)$개를 선택하는 다항계수(multinomial coefficient)로서
$$
c_r = \frac{n!}{(r-1)!\cdot 1!\cdot (n-r)!}
$$

$|\Delta x|\to 0$일 때
$$
F(x+|\Delta x|)-F(x)\approx f(x)|\Delta x|
$$
이므로
$$
P(x < X_{(r)} \le x+|\Delta x|)\approx\frac{n!}{(r-1)!(n-r)!}[F(x)]^{r-1}f(x)[1-F(x)]^{n-r}|\Delta x|
$$

따라서
$$
\text{pdf}_{X_{(r)}}(x)=\frac{n!}{(r-1)!(n-r)!}[F(x)]^{r-1}f(x)[1-F(x)]^{n-r}
$$
### 정리 4.3.2 (순서통계량의 주변확률밀도함수)
모집단의 누적분포함수를 $F(x)$, 확률밀도함수를 $f(x)$라 하자.

**(a) 단일 순서통계량의 확률밀도함수**
$$
\text{pdf}_{X_{(r)}}(x)
=\frac{n!}{(r-1)!(n-r)!}
[F(x)]^{r-1}f(x)[1-F(x)]^{n-r},
\quad 1\le r\le n
$$

**(b) 두 순서통계량의 결합확률밀도함수**
$$
\text{pdf}_{X_{(r)},X_{(s)}}(x,y)
=\frac{n!}{(r-1)!(s-r-1)!(n-s)!}
[F(x)]^{r-1}f(x)
[F(y)-F(x)]^{s-r-1}f(y)
[1-F(y)]^{n-s}
$$
$$
\quad 1\le r<s\le n,\ x<y
$$

#### 증명
정리 4.3.1에서 얻은 결합확률밀도함수
$$
\text{pdf}_Y(y_1,\dots,y_n)=n!f(y_1)\cdots f(y_n)I_{(y_1<\cdots<y_n)}
$$
를 적분하여 주변확률밀도함수를 계산하면 된다.

**(a)** $X_{(r)}$의 주변확률밀도함수를 구하기 위해 $y_r=x$로 고정하고 나머지 변수들에 대해 적분한다:
$$
\text{pdf}_{X_{(r)}}(x)
=\int_{-\infty<y_1<\cdots<y_{r-1}<x<y_{r+1}<\cdots<y_n<\infty}
n!f(y_1)\cdots f(y_{r-1})f(x)f(y_{r+1})\cdots f(y_n)\,dy_1\cdots dy_{r-1}dy_{r+1}\cdots dy_n
$$
이 적분을 계산하면

$$
\text{pdf}_{X_{(r)}}(x)
=n!f(x)\int_{-\infty}^{x}\cdots\int_{-\infty}^{y_2}f(y_1)dy_1\cdots dy_{r-1}
\int_{x}^{\infty}\cdots\int_{y_{n-1}}^{\infty}f(y_n)dy_n\cdots dy_{r+1}
$$

**첫 번째 적분 ($y_1, \ldots, y_{r-1}$ 부분):**
$-\infty<y_1<\cdots<y_{r-1}<x$ 영역에서 적분한다. $y_{r-1}$부터 역순으로 적분하면:

$$
\int_{-\infty}^{x}f(y_{r-1})\int_{-\infty}^{y_{r-1}}f(y_{r-2})\cdots\int_{-\infty}^{y_2}f(y_1)dy_1\cdots dy_{r-1}
$$

가장 안쪽 적분부터 계산:
$$
\int_{-\infty}^{y_2}f(y_1)dy_1=F(y_2)
$$

다음 적분:
$$
\int_{-\infty}^{y_3}f(y_2)F(y_2)dy_2=\int_{-\infty}^{y_3}F(y_2)dF(y_2)=\frac{[F(y_3)]^2}{2}
$$

이런 식으로 계속하면:
$$
\int_{-\infty}^{x}f(y_{r-1})\cdot\frac{[F(y_{r-1})]^{r-2}}{(r-2)!}dy_{r-1}
=\int_{-\infty}^{x}\frac{[F(y_{r-1})]^{r-2}}{(r-2)!}dF(y_{r-1})
=\frac{[F(x)]^{r-1}}{(r-1)!}
$$

마찬가지로 $x<y_{r+1}<\cdots<y_n<\infty$ 영역에서 적분하면:
$$
\int_{x}^{\infty}f(y_{r+1})\cdots\int_{y_{n-1}}^{\infty}f(y_n)dy_n\cdots dy_{r+1}
=\frac{[1-F(x)]^{n-r}}{(n-r)!}
$$

따라서
$$
\text{pdf}_{X_{(r)}}(x)
=n!f(x)\cdot\frac{[F(x)]^{r-1}}{(r-1)!}\cdot\frac{[1-F(x)]^{n-r}}{(n-r)!}
$$

**(b)** 마찬가지로 $y_r=x$, $y_s=y$로 고정하고 나머지 변수들에 대해 적분하면
$$
\text{pdf}_{X_{(r)},X_{(s)}}(x,y)
=\frac{n!}{(r-1)!(s-r-1)!(n-s)!}
[F(x)]^{r-1}f(x)
[F(y)-F(x)]^{s-r-1}f(y)
[1-F(y)]^{n-s}
$$

#### 예 4.3.2
(a)  
균등분포 $U(0,1)$에서의 순서통계량 $X_{(r)}$는
$$
X_{(r)}\sim\text{Beta}(r,n-r+1)
$$

(b)  
$Z_i = U_{(i)} - U_{(i-1)}, \quad i=1,\ldots,n$
로 정의하자. 여기서 $U_{(0)}=0$로 정의한다.

그러면 $(Z_1,\ldots,Z_{n}) \sim \text{Dirichlet}(1,\ldots,1)$이다. 즉, 
$$
\text{pdf}_{Z}(z_1,\ldots,z_{n}) = \frac{\Gamma(n+1)}{\Gamma(1)\cdots\Gamma(1)} I_{(z_i>0, \sum_{i=1}^{n}z_i<1)}
$$


**풀이**  
**(a)**  
$U(0,1)$의 경우 $F(x)=x$, $f(x)=1$ ($0<x<1$)이므로
$$
\text{pdf}_{X_{(r)}}(x)
=\frac{n!}{(r-1)!(n-r)!}
x^{r-1}(1-x)^{n-r}I_{(0,1)}(x)
$$
이는 $\text{Beta}(r,n-r+1)$의 확률밀도함수이다.

**(b)**  
균등분포 $U(0,1)$에서의 순서통계량 $U_{(1)}<\cdots<U_{(n)}$에 대하여
$$
Z_i = U_{(i)} - U_{(i-1)}, \quad i=1,\ldots,n+1
$$
로 정의하자. 여기서 $U_{(0)}=0$, $U_{(n+1)}=1$로 정의한다.

변환
$$
\begin{cases}
z_1 = u_1\\
z_2 = u_2 - u_1\\
\vdots\\
z_n = u_n - u_{n-1}\\
z_{n+1} = 1 - u_n
\end{cases}
$$
의 역변환은
$$
\begin{cases}
u_1 = z_1\\
u_2 = z_1 + z_2\\
\vdots\\
u_n = z_1 + z_2 + \cdots + z_n
\end{cases}
$$

야코비안을 계산하면
$$
J = \det\begin{pmatrix}
1 & 0 & \cdots & 0\\
1 & 1 & \cdots & 0\\
\vdots & \vdots & \ddots & \vdots\\
1 & 1 & \cdots & 1
\end{pmatrix} = 1
$$

<예 4.3.2> (a)로부터 $(U_{(1)},\ldots,U_{(n)})$의 결합확률밀도함수는
$$
\text{pdf}_{U_{(1)},\ldots,U_{(n)}}(u_1,\ldots,u_n) = n! I_{(0<u_1<\cdots<u_n<1)}
$$

정리 4.1.1에 의해
$$
\text{pdf}_{Z_1,\ldots,Z_{n+1}}(z_1,\ldots,z_{n+1}) = n! \cdot |J| \cdot I_{(z_i>0, \sum_{i=1}^{n+1}z_i=1)} \\
= \frac{\Gamma(n+1)}{\Gamma(1)\cdots\Gamma(1)} I_{(z_i>0, \sum_{i=1}^{n+1}z_i=1)}
$$

#### 예 4.3.3 지수분포에서의 순서통계량
$X_1,\dots,X_n\overset{iid}{\sim}\text{Exp}(1)$이고, $X_{(1)}< ... < X_{(n)}$에 대해
$$
Z_1=nX_{(1)},\quad
Z_2=(n-1)(X_{(2)}-X_{(1)}),\quad
\dots,\quad
Z_n=X_{(n)}-X_{(n-1)}
$$
라 하자. 이들은 서로 독립이며 각각 $\text{Exp}(1)$을 따름을 보여라.

**풀이**  
<예 4.3.1>로부터 $(X_{(1)},X_{(2)},X_{(3)})$의 결합확률밀도함수는
$$
\text{pdf}(y_1,y_2,y_3)=6e^{-(y_1+y_2+y_3)}I_{(0<y_1<y_2<y_3)}
$$

변환
$$
\begin{cases}
z_1=3y_1\\
z_2=2(y_2-y_1)\\
z_3=y_3-y_2
\end{cases}
$$
의 역변환은
$$
\begin{cases}
y_1=z_1/3\\
y_2=z_1/3+z_2/2\\
y_3=z_1/3+z_2/2+z_3
\end{cases}
$$

야코비안은
$$
J=\det\begin{pmatrix}
1/3 & 0 & 0\\
1/3 & 1/2 & 0\\
1/3 & 1/2 & 1
\end{pmatrix}=\frac{1}{6}
$$

따라서
$$
\text{pdf}_{Z_1,Z_2,Z_3}(z_1,z_2,z_3)
=6e^{-(z_1+z_2+z_3)}\cdot\frac{1}{6}I_{(z_1>0,z_2>0,z_3>0)}
=e^{-z_1}e^{-z_2}e^{-z_3}I_{(z_1>0,z_2>0,z_3>0)}
$$

즉 $Z_1,Z_2,Z_3$는 서로 독립이고 각각 $\text{Exp}(1)$을 따른다.


>참고: 일반화된 야코비안 계산 *(Generalized Jacobian Calculation)*
>순서통계량 $(X_{(1)}, \ldots, X_{(n)})$에서 간격(spacing)으로의 변환
>$$
>\begin{cases}
>Z_1 = n X_{(1)} \\
>Z_2 = (n-1)(X_{(2)} - X_{(1)}) \\
>\vdots \\
>Z_n = (X_{(n)} - X_{(n-1)})
>\end{cases}
>$$
>의 역변환은
>$$
>\begin{cases}
>X_{(1)} = \frac{Z_1}{n} \\
>X_{(2)} = \frac{Z_1}{n} + \frac{Z_2}{n-1} \\
>\vdots \\
>X_{(n)} = \frac{Z_1}{n} + \frac{Z_2}{n-1} + \cdots + \frac{Z_n}{1}
>\end{cases}
>$$
>
>야코비안 행렬은
>$$
>\frac{\partial X_{(i)}}{\partial Z_j} = 
>\begin{cases}
>\frac{1}{n-j+1} & \text{if } j \le i \\
>0 & \text{if } j > i
>\end{cases}
>$$
>
>따라서 야코비안 행렬은 하삼각행렬(lower triangular matrix)이다:
>$$
>J = \begin{pmatrix}
>\frac{1}{n} & 0 & 0 & \cdots & 0 \\
>\frac{1}{n} & \frac{1}{n-1} & 0 & \cdots & 0 \\
>\frac{1}{n} & \frac{1}{n-1} & \frac{1}{n-2} & \cdots & 0 \\
>\vdots & \vdots & \vdots & \ddots & \vdots \\
>\frac{1}{n} & \frac{1}{n-1} & \frac{1}{n-2} & \cdots & \frac{1}{1}
>\end{pmatrix}
>$$
>
>하삼각행렬의 행렬식은 대각원소들의 곱이므로
>$$
>|J| = \det(J) = \frac{1}{n} \cdot \frac{1}{n-1} \cdot \frac{1}{n-2} \cdots \frac{1}{1} = \frac{1}{n!}
>$$

### 지수분포에서의 순서통계량분포의 대의적 정의
지수분포 $\text{Exp}(1)$에서의 랜덤표본 $X_1,\dots,X_n$에 기초한 순서통계량을 $X_{(1)}<\cdots<X_{(n)}$이라 하면
$$
(X_{(r)})_{1\le r\le n}
\overset{d}{\equiv}
\left(\frac{Z_1}{n}+\cdots+\frac{Z_r}{n-r+1}\right)_{1\le r\le n},
\quad Z_r\overset{iid}{\sim}\text{Exp}(1)\ (r=1,\dots,n)
$$

### 정리 4.3.3 (확률적분변환, probability integral transformation)
확률변수 $X$가 연속형이고 그 누적분포함수 $F$가 **순증가(strictly increasing)** 함수라고 하자. 즉,
$$x_1 < x_2 \to F(x_1) < F(x_2)$$
이때 그 역함수를 $F^{-1}$라 하면 다음이 성립한다.  

**(a)** $F(X)\sim U(0,1)$
  - $F(X)$는 균등분포 $U(0,1)$를 따른다

**(b)** $U\sim U(0,1)\Rightarrow F^{-1}(U)\overset{d}{\equiv}X$
  - $F^{-1}(U)$의 누적함수는 $F$로서 $F^{-1}(U)$와 $X$는 같은 분포를 갖는다

#### 증명
$F$가 순증가함수이므로 역함수 $F^{-1}$가 존재하며 다음이 성립한다.
$$
F(x)\le u\ \Leftrightarrow\ x\le F^{-1}(u) \\
F^{-1}(u)\le x\ \Leftrightarrow\ u\le F(x) \\
F(F^{-1}(u))=u
$$

$U \sim U(0,1)$이므로 $P(U \le u) = u$ ($0 \le u \le 1$)

>1, 2번은 양변에 F나 F^-1 적용하면 유도됨. 3번은 역함수 정의

**(a)** $F(X) \sim U(0,1)$임을 보이기 위해, $Y = F(X)$로 정의하고 $Y$의 누적분포함수를 구한다.

$0 \le u \le 1$에 대하여
$$
P(F(X)\le u)=P(X\le F^{-1}(u))=F(F^{-1}(u))=u
$$

이는 $F(X)$가 균등분포 $U(0,1)$의 누적분포함수를 가짐을 의미한다. 따라서 $F(X) \sim U(0,1)$

**(b)** $U \sim U(0,1)$일 때, $F^{-1}(U)$의 누적분포함수를 구한다.

임의의 $x$에 대하여
$$
P(F^{-1}(U)\le x)=P(U\le F(x))=F(x)
$$

따라서 $F^{-1}(U)$의 누적분포함수가 $F$와 같으므로, $F^{-1}(U) \overset{d}{\equiv} X$가 성립한다.

**참고**: 이 정리는 난수 생성에 중요하게 활용된다. 균등분포를 따르는 난수 $U$를 생성한 후, 원하는 분포 $F$의 역함수를 적용하면 $F$를 따르는 난수를 얻을 수 있다.

#### 예 4.3.4
균등분포 $U(0,1)$에서의 순서통계량 $U_{(1)}<\cdots<U_{(n)}$과 지수분포 $\text{Exp}(1)$에서의 순서통계량 $X_{(1)}<\cdots<X_{(n)}$ 사이에는 다음 관계가 성립한다:
$$
(U_{(1)},\dots,U_{(n)})\overset{d}{\equiv}(1-e^{-X_{(n)}},\dots,1-e^{-X_{(1)}})
$$

**풀이**  
지수분포 $\text{Exp}(1)$의 누적분포함수는 $F(x)=1-e^{-x}$ ($x>0$)이므로, 정리 4.3.3에 의해
$$
F(X_i)=1-e^{-X_i}\sim U(0,1)
$$

$F$는 순증가함수이므로 $X_i$들의 대소관계가 $F(X_i)$들의 대소관계와 같다. 따라서
$$
F(X_{(1)})<F(X_{(2)})<\cdots<F(X_{(n)})
$$
즉
$$
(F(X_{(1)}),\dots,F(X_{(n)}))\overset{d}{\equiv}(U_{(1)},\dots,U_{(n)})
$$

### 정리 4.3.4 순서통계량 분포의 대의적 정의
모집단 분포가 연속형이고 누적분포함수가 $F(x)$일 때, $Z_1,\dots,Z_n\overset{iid}{\sim}\text{Exp}(1)$이라 하면
$$
(X_{(r)})_{1\le r\le n}
\overset{d}{\equiv}
\left(h\left(\frac{Z_1}{n}+\cdots+\frac{Z_r}{n-r+1}\right)\right)_{1\le r\le n} \\
h(y)=F^{-1}(1-e^{-y})
$$

#### 증명
<예 4.3.3>으로부터
$$
Z_1=nX_{(1)},\quad Z_1+Z_2=n X_{(1)}+(n-1)(X_{(2)}-X_{(1)})=(n-1)X_{(2)}+X_{(1)}
$$
일반적으로
$$
Z_1+\cdots+Z_r=(n-r+1)X_{(r)}+\cdots+2X_{(n-1)}+X_{(n)}
$$
이므로
$$
X_{(r)}=\frac{Z_1}{n}+\frac{Z_2}{n-1}+\cdots+\frac{Z_r}{n-r+1}
$$

<예 4.3.4>로부터
$$
U_{(r)}=1-e^{-X_{(r)}}
$$
이고, 정리 4.3.3에 의해
$$
X_{(r)}=F^{-1}(U_{(r)})
$$

따라서
$$
X_{(r)}=F^{-1}(1-e^{-X_{(r)}})=h\left(\frac{Z_1}{n}+\cdots+\frac{Z_r}{n-r+1}\right)
$$


## 4.4 다변량 정규분포 *(Multivariate Normal Distribution)*
통계 조사에서는 단일 특성보다 **서로 연관된 여러 특성**을 동시에 관측하는 경우가 많다.
이러한 다차원 자료에 대한 추론의 기본 모형으로 **다변량 정규분포(multivariate normal distribution)** 가 널리 사용된다.

### 정리 4.4.1 다변량 정규분포의 구성 *(Construction of Multivariate Normal Distribution)*
표준정규분포 $N(0,1)$를 따르고 서로 독립인 확률변수 $Z_1,Z_2,\dots,Z_n$
과 $n\times n$ 상수 행렬, $n$차원 벡터
$$
A=(a_{ij})_{1\le i,j\le n},\quad \mu=(\mu_1,\mu_2,\dots,\mu_n)^T
$$
에 대하여
$$
X=AZ+\mu,\quad X=(X_1,\dots,X_n)^T,\quad Z=(Z_1,\dots,Z_n)^T
$$
일때, 다음 두 가지가 성립한다.  
**(a) 확률밀도함수**  
행렬 $A$가 **정칙행렬(nonsingular matrix, invertible matrix)** 이면 $X=AZ+\mu$의 확률밀도함수는 ($\Sigma=AA^T$)
$$
\text{pdf}_X(x)
=
(\det(2\pi\Sigma))^{-1/2}
\exp\left\{
-\frac12(x-\mu)^T\Sigma^{-1}(x-\mu)
\right\},\quad x\in\mathbb{R}^n \\
$$

**(b) 적률생성함수 *(Moment Generating Function)***
$$
\text{mgf}_X(t)
=
\exp\left(
\mu^T t+\frac12 t^T\Sigma t
\right),\quad t\in\mathbb{R}^n
$$

#### 증명
$Z=(Z_1,\dots,Z_n)^T$의 확률밀도함수와 적률생성함수는
$$
\text{pdf}_Z(z)=(2\pi)^{-n/2}\exp\left(-\frac12 z^T z\right),\quad
\text{mgf}_Z(s)=\exp\left(\frac12 s^T s\right)
$$

**(a)** $X=u(Z)=AZ+\mu$라 하면 $A$가 정칙행렬이므로 $u$는 일대일 함수이며 역변환은
$$
u^{-1}(x)=A^{-1}(x-\mu)
$$
이고, 야코비안은
$$
|J_{u^{-1}}|=|\det(A^{-1})|=|\det(A)|^{-1}
$$

정리 4.1.1의 치환공식에 의해
$$
\text{pdf}_X(x)=\text{pdf}_Z(A^{-1}(x-\mu))|\det(A)|^{-1} \\
=(2\pi)^{-n/2}\exp\left(-\frac12(A^{-1}(x-\mu))^T(A^{-1}(x-\mu))\right)|\det(A)|^{-1}
$$

여기서 $(A^{-1}(x-\mu))^T(A^{-1}(x-\mu))=(x-\mu)^T(A^{-1})^TA^{-1}(x-\mu)$이고,
$\Sigma=AA^T$이므로 $\Sigma^{-1}=(AA^T)^{-1}=(A^T)^{-1}A^{-1}$이다.

또한 $|\det(A)|^{-1}=(\det(\Sigma))^{-1/2}$이므로
$$
\text{pdf}_X(x)
=
(\det(2\pi\Sigma))^{-1/2}
\exp\left\{
-\frac12(x-\mu)^T\Sigma^{-1}(x-\mu)
\right\}
$$

**(b)** 적률생성함수의 정의에 의해
$$
\text{mgf}_X(t)=E[e^{t^TX}]=E[e^{t^T(AZ+\mu)}]=E[e^{t^TAZ+t^T\mu}]=e^{t^T\mu}E[e^{(A^Tt)^TZ}]
$$

$Z$의 적률생성함수를 이용하면
$$
\text{mgf}_X(t)=e^{t^T\mu}\text{mgf}_Z(A^Tt)=e^{t^T\mu}\exp\left(\frac12(A^Tt)^T(A^Tt)\right)
$$

$(A^Tt)^T(A^Tt)=t^TAA^Tt=t^T\Sigma t$이므로
$$
\text{mgf}_X(t)=\exp\left(\mu^Tt+\frac12 t^T\Sigma t\right)
$$

### 다변량 정규분포의 정의 *(Definition of Multivariate Normal Distribution)*
$n$차원 확률벡터 $X$가 다음 조건들 중 하나를 만족하면
$$
X\sim N_n(\mu,\Sigma) \quad \text{or} \quad X\sim N(\mu,\Sigma)
$$
라 한다. 여기서 $\mu$는 $n$차원 평균벡터(mean vector)이고, $\Sigma$는 $n\times n$ 공분산행렬(covariance matrix)이다.

1. $X=AZ+\mu$, $Z\sim N_n(0,I)$, $AA^T=\Sigma$
  - A = nxm행렬이어도 됨
2. $X=\Sigma^{1/2}Z+\mu$, $Z\sim N_n(0,I)$ (여기서 $\Sigma^{1/2}$는 $\Sigma$의 제곱근 행렬)
3. $\text{mgf}_X(t)=\exp(\mu^Tt+\frac12 t^T\Sigma t)$
4. $\Sigma$가 정칙행렬일 때, X의 분포를 정칙다변량정규분포(nonsingular multivariate normal distribution)라 함.
    $$
    \text{pdf}_X(x)
    =
    (\det(2\pi\Sigma))^{-1/2}
    \exp\left\{-\frac12(x-\mu)^T\Sigma^{-1}(x-\mu)\right\}
    $$

### 정리 4.4.2 평균벡터와 분산행렬 *(Mean Vector and Covariance Matrix)*
(a) $X\sim N_n(\mu,\Sigma)$이면
$$
E(X)=\mu,\quad \text{Var}(X)=\Sigma
$$

(b)
$$
X\sim N(\mu,\Sigma)
\Leftrightarrow
X\overset{d}{\equiv}\Sigma^{1/2}Z+\mu,\quad Z\sim N_n(0,I)
$$

여기서 $\Sigma^{1/2}$는 대칭행렬 $\Sigma$의 **제곱근 행렬(square root matrix)** 로서 $(\Sigma^{1/2})^2=\Sigma^{1/2}\Sigma^{1/2}=\Sigma$를 만족한다.
  - 분산행렬 $\Sigma$는 음아닌 정부호의 행렬이고, $\Sigma^{1/2}\Sigma^{1/2}=\Sigma$인 $\Sigma^{1/2}$가 존재함
  - 음 아닌 정부호 행렬의 조건: 실수가 원소인 mxm대칭행렬 $\Sigma$에 대해 $a^T\Sigma a \geq 0, \forall a \in R^m$이면 $\Sigma$가 nonnegative definite행렬이라 한다. 이런 조건과 다음의 각 조건은 동등하다
    - $\Sigma$의 모든 고유값이 0이상
    - $\Sigma^{1/2}\Sigma^{1/2}=\Sigma$인 실수가 원소인 대칭행렬 $\Sigma^{1/2}$가 존재함
    - 참고: Statistics_02_추가_분산행렬의 스펙트럼 분해와 기하학적 해석.md

#### 증명
**(a)** $X=AZ+\mu$, $Z\sim N_n(0,I)$의 표현을 이용한다.  
**평균벡터:**
$$
E(X)=E(AZ+\mu)=AE(Z)+\mu=A\cdot 0+\mu=\mu
$$

**공분산행렬:**
$$
\text{Var}(X)=E[(X-E(X))(X-E(X))^T]=E[(AZ)(AZ)^T]
$$
$$
=E[AZZ^TA^T]=AE[ZZ^T]A^T
$$

$Z\sim N_n(0,I)$이므로 $E[ZZ^T]=I$이다. 따라서
$$
\text{Var}(X)=AIA^T=AA^T=\Sigma
$$

**(b)** 다변량 정규분포의 특성화는 적률생성함수로도 가능하다.  
$(\Rightarrow)$ $X\sim N(\mu,\Sigma)$이면 정리 4.4.1 (b)에 의해
$$
\text{mgf}_X(t)=\exp\left(\mu^Tt+\frac12 t^T\Sigma t\right)
$$

$(\Leftarrow)$ 역으로 $\text{mgf}_X(t)=\exp(\mu^Tt+\frac12 t^T\Sigma t)$이면, $\Sigma$의 제곱근 행렬 $\Sigma^{1/2}$가 존재하므로 $X\overset{d}{\equiv}\Sigma^{1/2}Z+\mu$로 표현할 수 있다. 여기서 $Z\sim N_n(0,I)$이다.

이는 다음과 같이 확인할 수 있다. $Y=\Sigma^{1/2}Z+\mu$로 정의하면
$$
\text{mgf}_Y(t)=E[e^{t^TY}]=E[e^{t^T(\Sigma^{1/2}Z+\mu)}]=e^{t^T\mu}E[e^{t^T\Sigma^{1/2}Z}]
$$
$$
=e^{t^T\mu}E[e^{(\Sigma^{1/2}t)^TZ}]=e^{t^T\mu}\text{mgf}_Z(\Sigma^{1/2}t)
$$

$Z\sim N_n(0,I)$이므로
$$
\text{mgf}_Z(s)=\exp\left(\frac12 s^Ts\right)
$$

따라서
$$
\text{mgf}_Y(t)=e^{t^T\mu}\exp\left(\frac12(\Sigma^{1/2}t)^T(\Sigma^{1/2}t)\right)=\exp\left(\mu^Tt+\frac12 t^T\Sigma t\right)
$$

적률생성함수가 같으므로 $X\overset{d}{\equiv}Y=\Sigma^{1/2}Z+\mu$이고, 따라서 $X\sim N(\mu,\Sigma)$이다.

#### 예 4.4.1 이변량 정규분포 *(Bivariate Normal Distribution)*
$$
(X_1,X_2)^T\sim N_2\left(
\begin{pmatrix}\mu_1\\ \mu_2\end{pmatrix},
\begin{pmatrix}
\sigma_1^2 & \rho\sigma_1\sigma_2\\
\rho\sigma_1\sigma_2 & \sigma_2^2
\end{pmatrix}
\right)
$$

여기서 공분산행렬의 행렬식을 계산하면
$$
\det(\Sigma) = \det\begin{pmatrix}
\sigma_1^2 & \rho\sigma_1\sigma_2\\
\rho\sigma_1\sigma_2 & \sigma_2^2
\end{pmatrix}
= \sigma_1^2\sigma_2^2 - \rho^2\sigma_1^2\sigma_2^2 = \sigma_1^2\sigma_2^2(1-\rho^2)
$$

이므로 $-1<\rho<1$일 때 $\det(\Sigma) > 0$이어서 공분산행렬이 정칙이다.

또한 역행렬은
$$
\Sigma^{-1} = \frac{1}{\sigma_1^2\sigma_2^2(1-\rho^2)}
\begin{pmatrix}
\sigma_2^2 & -\rho\sigma_1\sigma_2\\
-\rho\sigma_1\sigma_2 & \sigma_1^2
\end{pmatrix}
$$

확률밀도함수는
$$
f(x_1,x_2)
=
\frac{1}{2\pi\sigma_1\sigma_2\sqrt{1-\rho^2}}
\exp\left(-\frac12 Q\right) \\
Q=\frac1{1-\rho^2}
\left[
\left(\frac{x_1-\mu_1}{\sigma_1}\right)^2
-2\rho\frac{x_1-\mu_1}{\sigma_1}\frac{x_2-\mu_2}{\sigma_2}
+\left(\frac{x_2-\mu_2}{\sigma_2}\right)^2
\right]
$$

여기서 $\rho$는 상관계수(correlation coefficient)로서
$$
\rho=\frac{\text{Cov}(X_1,X_2)}{\sigma_1\sigma_2}
$$

상관계수 절댓값이 1인 경우에는 분산행렬 역행렬이 존재하지 않고, 정리2.2.3으로부터 다음과같이 두 변수 사이에 선형관계가 성립하는 것을 알 수 있다.  
**(i) $\rho=1$인 경우:**
$$
P\left(\frac{X_2-\mu_2}{\sigma_2}=\frac{X_1-\mu_1}{\sigma_1}\right)=1
$$
즉, $X_2=\mu_2+\frac{\sigma_2}{\sigma_1}(X_1-\mu_1)$ (완전한 양의 선형관계)

**(ii) $\rho=-1$인 경우:**
$$
P\left(\frac{X_2-\mu_2}{\sigma_2}=-\frac{X_1-\mu_1}{\sigma_1}\right)=1
$$
즉, $X_2=\mu_2-\frac{\sigma_2}{\sigma_1}(X_1-\mu_1)$ (완전한 음의 선형관계)  

이 경우 $(X_1,X_2)$의 결합분포는 일차원 직선 위에 집중되어 있어 2차원 확률밀도함수가 존재하지 않는다.

### 정리 4.4.3 다변량 정규분포의 성질 *(Properties of Multivariate Normal Distribution)*
**(a) 선형변환(linear transformation)**  
$X\sim N_n(\mu,\Sigma)$이고 $A$가 $m\times n$ 상수행렬, $b$가 $m$차원 상수벡터이면
$$
AX+b\sim N_m(A\mu+b,A\Sigma A^T)
$$

**(b) 공분산과 독립성(covariance and independence)**  
$X=(X_1^T,X_2^T)^T\sim N_n(\mu,\Sigma)$일 때
$$
\text{Cov}(X_1,X_2)=0
\Rightarrow
X_1\perp X_2
$$
이 성립한다. (이는 일반적인 분포에서는 성립하지 않고 **다변량 정규분포에서만** 성립하는 중요한 성질이다.)

**(c) 선형변환의 독립성**  
$A$, $B$가 상수행렬이고 $\text{Cov}(AX,BX)=A\Sigma B^T=0$ 이면 $AX\perp BX$

#### 증명
**(a)** $X=\Sigma^{1/2}Z+\mu$, $Z\sim N_n(0,I)$로 표현하면
$$
AX+b=A(\Sigma^{1/2}Z+\mu)+b=A\Sigma^{1/2}Z+(A\mu+b)
$$

$A\Sigma^{1/2}Z$는 표준정규분포를 따르는 확률변수들의 선형결합이므로 다변량 정규분포를 따르며, 평균은 $A\mu+b$이고 공분산행렬은
$$
(A\Sigma^{1/2})(A\Sigma^{1/2})^T=A\Sigma^{1/2}(\Sigma^{1/2})^TA^T=A\Sigma A^T
$$

**(b)** 적률생성함수를 이용한다. $X=(X_1^T,X_2^T)^T$에 대해
$$
\text{mgf}_{X_1,X_2}(t_1,t_2)=\exp\left(\mu^T\begin{pmatrix}t_1\\t_2\end{pmatrix}+\frac12\begin{pmatrix}t_1^T & t_2^T\end{pmatrix}\Sigma\begin{pmatrix}t_1\\t_2\end{pmatrix}\right)
$$

$\text{Cov}(X_1,X_2)=0$이면 $\Sigma$가 블록대각행렬이 되어
$$
\text{mgf}_{X_1,X_2}(t_1,t_2)=\text{mgf}_{X_1}(t_1)\cdot\text{mgf}_{X_2}(t_2)
$$
따라서 $X_1$과 $X_2$는 독립이다.

**(c)** (a)에 의해 $(AX,BX)^T$도 다변량 정규분포를 따르고, 공분산이 0이므로 (b)에 의해 독립이다.

#### 예 4.4.2 이변량 정규분포에서의 독립성
이변량 정규분포
$$
(X_1,X_2)^T\sim N_2\left(\begin{pmatrix}\mu_1\\\mu_2\end{pmatrix},\begin{pmatrix}\sigma_1^2 & \rho\sigma_1\sigma_2\\\rho\sigma_1\sigma_2 & \sigma_2^2\end{pmatrix}\right)
$$
에서는
$$
\text{Cov}(X_1,X_2)=\rho\sigma_1\sigma_2=0 \iff \rho=0 \iff X_1\perp X_2
$$

### 정리 4.4.4 주변분포와 조건부분포 *(Marginal and Conditional Distributions)*
$$
\begin{pmatrix}X_1\\X_2\end{pmatrix}
\sim
N\left(
\begin{pmatrix}\mu_1\\\mu_2\end{pmatrix},
\begin{pmatrix}
\Sigma_{11}&\Sigma_{12}\\
\Sigma_{21}&\Sigma_{22}
\end{pmatrix}
\right)
$$
일 때, 다음이 성립한다.

**(a) 주변분포(marginal distribution)**
$$
X_1\sim N(\mu_1,\Sigma_{11})
$$

**(b) 조건부분포(conditional distribution)**
$$
X_2\mid (X_1=x_1)
\sim
N\left(
\mu_2+\Sigma_{21}\Sigma_{11}^{-1}(x_1-\mu_1),
\Sigma_{22}-\Sigma_{21}\Sigma_{11}^{-1}\Sigma_{12}
\right)
$$

#### 증명
**(a)** 정리 4.4.3 (a)를 이용한다. $A=(I_{k_1},0)$로 선택하면
$$
AX=X_1\sim N(A\mu,A\Sigma A^T)=N(\mu_1,\Sigma_{11})
$$

**(b)** 조건부분포를 두 가지 방법으로 유도할 수 있다.  

**방법 1: 조건부 확률밀도함수 직접 계산**  
결합확률밀도함수를 주변확률밀도함수로 나누어 조건부 확률밀도함수를 구한다.

$$
\text{pdf}_{X_2|X_1}(x_2|x_1) = \frac{\text{pdf}_{X_1,X_2}(x_1,x_2)}{\text{pdf}_{X_1}(x_1)}
$$

결합확률밀도함수는
$$
\text{pdf}_{X_1,X_2}(x_1,x_2) = (\det(2\pi\Sigma))^{-1/2}\exp\left\{-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\right\}
$$

주변확률밀도함수는 (a)로부터
$$
\text{pdf}_{X_1}(x_1) = (\det(2\pi\Sigma_{11}))^{-1/2}\exp\left\{-\frac{1}{2}(x_1-\mu_1)^T\Sigma_{11}^{-1}(x_1-\mu_1)\right\}
$$

분할된 역행렬 공식(partitioned inverse formula)에 의해
$$
\Sigma^{-1} = \begin{pmatrix}
\Sigma_{11}^{-1}+\Sigma_{11}^{-1}\Sigma_{12}M^{-1}\Sigma_{21}\Sigma_{11}^{-1} & -\Sigma_{11}^{-1}\Sigma_{12}M^{-1} \\
-M^{-1}\Sigma_{21}\Sigma_{11}^{-1} & M^{-1}
\end{pmatrix}
$$
여기서 $M=\Sigma_{22}-\Sigma_{21}\Sigma_{11}^{-1}\Sigma_{12}$이다.

이차형식을 전개하면
$$
(x-\mu)^T\Sigma^{-1}(x-\mu) = (x_1-\mu_1)^T\Sigma_{11}^{-1}(x_1-\mu_1)
$$
$$
+ (x_2-\mu_2-\Sigma_{21}\Sigma_{11}^{-1}(x_1-\mu_1))^TM^{-1}(x_2-\mu_2-\Sigma_{21}\Sigma_{11}^{-1}(x_1-\mu_1))
$$

따라서
$$
\text{pdf}_{X_2|X_1}(x_2|x_1) = (\det(2\pi M))^{-1/2}\exp\left\{-\frac{1}{2}(x_2-\mu_{2|1})^TM^{-1}(x_2-\mu_{2|1})\right\}
$$
여기서 $\mu_{2|1}=\mu_2+\Sigma_{21}\Sigma_{11}^{-1}(x_1-\mu_1)$

즉, $X_2|X_1=x_1 \sim N(\mu_2+\Sigma_{21}\Sigma_{11}^{-1}(x_1-\mu_1), \Sigma_{22}-\Sigma_{21}\Sigma_{11}^{-1}\Sigma_{12})$

**방법 2: 선형변환을 이용한 유도**  
새로운 확률변수를 정의한다:
$$
Y = X_2 - \mu_2 - \Sigma_{21}\Sigma_{11}^{-1}(X_1-\mu_1)
$$

정리 4.4.3 (a)에 의해 $Y$도 정규분포를 따르며, 평균은
$$
E(Y) = E(X_2) - \mu_2 - \Sigma_{21}\Sigma_{11}^{-1}E(X_1-\mu_1) = \mu_2 - \mu_2 - 0 = 0
$$

공분산행렬은
$$
\text{Var}(Y) = \text{Var}(X_2) - \Sigma_{21}\Sigma_{11}^{-1}\text{Cov}(X_1,X_2)^T - \text{Cov}(X_1,X_2)\Sigma_{11}^{-1}\Sigma_{21}^T
$$
$$
+ \Sigma_{21}\Sigma_{11}^{-1}\text{Var}(X_1)\Sigma_{11}^{-1}\Sigma_{21}^T
$$
$$
= \Sigma_{22} - \Sigma_{21}\Sigma_{11}^{-1}\Sigma_{12} - \Sigma_{21}\Sigma_{11}^{-1}\Sigma_{12} + \Sigma_{21}\Sigma_{11}^{-1}\Sigma_{11}\Sigma_{11}^{-1}\Sigma_{12}
$$
$$
= \Sigma_{22} - \Sigma_{21}\Sigma_{11}^{-1}\Sigma_{12}
$$

또한 $Y$와 $X_1$의 공분산을 계산하면
$$
\text{Cov}(Y,X_1) = \text{Cov}(X_2,X_1) - \Sigma_{21}\Sigma_{11}^{-1}\text{Var}(X_1) = \Sigma_{21} - \Sigma_{21} = 0
$$

따라서 정리 4.4.3 (b)에 의해 $Y \perp X_1$이고, 이는 $Y$의 분포가 $X_1$의 값에 무관함을 의미한다.

결과적으로
$$
X_2 | X_1=x_1 = Y + \mu_2 + \Sigma_{21}\Sigma_{11}^{-1}(x_1-\mu_1)
$$
$$
\sim N(0+\mu_2+\Sigma_{21}\Sigma_{11}^{-1}(x_1-\mu_1), \Sigma_{22}-\Sigma_{21}\Sigma_{11}^{-1}\Sigma_{12})
$$

#### 예 4.4.3 이변량 정규분포의 조건부분포
이변량 정규분포
$$
(X_1,X_2)^T\sim N_2\left(\begin{pmatrix}\mu_1\\\mu_2\end{pmatrix},\begin{pmatrix}\sigma_1^2 & \rho\sigma_1\sigma_2\\\rho\sigma_1\sigma_2 & \sigma_2^2\end{pmatrix}\right)
$$
에서
$$
X_2\mid X_1=x_1
\sim
N\left(
\mu_2+\rho\frac{\sigma_2}{\sigma_1}(x_1-\mu_1),
\sigma_2^2(1-\rho^2)
\right)
$$

**풀이:**  
정리 4.4.4 (b)를 적용한다. 여기서
$$
\Sigma_{11}=\sigma_1^2,\quad \Sigma_{12}=\Sigma_{21}=\rho\sigma_1\sigma_2,\quad \Sigma_{22}=\sigma_2^2
$$

따라서
$$
\Sigma_{21}\Sigma_{11}^{-1}=\rho\sigma_1\sigma_2\cdot\frac{1}{\sigma_1^2}=\rho\frac{\sigma_2}{\sigma_1}
$$

조건부 평균은
$$
\mu_2+\Sigma_{21}\Sigma_{11}^{-1}(x_1-\mu_1)=\mu_2+\rho\frac{\sigma_2}{\sigma_1}(x_1-\mu_1)
$$

조건부 분산은
$$
\Sigma_{22}-\Sigma_{21}\Sigma_{11}^{-1}\Sigma_{12}=\sigma_2^2-\rho\sigma_1\sigma_2\cdot\frac{1}{\sigma_1^2}\cdot\rho\sigma_1\sigma_2=\sigma_2^2(1-\rho^2)
$$

### 정리 4.4.5 이차형식의 분포 *(Distribution of Quadratic Forms)*
**(a) 이차형식의 카이제곱분포**  
$X\sim N_k(\mu,\Sigma)$이고 $\Sigma$가 정칙행렬이면
$$
(X-\mu)^T\Sigma^{-1}(X-\mu)\sim\chi^2(k)
$$

**(b) 멱등행렬과 이차형식**  
$Z\sim N_n(0,I)$이고 $A$가 대칭 멱등행렬(symmetric idempotent matrix), 즉 $A^2=A$, $A^T=A$이면
$$
Z^TAZ\sim\chi^2(r),\quad r=\text{trace}(A)
$$

**각주**  
일반적으로 $x^TAx = x^T\left(\frac{A+A^T}{2}\right)x$이므로 $Z^TAZ$에서 행렬 $A$는 언제나 대칭행렬인 것으로 가정되어 있으며, (b)에서의 조건 $A^2=A$는 $Z^TAZ\sim\chi^2(r)$이기 위한 필요충분조건으로 알려져 있다.

#### 증명
**(a)** $X=\Sigma^{1/2}Z+\mu$, $Z\sim N_k(0,I)$로 표현하면
$$
X-\mu=\Sigma^{1/2}Z
$$

따라서
$$
(X-\mu)^T\Sigma^{-1}(X-\mu)=(\Sigma^{1/2}Z)^T\Sigma^{-1}(\Sigma^{1/2}Z)=Z^T(\Sigma^{1/2})^T\Sigma^{-1}\Sigma^{1/2}Z
$$

$\Sigma^{1/2}$는 대칭행렬이므로 $(\Sigma^{1/2})^T=\Sigma^{1/2}$이고, $\Sigma^{1/2}\Sigma^{1/2}=\Sigma$이므로
$$
(\Sigma^{1/2})^T\Sigma^{-1}\Sigma^{1/2}=\Sigma^{1/2}\Sigma^{-1}\Sigma^{1/2}=\Sigma^{-1/2}\Sigma\Sigma^{-1/2}=I
$$

따라서
$$
(X-\mu)^T\Sigma^{-1}(X-\mu)=Z^TZ=\sum_{i=1}^k Z_i^2
$$

$Z_i\overset{iid}{\sim} N(0,1)$이므로 카이제곱분포의 정의에 의해
$$
Z^TZ=\sum_{i=1}^k Z_i^2\sim\chi^2(k)
$$

**(b)** $A$가 대칭 멱등행렬이므로 스펙트럴 분해(spectral decomposition)를 이용하면
$$
A=P\Lambda P^T
$$
여기서 $P$는 직교행렬($P^TP=PP^T=I$), $\Lambda=\text{diag}(\lambda_1,\ldots,\lambda_n)$는 고유값 대각행렬이다.

$A^2=A$이므로
$$
P\Lambda P^TP\Lambda P^T=P\Lambda^2P^T=P\Lambda P^T
$$

양변에 $P^T$를 왼쪽에서, $P$를 오른쪽에서 곱하면 $\Lambda^2=\Lambda$를 얻는다. 따라서 모든 고유값 $\lambda_i$는 $\lambda_i^2=\lambda_i$를 만족하므로 $\lambda_i\in\{0,1\}$이다.

또한 $\text{trace}(A)=\text{trace}(\Lambda)=\sum_{i=1}^n\lambda_i=r$이므로, 정확히 $r$개의 고유값이 1이고 나머지 $n-r$개는 0이다.

일반성을 잃지 않고 처음 $r$개의 고유값이 1이라고 가정하면
$$
\Lambda=\begin{pmatrix}I_r & 0\\0 & 0\end{pmatrix}
$$

$W=P^TZ$로 정의하면, $P$가 직교행렬이고 $Z\sim N_n(0,I)$이므로
$$
W\sim N_n(P^T\cdot 0, P^TIP)=N_n(0,I)
$$

따라서
$$
Z^TAZ=Z^TP\Lambda P^TZ=(P^TZ)^T\Lambda(P^TZ)=W^T\Lambda W
$$
$$
=\sum_{i=1}^r W_i^2\cdot 1+\sum_{i=r+1}^n W_i^2\cdot 0=\sum_{i=1}^r W_i^2
$$

$W_i\overset{iid}{\sim} N(0,1)$이므로 카이제곱분포의 정의에 의해
$$
Z^TAZ=\sum_{i=1}^r W_i^2\sim\chi^2(r)
$$

#### 예 4.4.4 일원분류모형에서의 표본분포 *(Sampling Distribution in One-Way Classification Model)*
정리 4.2.7의 (a)에 따르면, 일원분류모형(i개 정규분포에서 각각 j개만큼 샘플 뽑아서 ij갯수 샘플이 있음)에서
정규분포 $N(\mu_i,\sigma^2/n_i)$를 따르는 서로 독립인 표본평균 $\bar{X}_i\ (i=1,\dots,k)$에 대하여

$$
\sum_{i=1}^k n_i(\bar{X}_i-\bar{X}-(\mu_i-\bar{\mu}))^2/\sigma^2 \sim \chi^2(k-1)
$$

가 성립한다. 이를 **정리 4.4.5**를 이용하여 유도한다.

#### 풀이
벡터 $Y=(\bar{X}_1,\dots,\bar{X}_k)^T$는 다변량 정규분포를 따른다:
$$
Y\sim N_k(\mu,\Sigma)
$$
여기서 $\mu=(\mu_1,\dots,\mu_k)^T$이고, $\bar{X}_i$들이 서로 독립이므로
$$
\Sigma=\text{diag}(\sigma^2/n_1,\dots,\sigma^2/n_k)=\sigma^2D(n_i^{-1})
$$

변환 $W=AY$를 고려하자. 여기서 $A$는 $k\times k$ 행렬로
$$
A=I-n^{-1}\mathbf{n}\mathbf{1}^T
$$
이다. 단, $\mathbf{n}=(n_1,\dots,n_k)^T$, $\mathbf{1}=(1,\dots,1)^T$, $n=\sum_{i=1}^k n_i$이다.

그러면
$$
W_i=\bar{X}_i-\frac{1}{n}\sum_{j=1}^k n_j\bar{X}_j=\bar{X}_i-\bar{X}
$$

따라서
$$
Y-\mu-A(Y-\mu)=(I-A)(Y-\mu)=n^{-1}\mathbf{n}\mathbf{1}^T(Y-\mu)
$$
로부터
$$
\bar{X}-\bar{\mu}=n^{-1}\sum_{i=1}^k n_i(\bar{X}_i-\mu_i)
$$

이제 이차형식을 계산하면
$$
\sum_{i=1}^k n_i(\bar{X}_i-\bar{X}-(\mu_i-\bar{\mu}))^2
=\sum_{i=1}^k n_i[(\bar{X}_i-\mu_i)-(\bar{X}-\bar{\mu})]^2
$$
$$
=(Y-\mu)^T\text{diag}(n_1,\dots,n_k)(I-n^{-1}\mathbf{1}\mathbf{n}^T)(Y-\mu)
$$
$$
=(Y-\mu)^T B(Y-\mu)
$$

여기서 $B=D(n_i)(I-n^{-1}\mathbf{1}\mathbf{n}^T)$는 대칭 멱등행렬이다:
$$
B^T=B,\quad B^2=D(n_i)(I-n^{-1}\mathbf{1}\mathbf{n}^T)D(n_i)(I-n^{-1}\mathbf{1}\mathbf{n}^T)=B
$$

표준화된 확률변수 $Z=\Sigma^{-1/2}(Y-\mu)\sim N_k(0,I)$에 대해
$$
\frac{1}{\sigma^2}\sum_{i=1}^k n_i(\bar{X}_i-\bar{X}-(\mu_i-\bar{\mu}))^2=Z^T\Sigma^{-1/2}B\Sigma^{-1/2}Z
$$

$\Sigma^{-1/2}B\Sigma^{-1/2}=\sigma^{-1}D(n_i^{1/2})(I-n^{-1}\mathbf{1}\mathbf{n}^T)D(n_i^{1/2})\sigma^{-1}$는 대칭 멱등행렬이고
$$
\text{trace}(\Sigma^{-1/2}B\Sigma^{-1/2})=\text{trace}(D(n_i^{1/2})(I-n^{-1}\mathbf{1}\mathbf{n}^T)D(n_i^{1/2})\sigma^{-2})
$$
$$
=\sigma^{-2}\text{trace}(D(n_i)-n^{-1}D(n_i^{1/2})\mathbf{1}\mathbf{n}^TD(n_i^{1/2}))
$$
$$
=\sigma^{-2}(n-n^{-1}\mathbf{n}^T D(n_i)\mathbf{1})=\sigma^{-2}(n-n)=k-1
$$

따라서 정리 4.4.5 (b)에 의해
$$
\frac{1}{\sigma^2}\sum_{i=1}^k n_i(\bar{X}_i-\bar{X}-(\mu_i-\bar{\mu}))^2\sim\chi^2(k-1)
$$

#### 예 4.4.5 표본분산의 표본분포 *(Sampling Distribution of Sample Variance)*
정규분포 $N(\mu,\sigma^2)$에서의 랜덤표본 $X_1,\dots,X_n$에 대하여

$$
\bar{X}=\frac1n\sum_{i=1}^n X_i,\quad
S^2=\frac1{n-1}\sum_{i=1}^n(X_i-\bar{X})^2
$$
라 하면, $\bar{X}$와 $S^2$는 서로 독립이며 $\frac{(n-1)S^2}{\sigma^2}\sim\chi^2(n-1)$

#### 풀이
벡터 $X=(X_1,\dots,X_n)^T$에 대하여 $X\sim N_n(\mu\mathbf{1},\sigma^2I)$  
여기서 $\mathbf{1}=(1,\ldots,1)^T$이다.

또한
$$
\bar{X}=n^{-1}\mathbf{1}^TX \\
(n-1)S^2=\sum_{i=1}^n(X_i-\bar{X})^2=X^T(I-n^{-1}\mathbf{1}\mathbf{1}^T)X
$$

행렬 $A=I-n^{-1}\mathbf{1}\mathbf{1}^T$는 대칭행렬이며 멱등행렬이다:
$$
A^T=A,\quad A^2=(I-n^{-1}\mathbf{1}\mathbf{1}^T)^2=I-2n^{-1}\mathbf{1}\mathbf{1}^T+n^{-2}\mathbf{1}(\mathbf{1}^T\mathbf{1})\mathbf{1}^T=I-n^{-1}\mathbf{1}\mathbf{1}^T=A
$$

또한
$$
\mathbf{1}^T(I-n^{-1}\mathbf{1}\mathbf{1}^T)=\mathbf{1}^T-n^{-1}\mathbf{1}^T\mathbf{1}\mathbf{1}^T=\mathbf{1}^T-\mathbf{1}^T=0
$$

이므로
$$
\text{Cov}(\bar{X},(I-n^{-1}\mathbf{1}\mathbf{1}^T)X)=\text{Cov}(n^{-1}\mathbf{1}^TX,(I-n^{-1}\mathbf{1}\mathbf{1}^T)X)
$$
$$
=n^{-1}\mathbf{1}^T\text{Var}(X)(I-n^{-1}\mathbf{1}\mathbf{1}^T)^T=n^{-1}\sigma^2\mathbf{1}^T(I-n^{-1}\mathbf{1}\mathbf{1}^T)=0
$$

따라서 정리 4.4.3 (c)에 의해 $\bar{X}$와 $(n-1)S^2$는 서로 독립이다.

또한 $Z=(X-\mu\mathbf{1})/\sigma\sim N_n(0,I)$로 표준화하면
$$
\frac{(n-1)S^2}{\sigma^2}
=\frac{1}{\sigma^2}(X-\mu\mathbf{1}+\mu\mathbf{1}-\bar{X}\mathbf{1})^T(I-n^{-1}\mathbf{1}\mathbf{1}^T)(X-\mu\mathbf{1}+\mu\mathbf{1}-\bar{X}\mathbf{1})
$$

$(I-n^{-1}\mathbf{1}\mathbf{1}^T)\mathbf{1}=\mathbf{1}-n^{-1}\mathbf{1}(\mathbf{1}^T\mathbf{1})=\mathbf{1}-\mathbf{1}=0$이므로
$$
\frac{(n-1)S^2}{\sigma^2}=\frac{1}{\sigma^2}(X-\mu\mathbf{1})^T(I-n^{-1}\mathbf{1}\mathbf{1}^T)(X-\mu\mathbf{1})=Z^TAZ
$$

여기서 $A=I-n^{-1}\mathbf{1}\mathbf{1}^T$는 대칭 멱등행렬이고
$$
\text{trace}(A)=\text{trace}(I)-\text{trace}(n^{-1}\mathbf{1}\mathbf{1}^T)=n-n^{-1}\text{trace}(\mathbf{1}^T\mathbf{1})=n-n^{-1}\cdot n=n-1
$$

따라서 정리 4.4.5 (b)에 의해
$$
\frac{(n-1)S^2}{\sigma^2}\sim\chi^2(n-1)
$$

### 선형회귀모형 *(Linear Regression Model)*
통계조사에서 변수 사이의 함수관계를 파악하고자 할 때가 많은데, 이런 목적에서 가장 기본적으로 사용되는 모형이다.  
설명변수(explanatory variable) $x_0, x_1, ..., x_p$와 오차항 e에 의해 반응변수(response variable) Y가 정해진다는 전제 하에 그 관계가 선형이라 가정하고 반복 관측된 Y값으로 선형관계의 구체적 형태를 추측하고자 하는 것이다.  

정규 오차항을 갖는 경우의 관측값은 다음과 같이 표현된다:
$$
Y_i = \beta_0x_{i0} + \beta_1 x_{i1} + \cdots + \beta_p x_{ip} + e_i, \quad i=1,\ldots,n
$$

여기서 오차항은
$$
e_i \overset{iid}{\sim} N(0,\sigma^2)
$$

이를 행렬 형태로 정리하면
$$
Y = X\beta + e
$$

여기서
- $Y = (Y_1,\ldots,Y_n)^T$: $n$차원 반응벡터
- $X = \begin{pmatrix} 1 & x_{11} & \cdots & x_{1p} \\ \vdots & \vdots & \ddots & \vdots \\ 1 & x_{n1} & \cdots & x_{np} \end{pmatrix}$: $n \times (p+1)$ 설계행렬
- $\beta = (\beta_0, \beta_1, \ldots, \beta_p)^T$: $(p+1)$차원 회귀계수벡터
- $e = (e_1,\ldots,e_n)^T \sim N_n(0,\sigma^2 I)$: $n$차원 오차벡터

따라서 $Y \sim N_n(X\beta, \sigma^2 I)$이다.

선형회귀모형에서 $X$는 $n\times(p+1)$ 설계행렬(design matrix)이며, 설계행렬 $X$의 계수(rank)가 $p+1$인 것으로 가정하고, 선형관계의 구체적 형태를 나타내는 $\beta = (\beta_0, \beta_1, \ldots, \beta_p)^T$를 **회귀계수(regression coefficient)** 라 하며, 이의 추측값으로는 흔히
$$
\hat{\beta} = (X^TX)^{-1}X^TY
$$
로 정의되는 **표본회귀계수(sample regression coefficient)** 또는 **최소제곱추정량(least squares estimator)** 을 사용한다.

이 추정량은 잔차제곱합(residual sum of squares)
$$
\text{RSS}(\beta) = (Y-X\beta)^T(Y-X\beta) = \sum_{i=1}^n(Y_i - \beta_0 - \beta_1 x_{i1} - \cdots - \beta_p x_{ip})^2
$$
을 최소화하는 $\beta$ 값으로 얻어진다.

#### 예 4.4.6 단순선형회귀모형에서 표본회귀계수
선형회귀모형에서 $p=1$이고, $x_0 = 1$인 경우, 즉
$$
Y_i = \beta_0 + \beta_1 x_i + e_i, \quad i=1,\ldots,n \\
e_i \overset{iid}{\sim} N(0, \sigma^2)
$$
로 나타내어지는 모형을 **단순선형회귀모형(simple linear regression model)** 이라 한다. 이 경우는
$$
X = \begin{pmatrix} 1 & x_1 \\ 1 & x_2 \\ \vdots & \vdots \\ 1 & x_n \end{pmatrix}, \quad
X^TX = \begin{pmatrix} n & \sum_{i=1}^n x_i \\ \sum_{i=1}^n x_i & \sum_{i=1}^n x_i^2 \end{pmatrix}, \quad
X^TY = \begin{pmatrix} \sum_{i=1}^n Y_i \\ \sum_{i=1}^n x_i Y_i \end{pmatrix}
$$

$$
(X^TX)^{-1} = \frac{1}{n\sum_{i=1}^n x_i^2 - (\sum_{i=1}^n x_i)^2} \begin{pmatrix} \sum_{i=1}^n x_i^2 & -\sum_{i=1}^n x_i \\ -\sum_{i=1}^n x_i & n \end{pmatrix}
$$

이므로 표본회귀계수 $\hat{\beta} = (\hat{\beta}_0, \hat{\beta}_1)^T = (X^TX)^{-1}X^TY$는 다음과 같이 주어진다.

$S_{xx} = \sum_{i=1}^n (x_i - \bar{x})^2 = \sum_{i=1}^n x_i^2 - n\bar{x}^2$, $S_{xy} = \sum_{i=1}^n (x_i - \bar{x})(Y_i - \bar{Y})$로 놓으면
$$
\hat{\beta}_1 = \frac{S_{xy}}{S_{xx}} = \frac{\sum_{i=1}^n (x_i - \bar{x})Y_i}{\sum_{i=1}^n (x_i - \bar{x})^2}
$$

$$
\hat{\beta}_0 = \bar{Y} - \hat{\beta}_1 \bar{x}
$$

### 평균오차제곱합 *(Mean Squared Error)*
선형회귀모형에서 오차항의 분산 $\sigma^2$의 추측값으로는 흔히 
$$
\hat{\sigma}^2 = \frac{(Y-X\hat{\beta})^T(Y-X\hat{\beta})}{n-p-1} = \frac{\sum_{i=1}^n(Y_i - \hat{\beta}_0x_{i0} - \hat{\beta}_1 x_{i1} - \cdots - \hat{\beta}_p x_{ip})^2}{n-p-1}
$$
로 정의되는 **평균오차제곱합(mean squared error)** 을 사용한다.

예 4.4.6에서의 단순선형회귀모형의 경우 $p=1$이므로
$$
\hat{\sigma}^2 = \frac{\sum_{i=1}^n(Y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i)^2}{n-2}
$$

### 정리 4.4.6 선형회귀모형에서의 표본분포 *(Sampling Distribution in Linear Regression Model)*
선형회귀모형에서 주어진 상수의 행렬 X의 계수가 p+1이라 가정하면 다음이 성립한다:

**(a)** 회귀계수 추정량의 분포
$$
\hat{\beta}\sim N_{p+1}(\beta,\sigma^2(X^TX)^{-1})
$$

**(b)** 추정량의 독립성  
$\hat{\beta}$와 오차분산 추정량 $\hat{\sigma}^2$는 서로 독립

**(c)** 오차분산 추정량의 분포
$$
\frac{(n-p-1)\hat{\sigma}^2}{\sigma^2}\sim\chi^2(n-p-1)
$$

#### 증명
**(a)** $Y\sim N_n(X\beta,\sigma^2I)$이므로 정리 4.4.3 (a)에 의해
$$
\hat{\beta}=(X^TX)^{-1}X^TY\sim N_{p+1}((X^TX)^{-1}X^TX\beta,\sigma^2(X^TX)^{-1}X^TI X(X^TX)^{-1})
$$
$$
=N_{p+1}(\beta,\sigma^2(X^TX)^{-1})
$$
**(b)** 추정량의 독립성  
투영행렬(projection matrix) $H=X(X^TX)^{-1}X^T$를 정의하면, $H$는 대칭 멱등행렬이다:
$$
H^2=H,\quad H^T=H
$$

이를 이용하면 예측값과 잔차를 다음과 같이 표현할 수 있다:
$$
\hat{Y}=X\hat{\beta}=HY
$$

$$
Y-X\hat{\beta}=(I-H)Y
$$

$\hat{\beta}$와 $(I-H)Y$의 공분산을 계산하면
$$
\text{Cov}(\hat{\beta},(I-H)Y)=(X^TX)^{-1}X^T\text{Var}(Y)(I-H)^T
$$
$$
=(X^TX)^{-1}X^T\cdot\sigma^2I\cdot(I-H)=\sigma^2(X^TX)^{-1}X^T(I-H)
$$

여기서 중요한 것은
$$
X^T(I-H)=X^T-X^TH=X^T-X^TX(X^TX)^{-1}X^T=X^T-X^T=0
$$

따라서
$$
\text{Cov}(\hat{\beta},(I-H)Y)=0
$$

정리 4.4.3 (c)에 의해, 다변량 정규분포를 따르는 확률변수들의 공분산이 0이면 독립이므로 $\hat{\beta}$와 $(I-H)Y$는 독립이다. $(I-H)Y$의 함수인 $\hat{\sigma}^2$도 $\hat{\beta}$와 독립이다.

**(c)** 오차분산 추정량의 분포  
$Z=(Y-X\beta)/\sigma\sim N_n(0,I)$로 표준화하면
$$
\frac{(n-p-1)\hat{\sigma}^2}{\sigma^2}=\frac{1}{\sigma^2}(Y-X\hat{\beta})^T(Y-X\hat{\beta})
$$

$Y-X\hat{\beta}=(I-H)Y$이고, $(I-H)$의 성질 $(I-H)X=X-HX=X-X=0$로부터
$$
(I-H)X\beta=0
$$

따라서
$$
Y-X\hat{\beta}=(I-H)Y=(I-H)(Y-X\beta)+(I-H)X\beta=(I-H)(Y-X\beta)
$$

이를 이용하면
$$
\frac{(n-p-1)\hat{\sigma}^2}{\sigma^2}=\frac{1}{\sigma^2}(Y-X\beta)^T(I-H)(Y-X\beta)=Z^T(I-H)Z
$$

행렬 $A=I-H$는 대칭 멱등행렬이며:
- 대칭성: $A^T=(I-H)^T=I-H^T=I-H=A$
- 멱등성: $A^2=(I-H)^2=I-2H+H^2=I-2H+H=I-H=A$

행렬 $A$의 대각합(trace)은
$$
\text{trace}(A)=\text{trace}(I-H)=\text{trace}(I)-\text{trace}(H)
$$

투영행렬 $H=X(X^TX)^{-1}X^T$의 대각합은
$$
\text{trace}(H)=\text{trace}(X(X^TX)^{-1}X^T)=\text{trace}(X^TX(X^TX)^{-1})=\text{trace}(I_{p+1})=p+1
$$

따라서
$$
\text{trace}(A)=n-(p+1)
$$

정리 4.4.5 (b)에 의해, $Z\sim N_n(0,I)$이고 $A$가 대칭 멱등행렬이면 $Z^TAZ\sim\chi^2(\text{trace}(A))$이므로
$$
\frac{(n-p-1)\hat{\sigma}^2}{\sigma^2}\sim\chi^2(n-p-1)
$$

#### 예 4.4.7 단순선형회귀모형에서의 표본분포 *(Sampling Distribution in Simple Linear Regression Model)*
단순선형회귀모형에서 정리 4.4.6을 적용하면 다음이 성립한다.

예 4.4.6에서 구한
$$
X^TX = \begin{pmatrix} n & n\bar{x} \\ n\bar{x} & \sum_{i=1}^n x_i^2 \end{pmatrix}, \quad
S_{xx} = \sum_{i=1}^n (x_i - \bar{x})^2 = \sum_{i=1}^n x_i^2 - n\bar{x}^2
$$

를 이용하면
$$
(X^TX)^{-1} = \frac{1}{nS_{xx}} \begin{pmatrix} \sum_{i=1}^n x_i^2 & -n\bar{x} \\ -n\bar{x} & n \end{pmatrix}
= \frac{1}{S_{xx}} \begin{pmatrix} \sum_{i=1}^n x_i^2/n & -\bar{x} \\ -\bar{x} & 1 \end{pmatrix}
$$

따라서 정리 4.4.6 (a)에 의해
$$
\hat{\beta} = \begin{pmatrix}\hat{\beta}_0\\\hat{\beta}_1\end{pmatrix}
\sim N_2\left(
\begin{pmatrix}\beta_0\\\beta_1\end{pmatrix},
\frac{\sigma^2}{S_{xx}}
\begin{pmatrix}
\sum_{i=1}^n x_i^2/n & -\bar{x}\\
-\bar{x} & 1
\end{pmatrix}
\right)
$$

특히 $\hat{\beta}_1 \sim N\left(\beta_1, \frac{\sigma^2}{S_{xx}}\right)$이다.

정리 4.4.6 (c)에 의해
$$
\frac{(n-2)\hat{\sigma}^2}{\sigma^2}\sim\chi^2(n-2)
$$

이고 정리 4.4.6 (b)에 의해 $\hat{\beta}$와 $\hat{\sigma}^2$는 서로 독립이다.

따라서 t분포의 대의적 정의로부터
$$
\frac{\hat{\beta}_1-\beta_1}{\sqrt{\hat{\sigma}^2/S_{xx}}}
= \frac{(\hat{\beta}_1-\beta_1)/\sqrt{\sigma^2/S_{xx}}}{\sqrt{(n-2)\hat{\sigma}^2/\sigma^2/(n-2)}}
\sim t(n-2)
$$

이 경우에 평균오차제곱합은 다음의 공식을 이용하여 계산할 수 있다:
$$
\hat{\sigma}^2 = \frac{\sum_{i=1}^n(Y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i)^2}{n-2}
= \frac{S_{YY}-(S_{xY})^2/S_{xx}}{n-2}
$$

여기서
$$
S_{YY}=\sum_{i=1}^n(Y_i-\bar{Y})^2, \quad S_{xY}=\sum_{i=1}^n(x_i-\bar{x})(Y_i-\bar{Y})
$$

### 다변량 정규분포의 성질 정리
$X \sim N_n(\mu, \Sigma)$일 때, 다음의 성질들이 성립한다.

**(1) 표준형 표현 *(Standard Form Representation)***
$$
X \sim N(\mu, \Sigma) \Leftrightarrow X \overset{d}{\equiv} \Sigma^{1/2}Z + \mu, \quad Z \sim N_n(0,I)
$$

**(2) 선형변환에 대한 불변성 *(Closure under Linear Transformations)***

임의의 $m \times n$ 상수행렬 $A$와 $m$차원 상수벡터 $b$에 대하여
$$
AX+b \sim N_m(A\mu+b, A\Sigma A^T)
$$

**(3) 독립 다변량 정규분포의 합 *(Sum of Independent Multivariate Normal Distributions)***

$X_1 \sim N_n(\mu_1, \Sigma_1)$, $X_2 \sim N_n(\mu_2, \Sigma_2)$이고 $X_1 \perp X_2$이면
$$
X_1 + X_2 \sim N_n(\mu_1+\mu_2, \Sigma_1+\Sigma_2)
$$

**(4) 공분산과 독립성의 동치 *(Equivalence of Zero Covariance and Independence)***

임의의 상수행렬 $A$, $B$에 대하여
$$
AX \perp BX \Leftrightarrow \text{Cov}(AX, BX) = A\Sigma B^T = 0
$$

**(5) 조건부분포의 정규성 *(Normality of Conditional Distribution)***

$$
\begin{pmatrix}X_1\\X_2\end{pmatrix}
\sim
N\left(
\begin{pmatrix}\mu_1\\\mu_2\end{pmatrix},
\begin{pmatrix}
\Sigma_{11}&\Sigma_{12}\\
\Sigma_{21}&\Sigma_{22}
\end{pmatrix}
\right)
$$

일 때,

(a) **주변분포:** $X_1 \sim N(\mu_1, \Sigma_{11})$

(b) **조건부분포:** 
$$
X_2 \mid X_1=x_1 \sim N\left(\mu_2 + \Sigma_{21}\Sigma_{11}^{-1}(x_1-\mu_1), \Sigma_{22} - \Sigma_{21}\Sigma_{11}^{-1}\Sigma_{12}\right)
$$

**(6) 이차형식의 카이제곱분포 *(Chi-squared Distribution of Quadratic Forms)***

$Z \sim N_n(0,I)$이고 $A$가 $n \times n$ 대칭행렬일 때
$$
Z^TAZ \sim \chi^2(r) \Leftrightarrow A^2 = A \text{ (멱등성)}, \quad r = \text{trace}(A)
$$

일반적으로, $X \sim N_n(\mu, \Sigma)$이고 $\Sigma$가 정칙행렬이면
$$
(X-\mu)^T\Sigma^{-1}(X-\mu) \sim \chi^2(n)
$$
## 대표적 표본분포 정리 
| 분포 (Distribution) | 대의적 정의 및 확률밀도함수 |
|:---|:---|
| **카이제곱분포**<br>*(Chi-squared distribution)*<br>$X\sim\chi^2(r)$<br>정규오차의 에너지(제곱 크기) | **대의적 정의:** $X\overset{d}{\equiv}\sum_{i=1}^r Z_i^2,\quad Z_i\overset{iid}{\sim} N(0,1)$<br><br>**확률밀도함수:**<br>$\text{pdf}_X(x)=\frac{1}{\Gamma(r/2)2^{r/2}}x^{r/2-1}e^{-x/2}I_{(0,\infty)}(x)$ |
| **t 분포**<br>*(Student's t distribution)*<br>$X\sim t(r)$<br>분산을 모르고 평균을 표준화했을 때의 불확실성 | **대의적 정의:** $X\overset{d}{\equiv}\frac{Z}{\sqrt{V/r}},\quad Z\sim N(0,1),\ V\sim\chi^2(r),\ Z\perp V$<br><br>**확률밀도함수:**<br>$\text{pdf}_X(x)=\frac{\Gamma((r+1)/2)}{\sqrt{\pi r}\Gamma(r/2)}\left(1+\frac{x^2}{r}\right)^{-(r+1)/2}$ |
| **F 분포**<br>*(F distribution)*<br>$X\sim F(r_1,r_2)$<br>두 변동성의 상대적 크기 비교 | **대의적 정의:** $X\overset{d}{\equiv}\frac{V_1/r_1}{V_2/r_2},\quad V_1\sim\chi^2(r_1),\ V_2\sim\chi^2(r_2),\ V_1\perp V_2$<br><br>**확률밀도함수:**<br>$\text{pdf}_X(x)=\frac{\Gamma((r_1+r_2)/2)}{\Gamma(r_1/2)\Gamma(r_2/2)}\left(\frac{r_1}{r_2}\right)^{r_1/2}\frac{x^{r_1/2-1}}{(1+r_1x/r_2)^{(r_1+r_2)/2}}I_{(0,\infty)}(x)$ |
| **베타분포**<br>*(Beta distribution)*<br>$X\sim\text{Beta}(\alpha_1,\alpha_2)$<br>성공확률이 얼마일지에 대한 불확실성 | **대의적 정의:** $X\overset{d}{\equiv}\frac{Y_1}{Y_1+Y_2},\quad Y_i\sim\text{Gamma}(\alpha_i,\beta),\ Y_1\perp Y_2$<br><br>**확률밀도함수:**<br>$\text{pdf}_X(x)=\frac{\Gamma(\alpha_1+\alpha_2)}{\Gamma(\alpha_1)\Gamma(\alpha_2)}x^{\alpha_1-1}(1-x)^{\alpha_2-1}I_{(0,1)}(x)$ |
| **디리클레분포**<br>*(Dirichlet distribution)*<br>$(Y_1,\ldots,Y_k)\sim$<br>$\text{Dirichlet}(\alpha_1,\ldots,\alpha_{k+1})$<br>범주별 성공확률에 대한 불확실성 | **대의적 정의:** $(Y_1,\ldots,Y_k)\overset{d}{\equiv}\left(\frac{X_1}{\sum_{j=1}^{k+1}X_j},\ldots,\frac{X_k}{\sum_{j=1}^{k+1}X_j}\right)$<br>$X_i\overset{ind}{\sim}\text{Gamma}(\alpha_i,\beta)$<br><br>**확률밀도함수:**<br>$\text{pdf}(y_1,\ldots,y_k)=\frac{\Gamma(\alpha_1+\cdots+\alpha_{k+1})}{\Gamma(\alpha_1)\cdots\Gamma(\alpha_{k+1})}\prod_{i=1}^k y_i^{\alpha_i-1}(1-y_1-\cdots-y_k)^{\alpha_{k+1}-1}$ |
