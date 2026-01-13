# 제4장 표본분포 *(Sampling Distributions)*

## 4.1 통계량의 분포 *(Distributions of Statistics)*
모집단의 개수가 유한개일 경우에 비복원추출에 의한 표본을 이용하여 모집단에 대한 추측을 하는 방법의 타당성과 효율성을 연구하려면, 비복원추출로 인한 추출 결과의 종속성(dependence)에서 야기되는 어려움이 많다.

한편, 복원추출(with replacement)에 의한 표본의 경우에는 추출 결과 사이의 독립성(independence)이 있기 때문에 추측 방법의 성질을 연구하는 것이 용이하다. 또한 초기하분포의 이항분포 근사에서 알 수 있듯이, 모집단 크기가 커짐에 따라 비복원추출에 따른 추측 방법의 성질이 복원추출에 의한 방법의 성질과 거의 같게 된다.

이러한 용이성과 근사성 때문에 통계적 추측 방법의 성질을 연구할 때에는 흔히 유한모집단에서의 복원추출을 개념화하여 동일한 모집단을 독립적으로 관측하는 것을 전제로 한다.

이러한 이유에서 앞으로는 **랜덤표본(random sample)** 이란 서로 독립이고 동일한 분포를 따르는 확률변수들을 뜻한다. 또한 이러한 연구에서 가능한 모집단의 분포들을 모형으로 설정할 때, 모형의 설정에 사용되는 매개변수를 **모수(parameter)** 라고 하며, 가능한 모수 전체의 집합을 **모수공간(parameter space)** 이라고 한다.

##### 예 4.1.1
(a) 공정의 불량률에 대한 통계 조사의 경우에 모집단분포는 베르누이분포 Bernoulli$(p)$로 나타낼 수 있고, 불량률 $p$가 모수이고 미지인 불량률의 집합
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

이들 예에서 알 수 있듯이, 모수는 모집단 분포를 결정짓는 미지의 특성치로서 흔히 $\theta$로 나타내고, 모수공간은 $\Omega$로 나타내며 모집단 분포는 확률밀도함수 $f(x;\theta)$로 나타낸다. 이러한 미지의 모집단 분포 또는 모수에 대한 추측을 목적으로 사용하는 랜덤표본의 관측 가능한 함수를 **통계량(statistic)** 이라고 한다.

### 랜덤표본과 통계량
모집단 분포가 확률밀도함수 $f(x;\theta)$, $\theta\in\Omega$인 모집단에서의 랜덤표본 $X_1,X_2,\dots,X_n$이란
$$
X_1,X_2,\dots,X_n \overset{iid}{\sim} f(x;\theta),\quad \theta\in\Omega
$$
를 뜻하고, 통계량 $u(X_1,X_2,\dots,X_n)$이란 랜덤표본의 함수로서 랜덤표본의 값이 주어지면 그 값이 정해지는 함수를 뜻한다.

##### 예 4.1.2
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

##### 예 4.1.3
(a) 불량률 $p$를 모수로 하는 베르누이분포 Bernoulli$(p)$에서의 랜덤표본을 $X_1,X_2,\dots,X_n$이라고 할 때, 표본 중 불량품의 개수인 $X_1+\cdots+X_n$은 이항분포 $Bin(n,p)$를 갖는다. 따라서 표본비율
$$
\hat p=\frac{X_1+\cdots+X_n}{n}
$$
은
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

##### 예 4.1.4
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

##### 예 4.1.5  위치모수와 척도모수를 이용한 확률분포
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

##### 예 4.1.6
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
\frac{\Gamma(\alpha_1+\cdots+\alpha*{k+1})}
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

TODO: 어제 내용 복습도 해야할듯...
## 4.4 다변량 정규분포 *(Multivariate Normal Distribution)*
통계 조사에서는 단일 특성보다 **서로 연관된 여러 특성**을 동시에 관측하는 경우가 많다.
이러한 다차원 자료에 대한 추론의 기본 모형으로 **다변량 정규분포(multivariate normal distribution)**가 널리 사용된다.

### 정리 4.4.1 다변량 정규분포의 구성 *(Construction of Multivariate Normal Distribution)*

표준정규분포 $N(0,1)$를 따르고 서로 독립인 확률변수
$$
Z_1,Z_2,\dots,Z_n
$$
과 상수 행렬 $A=(a_{ij})_{1\le i,j\le n}$, 벡터
$$
\mu=(\mu_1,\mu_2,\dots,\mu_n)^t
$$
에 대하여
$$
X=AZ+\mu,\quad X=(X_1,\dots,X_n)^t,\quad Z=(Z_1,\dots,Z_n)^t
$$
라고 하자.

(a) 확률밀도함수
행렬 $A$가 **정칙행렬(nonsingular matrix)**이면
$$
X=AZ+\mu
$$
의 확률밀도함수는
$$
pdf_X(x)
=
(\det(2\pi\Sigma))^{-1/2}
\exp!\left{
-\frac12(x-\mu)^t\Sigma^{-1}(x-\mu)
\right},\quad x\in\mathbb R^n
$$
이며
$$
\Sigma=AA^t
$$
이다.

(b) 적률생성함수 *(Moment Generating Function)*
$$
mgf_X(t)
=
\exp!\left(
\mu^t t+\frac12 t^t\Sigma t
\right),\quad t\in\mathbb R^n
$$

#### 증명
$Z=(Z_1,\dots,Z_n)^t$의 확률밀도함수와 적률생성함수는
$$
pdf_Z(z)=(2\pi)^{-n/2}\exp!\left(-\frac12 z^t z\right),\quad
mgf_Z(s)=\exp!\left(\frac12 s^t s\right)
$$
이다.

(a) $X=u(Z)=AZ+\mu$라 하면 $A$가 정칙행렬이므로 $u$는 일대일 함수이며
$$
u^{-1}(x)=A^{-1}(x-\mu),\quad
|J_{u^{-1}}|=|\det(A)|^{-1}
$$
이다. 치환공식에 의해
$$
pdf_X(x)=pdf_Z(A^{-1}(x-\mu))|\det(A)|^{-1}
$$
이고, $\Sigma=AA^t$를 이용하면 주어진 식을 얻는다.

(b)
$$
mgf_X(t)=E[e^{t^t(AZ+\mu)}]
=
mgf_Z(A^t t),e^{t^t\mu}
$$
에서 바로 따른다. □

### 다변량 정규분포의 정의 *(Definition of Multivariate Normal Distribution)*
확률벡터 $X$가 다음 조건들 중 하나를 만족하면
$$
X\sim N_n(\mu,\Sigma)
$$
라 한다.

1. $X=AZ+\mu$, $Z\sim N_n(0,I)$, $AA^t=\Sigma$
2. $X=\Sigma^{1/2}Z+\mu$, $Z\sim N_n(0,I)$
3. $mgf_X(t)=\exp(\mu^t t+\frac12 t^t\Sigma t)$
4. ($\Sigma$가 정칙행렬일 때)
   $$
   pdf_X(x)
   =
   (\det(2\pi\Sigma))^{-1/2}
   \exp!\left{-\frac12(x-\mu)^t\Sigma^{-1}(x-\mu)\right}
   $$

### 정리 4.4.2 평균벡터와 분산행렬 *(Mean Vector and Covariance Matrix)*

$X\sim N_n(\mu,\Sigma)$이면
$$
E(X)=\mu,\quad \mathrm{Var}(X)=\Sigma
$$

### 예 4.4.1 이변량 정규분포 *(Bivariate Normal Distribution)*
$$
(X_1,X_2)^t\sim N!\left(
\begin{pmatrix}\mu_1\ \mu_2\end{pmatrix},
\begin{pmatrix}
\sigma_1^2 & \rho\sigma_1\sigma_2\
\rho\sigma_1\sigma_2 & \sigma_2^2
\end{pmatrix}
\right)
$$

$-1<\rho<1$일 때 확률밀도함수는
$$
f(x_1,x_2)
=
\frac{1}{2\pi\sigma_1\sigma_2\sqrt{1-\rho^2}}
\exp!\left(-\frac12 Q\right)
$$
이며
$$
Q=\frac1{1-\rho^2}
\left[
\left(\frac{x_1-\mu_1}{\sigma_1}\right)^2
-2\rho\frac{x_1-\mu_1}{\sigma_1}\frac{x_2-\mu_2}{\sigma_2}
+\left(\frac{x_2-\mu_2}{\sigma_2}\right)^2
\right]
$$

### 정리 4.4.3 다변량 정규분포의 성질 *(Properties of Multivariate Normal Distribution)*

(a) 선형변환
$$
X\sim N(\mu,\Sigma)
\Rightarrow
AX+b\sim N(A\mu+b,A\Sigma A^t)
$$

(b) 공분산과 독립성
$$
\mathrm{Cov}(X_1,X_2)=0
\Rightarrow
X_1\perp X_2
$$
(다변량 정규분포에서만 성립)

(c)
$$
\mathrm{Cov}(AX,BX)=0
\Rightarrow
AX\perp BX
$$

### 예 4.4.2 이변량 정규분포에서의 독립성
이변량 정규분포에서는
$$
\mathrm{Cov}(X_1,X_2)=0 \iff X_1\perp X_2
$$

### 정리 4.4.4 주변분포와 조건부분포 *(Marginal and Conditional Distributions)*

$$
\begin{pmatrix}X_1\X_2\end{pmatrix}
\sim
N!\left(
\begin{pmatrix}\mu_1\\mu_2\end{pmatrix},
\begin{pmatrix}
\Sigma_{11}&\Sigma_{12}\
\Sigma_{21}&\Sigma_{22}
\end{pmatrix}
\right)
$$

(a) 주변분포
$$
X_1\sim N(\mu_1,\Sigma_{11})
$$

(b) 조건부분포
$$
X_2\mid X_1=x_1
\sim
N!\left(
\mu_2+\Sigma_{21}\Sigma_{11}^{-1}(x_1-\mu_1),
\Sigma_{22}-\Sigma_{21}\Sigma_{11}^{-1}\Sigma_{12}
\right)
$$

### 예 4.4.3 이변량 정규분포의 조건부분포
$$
X_2\mid X_1=x_1
\sim
N!\left(
\mu_2+\rho\frac{\sigma_2}{\sigma_1}(x_1-\mu_1),
\sigma_2^2(1-\rho^2)
\right)
$$

### 정리 4.4.5 이차형식의 분포 *(Distribution of Quadratic Forms)*
(a)
$$
X\sim N_k(\mu,\Sigma),\ \Sigma\ \text{정칙}
\Rightarrow
(X-\mu)^t\Sigma^{-1}(X-\mu)\sim\chi^2(k)
$$

(b)
$$
Z\sim N(0,I),\ A^2=A
\Rightarrow
Z^tAZ\sim\chi^2(r),\quad r=\mathrm{trace}(A)
$$

#### 예 4.4.4 일원분류모형에서의 표본분포 *(Sampling Distribution in One-Way Classification Model)*

정리 4.2.7의 (a)에 따르면, 일원분류모형에서
정규분포 $N(\mu_i,\sigma^2/n_i)$를 따르는 서로 독립인 표본평균 $\bar X_i\ (i=1,\dots,k)$에 대하여

$$
\sum_{i=1}^k n_i(\bar X_i-\bar X-(\mu_i-\mu))^2/\sigma^2 \sim \chi^2(k-1)
$$

가 성립한다.

이를 **정리 4.4.5**를 이용하여 유도한다.

#### 풀이
다음을 정의한다.

$$
Y_i=\bar X_i-\bar X-(\mu_i-\mu),\quad i=1,\dots,k
$$

그리고

$$
Y=(Y_1,\dots,Y_r)^t,\quad r=k-1
$$

여기서

$$
\bar X=\frac{\sum_{i=1}^k n_i\bar X_i}{n},\quad
\mu=\frac{\sum_{i=1}^k n_i\mu_i}{n},\quad
n=\sum_{i=1}^k n_i
$$

이다.

각 $Y_i$의 평균, 분산, 공분산은 다음과 같다.

$$
E(Y_i)=0
$$

$$
\mathrm{Var}(Y_i)=(n_i^{-1}-n^{-1})\sigma^2
$$

$$
\mathrm{Cov}(Y_i,Y_j)=-n^{-1}\sigma^2,\quad i\neq j
$$

따라서

$$
Y\sim N_r(0,\Sigma)
$$

이며 공분산행렬은

$$
\Sigma=\left[D(n_i^{-1})-n^{-1}\mathbf 1\mathbf 1^t\right]\sigma^2
$$

여기서

$$
D(n_i^{-1})=\begin{pmatrix}
n_1^{-1}&0&\cdots&0\
0&n_2^{-1}&\cdots&0\
\vdots&\vdots&\ddots&\vdots\
0&0&\cdots&n_r^{-1}
\end{pmatrix},\quad
\mathbf 1=(1,\dots,1)^t
$$

이다.

부록 II의 특수한 행렬 연산 성질로부터

$$
\left[D(n_i^{-1})-n^{-1}\mathbf 1\mathbf 1^t\right]^{-1}
= D(n_i)+n_k^{-1}(n_1,\dots,n_r)^t(n_1,\dots,n_r)
$$

이 성립한다.

따라서

$$
Y^t\Sigma^{-1}Y
=\sum_{i=1}^r n_iY_i^2/\sigma^2
+n_k^{-1}\left(\sum_{i=1}^r n_iY_i\right)^2/\sigma^2
=\sum_{i=1}^k n_iY_i^2/\sigma^2
$$

한편 정리 4.4.5의 (a)에 의해

$$
Y^t\Sigma^{-1}Y\sim\chi^2(r)
$$

이므로

$$
\sum_{i=1}^k n_i(\bar X_i-\bar X-(\mu_i-\mu))^2/\sigma^2\sim\chi^2(k-1)
$$

이다. □

#### 예 4.4.5 표본분산의 표본분포 *(Sampling Distribution of Sample Variance)*

정규분포 $N(\mu,\sigma^2)$에서의 랜덤표본 $X_1,\dots,X_n$에 대하여

$$
\bar X=\frac1n\sum_{i=1}^n X_i,\quad
S^2=\frac1{n-1}\sum_{i=1}^n(X_i-\bar X)^2
$$

라 하면, $\bar X$와 $S^2$는 서로 독립이며

$$
\frac{(n-1)S^2}{\sigma^2}\sim\chi^2(n-1)
$$

이다.

#### 풀이
벡터
$$
X=(X_1,\dots,X_n)^t
$$

에 대하여

$$
X\sim N_n(\mu\mathbf 1,\sigma^2I)
$$

이다.

또한

$$
\bar X=n^{-1}\mathbf 1^tX
$$

$$
(n-1)S^2=X^t(I-n^{-1}\mathbf 1\mathbf 1^t)X
$$

이다.

행렬 $I-n^{-1}\mathbf 1\mathbf 1^t$는 대칭행렬이며

$$
\mathbf 1^t(I-n^{-1}\mathbf 1\mathbf 1^t)=0
$$

이므로

$$
\mathrm{Cov}(\bar X,(I-n^{-1}\mathbf 1\mathbf 1^t)X)=0
$$

따라서 정리 4.4.3 (c)에 의해 $\bar X$와 $(n-1)S^2$는 서로 독립이다.

또한

$$
\frac{(n-1)S^2}{\sigma^2}
=\frac{(X-\mu\mathbf 1)^t(I-n^{-1}\mathbf 1\mathbf 1^t)(X-\mu\mathbf 1)}{\sigma^2}
$$

이며

$$
\frac{X-\mu\mathbf 1}{\sigma}\sim N_n(0,I)
$$

이고

$$
(I-n^{-1}\mathbf 1\mathbf 1^t)^2=I-n^{-1}\mathbf 1\mathbf 1^t,\quad
\mathrm{trace}(I-n^{-1}\mathbf 1\mathbf 1^t)=n-1
$$

이므로 정리 4.4.5 (b)에 의해

$$
\frac{(n-1)S^2}{\sigma^2}\sim\chi^2(n-1)
$$

이다. □

### 선형회귀모형 (정규 오차)
선형회귀모형을
$$
Y=X\beta+e,\quad e\sim N_n(0,\sigma^2I)
$$

라 하자.
여기서 $X$는 $n\times(p+1)$ 상수행렬이며 $\mathrm{rank}(X)=p+1$이다.

표본회귀계수는

$$
\hat\beta=(X^tX)^{-1}X^tY
$$

로 정의한다.

### 정리 4.4.6 선형회귀모형에서의 표본분포
위 모형에서 다음이 성립한다.

(a)
$$
\hat\beta\sim N_{p+1}(\beta,\sigma^2(X^tX)^{-1})
$$

(b)
$\hat\beta$와

$$
\hat\sigma^2=\frac{(Y-X\hat\beta)^t(Y-X\hat\beta)}{n-p-1}
$$

는 서로 독립이다.

(c)
$$
\frac{(n-p-1)\hat\sigma^2}{\sigma^2}\sim\chi^2(n-p-1)
$$

#### 증명

(a)
$Y\sim N_n(X\beta,\sigma^2I)$이므로 정리 4.4.3 (a)에 의해 성립한다.

(b)
$\Pi=X(X^tX)^{-1}X^t$라 하면 $\Pi^2=\Pi$, $\Pi^t=\Pi$이고

$$
Y-X\hat\beta=(I-\Pi)Y
$$

이며

$$
\mathrm{Cov}(\hat\beta,(I-\Pi)Y)=0
$$

이므로 정리 4.4.3 (c)에 의해 독립이다.

(c)
$I-\Pi$는 대칭 멱등행렬이고

$$
\mathrm{trace}(I-\Pi)=n-p-1
$$

이므로 정리 4.4.5 (b)에 의해 성립한다. □

#### 예 4.4.8 단순선형회귀모형에서의 표본분포
단순선형회귀모형
$$
Y_i=\beta_0+\beta_1x_i+e_i,\quad e_i\sim N(0,\sigma^2)
$$

에서

$$
\hat\beta=
\begin{pmatrix}
\hat\beta_0\
\hat\beta_1
\end{pmatrix}
\sim
N\left(
\begin{pmatrix}\beta_0\\beta_1\end{pmatrix},
\sigma^2
\begin{pmatrix}
\frac1n+\frac{\bar x^2}{S_{xx}} & -\frac{\bar x}{S_{xx}}\
-\frac{\bar x}{S_{xx}} & \frac1{S_{xx}}
\end{pmatrix}
\right)
$$

이며

$$
\frac{(n-2)\hat\sigma^2}{\sigma^2}\sim\chi^2(n-2)
$$

이고 $\hat\beta$와 $\hat\sigma^2$는 서로 독립이다.

## 대표적 표본분포 (Representative Sampling Distributions)

### 카이제곱분포 *(Chi-square distribution)*

$$
X\sim\chi^2(r)\iff X=\sum_{i=1}^r Z_i^2,\quad Z_i\sim N(0,1)
$$

$$
pdf_X(x)=\frac1{\Gamma(r/2)2^{r/2}}x^{r/2-1}e^{-x/2}I_{(0,\infty)}(x)
$$

### t 분포 *(Student's t distribution)*

$$
X\sim t(r)\iff X=\frac{Z}{\sqrt{V/r}},\quad Z\sim N(0,1),\ V\sim\chi^2(r)
$$

### F 분포 *(F distribution)*

$$
X\sim F(r_1,r_2)\iff X=\frac{V_1/r_1}{V_2/r_2},\quad V_i\sim\chi^2(r_i)
$$

### 베타분포 *(Beta distribution)*

$$
X\sim\mathrm{Beta}(\alpha_1,\alpha_2)
\iff X=\frac{Z_1}{Z_1+Z_2},\ Z_i\sim\mathrm{Gamma}(\alpha_i,\beta)
$$

### 디리클레분포 *(Dirichlet distribution)*

$$
(X_1,\dots,X_k)\sim\mathrm{Dirichlet}(\alpha_1,\dots,\alpha_{k+1})
$$