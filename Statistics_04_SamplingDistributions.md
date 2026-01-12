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
\iff
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
\iff
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
\iff
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
\iff
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
\iff
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
\iff
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
X \sim \text{Beta}(\alpha_1, \alpha_2) \iff X \overset{d}{\equiv} \frac{Y_1}{Y_1 + Y_2}
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
이고 $-y<y\iff y>0$이므로
$$
\text{pdf}_Y(y)
=\int_{-\infty}^{+\infty}\text{pdf}_{Y,Z}(y,z),dz
=\int_{-y}^{y}2,dz,I_{(0,1/2]}(y)
=4yI_{(0,1/2]}(y)
$$

(ii) $y>\frac12$인 경우  
$(-y)\vee(y-1)=y-1,\quad (1-y)\wedge y=1-y$
이고 $y-1<1-y\iff y<1$이므로
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
\iff
\text{pdf}_X(x)=\frac{1}{b-a}I_{(a,b)}(x)\\
\iff
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
\iff
(Y_1,\ldots,Y_k)\overset{d}{\equiv}\left(\frac{X_1}{X_1\oplus\cdots\oplus X_{k+1}},\ldots,\frac{X_k}{X_1\oplus\cdots\oplus X_{k+1}}\right)
$$

즉, 디리클레분포는 감마분포들의 합으로 정규화된 비율들의 결합분포이다.
$$
(Y_1,\dots,Y_k)^T\sim\text{Dirichlet}(\alpha_1,\dots,\alpha_k,\alpha_{k+1})
$$
$$
\iff
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

#TODO: 야코비안, 분포변환 공식 이해, 분포변환이랑 표본분포랑 무슨상관인지, ch3에서 배운 분포들이랑은 무슨 상관인지

## 대표적인 표본분포 *(Common Sampling Distributions)*


## 순서통계량의 분포


## 다변량 정규분포
