## 삼각함수 공식 정리

### 홀짝함수의 곱

| 첫 번째 함수 | 두 번째 함수 | 곱   |
| ------- | ------- | --- |
| 짝함수     | 짝함수     | 짝함수 |
| 홀함수     | 홀함수     | 짝함수 |
| 짝함수     | 홀함수     | 홀함수 |

따라서 $\cos(mx)\sin(nx)$는 짝함수와 홀함수의 곱이므로 홀함수다.

### 기본 삼각함수 관계식

$$
\boxed{\sin^2\theta+\cos^2\theta=1}
$$

이를 변형하면 $\sin^2\theta=1-\cos^2\theta$ 및 $\cos^2\theta=1-\sin^2\theta$ 를 얻는다.

또한 다음이 성립한다.

$$
\tan\theta=\frac{\sin\theta}{\cos\theta}
\qquad(\cos\theta\neq0) \\
1+\tan^2\theta=\sec^2\theta\\
1+\cot^2\theta=\csc^2\theta
$$

### 덧셈정리

대부분의 삼각공식은 덧셈정리로부터 유도된다.

$$
  \sin(A+B)=\sin A\cos B+\cos A\sin B \\
\sin(A-B)=\sin A\cos B-\cos A\sin B \\
\cos(A+B)=\cos A\cos B-\sin A\sin B \\
\cos(A-B)=\cos A\cos B+\sin A\sin B \\
$$

#### 증명

단위원 위의 점 $(\cos B,\sin B)$를 원점 주위로 $A$만큼 회전시키면 그 좌표는

$$
\begin{pmatrix}
\cos A&-\sin A\\
\sin A&\cos A
\end{pmatrix}
\begin{pmatrix}
\cos B\\
\sin B
\end{pmatrix}
=
\begin{pmatrix}
\cos A\cos B-\sin A\sin B\\
\sin A\cos B+\cos A\sin B
\end{pmatrix}
$$

이다. 한편 회전한 점의 좌표는 $(\cos(A+B),\sin(A+B))$이므로 좌표를 비교하면 (혹은 피타고라스 정리로 삼각형 대각선 길이 계산하는걸 생각해도 됨.)

$$
\cos(A+B)=\cos A\cos B-\sin A\sin B,
\qquad
\sin(A+B)=\sin A\cos B+\cos A\sin B
$$

를 얻는다. 여기서 $B$를 $-B$로 바꾸고
$\cos(-B)=\cos B$, $\sin(-B)=-\sin B$를 사용하면

$$
\cos(A-B)=\cos A\cos B+\sin A\sin B,
\qquad
\sin(A-B)=\sin A\cos B-\cos A\sin B
$$

가 성립한다. 따라서 네 덧셈정리가 모두 증명된다.

### 배각공식

덧셈정리에서 $A=B=\theta$로 놓으면 다음을 얻는다.

$$
\boxed{\sin(2\theta)=2\sin\theta\cos\theta}, \quad\boxed{\cos(2\theta)=\cos^2\theta-\sin^2\theta}
$$

기본 관계식 $\sin^2\theta+\cos^2\theta=1$을 이용하면 다음과 같이 바꿀 수 있다.

$$
\boxed{\cos(2\theta)=2\cos^2\theta-1}, \quad \boxed{\cos(2\theta)=1-2\sin^2\theta}
$$

배각공식을 변형하면 다음을 얻는다.

$$
\cos^2\theta=\frac{1+\cos(2\theta)}{2}, \quad \sin^2\theta=\frac{1-\cos(2\theta)}{2}
$$

이는 $\sin^2\theta$와 $\cos^2\theta$를 적분할 때 중요하다.

$$
\begin{aligned}
\int\cos^2x,dx
&=\int\frac{1+\cos(2x)}{2},dx\
&=\frac{x}{2}+\frac{\sin(2x)}{4}+C\\
\int\sin^2x,dx
&=\int\frac{1-\cos(2x)}{2},dx\
&=\frac{x}{2}-\frac{\sin(2x)}{4}+C
\end{aligned}
$$

### 응용: 곱을 합으로 바꾸는 공식

푸리에 급수에서 매우 중요한 공식이다.

$$
\sin A\cos B = \frac12{\sin(A+B)+\sin(A-B)} \\
\cos A\cos B = \frac12{\cos(A-B)+\cos(A+B)} \\
\sin A\sin B = \frac12{\cos(A-B)-\cos(A+B)}
$$

특히 $A=mx$, $B=nx$로 놓으면 다음을 얻는다.

$$
\sin(mx)\cos(nx) = \frac12 \left[\sin((m+n)x)+\sin((m-n)x)\right] \\
\cos(mx)\cos(nx) = \frac12 \left[\cos((m-n)x)+\cos((m+n)x)\right] \\
\sin(mx)\sin(nx) = \frac12 \left[\cos((m-n)x)-\cos((m+n)x)\right]
$$

반대로하면: 합을 곱으로 바꾸는 공식이 된다.

$$
\sin A+\sin B = 2\sin\left(\frac{A+B}{2}\right) \cos\left(\frac{A-B}{2}\right) \\
\sin A-\sin B = 2\cos\left(\frac{A+B}{2}\right) \sin\left(\frac{A-B}{2}\right) \\
\cos A+\cos B = 2\cos\left(\frac{A+B}{2}\right) \cos\left(\frac{A-B}{2}\right) \\
\cos A-\cos B = -2\sin\left(\frac{A+B}{2}\right) \sin\left(\frac{A-B}{2}\right)
$$

### 삼각함수의 미분과 적분

$n$이 상수일 때 다음이 성립한다.

$$
\frac{d}{dx}\sin(nx)=n\cos(nx), \quad \frac{d}{dx}\cos(nx)=-n\sin(nx) \\
\int\sin(nx),dx = -\frac{1}{n}\cos(nx)+C, \quad \int\cos(nx),dx = \frac{1}{n}\sin(nx)+C
$$

### 직교성의 의미

함수의 내적을 다음과 같이 정의한다.

$$
\langle f,g\rangle = \int_{-\pi}^{\pi}f(x)g(x),dx
$$

두 함수의 내적이 $0$이면 두 함수가 서로 직교한다고 한다.

$$
\langle f,g\rangle=0 \quad\Longrightarrow\quad f\perp g
$$

이는 유클리드 공간에서 두 벡터의 내적이 $0$이면 서로 수직인 것과 같은 개념이다.

(직교행렬: $AA^T = I$)

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

이때 gamma 분포의 정의에 따라, $Y_i$의 합인 $\sum Y_i$는 $\text{Gamma}(n, \theta)$이 된다. 

$-2\sum \log X_i$의 $\chi^2$ 분포 유도: 앞에서 $Y_i=-\theta\log X_i\sim \mathrm{Exp}(\theta)$ (scale $\theta$) 이므로

$$S:=\sum_{i=1}^n Y_i \sim \mathrm{Gamma}(n,\theta)$$

감마분포의 스케일 성질 $cS\sim \mathrm{Gamma}(n,c\theta)$ ($c>0$)를 쓰면

$$2S=-2\theta\sum_{i=1}^n\log X_i \sim \mathrm{Gamma}(n,2\theta)$$

여기서 $\chi^2$와의 정확한 연결은 다음과 같다.

$$\chi^2(\nu)\equiv \mathrm{Gamma}\left(\frac{\nu}{2},\,2\right)$$

따라서 $\mathrm{Gamma}(n,2\theta)$가 $\chi^2(2n)$와 **동일**하려면 $\theta=1$이어야 한다.  
일반 $\theta$에 대해 $\chi^2$ 피벗은 $\theta$로 나눈 형태이다:

$$\frac{2S}{\theta}=\frac{-2\theta\sum \log X_i}{\theta}=-2\sum_{i=1}^n \log X_i$$

그런데 $Z_i:=-\log X_i\sim \mathrm{Exp}(1)$, 독립이므로

$$\sum_{i=1}^n Z_i \sim \mathrm{Gamma}(n,1)\quad\Rightarrow\quad 2\sum_{i=1}^n Z_i \sim \mathrm{Gamma}(n,2)=\chi^2(2n)$$

즉 최종적으로 $-2\sum_{i=1}^n \log X_i \sim \chi^2(2n)$ 

그리고 동치로 $-2\theta\sum_{i=1}^n \log X_i \sim \mathrm{Gamma}(n,2\theta)$

**베타-감마 관계 증명**  
$Y_1 \sim \text{Gamma}(\alpha,\theta)$, $Y_2 \sim \text{Gamma}(\beta,\theta)$ 독립이면, $U = \frac{Y_1}{Y_1+Y_2}$는 $\text{Beta}(\alpha,\beta)$를 따른다.

$Y_1$과 $Y_2$의 결합확률밀도함수(pdf)는 

$$f_{Y_1,Y_2}(y_1,y_2) = \frac{1}{\Gamma(\alpha)\theta^\alpha} y_1^{\alpha-1} e^{-y_1/\theta} \cdot \frac{1}{\Gamma(\beta)\theta^\beta} y_2^{\beta-1} e^{-y_2/\theta}$$

$$= \frac{1}{\Gamma(\alpha)\Gamma(\beta)\theta^{\alpha+\beta}} y_1^{\alpha-1} y_2^{\beta-1} e^{-(y_1+y_2)/\theta}$$

$U = \frac{Y_1}{Y_1+Y_2}$와 $V = Y_1 + Y_2$로 변수변환을 하면, 역변환은 $Y_1 = UV$, $Y_2 = (1-U)V$가 된다.  
이때 야코비안은

$$J = \begin{vmatrix}\frac{\partial Y_1}{\partial U} & \frac{\partial Y_1}{\partial V} \\ \frac{\partial Y_2}{\partial U} & \frac{\partial Y_2}{\partial V}\end{vmatrix} = \begin{vmatrix}V & U \\ -V & 1-U\end{vmatrix} = V$$

따라서 $U$의 확률밀도함수(pdf)는 다음과 같이 계산된다.

$$
\begin{aligned}
f_U(u) &= \int_0^\infty f_{Y_1,Y_2}(uv,(1-u)v) \cdot J \, dv\\
&= \int_0^\infty \frac{1}{\Gamma(\alpha)\Gamma(\beta)\theta^{\alpha+\beta}} (uv)^{\alpha-1} ((1-u)v)^{\beta-1} e^{-v/\theta} \cdot v \, dv\\
&= \frac{u^{\alpha-1}(1-u)^{\beta-1}}{\Gamma(\alpha)\Gamma(\beta)\theta^{\alpha+\beta}} \int_0^\infty v^{\alpha+\beta-1} e^{-v/\theta} dv\\
&= \frac{u^{\alpha-1}(1-u)^{\beta-1}}{\Gamma(\alpha)\Gamma(\beta)\theta^{\alpha+\beta}} \cdot \Gamma(\alpha+\beta) \theta^{\alpha+\beta}\\
&= \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} u^{\alpha-1}(1-u)^{\beta-1}
\end{aligned}
$$

따라서 $U$는 $\text{Beta}(\alpha,\beta)$를 따른다. 특히 $\text{Beta}(\theta,1)$의 경우, $X \sim \text{Beta}(\theta,1)$이면 $-\log X \sim \text{Exp}(1)$이 된다. 왜냐하면 $\text{Beta}(\theta,1)$의 확률밀도함수(pdf)는

$$f_X(x) = \frac{\Gamma(\theta+1)}{\Gamma(\theta)\Gamma(1)} x^{\theta-1}(1-x)^{0} = \theta x^{\theta-1}$$

따라서 $Y = -\log X$의 확률밀도함수(pdf)는 다음과 같이 계산된다.

$$f_Y(y) = f_X(e^{-y}) \cdot \left| \frac{d}{dy} e^{-y} \right| = \theta e^{-\theta y}$$

즉 $Y$는 $\text{Exp}(1)$를 따른다.

**F-분포와의 연결 증명**: $Y_1 \sim \chi^2_{2m}$, $Y_2 \sim \chi^2_{2n}$ 독립이면, $F = \frac{Y_1/m}{Y_2/n}$는 $F(m,n)$을 따른다.

$Y_1$과 $Y_2$의 확률밀도함수(pdf)는 각각

$$f_{Y_1}(y_1) = \frac{1}{2^m \Gamma(m)} y_1^{m-1} e^{-y_1/2}, \quad f_{Y_2}(y_2) = \frac{1}{2^n \Gamma(n)} y_2^{n-1} e^{-y_2/2}$$

따라서 $F = \frac{Y_1/m}{Y_2/n}$와 $V = Y_2$로 변수변환을 하면, 역변환은 $Y_1 = mF \cdot \frac{V}{n}$, $Y_2 = V$가 된다. 이때 야코비안은

$$J = \begin{vmatrix}\frac{\partial Y_1}{\partial F} & \frac{\partial Y_1}{\partial V} \\ \frac{\partial Y_2}{\partial F} & \frac{\partial Y_2}{\partial V}\end{vmatrix} = \begin{vmatrix}\frac{mV}{n} & \frac{mF}{n} \\ 0 & 1\end{vmatrix} = \frac{mV}{n}$$

따라서 $F$의 확률밀도함수(pdf)는 다음과 같이 계산된다.

$$
\begin{aligned}
f_F(f) &= \int_0^\infty f_{Y_1,Y_2}\left(\frac{mV}{n}f, V\right) \cdot J \, dV \\
&= \int_0^\infty \frac{1}{2^m \Gamma(m)} \left(\frac{mV}{n}f\right)^{m-1} e^{-\frac{mV}{2n}f} \cdot \frac{1}{2^n \Gamma(n)} V^{n-1} e^{-V/2} \cdot \frac{mV}{n} \, dV \\
&= \frac{m^m f^{m-1}}{n^m 2^{m+n} \Gamma(m) \Gamma(n)} \int_0^\infty V^{m+n-1} e^{-\frac{V}{2}\left(1+\frac{mf}{n}\right)} dV \\
&= \frac{m^m f^{m-1}}{n^m 2^{m+n} \Gamma(m) \Gamma(n)} \cdot \Gamma(m+n) \left(\frac{2}{1+\frac{mf}{n}}\right)^{m+n}\\
&= \frac{\Gamma(m+n)}{\Gamma(m)\Gamma(n)} \left(\frac{m}{n}\right)^m \frac{f^{m-1}}{\left(1+\frac{mf}{n}\right)^{m+n}}
\end{aligned}
$$

따라서 $F$는 $F(m,n)$을 따른다.
 
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


## 8. 이차형식 전개 (Quadratic Form Expansion)

다변량 통계에서 자주 등장한다.

$$\mathbf{x}'A\mathbf{x} = \sum_i\sum_j a_{ij}x_ix_j$$

### 활용

* 다변량 정규분포
* Wishart 분포
* 회귀분석

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

### 정규분포의 **이차형식의 독립성** 특성

정규벡터 $\mathbf{Z} \sim N(\mathbf{0}, I_n)$에서 두 개의 idempotent 행렬 $A, B$에 대해

$$\mathbf{Z}^T A \mathbf{Z} \perp \mathbf{Z}^T B \mathbf{Z}$$

(독립)이 되는 필요충분조건은 $AB = 0$

>#### 증명
>**필요조건**: 이차형식들이 독립이면 $AB = 0$임을 보인다.
>
>여기서 $A,B$는 대칭 멱등행렬이라고 하자. 즉,
> $A^T=A$, $B^T=B$, $A^2=A$, $B^2=B$이다. 다음과 같이 놓는다.
> $$Q_A=\mathbf Z^T A\mathbf Z,\qquad Q_B=\mathbf Z^T B\mathbf Z.$$
>
> $Q_A$와 $Q_B$가 독립이라고 가정하면, 독립인 확률변수의 공분산은 0이므로
> $$\text{Cov}(Q_A,Q_B)=0$$
> 이다. 표준정규벡터 $\mathbf Z\sim N(\mathbf 0,I)$에 대한 이차형식의 공분산 공식에 의해
> $$
> \text{Cov}(\mathbf Z^T A\mathbf Z,\mathbf Z^T B\mathbf Z)
> =2\text{tr}(AB)
> $$
> 이다. (대칭이 아닌 경우에는 일반적으로 $2\text{tr}(A B)$ 대신
> $2\text{tr}(A B)$에 대칭부분을 반영한 형태가 필요하지만, 여기서는
> $A,B$가 대칭이므로 위 공식이 그대로 적용된다.) 따라서
> $$2\text{tr}(AB)=0,
> \qquad\text{즉}\qquad \text{tr}(AB)=0$$
> 을 얻는다.
>
> 단순히 $\text{tr}(AB)=0$이라는 사실만으로는 일반 행렬에 대해
> $AB=0$이라고 결론 내릴 수 없다. 이제 대칭 멱등성 조건을 사용한다.
> Frobenius 노름의 제곱을 계산하면
> $$
> \begin{aligned}
> \|AB\|_F^2
> &=\text{tr}\left((AB)^T(AB)\right)\\
> &=\text{tr}(BAAB) && (A^T=A,\ B^T=B)\\
> &=\text{tr}(BAB) && (A^2=A)\\
> &=\text{tr}(ABB) && (trace의 순환성)\\
> &=\text{tr}(AB) && (B^2=B)\\
> &=0.
> \end{aligned}
> $$
> Frobenius 노름은 행렬 원소들의 제곱합의 제곱근이므로
> $$\|AB\|_F^2=\sum_{i,j}(AB)_{ij}^2=0$$
> 이면 모든 원소가 0이다. 따라서
> $$\boxed{AB=0}$$
> 이다. 
>
>**충분조건**: $AB = 0$이면 독립임을 보인다.
>
>$\text{Cov}(A\mathbf{Z}, B\mathbf{Z}) = A \mathbb{E}[\mathbf{Z}\mathbf{Z}^T] B^T = AB = 0$
>
> $\mathbf Z^T A\mathbf Z = \mathbf Z^T A^T A \mathbf Z = \|AZ\|^2$ 으로 $AZ$의 함수이다. $BZ$도 마찬가지.  
> 따라서 $\mathbf{Z}^T A \mathbf{Z} \perp \mathbf{Z}^T B \mathbf{Z}$


### 예시: ANOVA/회귀분석 χ² 분해의 직교투영 해석

다변량 정규분포에서 이차형식의 독립성을 이용하면, 회귀분석의 분산분해 구조를 엄밀하게 증명할 수 있다.

#### 설정

- $\mathbf{Y} = (Y_1, \ldots, Y_n)^T \sim N(\mathbf{X}\boldsymbol{\beta}, \sigma^2 I_n)$
- $\mathbf{X} = [\mathbf{X}_0 \mid \mathbf{X}_1]$ (부분모형 비교)
- $\Pi_0 = \mathbf{X}_0(\mathbf{X}_0^T\mathbf{X}_0)^{-1}\mathbf{X}_0^T$ (귀무모형 투영)
- $\Pi_{0,1} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$ (전체모형 투영)
- $\Pi_{1|0} = \Pi_{0,1} - \Pi_0$ (제거 효과 투영, idempotent)

#### 핵심: 직교성과 독립성

**Step 1: 투영행렬의 직교성**

$$\Pi_{1|0}(I - \Pi_{0,1}) = 0$$

**증명**: $(I - \Pi_{0,1})$은 $\text{range}(\mathbf{X})$의 직교여공간으로의 투영이고, $\Pi_{1|0}$는 $\text{range}(\mathbf{X})$ 내의 연산이므로

$$\Pi_{1|0}(I - \Pi_{0,1}) = (\Pi_{0,1} - \Pi_0)(I - \Pi_{0,1})\\
= \Pi_{0,1} - \Pi_{0,1}^2 - \Pi_0 + \Pi_0\Pi_{0,1}\\
= \Pi_{0,1} - \Pi_{0,1} - \Pi_0 + \Pi_0 = 0$$

**Step 2: 이차형식의 독립성 적용**

$\mathbf{Y}$를 표준화하면 $\mathbf{Z} = (\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})/\sigma \sim N(\mathbf{0}, I_n)$

두 idempotent 행렬 $A = \Pi_{1|0}$, $B = I - \Pi_{0,1}$에 대해 $AB = 0 \iff \mathbf{Z}^T A \mathbf{Z} \perp \mathbf{Z}^T B \mathbf{Z}$ (정규벡터의 이차형식 독립성 원리)

이를 원래 변수로 돌리면 $R(1|0) = \mathbf{Y}^T\Pi_{1|0}\mathbf{Y} \perp SSE = \mathbf{Y}^T(I-\Pi_{0,1})\mathbf{Y}$


## 26. 기댓값과 대각합(trace) 연산 순서 교환 (Expectation-Trace Interchange)

선형연산자 $\text{trace}$와 기댓값 $E[\cdot]$는 교환 가능하다. 
$E[\text{trace}(\mathbf{A})] = \text{trace}(E[\mathbf{A}])$

대각합은 선형연산자(linear operator)이므로, 선형성(linearity of expectation)에 의해 기댓값과 교환 가능.

### 일반화: 선형연산자와 기댓값

모든 선형연산자 $L$에 대해 $E[L(\mathbf{X})] = L(E[\mathbf{X}])$

**예시**
- $\text{trace}(\mathbf{X})$: 선형 ✓
- $\|\mathbf{X}\|_F^2 = \text{trace}(\mathbf{X}^\top\mathbf{X})$: 이차형식이므로 선형 아님 ✗

$E[\text{trace}[(I - \Pi) \mathbf{e} \mathbf{e}^\top]] = \text{trace}[E[(I - \Pi) \mathbf{e} \mathbf{e}^\top]]$

**예시3**
유한한 합에 대해서는 $E\left[\sum_r c_rX_r\right] = \sum_r c_rE[X_r]$ 가 항상 성립한다. 여기서 $c_r$은 확률변수가 아닌 상수다. **확률변수들이 독립일 필요도 없다.**


먼저 $x^TAx=\sum_{i=1}^n\sum_{j=1}^n a_{ij}x_ix_j$ 이고 $x^TA\mu = \sum_{k=1}^n x_k(A\mu)_k.$

따라서 두 식을 곱하면

$$
\begin{aligned}
(x^TAx)(x^TA\mu)
&=
\left(\sum_{i,j}a_{ij}x_ix_j\right)
\left(\sum_k(A\mu)_kx_k\right).
\end{aligned}
$$

여기서 일반적인 합의 분배법칙 $\left(\sum_{i,j}u_{ij}\right) \left(\sum_kv_k\right) = \sum_{i,j,k}u_{ij}v_k$ 를 적용하면

$$
\begin{aligned}
(x^TAx)(x^TA\mu)
&=
\sum_{i,j,k}
(a_{ij}x_ix_j)((A\mu)_kx_k)\\
&=
\sum_{i,j,k}
a_{ij}(A\mu)_kx_ix_jx_k.
\end{aligned}
$$

따라서 기댓값을 취하면

$$
E[(x^TAx)(x^TA\mu)] = E\left[
\sum_{i,j,k}
a_{ij}(A\mu)_kx_ix_jx_k
\right].
$$

이제 **기댓값과 유한합은 순서를 바꿀 수 있다.**

$$
=\sum_{i,j,k} E\left[ a_{ij}(A\mu)_kx_ix_jx_k \right].
=\sum_{i,j,k} a_{ij}(A\mu)_k E[x_ix_jx_k].
$$

여기서 마지막으로 $x\sim N(0,V)$이므로 $E[x_ix_jx_k]=0$ 이고, 따라서

$$
\sum_{i,j,k}a_{ij}(A\mu)_k
\underbrace{E[x_ix_jx_k]}_{0}
=0.
$$

주의할 점은 **곱은 이렇게 분리할 수 없다는 것**이다. $X,Y$가 독립인 경우 등에만 $E[XY]=E[X]E[Y]$가 성립한다.


## 27. trace와 Variance

$\mathbf d^T\mathbf d=\text{tr}(\mathbf d\mathbf d^T)$인가

$$
\mathbf d=
\begin{pmatrix}
d_1\\
d_2\\
\vdots\\
d_p
\end{pmatrix}
$$

라고 하자. 그러면 $\mathbf d^T\mathbf d = d_1^2+d_2^2+\cdots+d_p^2$ 이다.

한편 외적 $\mathbf d\mathbf d^T$은

$$
\mathbf d\mathbf d^T =
\begin{pmatrix}
d_1^2&d_1d_2&\cdots&d_1d_p\\
d_2d_1&d_2^2&\cdots&d_2d_p\\
\vdots&\vdots&\ddots&\vdots\\
d_pd_1&d_pd_2&\cdots&d_p^2
\end{pmatrix}.
$$

이 행렬의 trace는 대각성분의 합이므로 $\text{tr}(\mathbf d\mathbf d^T) =d_1^2+d_2^2+\cdots+d_p^2=\mathbf d^T\mathbf d.$

따라서

$$
\boxed{
\|\mathbf d\|^2 =\mathbf d^T\mathbf d =\text{tr}(\mathbf d\mathbf d^T)
}
$$

**$E[\mathbf d\mathbf d^T]$와 공분산행렬의 관계**

확률벡터 $\mathbf d$의 평균을 $\boldsymbol\mu_d=E(\mathbf d)$ 라고 하자. 공분산행렬을 전개하면

$$
\begin{aligned}
\text{Cov}(\mathbf d)
&= E\left[
\mathbf d\mathbf d^T -\mathbf d\boldsymbol\mu_d^T -\boldsymbol\mu_d\mathbf d^T +\boldsymbol\mu_d\boldsymbol\mu_d^T
\right]\\
&= E[\mathbf d\mathbf d^T] -\boldsymbol\mu_d\boldsymbol\mu_d^T.
\end{aligned}
$$

따라서 $E[\mathbf d\mathbf d^T] = \text{Cov}(\mathbf d) + E(\mathbf d)E(\mathbf d)^T$  
이 식의 trace를 취하면

$$
E(\mathbf d^T\mathbf d) =\text{tr}\{\text{Cov}(\mathbf d)\} + \text{tr}\{E(\mathbf d)E(\mathbf d)^T\}.
$$

그런데 임의의 벡터 $\mathbf a$에 대해 $\text{tr}(\mathbf a\mathbf a^T) =\mathbf a^T\mathbf a =\|\mathbf a\|^2 $이므로

$$
\boxed{E\|\mathbf d\|^2 = \text{tr}\{\text{Cov}(\mathbf d)\} + \|E(\mathbf d)\|^2}
$$

### 추정량의 MSE 분해

이제 $\mathbf d=\hat{\boldsymbol\beta}-\boldsymbol\beta$ 라고 놓으면 $E(\mathbf d) = E(\hat{\boldsymbol\beta})-\boldsymbol\beta = \text{Bias}(\hat{\boldsymbol\beta}).$ 또한 $\boldsymbol\beta$는 상수이므로 $\text{Cov}(\mathbf d) = \text{Cov}(\hat{\boldsymbol\beta}).$

따라서 일반적인 벡터 추정량의 MSE는

$$
\boxed{
E\|\hat{\boldsymbol\beta}-\boldsymbol\beta\|^2
= \text{tr}\{\text{Cov}(\hat{\boldsymbol\beta})\} + \|\text{Bias}(\hat{\boldsymbol\beta})\|^2
}
$$

이를 성분별로 쓰면

$$
\boxed{
E\|\hat{\boldsymbol\beta}-\boldsymbol\beta\|^2
= \sum_{j=1}^p\text{Var}(\hat\beta_j) + \sum_{j=1}^p
\left(E[\hat\beta_j]-\beta_j\right)^2
}
$$

### OLS의 경우

OLS는 $E(\hat{\boldsymbol\beta})=\boldsymbol\beta$ 인 불편추정량이므로 
$E(\mathbf d)=0.$  
따라서 $E[\mathbf d\mathbf d^T] = \text{Cov}(\hat{\boldsymbol\beta})$ 이고,

$$
\boxed{
MSE(\hat{\boldsymbol\beta})
= \text{tr}\{\text{Cov}(\hat{\boldsymbol\beta})\}
= \sum_{j=1}^p\text{Var}(\hat\beta_j)
}
$$

### Ridge의 경우

Ridge 추정량은 일반적으로 편향되어 있으므로 $E(\mathbf d)\neq0.$

$$
E[\mathbf d\mathbf d^T] \neq \text{Cov}(\hat{\boldsymbol\beta}_R)
$$

이며 반드시

$$
\boxed{
MSE(\hat{\boldsymbol\beta}_R)
= \text{tr}\{\text{Cov}(\hat{\boldsymbol\beta}_R)\} + \|\text{Bias}(\hat{\boldsymbol\beta}_R)\|^2
}
$$

로 계산해야 한다.

### 일반적인 이차형식에서의 기댓값

더 일반적으로 확률벡터 $\mathbf z$와 상수행렬 $A$에 대해 $Q=\mathbf z^TA\mathbf z$ 라고 하자. 다음 trace 표현이 성립한다. $\mathbf z^TA\mathbf z =\text{tr}(\mathbf z^TA\mathbf z) =\text{tr}(A\mathbf z\mathbf z^T).$

따라서 $E(\mathbf z^TA\mathbf z) = \text{tr}\left(AE[\mathbf z\mathbf z^T]\right)$.  이때, $E[\mathbf z\mathbf z^T] = \Sigma+\boldsymbol\mu\boldsymbol\mu^T$ 이므로

$$
\boxed{
E(\mathbf z^TA\mathbf z) =
\text{tr}(A\Sigma) + \boldsymbol\mu^TA\boldsymbol\mu
}
$$

가 된다. 여기서 $\boldsymbol\mu=E(\mathbf z), \quad \Sigma=\text{Cov}(\mathbf z)$ 이다.

### 기댓값과 trace의 주요 공식

| 공식                                                                                   | 조건·의미          |
| ------------------------------------------------------------------------------------ | -------------- |
| $\text{tr}(A)=\sum_iA_{ii}$                                                | trace의 정의      |
| $\text{tr}(A+B)=\text{tr}(A)+\text{tr}(B)$                 | 선형성            |
| $\text{tr}(cA)=c\text{tr}(A)$                                      | $c$는 스칼라     |
| $\text{tr}(AB)=\text{tr}(BA)$                                      | 곱의 차원이 맞아야 함   |
| $\text{tr}(ABC)=\text{tr}(BCA)=\text{tr}(CAB)$             | 순환이동만 가능       |
| $E[\text{tr}(M)]=\text{tr}(E[M])$                                  | 유한한 해당 기댓값 필요  |
| $\mathbf x^TA\mathbf x=\text{tr}(A\mathbf x\mathbf x^T)$                   | 이차형식의 trace 표현 |
| $E[\mathbf x\mathbf x^T]=\text{Cov}(\mathbf x)+E[\mathbf x]E[\mathbf x]^T$ | 이차적률과 공분산 관계   |
| $E[\mathbf x^TA\mathbf x]=\text{tr}(A\Sigma)+\mu^TA\mu$                    | 이차형식의 기댓값      |
| $\text{tr}(Q^TAQ)=\text{tr}(A)$                                    | $Q$가 직교행렬일 때 |
| $\text{tr}(A)=\sum_i\lambda_i(A)$                                          | 중복도를 포함한 고윳값 합 |

주의할 점은 $E[\text{tr}(M)] = \text{tr}(E[M])$ 은 항상 선형성으로 성립하지만, 일반적으로 $E[AB]\neq E[A]E[B]$ 이다. 특히 $E[\mathbf d\mathbf d^T] \neq E[\mathbf d]E[\mathbf d]^T$ 이며, 두 행렬의 차이가 바로 공분산행렬이다.

### 고윳값과 trace
: linear_algebra_05_EigenValue_Diagonalization.md에 고윳값의 합과 곱 참고

## 15. 분산분해 (ANOVA Decomposition)

$$SST = SSR + SSE$$

즉

$$\sum (y_i-\bar{y})^2 = \sum (\hat{y}_i-\bar{y})^2 + \sum (y_i-\hat{y}_i)^2$$

**증명** 

회귀잔차 성질에 의해 $\sum(y_i-\hat{y}_i)(\hat{y}_i-\bar{y})=0$이므로

$$
\begin{aligned}
\sum (y_i-\bar{y})^2 &= \sum (y_i-\hat{y}_i+\hat{y}_i-\bar{y})^2 \\
&= \sum [(y_i-\hat{y}_i)+(\hat{y}_i-\bar{y})]^2 \\
&= \sum (y_i-\hat{y}_i)^2 + 2\sum(y_i-\hat{y}_i)(\hat{y}_i-\bar{y}) + \sum(\hat{y}_i-\bar{y})^2 \\
&= \sum (y_i-\hat{y}_i)^2 + \sum(\hat{y}_i-\bar{y})^2
\end{aligned}
$$

## 20. Conditioning Trick: $E[X] = E[E[X|Y]]$

$$
\begin{aligned}
\text{Var}(X) &= E(X^2) - E(X)^2 \\
&= E[E(X^2|Y)] - E[E(X|Y)]^2 \\
&= E[E(X^2|Y) - E(X|Y)^2] + E[E(X|Y)^2] - E[E(X|Y)]^2 \\
&= E[\text{Var}(X|Y)] + \text{Var}(E[X|Y])
\end{aligned}
$$


## 22. 대칭성 이용 (Symmetry)

### 기본 형태

확률밀도함수 $\phi(x)$가 0을 중심으로 대칭일 때:

$$\int_{-\infty}^{\infty} x \phi(x)dx = 0 \quad (\text{홀함수와 짝함수의 곱})$$

**증명**: $\phi(-x) = \phi(x)$ (짝함수)이고, $x$는 홀함수이므로 $x\phi(x)$는 홀함수. 따라서 대칭 구간에서의 적분은 0.

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

$$\frac{d}{d\theta}\int g(x;\theta)f(x)dx = \int \frac{\partial g(x;\theta)}{\partial\theta}f(x)dx$$

**특수 경우: Leibniz Rule**

$$\frac{d}{d\theta}\int_{a(\theta)}^{b(\theta)} g(x;\theta)dx = \int_{a(\theta)}^{b(\theta)} \frac{\partial g}{\partial\theta}dx + g(b;\theta)b'(\theta) - g(a;\theta)a'(\theta)$$


## 24. 사건 포함관계로 확률을 쪼개는 트릭 (Event Inclusion Bound)

확률수렴, Slutsky류 증명에서 자주 쓰는 트릭.  
핵심은 **복잡한 사건을 더 다루기 쉬운 두 사건의 합집합으로 포함시키는 것**이다.

### 기본 형태

임의의 $\varepsilon>0$, $M>0$에 대해  $\{|X_n Z_n|>\varepsilon\} \subset \{|X_n|>M\} \cup \left\{|Z_n|>\frac{\varepsilon}{M}\right\}$

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

  $$ \left\{\left|\frac{X_n}{Y_n}\right| > \varepsilon\right\} \subset \{|X_n| > \varepsilon/ M\} \cup \{|Y_n| < 1/M\}$$

증명: 대우로 보인다.  
예를 들어, $|X_n| \le M$이고 $|Z_n| \le \frac{\varepsilon}{M}$이면 $|X_n Z_n| \le M \cdot \frac{\varepsilon}{M} = \varepsilon$  
즉, $|X_n Z_n| > \varepsilon$가 되려면 둘 중 하나는 반드시 조건을 벗어나야 한다.

### 관련 증명 습관

- 직접 증명보다 **대우**를 먼저 본다.
- 사건 포함관계를 만들면 곧바로 **확률 부등식**으로 바꾼다.
- 임의의 $M$을 도입해 **bounded part**와 **small remainder**를 분리한다.
- 이후 union bound, convergence in probability, tightness와 연결한다.

## 로그-가중 적분 공식 ($\mathrm{Beta}(\theta,1)$ 핵심 항)

$$\theta>0,\qquad \int_0^1 \log x \cdot \theta x^{\theta-1}\,dx = -\frac{1}{\theta}$$

즉, $X\sim \mathrm{Beta}(\theta,1)$이면

$$E[\log X]=-\frac{1}{\theta},\qquad E[-\log X]=\frac{1}{\theta}$$

>**증명**
>
>$$u=x^\theta \;\Rightarrow\; du=\theta x^{\theta-1}dx,\quad \log x=\frac1\theta\log u \\
>\int_0^1 \log x\cdot \theta x^{\theta-1}dx
>= \frac1\theta\int_0^1 \log u\,du
>= \frac1\theta(-1)
>= -\frac1\theta$$
>
>>**적분 참고** 
>>
>>$\int_0^1\log u\,du=\lim_{a\downarrow0}\int_a^1\log u\,du
>>=\lim_{a\downarrow0}\left[u\log u-u\right]_a^1
>>=-1-\lim_{a\downarrow0}(a\log a-a)=-1.$
>>
>>왜냐하면 $a\log a\to0$이기 때문이다. 실제로 $a=1/t$로 두면
>>$a\log a=-\frac{\log t}{t}\to0$ ($t\to\infty$)이다.
>>
>>따라서 $\frac1\theta\int_0^1\log u\,du=-\frac1\theta$이다.

### 빠른 유도 2 (파라미터 미분)

$$\int_0^1 x^{\theta-1}dx=\frac1\theta$$

양변을 $\theta$로 미분하면 (analysis에 정리 8.4.6. 적분기호 속의 미분)

$$\int_0^1 x^{\theta-1}\log x\,dx=-\frac1{\theta^2}$$

여기에 $\theta$를 곱해

$$\int_0^1 \log x\cdot \theta x^{\theta-1}dx=-\frac1\theta$$

### 활용 포인트

- Beta/Gamma 계열 로그우도 미분(Score) 계산
- Fisher Information 계산
- $(0,1)$ 구간 로그모멘트 계산의 기본 블록


## 27. 미분 관련 기호 및 개념 총정리

### 1. 그래디언트 (Gradient) ∇

**정의**: 스칼라함수 $f: \mathbb{R}^n \to \mathbb{R}$의 편미분을 모은 열**벡터**

$$\nabla f(\mathbf{x}) = \begin{pmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{pmatrix}$$

**성질**
- $\nabla f$는 함수값이 가장 빠르게 증가하는 방향
- $\nabla f = \mathbf{0}$인 점이 극값(critical point)

**활용**: 최적화, MLE 계산

### 2. 헤시안 행렬 (Hessian Matrix) H, $H_f$

**정의**: 스칼라함수 $f: \mathbb{R}^n \to \mathbb{R}$의 2차 편미분을 모은 정사각행렬

$$H_f(\mathbf{x}) = \begin{pmatrix} 
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{pmatrix}$$

**성질**
- Schwarz 정리: 연속 2차 미분 조건 하에서 $H_f$ 대칭 ($H_f = H_f^T$)
- $H_f$의 고유값이 모두 양수 → $f$ 볼록(convex)
- $H_f$의 고유값이 모두 음수 → $f$ 오목(concave)

**활용**: Newton-Raphson 방법, 극값의 성질 판정, Fisher Information과 연결

### 3. 야코비안 행렬 (Jacobian Matrix) J, $J_f$

**정의**: 벡터함수 $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$의 모든 편미분을 모은 $m \times n$ 행렬

$$J_{\mathbf{f}}(\mathbf{x}) = \begin{pmatrix} 
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{pmatrix}$$

**특수 경우**
- $m=1$ (스칼라 함수): $J_f = (\nabla f)^T$ (행벡터)
- $n=1$ (곡선): $J_{\mathbf{f}} = (\mathbf{f}'(t))^T$

**활용**: 변수변환(change of variables), 야코비안 행렬식 → 확률변수 변환의 밀도함수

### 5. 라플라시안 (Laplacian) $\nabla^2$, $\Delta$

**정의**: 스칼라함수 $f: \mathbb{R}^n \to \mathbb{R}$의 2차 편미분의 합

$$\nabla^2 f = \Delta f = \sum_{i=1}^{n} \frac{\partial^2 f}{\partial x_i^2} = \text{trace}(H_f)$$

**성질**
- 헤시안의 대각합(trace)
- PDE(편미분방정식)의 기본 연산자

### 6. 발산 (Divergence) $\nabla \cdot \mathbf{F}$, $\text{div}(\mathbf{F})$

**정의**: 벡터장 $\mathbf{F}: \mathbb{R}^n \to \mathbb{R}^n$, $\mathbf{F} = (F_1, \ldots, F_n)^T$에 대해

$$\nabla \cdot \mathbf{F} = \sum_{i=1}^{n} \frac{\partial F_i}{\partial x_i}$$

**성질**
- 스칼라값 반환
- 벡터장의 "발산" 정도 측정

**활용**: 확률밀도함수 표준화 조건 (divergence theorem)

### 7. 회전 (Curl) $\nabla \times \mathbf{F}$, $\text{curl}(\mathbf{F})$

**정의** ($\mathbb{R}^3$): 벡터장 $\mathbf{F} = (F_1, F_2, F_3)^T$에 대해

$$\nabla \times \mathbf{F} = \begin{pmatrix} 
\frac{\partial F_3}{\partial x_2} - \frac{\partial F_2}{\partial x_3} \\
\frac{\partial F_1}{\partial x_3} - \frac{\partial F_3}{\partial x_1} \\
\frac{\partial F_2}{\partial x_1} - \frac{\partial F_1}{\partial x_2}
\end{pmatrix}$$

**활용**: 벡터 미분(덜 자주 사용)

### 8. 연쇄법칙 (Chain Rule) 형태별 정리

**스칼라 → 스칼라**

$$\frac{df}{dt} = \frac{\partial f}{\partial x}\frac{dx}{dt} + \frac{\partial f}{\partial y}\frac{dy}{dt}$$

**벡터 → 스칼라** (합성: $f \circ \mathbf{g}$)

$$\frac{df}{dt} = \nabla f(\mathbf{g}(t))^T \cdot \mathbf{g}'(t) = (\nabla f)^T \mathbf{g}'$$

**벡터 → 벡터** (합성: $\mathbf{f} \circ \mathbf{g}$)

$$J_{\mathbf{f} \circ \mathbf{g}}(\mathbf{x}) = J_{\mathbf{f}}(\mathbf{g}(\mathbf{x})) \cdot J_{\mathbf{g}}(\mathbf{x})$$

### 9. 2차 미분과 헤시안의 관계

**이계 방향미분 (directional second derivative)**

$$\mathbf{d}^T H_f(\mathbf{x}) \mathbf{d} = \lim_{h \to 0} \frac{f(\mathbf{x} + h\mathbf{d}) - 2f(\mathbf{x}) + f(\mathbf{x} - h\mathbf{d})}{h^2}$$

**Taylor 전개** (1차)

$$f(\mathbf{x} + \Delta \mathbf{x}) \approx f(\mathbf{x}) + (\nabla f)^T \Delta \mathbf{x}$$

**Taylor 전개** (2차)

$$f(\mathbf{x} + \Delta \mathbf{x}) \approx f(\mathbf{x}) + (\nabla f)^T \Delta \mathbf{x} + \frac{1}{2}(\Delta \mathbf{x})^T H_f (\Delta \mathbf{x})$$

### 10. 통계학에서의 미분 기호

| 기호 | 의미 | 용도 |
|------|------|------|
| $\frac{\partial \ell}{\partial \theta}$ | Score function | MLE 계산 |
| $H_{\ell} = -\frac{\partial^2 \ell}{\partial \theta^2}$ | Hessian of log-likelihood | Fisher Information |
| $\mathcal{I}(\theta) = E[H_\ell]$ | Fisher Information Matrix | 점근분포, 효율성 |
| $J(\hat{\theta}) = -H_\ell\|_{\hat{\theta}}$ | Observed Information | 수치 표준오차 계산 |

### 11. 행렬 미분 (Matrix Calculus) 기호

**벡터 w.r.t 벡터**

$$\frac{\partial \mathbf{f}}{\partial \mathbf{x}} = J_{\mathbf{f}}^T = \begin{pmatrix} \frac{\partial \mathbf{f}^T}{\partial x_1} \\ \vdots \\ \frac{\partial \mathbf{f}^T}{\partial x_n} \end{pmatrix}$$

**스칼라 w.r.t 행렬**

$$\frac{\partial f}{\partial \mathbf{X}} = \begin{pmatrix} 
\frac{\partial f}{\partial X_{11}} & \cdots & \frac{\partial f}{\partial X_{1n}} \\
\vdots & \ddots & \vdots \\
\frac{\partial f}{\partial X_{m1}} & \cdots & \frac{\partial f}{\partial X_{mn}}
\end{pmatrix}$$

**유용한 공식**
- $\frac{\partial}{\partial \mathbf{x}}(\mathbf{a}^T \mathbf{x}) = \mathbf{a}$
- $\frac{\partial}{\partial \mathbf{x}}(\mathbf{x}^T \mathbf{A} \mathbf{x}) = (\mathbf{A} + \mathbf{A}^T)\mathbf{x}$
- $\frac{\partial}{\partial \mathbf{X}}\text{trace}(\mathbf{AXB}) = \mathbf{A}^T \mathbf{B}^T$


# 주요 부등식 정리 (Summary of Key Inequalities)


## 로그 부등식 (Logarithmic Inequality)

모든 $t \geq 0$에 대해 $\log t \leq t - 1$  

등호는 $t = 1$일 때만 성립.

>**증명**
>
>$f(t) = t - 1 - \log t$로 정의하면 $f'(t) = 1 - \frac{1}{t} = \frac{t-1}{t}$
>
>- $0 < t < 1$일 때 $f'(t) < 0$ (감소)
>- $t > 1$일 때 $f'(t) > 0$ (증가)
>
>따라서 $t = 1$에서 최솟값 $f(1) = 0$을 가지므로 $f(t) \geq 0$, 즉 $\log t \leq t - 1$

**활용**

- KL divergence 비음성 증명
- 정보이론 부등식 기초
- MLE 수렴성 증명
- Gibbs 부등식의 선행 정리

**확장형**  
치환 $t \to t/\alpha$를 사용하면 일반화 가능.

$t > 0$에 대해 $\log t \leq \frac{t}{\alpha} - 1 + \log \alpha \quad (\alpha > 0)$


## 삼각부등식 (Triangle Inequality)
임의의 실수 $a, b$에 대해 $|a + b| \leq |a| + |b|$

확률변수 $X, Y$에 대해서도 $|X + Y| \leq |X| + |Y| \implies E[|X + Y|] \leq E[|X|] + E[|Y|]$

### 역삼각부등식 
TODO:

삼각부등식 $|a| \leq |a-b| + |b|$에서 $|a| - |b| \leq |a-b|$이고, $|b| - |a| \leq |b-a| = |a-b|$이므로

$$ -|a-b| \leq |a| - |b| \leq |a-b| $$

## 산술 기하 조화 부등식 (AM-GM-HM Inequality)
양의 실수 $a, b$에 대해 

$$\sqrt{ab} \leq \frac{a+b}{2} \leq \frac{2}{\frac{1}{a} + \frac{1}{b}}$$


## 절댓값 차이 부등식
임의의 실수 $a, b$에 대해

$$
||a| - |b|| \leq |a - b|
$$

**증명**  
역삼각부등식의 왼쪽, 오른쪽 부등식 두 경우를 합치면 $||a| - |b|| \leq |a-b|$.

확률변수 $X, Y$에 대해서도 $||X| - |Y|| \leq |X - Y|$

## 절댓값의 곱과 합 부등식
임의의 실수 $a, b$에 대해 $|ab| \leq \frac{a^2 + b^2}{2}$

이는 $2ab \leq a^2 + b^2$에서 유도된다.

## 최대/최소와 절댓값 부등식
임의의 실수 $a, b$에 대해

$$
\max(a, b) \leq |a| + |b|,\qquad \min(a, b) \geq -(|a| + |b|)
$$

## Bernoulli 부등식 (Bernoulli's Inequality)
$x > -1$, $r \geq 1$일 때

$$
(1 + x)^r \geq 1 + r x
$$


## Chernoff 부등식 (Chernoff Bound)
$X$ 임의의 확률변수, $t > 0$에 대해

$$
P(X \geq a) \leq \frac{E[e^{tX}]}{e^{ta}}
$$

## Kolmogorov 부등식 (부분합 최대치)
$S_n = X_1 + \cdots + X_n$이 독립이고 $E[X_i] = 0$이면

$$
P\left(\max_{1 \leq k \leq n} |S_k| \geq \lambda\right) \leq \frac{E[S_n^2]}{\lambda^2}
$$

## Paley–Zygmund 부등식
$X \geq 0$, $E[X^2] < \infty$, $0 < \theta < 1$일 때

$$
P(X \geq \theta E[X]) \geq (1-\theta)^2 \frac{(E[X])^2}{E[X^2]}
$$

## Jensen–Shannon 부등식 (정보이론)
두 분포 $P, Q$에 대해

$$
\frac{1}{2} D_{KL}(P \| M) + \frac{1}{2} D_{KL}(Q \| M) \leq \log 2
$$

($M = \frac{1}{2}(P+Q)$)

## Log-Sum 부등식
$a_i, b_i > 0$에 대해

$$
\sum_i a_i \log \frac{a_i}{b_i} \geq \left(\sum_i a_i\right) \log \frac{\sum_i a_i}{\sum_i b_i}
$$

## Gibbs 부등식 (상대엔트로피 비음성)
확률분포 $p, q$에 대해

$$
D_{KL}(p \| q) \geq 0
$$

등호는 $p = q$일 때만 성립.

## Bonferroni 부등식 (확률의 하한)
사건 $A_1, \ldots, A_n$에 대해

$$
P\left(\bigcup_{i=1}^n A_i\right) \geq \sum_{i=1}^n P(A_i) - \sum_{i<j} P(A_i \cap A_j)
$$

## Union Bound (Boole's Inequality)
임의의 사건 $A_1, \ldots, A_n$에 대해

$$
P\left(\bigcup_{i=1}^n A_i\right) \leq \sum_{i=1}^n P(A_i)
$$

## FKG 부등식 (양의 상관관계)
$X, Y$가 증가함수일 때

$$
E[XY] \geq E[X] E[Y]
$$

## 코시-슈바르츠 부등식 (Cauchy-Schwarz Inequality)

$a, b, c, d \in \mathbb{R}$에 대해

$$
ac + bd \leq \sqrt{a^2 + b^2} \sqrt{c^2 + d^2}
$$

확률변수 $X, Y$에 대해 $E[X^2], E[Y^2] < \infty$이면

$$
|E[XY]| \leq \sqrt{E[X^2]} \sqrt{E[Y^2]}
$$

Variance, Covariance 관련 부등식으로 표현하면, 

$$Var(X) \geq 0, \quad |Cov(X,Y)| \leq \sqrt{Var(X)} \sqrt{Var(Y)}$$

## 영(Young)의 부등식 (Young's Inequality)
$a, b \geq 0$, $p, q > 1$, $1/p + 1/q = 1$일 때

$$

Ab \leq \frac{a^p}{p} + \frac{b^q}{q}
$$
