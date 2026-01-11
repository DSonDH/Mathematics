# 1. 쐐기곱 (Wedge product)
## 1.1 해석학에서의 위치
쐐기곱은 다음 맥락에서 **표준 개념**이다.

* 다변수 해석학 (multivariable analysis)
* 미분형식 (differential forms)
* 적분론 (integration theory)
* 미분기하 (differential geometry)

특히
$$dx\,dy,\; dx\,dy\,dz$$
같은 표기는 **엄밀히는**
$$dx\wedge dy,\quad dx\wedge dy\wedge dz$$
의 축약 표기다.

## 1.2 1-형식과 2-형식

### 1-형식 (differential 1-form)

함수 $f(x_1,\dots,x_n)$에 대해
$$df=\sum_{i=1}^n \frac{\partial f}{\partial x_i}dx_i$$
는 1-형식이다. 이는 함수의 **전미분(total differential)** 이며, 각 방향으로의 변화율을 나타낸다.

예:
$$dx,\; dy,\; dr,\; d\theta$$
는 모두 1-형식이다. 각각은 해당 좌표 방향으로의 **무한소 변화**를 나타낸다.

### 2-형식 (differential 2-form)

두 1-형식을 쐐기곱하면 2-형식이 된다.
$$dx\wedge dy$$
이는 **방향과 면적을 가진 객체**다. 단순한 면적이 아니라, $x$축 방향과 $y$축 방향이 만드는 **방향 있는 평행사변형의 면적**을 의미한다.

## 1.3 쐐기곱의 정의와 성질

1-형식 $\alpha,\beta$에 대해 쐐기곱 $\alpha\wedge\beta$는 다음 성질을 가진다.

### (1) 쌍선형성

$$\left(a\alpha+b\beta\right)\wedge\gamma
= a\left(\alpha\wedge\gamma\right)+b\left(\beta\wedge\gamma\right)$$

이는 분배법칙이 성립함을 의미한다.

### (2) 반대칭성

$$\alpha\wedge\beta=-\beta\wedge\alpha$$

특히
$$\alpha\wedge\alpha=0$$
이다.

그래서
$$d\theta\wedge d\theta=0$$
이 된다. 이는 **같은 방향끼리는 면적을 만들 수 없다**는 기하학적 직관과 일치한다.

### (3) 기하적 의미

$$\alpha\wedge\beta$$
는 두 방향이 만드는 **방향 있는 면적 요소**다.

* 같은 방향 → 면적 0
* 독립 방향 → 면적 생성
* 순서 바꿈 → 부호 반전 (방향 반전)

## 1.4 왜 쐐기곱이 필요한가

다변수에서 면적과 부피는

* "길이의 곱"이 아니라
* **서로 다른 방향 성분이 만드는 부피**

이기 때문이다. 예를 들어, 두 벡터가 평행하면 그들이 만드는 평행사변형의 면적은 0이다.

그래서 단순 곱
$$dx\cdot dy$$
가 아니라
$$dx\wedge dy$$
가 등장한다. 쐐기곱은 이러한 **방향성과 독립성을 자동으로 추적**한다.

# 2. 변수변환 정리 (Change of variables theorem)

이제 쐐기곱을 이용해 변수변환을 정리한다.

## 2.1 변수변환의 핵심 구조

변환
$$T(u_1,\dots,u_n)=(x_1,\dots,x_n)$$
이 주어졌을 때, 적분에서 핵심 질문은 다음이다.

> $n$-차원 부피요소가 어떻게 변하는가?

답이 바로 **야코비안(Jacobian)** 이다. 야코비안은 **선형변환이 부피를 얼마나 확대/축소하는지**를 측정한다.

## 2.2 쐐기곱으로 본 야코비안

각 좌표에 대해 전미분을 취하면
$$dx_i=\sum_{j=1}^n \frac{\partial x_i}{\partial u_j}du_j$$
이므로, 이들을 모두 쐐기곱하면
$$dx_1\wedge\cdots\wedge dx_n
= \det\left(\frac{\partial(x_1,\dots,x_n)}{\partial(u_1,\dots,u_n)}\right)
du_1\wedge\cdots\wedge du_n$$

이 등식이 **야코비안의 본질**이다. 행렬식(determinant)이 자연스럽게 나타나는 이유는 쐐기곱의 반대칭성 때문이다.

즉,
$$\boxed{
dx_1\wedge\cdots\wedge dx_n
= J_T(u)\,du_1\wedge\cdots\wedge du_n
}$$

여기서 $J_T(u)$는 야코비 행렬식이다.

## 2.3 2차원 변수변환 정리

2차원에서는
$$(x,y)=(x(u,v),y(u,v))$$
일 때
$$dx\wedge dy
= \det
\begin{pmatrix}
\partial x/\partial u & \partial x/\partial v\\
\partial y/\partial u & \partial y/\partial v
\end{pmatrix}
du\wedge dv$$

이를 적분에 쓰면
$$\iint_D f(x,y)\,dx\,dy
= \iint_{D'} f(x(u,v),y(u,v))
\left|\det\frac{\partial(x,y)}{\partial(u,v)}\right|
du\,dv$$

절댓값 $|\det|$이 붙는 이유는 적분에서는 **부피의 크기**만 필요하고, 방향은 적분 영역의 방향으로 이미 결정되기 때문이다.

## 2.4 극좌표는 변수변환의 예제

$$x=r\cos\theta,\quad y=r\sin\theta$$

전미분을 계산하면:
$$dx=\cos\theta\,dr-r\sin\theta\,d\theta$$
$$dy=\sin\theta\,dr+r\cos\theta\,d\theta$$

쐐기곱을 계산하면:
$$\begin{aligned}
dx\wedge dy
&=(\cos\theta\,dr-r\sin\theta\,d\theta)
\wedge
(\sin\theta\,dr+r\cos\theta\,d\theta)\\
&=\cos\theta\sin\theta\,(dr\wedge dr)
+\cos^2\theta\,r\,(dr\wedge d\theta)\\
&\quad -r\sin^2\theta\,(d\theta\wedge dr)
-r^2\sin\theta\cos\theta\,(d\theta\wedge d\theta)\\
&=0 + r\cos^2\theta\,(dr\wedge d\theta)
+r\sin^2\theta\,(dr\wedge d\theta) + 0\\
&=r(\cos^2\theta+\sin^2\theta)\,dr\wedge d\theta\\
&=r\,dr\wedge d\theta
\end{aligned}$$

여기서 $dr\wedge dr=0$, $d\theta\wedge d\theta=0$, 그리고 $d\theta\wedge dr = -dr\wedge d\theta$를 사용했다.

따라서
$$dx\,dy=r\,dr\,d\theta$$
가 된다. 이것이 극좌표에서 야코비안이 $r$인 이유다.
