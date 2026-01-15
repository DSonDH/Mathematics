### 정리 2.5.5 (분산행렬의 스펙트럼 분해와 기하학적 해석)
#### 대각화 정리 (Spectral Decomposition Theorem)
확률벡터 $\mathbf{X}\in\mathbb{R}^k$의 분산행렬 $\Sigma=\mathrm{Var}(\mathbf{X})$는 대칭이며 음이 아닌 정부호(non-negative definite)이므로, 다음과 같은 스펙트럼 분해(spectral decomposition)가 가능하다.

$$
\Sigma = P\Lambda P^T
$$

여기서
* $P=(p_1,\dots,p_k)$는 **직교행렬(orthogonal matrix)**, 즉 $P^TP=PP^T=I$를 만족한다.
* $\Lambda=\mathrm{diag}(\lambda_1,\dots,\lambda_k)$는 대각행렬이며, 대각성분 $\lambda_i$는 $\Sigma$의 **고유값(eigenvalue)**이다.
* $p_i$는 고유값 $\lambda_i$에 대응하는 **단위 고유벡터(unit eigenvector)**이다.

#### 음이 아닌 정부호 행렬의 조건
정리 2.5.4에 의해 $\Sigma$는 음이 아닌 정부호이므로, 모든 고유값이 음이 아니다.
$$
\lambda_i\ge 0,\quad i=1,\dots,k
$$

또한 다음이 성립한다.
* $\Sigma$가 **양의 정부호(positive definite)**  
  ⇔ 모든 고유값이 양수: $\lambda_i>0,\ \forall i$  
  ⇔ $\Sigma$가 정칙행렬(invertible)

* $\Sigma$가 **특이(singular)**  
  ⇔ 적어도 하나의 고유값이 0: $\exists i,\ \lambda_i=0$  
  ⇔ 확률변수들 사이에 선형종속 관계가 존재(확률 1로)

#### 분산행렬의 기하학적 해석
고유벡터와 고유값은 확률분포의 기하학적 구조를 나타낸다.

**(1) 주성분 방향(Principal Directions)**  
고유벡터 $p_i$는 데이터의 **주된 변동 방향(principal direction of variation)** 을 나타낸다.
* $p_1$: 가장 큰 고유값 $\lambda_1$에 대응하는 방향 → 분산이 가장 큰 방향
* $p_k$: 가장 작은 고유값 $\lambda_k$에 대응하는 방향 → 분산이 가장 작은 방향
* 고유벡터들은 서로 직교하므로, 주성분 방향들은 서로 수직이다.

**(2) 산포의 크기(Magnitude of Dispersion)**  
고유값 $\lambda_i$는 $p_i$ 방향으로의 **분산의 크기**를 나타낸다.
$$
\mathrm{Var}(p_i^T\mathbf{X})=p_i^T\Sigma p_i=p_i^T(P\Lambda P^T)p_i=\lambda_i
$$

즉, 확률벡터 $\mathbf{X}$를 고유벡터 $p_i$ 방향으로 정사영한 확률변수 $p_i^T\mathbf{X}$의 분산은 정확히 $\lambda_i$이다.

**(3) 총변동(Total Variation)**  
분산행렬의 대각합(trace)은 총변동을 나타낸다.
$$
\mathrm{tr}(\Sigma)=\sum_{i=1}^k\mathrm{Var}(X_i)=\sum_{i=1}^k\lambda_i
$$

고유값의 합은 원래 좌표계에서의 총분산과 같으며, 이는 좌표 변환에 불변(invariant)이다.

**(4) 일반화된 분산(Generalized Variance)**  
행렬식(determinant)은 다차원 산포의 부피를 나타낸다.
$$
\det(\Sigma)=\prod_{i=1}^k\lambda_i
$$

이는 확률분포의 **집중도(concentration)** 를 측정하는 지표로 사용되며,
* $\det(\Sigma)=0$이면 분포가 더 낮은 차원의 부분공간에 집중되어 있음을 의미한다.
* $\det(\Sigma)$가 클수록 분포가 더 퍼져 있음을 의미한다.

#### 예시 2.5.9 (분산행렬의 고유분석)
예시 2.5.7에서 얻은 분산행렬
$$
\Sigma=
\begin{pmatrix}
1/4 & 1/4\\
1/4 & 5/4
\end{pmatrix}
$$
의 고유값과 고유벡터를 구하자.

특성방정식(characteristic equation)은
$$
\det(\Sigma-\lambda I)
=\det\begin{pmatrix}
1/4-\lambda & 1/4\\
1/4 & 5/4-\lambda
\end{pmatrix}
=(1/4-\lambda)(5/4-\lambda)-1/16=0
$$

전개하면
$$
\lambda^2-\frac{3}{2}\lambda+\frac{1}{4}=0
$$

근의 공식으로
$$
\lambda=\frac{3/2\pm\sqrt{9/4-1}}{2}=\frac{3\pm\sqrt{5}}{4}
$$

따라서 고유값은
$$
\lambda_1=\frac{3+\sqrt{5}}{4}\approx 1.309,\quad
\lambda_2=\frac{3-\sqrt{5}}{4}\approx 0.191
$$

각 고유값에 대응하는 정규화된 고유벡터는
$$
p_1=\frac{1}{\sqrt{2+2\sqrt{5}}}
\begin{pmatrix}
1\\
\sqrt{5}
\end{pmatrix}
\approx
\begin{pmatrix}
0.447\\
1.000
\end{pmatrix}
(\text{정규화 전}),\quad
p_2=\frac{1}{\sqrt{2-2\sqrt{5}}}
\begin{pmatrix}
-\sqrt{5}\\
1
\end{pmatrix}
$$

**해석**
* $p_1$ 방향(대략 $(0.447, 1.000)$ 방향)이 가장 큰 분산($\lambda_1\approx 1.309$)을 갖는 주성분 방향이다.
* $p_2$ 방향은 $p_1$에 직교하며, 가장 작은 분산($\lambda_2\approx 0.191$)을 갖는다.
* 총분산: $\lambda_1+\lambda_2=3/2=\mathrm{Var}(X_1)+\mathrm{Var}(X_2)=1/4+5/4$
* 일반화된 분산: $\det(\Sigma)=\lambda_1\lambda_2=(3+\sqrt{5})(3-\sqrt{5})/16=4/16=1/4$

이는 $(X_1,X_2)$의 결합분포가 $p_1$ 방향으로 더 넓게 퍼져 있고, $p_2$ 방향으로는 상대적으로 집중되어 있음을 의미한다.

> **참고: 주성분 분석(Principal Component Analysis, PCA)**  
> 분산행렬의 고유벡터와 고유값을 이용한 좌표 변환
> $$
> \mathbf{Y}=P^T\mathbf{X}
> $$
> 을 **주성분 변환(principal component transformation)** 이라 하며,  
> 변환된 확률벡터 $\mathbf{Y}$의 성분들을 **주성분(principal components)** 이라 한다.
>
> 이 변환 후 분산행렬은 대각행렬이 되어
> $$
> \mathrm{Var}(\mathbf{Y})=P^T\Sigma P=\Lambda
> $$
> 즉, 주성분들은 서로 비상관(uncorrelated)이며,  
> 각 주성분의 분산은 고유값으로 주어진다.
>
> 이는 고차원 데이터의 차원 축소(dimensionality reduction)와  
> 특징 추출(feature extraction)의 기초가 되며,  
> 통계학, 기계학습, 데이터 과학에서 광범위하게 활용된다.

### 정리4.4.2 참고: 음이닌 정부호 행렬과 제곱근 행렬
실수가 원소인 $m\times m$ 대칭행렬 $\Sigma$에 대하여, 다음 조건들은 서로 동치이다:

1. $a^T\Sigma a \geq 0$ for all $a \in \mathbb{R}^m$ ($\Sigma$가 **음이닌 정부호 행렬(nonnegative definite matrix)**)
2. $\Sigma$의 모든 고유값이 0 이상
3. $\Sigma^{1/2}\Sigma^{1/2}=\Sigma$인 실수 대칭행렬 $\Sigma^{1/2}$가 존재

**증명**

**(1) ⇒ (2):**  
$\lambda$를 $\Sigma$의 고유값, $v$를 대응하는 고유벡터라 하자. 즉 $\Sigma v = \lambda v$, $v \neq 0$이다.

양변에 $v^T$를 왼쪽에서 곱하면
$$
v^T\Sigma v = v^T(\lambda v) = \lambda(v^Tv) = \lambda\|v\|^2
$$

조건 (1)에 의해 $v^T\Sigma v \geq 0$이고, $\|v\|^2 > 0$이므로 $\lambda \geq 0$이다.

**(2) ⇒ (3):**  
$\Sigma$는 실수 대칭행렬이므로 **스펙트럴 분해(spectral decomposition)**가 가능하다:
$$
\Sigma = Q\Lambda Q^T
$$
여기서 $Q$는 직교행렬($Q^TQ=I$), $\Lambda=\text{diag}(\lambda_1,\ldots,\lambda_m)$는 고유값 대각행렬이다.

조건 (2)에 의해 모든 $\lambda_i \geq 0$이므로, $\Lambda^{1/2}=\text{diag}(\sqrt{\lambda_1},\ldots,\sqrt{\lambda_m})$를 정의할 수 있다.

이제 $\Sigma^{1/2}=Q\Lambda^{1/2}Q^T$로 정의하면
$$
\Sigma^{1/2}\Sigma^{1/2} = (Q\Lambda^{1/2}Q^T)(Q\Lambda^{1/2}Q^T) = Q\Lambda^{1/2}(Q^TQ)\Lambda^{1/2}Q^T = Q\Lambda Q^T = \Sigma
$$

또한 $(\Sigma^{1/2})^T = (Q\Lambda^{1/2}Q^T)^T = Q(\Lambda^{1/2})^TQ^T = Q\Lambda^{1/2}Q^T = \Sigma^{1/2}$이므로 $\Sigma^{1/2}$는 대칭행렬이다.

**(3) ⇒ (1):**  
$\Sigma^{1/2}\Sigma^{1/2}=\Sigma$인 실수 대칭행렬 $\Sigma^{1/2}$가 존재한다고 하자.

임의의 $a \in \mathbb{R}^m$에 대하여
$$
a^T\Sigma a = a^T(\Sigma^{1/2}\Sigma^{1/2})a = a^T\Sigma^{1/2}(\Sigma^{1/2})^Ta = (\Sigma^{1/2}a)^T(\Sigma^{1/2}a) = \|\Sigma^{1/2}a\|^2 \geq 0
$$

따라서 $\Sigma$는 음이닌 정부호 행렬이다.
