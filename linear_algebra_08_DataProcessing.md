# 1. 우선순위 평가 (Priority Ranking)

## (1) 인접행렬 (Adjacency Matrix)

### 1) 개념

요소 간의 연결 관계를 나타내는 정사각 행렬을 말한다.
예) 연결 그래프가 있을 때 인접행렬 $A$는 다음과 같다.

$$
A=
\begin{pmatrix}
0 & 1 & 1\\
1 & 0 & 1\\
1 & 1 & 0
\end{pmatrix}
$$

### 2) 권위벡터와 허브벡터 (Authority Vector and Hub Vector)

$n\times n$ 인접행렬 $A=(a_{ij})$에 대하여

권위벡터 $u$ (authority vector)는
$u_i = \sum_{i=1}^n a_{ij}$ (즉, 한 칼럼 값들 더함 : 받은값)  
허브벡터 $v$ (hub vector)는
$v_j = \sum_{j=1}^n a_{ij}$ (즉, 한 행 값들 더함: 보낸값)  

각 성분은 권위가중치(authority weight)와 허브가중치(hub weight)라 한다.

## (2) 순위평가 원리 (Ranking Principle)
순위평가 반복식의 정규화 표현:  
인접행렬 $A$와 초기권위벡터 $u_0$, 초기허브벡터 $v_0$에 대하여 정규화된 권위벡터 $u_k$는
$$
u_k =
\begin{cases}
{u_0}, & k=0 \\
\dfrac{A^T v_{k}}{|A^T v_{k}|}, & k>0
\end{cases}
$$

정규화된 허브벡터 $v_k$는

$$
v_k =
\begin{cases}
{v_0}, & k=0 \\
\dfrac{A u_{k-1}}{|A u_{k-1}|}, & k>0
\end{cases}
$$
이다.  
정규화는 크기가 너무 커지거나 작아지는걸 방지하기 위함임.  

정규화된 $u_k$와 $v_k$의 점화식은 다음과 같다.

$$
u_k
= \frac{A^T v_{k}}{|A^T v_{k}|}
= A^T
\left(
\frac{A u_{k-1}}{|A u_{k-1}|}
\right)
\Bigg/
\left|
A^T
\left(
\frac{A u_{,k-1}}{|A u_{,k-1}|}
\right)
\right|
$$

즉,
$$
u_k=
\frac{(A^T A)u_{k-1}}{|(A^T A)u_{k-1}|}
$$
이고,
$$
v_k=\frac{(A A^T)v_{k-1}}{|(A A^T)v_{k-1}|}
$$
이다.  
(이 식을 코딩해서 쭉 돌리고,) 이 벡터들이 안정화되었다고 판단되는 시점부터
각각 최종 중요도(final importance)로 판단한다.

## (3) 사례 (Example)
10개의 인터넷 페이지 $p_1$~$p_{10}$ 간의 인접행렬 $A$가 다음과 같다고 한다.

$$
A=
\begin{pmatrix}
0&1&0&0&1&0&0&1&0&0\\
0&0&0&0&1&0&0&0&0&0\\
0&0&0&1&0&0&0&0&0&0\\
0&0&0&0&1&0&0&0&0&0\\
0&0&0&0&0&1&0&0&0&0\\
0&0&0&0&0&0&1&0&0&1\\
0&0&0&0&0&0&0&1&0&1\\
0&0&0&0&0&0&0&0&1&0\\
0&0&0&0&0&0&0&0&0&0\\
1&0&0&0&0&0&0&0&0&0
\end{pmatrix}
$$

반복 계산(iterative computation)에 따라 권위벡터가 안정화될 때까지 계산하면 다음 벡터들이 얻어진다.

예)

$$
u_5 =
\begin{pmatrix}
0.2722\\
0.2045\\
0.4165\\
0.2914\\
\vdots
\end{pmatrix},
\qquad
u_{10}=
\begin{pmatrix}
0.4199\\
0.2390\\
0.3535\\
0.2854\\
\vdots
\end{pmatrix}
$$

이로부터 권위가중치(authority weight)가 가장 높은 페이지는 $p_1$,
그 다음은 $p_3$, 그 다음은 $p_4$ 순임을 알 수 있다. (검색 엔진의 기초 원리)  

# 2. 자료압축 (Data Compression)
## (1) 특잇값 분해 (Singular Value Decomposition)
### 1) 분해 (Matrix Factorization)
한 행렬을 여러 행렬들의 곱으로 표현하는 것.  
예) QR분해(QR decomposition), LU분해(LU decomposition), LDU분해, 고유값분해(eigendecomposition),
해셴분해(Hessian decomposition), 슈르분해(Schur decomposition), 특잇값분해(SVD) 등.

### 2) 특잇값 (Singular Values)
$m\times n$ 행렬 $A$에 대하여 $A^TA$의 고윳값
$
\lambda_1,\lambda_2,\ldots,\lambda_n
$
이라 하면

$$
\sigma_1 = \sqrt{\lambda_1},\qquad
\sigma_2 = \sqrt{\lambda_2},\qquad
\ldots,\qquad
\sigma_n = \sqrt{\lambda_n}
$$

을 $A$의 특잇값(singular values)이라 한다.

### 3) 특잇값 분해 (Singular Value Decomposition)
정방행렬이 아닐 때에도
$$
A = U \Sigma V^T
$$

형태로 분해할 수 있다.
여기서 $U$와 $V$는 직교행렬(orthogonal matrices),
$\Sigma$는 특잇값이 대각에 놓이고 나머지 성분은 0인 $m \times n$ 대각행렬(diagonal matrix)이다.

예) A를 특잇값 분해 해보자
$$
A=
\begin{pmatrix}
1 & 1\\
0 & 1\\
1 & 0
\end{pmatrix}
$$

$$
U=
\begin{pmatrix}
\dfrac{\sqrt{6}}{3} & 0 & \dfrac{1}{\sqrt{3}}\\[4pt]
\dfrac{\sqrt{6}}{6} & \dfrac{\sqrt{2}}{2} & -\dfrac{1}{\sqrt{3}}\\[4pt]
\dfrac{\sqrt{6}}{6} & -\dfrac{\sqrt{2}}{2} & \dfrac{1}{\sqrt{3}}
\end{pmatrix}
$$
U 열벡터: $A\mathbf{v}_1 / \sigma_1$, $A\mathbf{v}_2 / \sigma_2$, ..., $A\mathbf{v}_n / \sigma_n$ 형태로 구해진다.  
U의 마지막 열벡터: 이전 열벡터들과 정규직교인 벡터  
U의 벡터들이 기저를 이룬다.  

$$
\Sigma=
\begin{pmatrix}
\sqrt{3} & 0\\
0 & 1\\
0 & 0
\end{pmatrix}
$$

$$
V^T=
\begin{pmatrix}
\dfrac{\sqrt{2}}{2} & \dfrac{\sqrt{2}}{2}\\[4pt]
\dfrac{\sqrt{2}}{2} & -\dfrac{\sqrt{2}}{2}
\end{pmatrix}
$$
V의 행들은 고유벡터에서 파생된 것.  

## (2) 축소된 특잇값 분해 (Reduced SVD)

특잇값 분해에서 0인 특잇값의 행 또는 열 제외한 형태를 축소된 특잇값 분해라 한다.

$$
A = U_r \Sigma_r V_r^T \\
= (u_1\ u_2\ \ldots\ u_r)
\begin{pmatrix}
\sigma_1 & 0 & \cdots & 0 \\
0 & \sigma_2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \sigma_r
\end{pmatrix}
\begin{pmatrix}
v_1^T \\
v_2^T \\
\vdots \\
v_r^T
\end{pmatrix}
$$
$r$은 랭크(rank)이다.

축소된 특잇값 분해를 사용하면
$$
A = \sigma_1 u_1 v_1^T + \sigma_2 u_2 v_2^T + \cdots + \sigma_r u_r v_r^T
$$

와 같이 중요한 성분들만을 이용하여 행렬을 근사할 수 있다.
$m\times n$을 $k\times n$으로 줄일 수 있다.  

## (3) 자료압축 원리 (Principle of Compression)
압축되지 않은 $m\times n$ 행렬 $A$를 저장하기 위한 공간은 $mn$이다.  
축소된 특잇값 분해를 사용하여
$$
A \approx \sigma_1 u_1 v_1^T + \cdots + \sigma_k u_k v_k^T
$$
라 하면, 저장해야 하는 용량은
$$
k + km + kn = k(1+m+n)
$$
이다.

$k$가 충분히 작으면
$$
k(1+m+n) \ll mn
$$
이므로 자료압축이 가능하다.
