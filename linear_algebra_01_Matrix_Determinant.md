# 1강. 행렬과 행렬식

## 1. 행렬 (Matrix)

### (1) 용어정리

* **성분(element)**: 행렬 안에 배열된 구성원 (수, 원소)
* **행(row)**: 행렬의 가로줄
* **열(column)**: 행렬의 세로줄
* **$m \times n$ 행렬**: $m$개의 행과 $n$개의 열로 이루어진 행렬

#### 주요 용어

* **주대각선(main diagonal)**: 행렬의 왼쪽 위에서 오른쪽 아래로 가는 선
* **대각성분(diagonal element)**: 주대각선에 걸치는 항
* **영행렬(zero matrix)**: 모든 성분이 0인 행렬
* **전치행렬(transpose matrix)**: $A = (a_{ij})$에 대하여 $A^T = (a_{ji})$
* **대칭행렬(symmetric matrix)**: $A^T = A$인 행렬
* **정사각행렬(square matrix)**: 행의 개수와 열의 개수가 같은 행렬
* **단위행렬(identity matrix)**: 모든 대각성분이 1이고, 그 외의 성분은 0인 정사각행렬

### (2) 행렬의 연산

#### ① 덧셈과 뺄셈

$m \times n$ 행렬 $A = (a_{ij})$, $B = (b_{ij})$에 대해

$$A \pm B = (a_{ij} \pm b_{ij})$$

#### ② 상수배

상수 $c$에 대하여

$$cA = (ca_{ij})$$

#### ③ 곱셈

$m \times n$ 행렬 $A = (a_{ij})$, $n \times r$ 행렬 $B = (b_{jk})$일 때

$$AB = (c_{ik}), \quad c_{ik} = \sum_{j=1}^{n} a_{ij}b_{jk}$$

> ⚠️ 행렬의 곱셈은 **교환법칙이 성립하지 않는다.**
> 즉, $AB \neq BA$인 경우가 일반적이다.

>**주의: 행렬의 거듭제곱 표기**
>* $A^2 = A^\top A$를 뜻한다
>* $AA^T$는 $A$와 전치행렬의 곱으로, $A^2$와 다른 개념이다: $AA^T \neq A^2$ (일반적으로)


## 2. 연립일차방정식 (System of Linear Equations)

### (1) 행렬의 표현

예를 들어,

$$
\begin{cases}
x + 2y = 5 \\
2x + 3y = 8
\end{cases}
$$

는 행렬식으로

$$
\begin{pmatrix}
1 & 2 \\
2 & 3
\end{pmatrix}
\begin{pmatrix}
x \\ y
\end{pmatrix}
= \begin{pmatrix}
5 \\ 8
\end{pmatrix}
$$

으로 표현된다.

### (2) 가우스 조던 소거법 (Gauss–Jordan Elimination)
다음 세 가지의 기본 행 연산을 통해 연립방정식을 변환하여 해를 구한다.

1. 한 행을 상수배한다.
2. 한 행을 상수배하여 다른 행에 더한다.
3. 두 행을 맞바꾼다. (pivoting)  
주대각성분이 1이고 나머지 행 성분은 1, 맨 오른쪽 값은 남아있는 형태를 기약행사다리꼴이라 함.  
이는 각 행 별 답이 맨 오른쪽 값인 것.  
(참고: 이런 기본 행 연산을 행렬곱으로 표현 할 수 있고, 이 행렬을 기본연산행렬(elementary operator matrix)라 함. 이들은 행렬식 전개를 단순화 하는데 유용하게 쓰인다.)

### (3) 역행렬 이용
연립일차방정식 $AX = B$에서 $A$의 역행렬 $A^{-1}$이 존재하면

$$X = A^{-1}B$$

예:

$$
\begin{pmatrix}
1 & 2 \\
2 & 3
\end{pmatrix}
\begin{pmatrix}
x \\ y
\end{pmatrix}
= \begin{pmatrix}
5 \\ 8
\end{pmatrix}
\Leftrightarrow
\begin{pmatrix}
x \\ y
\end{pmatrix}
= \begin{pmatrix}
1 & 2 \\
2 & 3
\end{pmatrix}^{-1}
\begin{pmatrix}
5 \\ 8
\end{pmatrix}
$$

## 3. 행렬식 (Determinant)
### (1) 행렬식이란?
정사각행렬 $A$를 하나의 스칼라 값으로 대응시키는 함수. $\det A = |A|$

#### 기본 성질
* $0 \times 0$: $\det() = 0$
* $1 \times 1$: $\det(a) = a$
* $2 \times 2$: 

$$\det\begin{pmatrix} a & b \\ c & d \end{pmatrix} = ad - bc$$

$M_{ij}$는 i행 j열을 제외한 나머지 행렬성분들로 구성된 행렬의 행렬식 (Minor matrix 약자) 
* $3 \times 3$:
   
   $$
   \det\begin{pmatrix}
   a_{11} & a_{12} & a_{13} \\
   a_{21} & a_{22} & a_{23} \\
   a_{31} & a_{32} & a_{33}
   \end{pmatrix}
   = a_{11}M_{11} - a_{12}M_{12} + a_{13}M_{13}
   $$

* $4 \times 4$:
   
   $$\det A = a_{11}M_{11} - a_{12}M_{12} + a_{13}M_{13} - a_{14}M_{14}   $$

행을 기준으로 안하고, $a_{11}, a_{21}, a_{31}$같은 열을 기준으로 전개해도 무방.

### (2) 역행렬 (Inverse Matrix)
행렬식이 0이 아닌 정사각행렬 $A$의 역행렬은

$$
A^{-1} = \frac{1}{\det A}
\begin{pmatrix}
C_{11} & C_{21} & \cdots \\
C_{12} & C_{22} & \cdots \\
\vdots & \vdots & \ddots
\end{pmatrix}
$$

단, $C_{ij} = (-1)^{i+j}M_{ij}$ (여인수, cofactor)  
$M_{ij}$ : 소행렬식(minor)  

예:

$$
\begin{pmatrix}
a & b \\ c & d
\end{pmatrix}^{-1}
= \frac{1}{ad-bc}
\begin{pmatrix}
d & -b \\ -c & a
\end{pmatrix}
$$

AB = I 이면, AB = BA = I 이다.  

### (3) 크래머 공식 (Cramer's Rule)

연립일차방정식 $AX = B$에서, $A$가 행렬식이 0이 아닌 정사각행렬일 때
각 변수 $x_j$는 다음으로 구할 수 있다.

$$x_j = \frac{\det A_j}{\det A}$$

단, $A_j$는 $A$의 $j$번째 열을 $B$의 열로 바꾼 행렬이다.

### 행렬식의 기본성질들
#### 전치행렬의 행렬식

$$\det(A^T) = \det(A)$$

#### 역행렬의 행렬식
$A$의 역행렬 $A^{-1}$이 존재하면:

$$\det(A^{-1}) = \frac{1}{\det(A)}$$

#### 여인수 (Cofactor)
$C_{ij} = (-1)^{i+j}M_{ij}$로 정의되며, $i$번째 행과 $j$번째 열을 제거한 소행렬식에 부호를 붙인 값이다.

#### 행 연산과 행렬식
한 행(열)에 다른 행(열)의 배수를 더해도 행렬식 값은 변하지 않는다.

#### 곱의 행렬식

$$\det(AB) = \det(A)\det(B)$$

**따름정리**
* 직교행렬 $A$에서 $\det(A) = \pm 1$ 
    (∵ $AA^T = I$로부터 $\det(A^2) = \det(A)\det(A^T) = \det(I) = 1$)
* 멱등행렬 $A$에서 $\det(A) = 0$ 또는 $1$ 
    (∵ $A^2 = A$로부터 $\det(A^2) = \det(A)$, 즉 $(\det A)^2 = \det A$)

#### 대각 전개
행렬식을 특정 행이나 열을 기준으로 여인수를 이용하여 전개하는 방법

$$\det A = \sum_{j=1}^{n} a_{ij}C_{ij} = \sum_{i=1}^{n} a_
{ij}C_{ij}$$


**여인수를 이용한 역행렬 공식**  
$\det(A) \neq 0$일 때, 역행렬의 $(i,j)$ 성분은:

$$(A^{-1})_{ij} = \frac{(-1)^{i+j}M_{ji}}{\det(A)}$$

즉, $j$행 $i$열을 제거한 소행렬식에 부호를 붙이고 $\det(A)$로 나눈 값이다.  
**행렬 형태**:

$$A^{-1} = \frac{1}{\det(A)}\begin{pmatrix}
C_{11} & C_{21} & \cdots & C_{n1} \\
C_{12} & C_{22} & \cdots & C_{n2} \\
\vdots & \vdots & \ddots & \vdots \\
C_{1n} & C_{2n} & \cdots & C_{nn}
\end{pmatrix}$$

여기서 $C_{ij} = (-1)^{i+j}M_{ij}$는 여인수(cofactor)이며, **여인수행렬의 전치**를 $\det(A)$로 나눈다.

#### 라플라스 전개 (Laplace Expansion)
여러 행이나 열을 동시에 선택하여 행렬식을 전개하는 일반화된 방법

#### 행렬식의 합과 차
일반적으로 $\det(A + B) \neq \det(A) + \det(B)$이다.
단, 특정 행만 다른 경우:

$$\det\begin{pmatrix} a_1 + b_1 \\ a_2 \\ \vdots \end{pmatrix} = \det\begin{pmatrix} a_1 \\ a_2 \\ \vdots \end{pmatrix} + \det\begin{pmatrix} b_1 \\ a_2 \\ \vdots \end
{pmatrix}$$

### 반데르몬드 행렬식 (Vandermonde Determinant)
**정의**  
서로 다른 수 $x_1, x_2, \ldots, x_n$에 대한 반데르몬드 행렬(Vandermonde matrix) 에서,  
반데르몬드 행렬식 (Vandermonde Determinant)은

$$V = \begin{pmatrix}
1 & x_1 & x_1^2 & \cdots & x_1^{n-1} \\
1 & x_2 & x_2^2 & \cdots & x_2^{n-1} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_n & x_n^2 & \cdots & x_n^{n-1}
\end{pmatrix} \\
\det(V) = \prod_{1 \leq i < j \leq n} (x_j - x_i)$$

**성질 및 응용**
* $x_i = x_j$ ($i \neq j$)이면 $\det(V) = 0$ (행렬식 공식과 일치)
* **다항식 보간**: 반데르몬드 행렬은 라그랑주 보간(Lagrange interpolation)에서 핵심
* **선형독립성**: $x_1, \ldots, x_n$이 모두 다르면 $V$는 정칙 (계수 연립방정식의 유일해 존재)
* **수치해석**: 라그랑주 다항식의 계수 계산에 사용되지만, 수치 안정성이 낮음


## 4. 행렬 연산 정리

### (1) 분할행렬 (Partitioned Matrix)
행렬을 여러 개의 부분행렬(submatrix)로 나누어 표현한 것

#### 분할행렬의 전치
분할행렬 $A = \begin{pmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{pmatrix}$에 대해

$$A^T = \begin{pmatrix} A_{11}^T & A_{21}^T \\ A_{12}^T & 
A_{22}^T \end{pmatrix}$$

#### 분할행렬의 덧셈
분할행렬 $A = \begin{pmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{pmatrix}$, $B = \begin{pmatrix} B_{11} & B_{12} \\ B_{21} & B_{22} \end{pmatrix}$에 대해

$$A + B = \begin{pmatrix} A_{11}+B_{11} & A_{12}+B_{12} \\ 
A_{21}+B_{21} & A_{22}+B_{22} \end{pmatrix}$$
단, 각 블록의 차원이 일치해야 한다.

#### 분할행렬의 곱셈
$A = \begin{pmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{pmatrix}$, $B = \begin{pmatrix} B_{11} & B_{12} \\ B_{21} & B_{22} \end{pmatrix}$일 때

$$AB = \begin{pmatrix} A_{11}B_{11} + A_{12}B_{21} & A_{11}B_{12} + A_{12}B_{22} \\ A_{21}B_{11} + A_{22}B_{21} & A_
{21}B_{12} + A_{22}B_{22} \end{pmatrix}$$

단, 인접한 블록의 차원이 곱셈 조건을 만족해야 한다.

#### 분할행렬의 행렬식
$A = \begin{pmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{pmatrix}$이고 $A_{ij}$의 차수가 $n_i \times n_j$인 정사각행렬일 때:

* $A_{11}$이 정칙이면: 

$$\det(A) = \det(A_{11})\det(A_{22} - A_{21}A_{11}^{-1}A_
{12})$$

* $A_{22}$가 정칙이면: 

$$\det(A) = \det(A_{22})\det(A_{11} - A_{12}A_{22}^{-1}A_
{21})$$

* $A_{12} = 0$ (또는 $A_{21} = 0$)인 블록 삼각행렬: 

$$\det(A) = \det(A_{11})\det(A_{22})$$

**참고**: 우변의 $A_{22} - A_{21}A_{11}^{-1}A_{12}$ 또는 $A_{11} - A_{12}A_{22}^{-1}A_{21}$를 **슈어 보수행렬(Schur complement)** 이라 한다.

**증명** ($A_{11}$이 정칙인 경우):  
다음 행렬 변환을 고려한다:

$$\begin{pmatrix} I & 0 \\ -A_{21}A_{11}^{-1} & I \end{pmatrix}\begin{pmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{pmatrix} = \begin{pmatrix} A_{11} & A_{12} \\ 
0 & A_{22} - A_{21}A_{11}^{-1}A_{12} \end{pmatrix}$$

양변에 행렬식을 취하면:

$$\det\begin{pmatrix} I & 0 \\ -A_{21}A_{11}^{-1} & I \end{pmatrix} \cdot \det(A) = \det\begin{pmatrix} A_{11} & A_{12} \\ 0 & A_{22} - A_{21}A_{11}^{-1}A_{12} \end{pmatrix}
$$

좌측 블록 삼각행렬의 행렬식은 대각블록의 곱이므로:

$$\det\begin{pmatrix} I & 0 \\ -A_{21}A_{11}^{-1} & I \end
{pmatrix} = \det(I) \cdot \det(I) = 1$$

우측 블록 삼각행렬의 행렬식도:

$$\det\begin{pmatrix} A_{11} & A_{12} \\ 0 & A_{22} - A_{21}A_{11}^{-1}A_{12} \end{pmatrix} = \det(A_{11})\det(A_
{22} - A_{21}A_{11}^{-1}A_{12})$$

따라서:

$$\det(A) = \det(A_{11})\det(A_{22} - A_{21}A_{11}^{-1}A_
{12})$$

#### 분할행렬의 역행렬
$A = \begin{pmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{pmatrix}$이고 $A_{ij}$의 차수가 $n_i \times n_j$인 정사각행렬일 때, 역행렬은 다음과 같이 구할 수 있다.

**$A_{11}$이 정칙인 경우**  
($S = A_{22} - A_{21}A_{11}^{-1}A_{12}$, $G = A_{11} - A_{12}A_{22}^{-1}A_{21}$):

$$
A^{-1} = \begin{pmatrix} A_{11}^{-1} + A_{11}^{-1}A_{12}S^{-1}A_{21}A_{11}^{-1} & -A_{11}^{-1}A_{12}S^{-1} \\ -S^{-1}A_{21}A_{11}^{-1} & S^{-1} \end{pmatrix} \\
= \begin{pmatrix} G & - GA_{12}A_{22}^{-1} \\ -A_{22}^{-1}A_{21}G & A_{22}^{-1} + A_{22}^{-1}A_{21}GA_{12}A_{22}^{-1} \end{pmatrix} 
$$

**증명**:

$$\begin{pmatrix} I & 0 \\ -A_{21}A_{11}^{-1} & I \end{pmatrix}\begin{pmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{pmatrix}\begin{pmatrix} I & -A_{11}^{-1}A_{12} \\ 0 & I \end{pmatrix} = \begin{pmatrix} A_{11} & 0 \\ 0 & 
S \end{pmatrix}$$

따라서,

$$\begin{pmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{pmatrix}^{-1} = \begin{pmatrix} I & -A_{11}^{-1}A_{12} \\ 0 & I \end{pmatrix}\begin{pmatrix} A_{11}^{-1} & 0 \\ 0 & S^{-1} \end{pmatrix}\begin{pmatrix} I & 0 \\ A_{21}A_{11}^
{-1} & I \end{pmatrix}$$

### 특수한 행렬의 역행렬과 행렬식
#### 행렬 섭동 공식 (Matrix Perturbation Formulas)
**행렬식 보조정리 (Determinant Lemma)**

$m \times 1$ 열벡터 $a$, $b$에 대해:

$$\det(I + ab^T) = 1 + b^Ta$$

(귀납법으로도 증명할 수 있다...!)

**셔먼-모리슨 공식 (Sherman-Morrison Formula)**

$m \times 1$ 열벡터 $a$, $b$에 대해 $1 + b^Ta \neq 0$이면:

$$(I + ab^T)^{-1} = I - \frac{ab^T}{1 + b^Ta}$$

**일반화: 우드버리 행렬 항등식 (Woodbury Matrix Identity)**

행렬 $A$가 정칙이고, $U$는 $m \times k$, $V$는 $n \times k$ 행렬일 때:

$$(A + UV^T)^{-1} = A^{-1} - A^{-1}U(I_k + V^TA^{-1}U)^{-1}
V^TA^{-1}$$

### (2) 대각합 (Trace)
정사각행렬의 대각성분의 합

$$\text{tr}(A) = \sum_{i=1}^{n} a_{ii}$$

#### 대각합의 성질
* $\text{tr}(A^T) = \text{tr}(A)$
* $\text{tr}(A + B) = \text{tr}(A) + \text{tr}(B)$
* $\text{tr}(AB) = \text{tr}(BA)$
* $\text{tr}(ABC) = \text{tr}(BCA) = \text{tr}(CAB)$ (순환 성질)  
    **증명**:  
    임의의 $n \times n$ 행렬 $A, B, C$에 대해,
    
    $$\text{tr}(ABC) = \sum_{i=1}^n (ABC)_{ii}$$

    행렬곱의 성분은
    
    $$(ABC)_{ii} = \sum_{j=1}^n \sum_{k=1}^n a_{ij}b_{jk}c_{ki}$$

    따라서,
    
    $$\text{tr}(ABC) = \sum_{i=1}^n \sum_{j=1}^n \sum_{k=1}^n a_{ij}b_{jk}c_{ki}$$

    지수 $i, j, k$를 순환적으로 바꾸면,
    
    $$\sum_{i,j,k} a_{ij}b_{jk}c_{ki} = \sum_{j,k,i} b_{jk}c_{ki}a_{ij} = \sum_{k,i,j} c_{ki}a_{ij}b_{jk}$$

    즉,
    
    $$\text{tr}(ABC) = \text{tr}(BCA) = \text{tr}(CAB)$$
    
### 스칼라의 대각합 표현

임의의 벡터 $x \in \mathbb{R}^n$과 행렬 $A \in \mathbb{R}^{n \times n}$에 대해 다음이 성립한다:

$$x^TAx = \operatorname{tr}(Axx^T)$$

**설명**: $x^TAx$는 스칼라이고, 스칼라는 $1 \times 1$ 행렬이므로 자기 자신의 대각합과 같다. 여기에 대각합의 순환 성질을 적용하면 우변으로 변환할 수 있다.

**증명**:

**1단계: 좌변 전개**

$$x^TAx = \sum_{i=1}^{n} \sum_{j=1}^{n} x_i a_{ij} x_j$$

**2단계: 우변 전개 - 행렬 $Axx^T$의 $(i,i)$ 성분 계산**

먼저 행렬 $Axx^T$의 $(i,i)$ 성분을 계산한다:

$$(Axx^T)_{ii} = \sum_{j=1}^{n} a_{ij}(xx^T)_{ji}$$

여기서 $(xx^T)_{ji} = x_j x_i$이므로:

$$(Axx^T)_{ii} = \sum_{j=1}^{n} a_{ij} x_j x_i$$

**3단계: 대각합(trace) 계산**

$$\text{tr}(Axx^T) = \sum_{i=1}^{n} (Axx^T)_{ii} = \sum_{i=1}^{n} \sum_{j=1}^{n} a_{ij} x_j x_i$$

**4단계: 좌변과 우변이 같음을 확인**

$$\sum_{i=1}^{n} \sum_{j=1}^{n} a_{ij} x_j x_i = \sum_{i=1}^{n} \sum_{j=1}^{n} x_i a_{ij} x_j = x^TAx$$

(곱셈의 교환법칙 적용: $x_i a_{ij} x_j = a_{ij} x_j x_i$)

따라서:

$$x^TAx = \text{tr}(Axx^T)$$

#### 기댓값과 정사영행렬의 상호작용
정사영행렬 $\Pi$와 확률벡터 $\mathbf{e}$에 대해:

$$E[\mathbf{e}^\top (I - \Pi) \mathbf{e}] = \operatorname{trace}[(I - \Pi) E[\mathbf{e} \mathbf{e}^\top]]$$

**증명**:

$$E[\mathbf{e}^\top (I - \Pi) \mathbf{e}] = E[\operatorname
{trace}[\mathbf{e}^\top (I - \Pi) \mathbf{e}]]$$

대각합의 순환 성질을 이용하면:

$$= E[\operatorname{trace}[(I - \Pi) \mathbf{e} \mathbf{e}
^\top]]$$

기댓값과 대각합 연산의 순서를 바꾸면:

$$= \operatorname{trace}[E[(I - \Pi) \mathbf{e} \mathbf{e}
^\top]]$$

$(I - \Pi)$는 상수행렬이므로:

$$= \operatorname{trace}[(I - \Pi) E[\mathbf{e} \mathbf{e}
^\top]]$$

**경우 1**: $E[\mathbf{e}] = 0$이면 (중심화된 오차)

$$\operatorname{Cov}(\mathbf{e}) = E[\mathbf{e} \mathbf{e}
^\top]$$

따라서:

$$E[\mathbf{e}^\top (I - \Pi) \mathbf{e}] = \operatorname
{trace}[(I - \Pi) \operatorname{Cov}(\mathbf{e})]$$

**경우 2**: $\operatorname{Cov}(\mathbf{e}) = \sigma^2 I$ (등분산 오차)이면:

$$E[\mathbf{e}^\top (I - \Pi) \mathbf{e}] = \sigma^2 \operatorname{trace}(I - \Pi) = \sigma^2 [n - \operatorname
{rank}(\Pi)]$$

**응용 (회귀분석에서)**:
최소제곱추정에서 잔차제곱합의 기댓값:

$$E[\text{RSS}] = E[\mathbf{e}^\top (I - \Pi_X) \mathbf{e}] = \operatorname{trace}[(I - \Pi_X) \sigma^2 I] = 
\sigma^2(n - p)$$

여기서 $\Pi_X = X(X^TX)^{-1}X^T$는 $X$의 열공간으로의 정사영행렬이고, $p = \operatorname{rank}(X)$는 매개변수 개수이다.

따라서 불편추정량:

$$\hat{\sigma}^2 = \frac{\text{RSS}}{n-p}$$

### 전치와 덧셈

$$(A + B)^T = A^T + B^T$$

### 전치와 곱셈

$$(ABC)^T = C^T B^T A^T$$

일반적으로 $(A_1 A_2 \cdots A_n)^T = A_n^T \cdots A_2^T A_1^T$

### 분할행렬의 곱셈
$A = \begin{pmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{pmatrix}$, $B = \begin{pmatrix} B_{11} \\ B_{21} \end{pmatrix}$일 때

$$AB = \begin{pmatrix} A_{11}B_{11} + A_{12}B_{21} \\ A_{21}B_{11} + A_{22}B_{21} \end{pmatrix}$$

분할이 적절하게 이루어진 경우, 일반 행렬곱과 동일한 방식으로 계산 가능

### 하다마드 곱 (Hadamard Product)
같은 크기의 행렬 $A$, $B$에 대해 성분별 곱

$$A \circ B = (a_{ij} \cdot b_{ij})$$

### (5) 행렬의 직합 (Direct Sum)
블록 대각행렬 형태로 행렬을 결합

$$A \oplus B = \begin{pmatrix} A & 0 \\ 0 & B \end{pmatrix}$$

#### 직합의 일반화

$$A \oplus B \oplus C = \begin{pmatrix} A & 0 & 0 \\ 0 & B & 0 \\ 0 & 0 & C \end{pmatrix} \\
\bigoplus_{l=1}^{k} A_l = \text{diag}\{A_1, A_2, \ldots, 
A_k\}$$

#### 직합의 성질
* $(A \oplus B) + (C \oplus D) = (A+C) \oplus (B+D)$
* $(A \oplus B)(C \oplus D) = (AC) \oplus (BD)$

### (6) 행렬의 직적 (Direct Product)
크로네커 곱(Kronecker Product) 또는 쪼이푸스 곱(Zehfuss Product)이라고도 함

$A = (a_{ij})_{p \times q}$, $B$가 $m \times n$ 행렬일 때

$$A \otimes B = \begin{pmatrix} 
a_{11}B & a_{12}B & \cdots & a_{1q}B \\
a_{21}B & a_{22}B & \cdots & a_{2q}B \\
\vdots & \vdots & \ddots & \vdots \\
a_{p1}B & a_{p2}B & \cdots & a_{pq}B\end{pmatrix}$$

#### 직적의 성질
* **전치**: $(A \otimes B)^T = A^T \otimes B^T$
* **벡터의 직적**: $x^T \otimes y = yx^T$ (외적)
* **스칼라 곱**: $\lambda \otimes A = \lambda A = A \otimes \lambda$
* **곱셈**: $(A \otimes B)(X \otimes Y) = AX \otimes BY$ (단, 곱이 정의될 때)
* **대각행렬**: $D_k = \text{diag}\{d_1, d_2, \ldots, d_k\}$일 때
    
$$D_k \otimes A = d_1A \oplus d_2A \oplus \cdots 
\oplus d_kA$$

* **대각합**: $\text{tr}(A \otimes B) = \text{tr}(A) \cdot \text{tr}(B)$
* **행렬식**: $A$가 $p \times p$, $B$가 $m \times m$ 정방행렬일 때
    
$$\det(A \otimes B) = \det(A)^m \cdot \det(B)^p$$

> ⚠️ 분할행렬에 대해 $[A_1\ A_2] \otimes B = [A_1 \otimes B\ A_2 \otimes B]$는 성립하지만, $ A\otimes [B_1\ B_2] \neq [A \otimes B_1\ A \otimes B_2]$


## 정칙행렬 (Regular Matrix, Non-singular Matrix)
역행렬이 존재하는 정사각행렬

### 정칙행렬의 조건
정사각행렬 $A$가 정칙행렬이 되기 위한 필요충분조건:
* $\det(A) \neq 0$
* $A^{-1}$이 존재한다
* $AX = 0$의 유일한 해가 $X = 0$이다 (자명해만 존재)

### 정칙행렬의 성질
* $A$가 정칙행렬이면 $A^{-1}$도 정칙행렬이다
* $A$, $B$가 정칙행렬이면 $AB$도 정칙행렬이며, $(AB)^{-1} = B^{-1}A^{-1}$
* $A$가 정칙행렬이면 $\det(A^{-1}) = \frac{1}{\det(A)}$
* $A$가 정칙행렬이면 $A^T$도 정칙행렬이며, $(A^T)^{-1} = (A^{-1})^T$

### 특이행렬 (Singular Matrix)
역행렬이 존재하지 않는 정사각행렬 (즉, $\det(A) = 0$인 행렬)

## 역행렬

### 역행렬의 성질
* $(AB)^{-1} = B^{-1}A^{-1}$
* $(A^T)^{-1} = (A^{-1})^T$
* $A$가 정칙이면 $\det(A^{-1}) = \frac{1}{\det(A)}$
* $(A^{-1})^{-1} = A$
* $(A \otimes B)^{-1} = A^{-1} \otimes B^{-1}$
* $(A \oplus B)^{-1} = A^{-1} \oplus B^{-1}$

### 분할행렬의 행렬식과 역행렬
$A = \begin{pmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{pmatrix}$이고 $A_{ij}$의 차수가 $n_i \times n_j$라면

**행렬식**:
* $A_{11}$이 정칙이면: $\det(A) = \det(A_{11})\det(A_{22} - A_{21}A_{11}^{-1}A_{12})$
* $A_{22}$가 정칙이면: $\det(A) = \det(A_{22})\det(A_{11} - A_{12}A_{22}^{-1}A_{21})$

> 💡 **슈어 보수행렬 (Schur Complement)**  
> $S = A_{22} - A_{21}A_{11}^{-1}A_{12}$를 분할행렬 $A$에서의 $A_{11}$에 대한 슈어 보수행렬이라 하며, 이는 $A_{11}$이 정칙인 경우에만 정의된다. 비정칙이면 일반화 슈어 보수행렬로 정의한다.  
>
>관련 식: $I + X + X^2 + \cdots + X^{n-1} = (X^n - I)(X - I)^{-1} = (X - I)^{-1}(X^n - I)$

**역행렬**: $A^{-1} = \begin{pmatrix} B_{11} & B_{12} \\ B_{21} & B_{22} \end{pmatrix}$일 때
* $B_{11} = (A_{11} - A_{12}A_{22}^{-1}A_{21})^{-1}$
* $B_{12} = -B_{11}A_{12}A_{22}^{-1}$
* $B_{21} = -A_{22}^{-1}A_{21}B_{11}$
* $B_{22} = A_{22}^{-1} + A_{22}^{-1}A_{21}B_{11}A_{12}A_{22}^{-1}$


# 특수한 행렬
## (1) 대칭행렬 (Symmetric Matrix)
전치행렬이 자기 자신과 같은 정사각행렬

$$A^T = A$$

### 대칭행렬들의 곱
$(AB)^T = B^TA^T$
* 두 대칭행렬 $A$, $B$의 곱 $AB$가 대칭행렬이 되려면 $AB = BA$이어야 한다
* $A$가 대칭행렬이면 $A^n$도 대칭행렬이다

### $AA^T$와 $A^TA$의 성질
임의의 행렬 $A$에 대해:
* $AA^T$와 $A^TA$는 항상 대칭행렬이다
* $AA^T$의 $(i,j)$ 성분은 $A$의 $i$번째 행과 $j$번째 행의 내적
* $A^TA$의 $(i,j)$ 성분은 $A$의 $i$번째 열과 $j$번째 열의 내적
* $AA^T = 0$ (영행렬)이면 $A$는 영행렬이다
* $\text{tr}(AA^T) = 0$이면 $A$는 영행렬이다
* $\text{tr}(AA^T) = \sum_{i,j} a_{ij}^2$ (모든 성분의 제곱합)

**응용**:

$$PXX^T = QXX^T \Rightarrow PX = QX$$

## 기초벡터 (Standard Basis Vector)
$i$번째 성분만 1이고 나머지는 0인 벡터

$$\mathbf{e}_i = (0, \ldots, 0, 1, 0, \ldots, 0)^T$$

**성질**:
* $\mathbf{e}_i^T \mathbf{e}_j = \delta_{ij}$ (크로네커 델타)
* $\mathbf{I}_n = [\mathbf{e}_1\ \mathbf{e}_2\ \cdots\ \mathbf{e}_n]$
* 임의의 벡터 $x$는 $x = \sum_{i=1}^{n} x_i \mathbf{e}_i$로 표현
* $\mathbf{e}_i^T x = x_i$ ($x$의 $i$번째 성분 추출)
* $\mathbf{e}_i \mathbf{e}_j^T$는 $(i,j)$ 위치만 1이고 나머지는 0인 행렬

## 왜대칭행렬 (Skew-symmetric Matrix)
전치행렬이 자기 자신의 음수와 같은 정사각행렬

$$A^T = -A$$

**성질**:
* 왜대칭행렬의 대각성분은 모두 0이다
* 임의의 정사각행렬 $A$는 대칭행렬과 왜대칭행렬의 합으로 표현할 수 있다:
   
$$A = \frac{1}{2}(A + A^T) + \frac{1}{2}(A - A^T)$$

### 대칭행렬의 대각화
#### (1) 스펙트럼 정리 (Spectral Theorem)
**대칭행렬 $A$는 항상 직교행렬로 대각화 가능하다**  

$A$가 $n \times n$ 대칭행렬이면, 다음을 만족하는 직교행렬 $Q$와 대각행렬 $\Lambda$가 존재한다:

$$A = Q\Lambda Q^T$$

- $Q = [\mathbf{q}_1 \ \mathbf{q}_2 \ \cdots \ \mathbf{q}_n]$: 정규 고유벡터들로 이루어진 직교행렬
- $\Lambda = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_n)$: 고유값들의 대각행렬
- $Q^TQ = QQ^T = I$ (직교성)

#### (2) 대칭행렬의 고유값과 고유벡터
**성질 1**: 대칭행렬의 고유값은 모두 실수이다.

**증명**: $A = A^T$인 대칭행렬 $A$에 대해 $A\mathbf{v} = \lambda\mathbf{v}$ ($\lambda \in \mathbb{C}$, $\mathbf{v} \neq 0$)라 하자.

양변에 $\mathbf{v}^*$를 좌측에서 곱하면:

$$\mathbf{v}^*A\mathbf{v} = \lambda\mathbf{v}^*\mathbf{v}$$

양변의 켤레 전치를 취하면:

$$\mathbf{v}^TA^T\mathbf{v} = \bar{\lambda}\mathbf{v}^T\mathbf{v}$$

$A = A^T$이므로:

$$\mathbf{v}^TA\mathbf{v} = \bar{\lambda}\mathbf{v}^T\mathbf{v}$$

따라서:

$$\lambda\mathbf{v}^T\mathbf{v} = \bar{\lambda}\mathbf{v}^T\mathbf{v}$$

$\mathbf{v} \neq 0$이므로 $\mathbf{v}^T\mathbf{v} > 0$, 그러므로 $\lambda = \bar{\lambda}$, 즉 $\lambda \in \mathbb{R}$

**성질 2**: 서로 다른 고유값에 대응하는 고유벡터는 직교한다.

**증명**: $\lambda_i \neq \lambda_j$에 대응하는 고유벡터를 각각 $\mathbf{v}_i$, $\mathbf{v}_j$라 하면:

$$A\mathbf{v}_i = \lambda_i\mathbf{v}_i, \quad A\mathbf{v}
_j = \lambda_j\mathbf{v}_j$$

첫 번째 식에 $\mathbf{v}_j^T$를 좌측에서 곱하면:

$$\mathbf{v}_j^TA\mathbf{v}_i = \lambda_i\mathbf{v}_j^T\mathbf{v}_i$$

$A$가 대칭이므로:

$$\mathbf{v}_j^TA\mathbf{v}_i = (A\mathbf{v}_j)^T\mathbf{v}_i = \lambda_j\mathbf{v}_j^T\mathbf{v}_i$$

따라서:

$$\lambda_i\mathbf{v}_j^T\mathbf{v}_i = \lambda_j\mathbf{v}_j^T\mathbf{v}_i \\

(\lambda_i - \lambda_j)\mathbf{v}_j^T\mathbf{v}_i = 0$$

$\lambda_i \neq \lambda_j$이므로 $\mathbf{v}_i \perp \mathbf{v}_j$

#### (3) 대각화 과정
$n \times n$ 대칭행렬 $A$를 대각화하는 단계:

**단계 1**: $\det(A - \lambda I) = 0$을 풀어 고유값 $\lambda_1, \lambda_2, \ldots, \lambda_n$을 구한다.

**단계 2**: 각 고유값 $\lambda_i$에 대해 $(A - \lambda_i I)\mathbf{x} = 0$을 풀어 고유벡터를 구한다.

**단계 3**: 응집된 고유값(중복도가 2 이상)에 대해 그람-슈미트 정규직교화를 수행한다.

**단계 4**: 모든 고유벡터를 정규화(단위벡터로)하여 정규 고유벡터를 얻는다:

$$\mathbf{q}_i = \frac{\mathbf{v}_i}{\|\mathbf{v}_i\|}$$

**단계 5**: 정규 고유벡터들을 열로 하는 직교행렬 $Q = [\mathbf{q}_1 \ \mathbf{q}_2 \ \cdots \ \mathbf{q}_n]$를 구성한다.

**단계 6**: 고유값을 대각원소로 하는 대각행렬 $\Lambda = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_n)$를 구성한다.

**결과**: $A = Q\Lambda Q^T$

#### (4) 예제: $2 \times 2$ 대칭행렬의 대각화
**예제**: $A = \begin{pmatrix} 1 & 2 \\ 2 & 1 \end{pmatrix}$를 대각화하시오.

**1단계**: 고유값 구하기

$$\det(A - \lambda I) = \det\begin{pmatrix} 1-\lambda & 2 \\ 2 & 1-\lambda \end{pmatrix} \\
= (1-\lambda)^2 - 4 = \lambda^2 - 2\lambda - 3 = (\lambda 
- 3)(\lambda + 1) = 0$$

따라서 $\lambda_1 = 3$, $\lambda_2 = -1$

**2단계**: 고유벡터 구하기  
$\lambda_1 = 3$일 때:

$$(A - 3I)\mathbf{x} = \begin{pmatrix} -2 & 2 \\ 2 & -2 
\end{pmatrix}\begin{pmatrix} x_1 \\ x_2 \end{pmatrix} = 0$$

$-2x_1 + 2x_2 = 0 \Rightarrow x_1 = x_2$

고유벡터: $\mathbf{v}_1 = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$

$\lambda_2 = -1$일 때:

$$(A + I)\mathbf{x} = \begin{pmatrix} 2 & 2 \\ 2 & 2 \end
{pmatrix}\begin{pmatrix} x_1 \\ x_2 \end{pmatrix} = 0$$

$2x_1 + 2x_2 = 0 \Rightarrow x_1 = -x_2$

고유벡터: $\mathbf{v}_2 = \begin{pmatrix} 1 \\ -1 \end{pmatrix}$

**3단계**: 정규화  

$$\mathbf{q}_1 = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix}, \quad \mathbf{q}_2 = \frac{1}{\sqrt{2}}
\begin{pmatrix} 1 \\ -1 \end{pmatrix}$$

**4단계**: 직교행렬과 대각행렬 구성  

$$Q = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}, \quad \Lambda = \begin{pmatrix} 3 & 0 \\ 
0 & -1 \end{pmatrix}$$

**검증**:

$$Q\Lambda Q^T = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}\begin{pmatrix} 3 & 0 \\ 0 & -1 \end{pmatrix}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} \\
= \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}\begin{pmatrix} 3 & 3 \\ -1 & 1 \end{pmatrix} = \frac{1}{2}\begin{pmatrix} 2 & 4 \\ 4 & 2 \end{pmatrix} = \begin
{pmatrix} 1 & 2 \\ 2 & 1 \end{pmatrix} = A$$

#### (5) 대칭행렬 대각화의 응용
**응용 1**: 이차형식의 표준형  
대칭행렬 $A$에 대한 이차형식 $Q(\mathbf{x}) = \mathbf{x}^TA\mathbf{x}$는 $A = Q\Lambda Q^T$를 이용하여:

$$Q(\mathbf{x}) = \mathbf{x}^TQ\Lambda Q^T\mathbf{x}$$

$\mathbf{y} = Q^T\mathbf{x}$로 치환하면 ($\mathbf{x} = Q\mathbf{y}$):

$$Q(\mathbf{x}) = \mathbf{y}^T\Lambda\mathbf{y} = \sum_{i=1}^{n} \lambda_i y_i^2$$

그러므로:
- 모든 $\lambda_i > 0 \Rightarrow A$는 양정치
- 모든 $\lambda_i \geq 0 \Rightarrow A$는 양반정치
- $\lambda_i$가 양수와 음수를 섞어가짐 $\Rightarrow A$는 부정치

**응용 2**: 행렬의 거듭제곱  

$$A^k = Q\Lambda^k Q^T = Q\begin{pmatrix} \lambda_1^k & & 
\\ & \ddots & \\ & & \lambda_n^k \end{pmatrix}Q^T$$

따라서 $A^k$를 효율적으로 계산할 수 있다.

**응용 3**: 행렬 함수  

$$f(A) = Qf(\Lambda)Q^T = Q\begin{pmatrix} f(\lambda_1) & & \\ & \ddots & \\ & & f(\lambda_n) \end{pmatrix}Q^T$$

예: $\sqrt{A}$, $\log(A) = Q\begin{pmatrix} \log\lambda_1 & & \\ & \ddots & \\ & & \log\lambda_n \end{pmatrix}Q^T$

**응용 4**: 주성분분석(PCA)  
공분산행렬 $\Sigma$를 대각화하면, 고유벡터 방향이 주성분 방향이 되고 고유값이 분산을 나타낸다.

**응용 5**: 동역학계 안정성 분석  
선형 동역학계 $\frac{d\mathbf{x}}{dt} = A\mathbf{x}$ (단, $A$는 대칭)의 안정성은 고유값의 부호로 판단한다.

#### (6) 대칭행렬과 대각화의 핵심 성질
| 성질 | 설명 |
|---|---|
| **항상 대각화 가능** | 모든 대칭행렬은 직교행렬로 대각화 |
| **실수 고유값** | 대칭행렬의 모든 고유값은 실수 |
| **직교 고유벡터** | 서로 다른 고유값의 고유벡터는 항상 직교 |
| **완전집합** | $n \times n$ 대칭행렬은 $n$개의 선형독립 고유벡터를 가짐 |
| **직교대각화** | $A = Q\Lambda Q^T$ (일반 대각화 $A = PDP^{-1}$와 다름) |
| **계산 효율** | 행렬식 $\det(A) = \prod \lambda_i$, 대각합 $\text{tr}(A) = \sum \lambda_i$ 쉽게 계산 |



## (2) 합벡터 (Summing Vector)
모든 성분이 1인 벡터 $\mathbf{1}_n = (1, 1, \ldots, 1)^T$

### 합벡터의 성질
* $\mathbf{1}_n^T x = \sum_{i=1}^{n} x_i$ (벡터의 모든 성분의 합)
* $\mathbf{1}_n \mathbf{1}_n^T = \mathbf{J}_n$ (모든 성분이 1인 $n \times n$ 행렬)
* $A\mathbf{1}_n$은 $A$의 각 행의 합을 성분으로 하는 벡터
* $\mathbf{1}_n^T A$는 $A$의 각 열의 합을 성분으로 하는 행벡터

### $\mathbf{J}$ 행렬의 성질
$\mathbf{J}_{r \times s} = \mathbf{1}_r \mathbf{1}_s^T$ (모든 성분이 1인 $r \times s$ 행렬)

**정방행렬 $\mathbf{J}_n$의 성질**:
* $\mathbf{J}_n^2 = n\mathbf{J}_n$
* $\mathbf{J}_n^k = n^{k-1}\mathbf{J}_n$ (모든 양의 정수 $k$)
* $\text{rank}(\mathbf{J}_n) = 1$
* $\mathbf{J}_n$은 대칭행렬이다

### 평균행렬 (Averaging Matrix)

$$\bar{\mathbf{J}}_n = \frac{1}{n}\mathbf{J}_n$$

**평균행렬 성질**:
* $\bar{\mathbf{J}}_n^2 = \bar{\mathbf{J}}_n$ (멱등행렬)
* $\bar{\mathbf{J}}_n x = \bar{x}\mathbf{1}_n$ (여기서 $\bar{x} = \frac{1}{n}\sum x_i$는 평균)

### 중심화행렬 (Centering Matrix)

$$C_n = I_n - \bar{\mathbf{J}}_n = I_n - \frac{1}{n}\mathbf{J}_n$$

**중심화행렬 성질**:
* $C_n = C_n^T$ (대칭행렬)
* $C_n^2 = C_n$ (멱등행렬)
* $C_n\mathbf{1}_n = \mathbf{0}$
* $C_n\mathbf{J}_n = \mathbf{J}_n C_n = \mathbf{0}$
* $C_n x$는 $x$의 각 성분에서 평균을 뺀 벡터 (중심화된 벡터)

**예제**: 벡터 $x$에 대하여
* 평균: $\bar{x} = \frac{1}{n}\mathbf{1}_n^T x = \frac{1}{n}x^T\mathbf{J}_n x$
* 편차의 제곱합: $\sum(x_i - \bar{x})^2 = x^T C_n x$

**예제 2: 편차제곱합의 행렬 표현**

벡터 $W=(W_1,\dots,W_n)^T$, 합벡터 $\mathbf{1}=(1,\dots,1)^T$, 평균 $\bar W=\frac{1}{n}\mathbf{1}^TW$라 하면

$$S_{WW} = \sum_{i=1}^n (W_i-\bar W)^2 = W^TW-\frac{(\mathbf{1}^TW)^2}{\mathbf{1}^T\mathbf{1}}$$

또한 $\mathbf{1}^T\mathbf{1}=n$이므로

$$S_{WW} = W^T\!\left(I-\mathbf{1}(\mathbf{1}^T\mathbf{1})^{-1}\mathbf{1}^T\right)\!W = W^T\!\left(I-\frac{1}{n} \mathbf{1}\mathbf{1}^T\right)\!W = W^T C_n W$$

즉, 편차제곱합은 중심화행렬 $C_n$에 의한 이차형식으로 정리된다.


## (3) 멱등행렬 (Idempotent Matrix)
자신의 제곱이 자신과 같은 행렬

$$K^2 = K$$

### 기본 성질
* 모든 멱등행렬은 정사각행렬이다
* 단위행렬 $I$와 영행렬 $O$는 멱등행렬이다
* $K$가 멱등행렬이면 $I - K$도 멱등행렬이다
* $K$가 멱등행렬이면 임의의 양의 정수 $r$에 대해 $K^r = K$
* $K - I$는 일반적으로 멱등행렬이 아니다
* $K$가 멱등행렬이고 $K \neq I$이면 $K$는 특이행렬이다 (∵ $\det(K^2) = \det(K)$이므로 $(\det K)^2 = \det K$, 즉 $\det K = 0$ 또는 $1$. 만약 $\det K = 1$이면 $K$는 정칙이고 $K^2 = K$로부터 $K = I$)
* $K$의 고유값은 0 또는 1이다
* $\text{rank}(K) = \text{tr}(K)$
  - 일반적으로 행렬의 계수를 찾는것은 쉽지 않지만, 멱등행렬이면 tr(K)로 쉽게 계산가능.

- **멱등행렬의 곱**: 두 멱등행렬 $K_1$, $K_2$가 곱셈에 대해 교환가능하면 ($K_1K_2 = K_2K_1$), 그 곱 $K_1K_2$도 멱등행렬이다.
  - **증명**: $(K_1K_2)^2 = K_1K_2K_1K_2 = K_1K_1K_2K_2 = K_1K_2$

### 멱등행렬의 예
* $I - \bar{\mathbf{J}}_n$ (중심화행렬)
* $A(A^TA)^{-1}A^T$ ($A$가 열 최대계수 행렬일 때)
- 멱영행렬 (Nilpotent Matrix): $A^2 = O$인 행렬 A
- 멱항행렬 (Unipotent Matrix): $A^2 = I$인 행렬 A

## (4) 일반화역행렬 (Generalized Inverse)
행렬 $A$에 대해 $AGA = A$를 만족하는 행렬 $G$를 $A$의 일반화역행렬이라 하고 $A^-$로 표기한다.

### 성질
* $GA$는 멱등행렬이다 (증명: $(GA)^2 = GAGA = GA$)
* $AG$는 멱등행렬이다
* $A$가 정칙행렬이면 $A^- = A^{-1}$이다
* 일반화역행렬은 유일하지 않다

### 무어-펜로즈 역행렬 (Moore-Penrose Inverse)
다음 네 조건을 모두 만족하는 유일한 일반화역행렬 $A^+$:
1. $AA^+A = A$
2. $A^+AA^+ = A^+$
3. $(AA^+)^T = AA^+$
4. $(A^+A)^T = A^+A$


## (5) 직교행렬 (Orthogonal Matrix)
$A^TA = AA^T = I$를 만족하는 정사각행렬, 즉 $A^T = A^{-1}$

### 직교행렬의 성질
* 직교행렬의 행벡터들은 서로 직교하는 단위벡터이다
* 직교행렬의 열벡터들은 서로 직교하는 단위벡터이다
* $A$, $B$가 직교행렬이면 $AB$도 직교행렬이다

**성질 1: 고유값의 절댓값**

$$|\lambda| = 1$$

즉, 직교행렬의 모든 고유값은 단위원 위에 있다.

**성질 2: 대각합 불변성**

$$\text{tr}(P^TAP) = \text{tr}(A)$$

**성질 3: 행렬식 불변성**

$$\det(P^TAP) = \det(A)$$

**성질 4: 행렬식의 부호**

$$\det(P) = \pm 1$$

**성질 5: 거리 보존**

$$\|P\mathbf{x}\|_2 = \|\mathbf{x}\|_2, \quad \|PA\|_F = \|
A\|_F$$

**성질 6: 각도 보존**

$$\cos\theta = \frac{(P\mathbf{x})^T(P\mathbf{y})}{\|P\mathbf{x}\|_2 \|P\mathbf{y}\|_2} = \frac{\mathbf{x}
^T\mathbf{y}}{\|\mathbf{x}\|_2 \|\mathbf{y}\|_2}$$

### 참고: 정직교 행과 열을 가지는 행렬
**정규벡터** (단위벡터, normal vector또는 unit vector)
: norm이 1인 벡터.  
**직교** : 영벡터가 아닌 벡터 x, y에서 $x^Ty=0$일때. 
두 정규벡터가 직교하면 **정직교벡터(orthonormal vector)** 라 함.  
벡터집합(set of vectors): 같은 차수의 벡터들을 모아놓은 것.  
정직교집합(orthonormal set): 정규벡터들을 원소로 가지고, 임의의 서로 다른 두 벡터는 직교하는 집합.  

**정직교 행을 가지는 행렬**  
$P_{r \times c}$의 행벡터들이 정직교집합을 이룰 때, $P$는 **정직교 행을 가진다**(has orthonormal rows)고 하며, 이때

$$PP^T = I_r$$

이 성립한다. 그러나 일반적으로 $P^TP \neq I_c$이다.

**정직교 열을 가지는 행렬**  
$P_{r \times c}$의 열벡터들이 정직교집합을 이룰 때, $P$는 **정직교 열을 가진다**(has orthonormal columns)고 하며, 이때

$$P^TP = I_c$$

이 성립한다. 그러나 일반적으로 $PP^T \neq I_r$이다.

**직교행렬로의 확장**  
만약 $r = c$ (정사각행렬)이고 정직교 행(또는 열)을 가지면:

$$PP^T = P^TP = I$$

가 성립하며, 이때 $P$를 **직교행렬**(orthogonal matrix)이라 한다.

> 💡 **요약**
> * 정직교 행 $\Rightarrow$ $PP^T = I_r$
> * 정직교 열 $\Rightarrow$ $P^TP = I_c$  
> * 정사각행렬 + 정직교 행(또는 열) $\Rightarrow$ 직교행렬 ($PP^T = P^TP = I$)

### 그람-슈미트 정규직교화 (Gram-Schmidt Orthonormalization)
선형독립인 벡터들을 직교하는 단위벡터들로 변환하는 알고리즘. QR 분해, 직교행렬 구성, 대칭행렬 대각화 등에 필수적으로 사용된다.

주어진 벡터 $\{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_n\}$에서:
1. 첫 벡터를 정규화
2. 다음 벡터에서 이전 벡터들의 성분을 제거(사영 제거)
3. 결과를 정규화
4. 반복

**알고리즘**  
**Step 1**: 첫 번째 정규직교벡터

$$\mathbf{u}_1 = \mathbf{v}_1, \quad \mathbf{q}_1 = \frac
{\mathbf{u}_1}{\|\mathbf{u}_1\|}$$

**Step 2**: $k=2,3,\ldots,n$에 대해  
직교화 (제거 단계):

$$\mathbf{u}_k = \mathbf{v}_k - \sum_{i=1}^{k-1} \langle 
    \mathbf{v}_k, \mathbf{q}_i \rangle \mathbf{q}_i$$

정규화:

$$\mathbf{q}_k = \frac{\mathbf{u}_k}{\|\mathbf{u}_k\|}$$

**결과**: $\{\mathbf{q}_1, \mathbf{q}_2, \ldots, \mathbf{q}_n\}$은 정직교벡터들의 집합

**수식 정리**  
$k$번째 단계에서:

$$\mathbf{u}_k = \mathbf{v}_k - \sum_{i=1}^{k-1} (\mathbf
{v}_k \cdot \mathbf{q}_i)\mathbf{q}_i$$

여기서 $\mathbf{v}_k \cdot \mathbf{q}_i$는 $\mathbf{v}_k$를 $\mathbf{q}_i$ 방향으로 사영한 크기.

 예제: $\mathbb{R}^3$에서 3개 벡터

주어진 벡터:

$$\mathbf{v}_1 = \begin{pmatrix} 1 \\ 0 \\ 1 \end{pmatrix}, \quad \mathbf{v}_2 = \begin{pmatrix} 1 \\ 1 \\ 0 \end{pmatrix}, \quad \mathbf{v}_3 = \begin{pmatrix} 0 \\ 
1 \\ 1 \end{pmatrix}$$

**Step 1**: 

$$\mathbf{u}_1 = \begin{pmatrix} 1 \\ 0 \\ 1 \end{pmatrix}, \quad \|\mathbf{u}_1\| = \sqrt{2} \\
\mathbf{q}_1 = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 0 \\ 
1 \end{pmatrix}$$

**Step 2**:

$$\mathbf{v}_2 \cdot \mathbf{q}_1 = \frac{1}{\sqrt{2}}(1 + 0) = \frac{1}{\sqrt{2}} \\
\mathbf{u}_2 = \begin{pmatrix} 1 \\ 1 \\ 0 \end{pmatrix} - \frac{1}{\sqrt{2}} \cdot \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 0 \\ 1 \end{pmatrix} = \begin{pmatrix} 1/2 \\ 1 \\ -1/2 \end{pmatrix} \\
\|\mathbf{u}_2\| = \sqrt{1/4 + 1 + 1/4} = \sqrt{3/2} = \frac{\sqrt{6}}{2} \\
\mathbf{q}_2 = \frac{1}{\sqrt{6}}\begin{pmatrix} 1 \\ 2 \\ 
-1 \end{pmatrix}$$

**Step 3**: 

$$\mathbf{v}_3 \cdot \mathbf{q}_1 = \frac{1}{\sqrt{2}}, \quad \mathbf{v}_3 \cdot \mathbf{q}_2 = \frac{1}{\sqrt{6}} \\
\mathbf{u}_3 = \begin{pmatrix} 0 \\ 1 \\ 1 \end{pmatrix} - \frac{1}{\sqrt{2}} \cdot \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 0 \\ 1 \end{pmatrix} - \frac{1}{\sqrt{6}} \cdot \frac{1}{\sqrt{6}}\begin{pmatrix} 1 \\ 2 \\ -1 \end{pmatrix} \\
= \begin{pmatrix} 0 \\ 1 \\ 1 \end{pmatrix} - \begin{pmatrix} 1/2 \\ 0 \\ 1/2 \end{pmatrix} - \begin{pmatrix} 1/6 \\ 1/3 \\ -1/6 \end{pmatrix} = \begin{pmatrix} -2/3 \\ 2/3 \\ 2/3 \end{pmatrix} \\
\mathbf{q}_3 = \frac{1}{\sqrt{3}}\begin{pmatrix} -1 \\ 1 
\\ 1 \end{pmatrix}$$

**응용**  
**(1) QR 분해**

$$A = QR$$

여기서 $Q = [\mathbf{q}_1\ \mathbf{q}_2\ \cdots\ \mathbf{q}_n]$ (정직교 열), $R$은 상삼각행렬

**(2) 정사각행렬의 직교대각화**
대칭행렬의 고유벡터들이 중근을 가질 때, 같은 고유값의 고유벡터들을 직교화

**(3) 최소제곱 문제**

$$\arg\min_x \|Ax - b\|_2$$
QR 분해를 이용하면 $R$이 상삼각이므로 백대입으로 효율적 계산

 행렬 형태 표현

$A = [\mathbf{v}_1\ \mathbf{v}_2\ \cdots\ \mathbf{v}_n]$이면:

$$A = QR$$

여기서:
- $Q$: 정규직교화된 벡터들의 합성, 정직교 열을 가짐 ($Q^TQ = I$)
- $R$: 상삼각행렬, $r_{ij} = \mathbf{v}_j \cdot \mathbf{q}_i$ (사영 크기)


$$R = \begin{pmatrix}
\|\mathbf{u}_1\| & \mathbf{v}_2 \cdot \mathbf{q}_1 & \mathbf{v}_3 \cdot \mathbf{q}_1 & \cdots \\
0 & \|\mathbf{u}_2\| & \mathbf{v}_3 \cdot \mathbf{q}_2 & \cdots \\
0 & 0 & \|\mathbf{u}_3\| & \cdots \\
\vdots & \vdots & \vdots & \ddots
\end{pmatrix}$$

### 직교행렬 예시
**QR 분해 (QR Decomposition)**  
행렬 $A$가 $n \times p$ 차원이고 계수 $p$ ($n \geq p$)를 가진다고 하자. 그러면 다음을 만족하는 행렬들이 존재한다:

$$A = QR$$

* $Q$: $n \times p$ 차원이며 정직교 열을 가지는 행렬 ($Q^TQ = I_p$)
* $R$: $p \times p$ 차원 상삼각행렬 (upper triangular matrix)

**예시**: $A = \begin{pmatrix} 1 & 1 \\ 0 & 1 \\ 1 & 0 \end{pmatrix}$

그람-슈미트 과정을 통해:

$$Q = \begin{pmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{6}} \\ 0 & \frac{2}{\sqrt{6}} \\ \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{6}} \end{pmatrix}, \quad R = \begin{pmatrix} \sqrt{2} & \frac{1}{\sqrt{2}} \\ 0 & \sqrt{\frac
{3}{2}} \end{pmatrix}$$

검증: $Q^TQ = I_2$이고 $QR = A$

**헬머트 행렬 (Helmert Matrix)**  
$n \times n$ 헬머트 행렬 $H_n$은 첫 행이 $\frac{1}{\sqrt{n}}\mathbf{1}_n^T$이고, 나머지 행들이 순차적 대비(sequential contrast)를 나타내는 직교행렬이다.

3차원 헬머트 행렬의 예:

$$H_3 = \begin{pmatrix}
\frac{1}{\sqrt{3}} & \frac{1}{\sqrt{3}} & \frac{1}{\sqrt{3}} \\
\frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} & 0 \\
\frac{1}{\sqrt{6}} & \frac{1}{\sqrt{6}} & -\frac{2}{\sqrt{6}}
\end{pmatrix}$$

검증: 각 행의 norm이 1이고 서로 직교하므로 $H_3^TH_3 = I_3$

**기븐스 행렬 (Givens Matrix)**  
$(i,j)$ 평면에서 각도 $\theta$만큼 회전시키는 $n \times n$ 직교행렬. 단위행렬에서 $(i,i)$, $(i,j)$, $(j,i)$, $(j,j)$ 위치만 다음과 같이 변경:

$$G(i,j,\theta) = \begin{pmatrix}
1 & \cdots & 0 & \cdots & 0 & \cdots & 0 \\
\vdots & \ddots & \vdots & & \vdots & & \vdots \\
0 & \cdots & \cos\theta & \cdots & -\sin\theta & \cdots & 0 \\
\vdots & & \vdots & \ddots & \vdots & & \vdots \\
0 & \cdots & \sin\theta & \cdots & \cos\theta & \cdots & 0 \\
\vdots & & \vdots & & \vdots & \ddots & \vdots \\
0 & \cdots & 0 & \cdots & 0 & \cdots & 1
\end{pmatrix}$$

주로 QR 분해나 고유값 계산에서 특정 성분을 0으로 만드는 데 사용된다.

**하우스홀더 행렬 (Householder Matrix)**  
정방행렬을 삼각화하는 데 유용한 또 다른 직교행렬.  
단위벡터 $v$에 대해 $v$에 수직인 초평면에 대한 반사를 나타내는 행렬:

$$H = I - 2vv^T$$

**직교행렬임을 증명**:

$$H^TH = (I - 2vv^T)^T(I - 2vv^T) = (I - 2vv^T)(I - 2vv^T)\\
= I - 4vv^T + 4vv^Tvv^T \\= I - 4vv^T + 4v(v^Tv)v^T \\ 
= I - 4vv^T + 4vv^T = I$$

하우스홀더 행렬은 대칭이면서 직교행렬이므로 $H^T = H = H^{-1}$이다. 주로 벡터를 좌표축 방향으로 반사시켜 QR 분해를 수행하는 데 활용된다.

## (8) 제곱근 행렬 (Matrix Square Root)
정의: 정사각행렬 $A$에 대해 $B^2 = A$를 만족하는 행렬 $B$를 $A$의 제곱근 행렬이라 하고, $B = A^{1/2}$ 또는 $B = \sqrt{A}$로 표기한다.

존재성과 유일성  
**양정치행렬의 경우**:
대칭인 양정치행렬 $A \succ 0$에 대해 고유값 분해 $A = Q\Lambda Q^T$를 이용하면:

$$A^{1/2} = Q\Lambda^{1/2}Q^T$$

여기서 $\Lambda^{1/2} = \text{diag}(\sqrt{\lambda_1}, \sqrt{\lambda_2}, \ldots, \sqrt{\lambda_n})$이고, 모든 $\lambda_i > 0$이므로 항상 존재하며 유일하다 (양정치 범위에서).

**일반 행렬의 경우**:
- 일반적인 정사각행렬은 제곱근이 존재하지 않을 수 있다
- 제곱근이 존재하더라도 여러 개일 수 있다
- 예: $I$의 제곱근은 $I$ 뿐만 아니라 $-I$도 포함

성질  
* $(A^{1/2})^T = (A^T)^{1/2}$ (단, $A$가 대칭행렬일 때)
* $(A^{1/2})^{-1} = (A^{-1})^{1/2}$ (단, $A \succ 0$일 때)
* $(A^{1/2})^2 = A$
* $\det(A^{1/2}) = \sqrt{\det(A)}$ (단, $\det(A) \geq 0$일 때)
* $A$가 양정치이고 $B$도 양정치이면, $A^{1/2}B^{1/2}$과 $B^{1/2}A^{1/2}$는 일반적으로 다르다 (교환가능하지 않음)

### 계산 방법
**방법 1: 고유값 분해**
양정치 대칭행렬 $A$에 대해:
1. 고유값 분해: $A = Q\Lambda Q^T$
2. 제곱근 계산: $A^{1/2} = Q\Lambda^{1/2}Q^T$

**예제**:

$$A = \begin{pmatrix} 4 & 0 \\ 0 & 9 \end{pmatrix} \Rightarrow A^{1/2} = \begin{pmatrix} 2 & 0 \\ 0 & 3 \end
{pmatrix}$$

**방법 2: Newton-Schulz 반복** (수치 계산)
초기값 $X_0 = I$로 시작하여:

$$X_{n+1} = \frac{1}{2}\left(X_n + X_n^{-1}A\right)$$

이 반복은 $A^{1/2}$로 수렴한다.

**방법 3: Cholesky 분해**
양정치행렬 $A$에 대해 Cholesky 분해 $A = LL^T$를 이용하면:

$$A^{1/2} = L$$

여기서 $L$은 하삼각 분해 행렬이다.

### 응용
**확률통계에서의 응용**:
공분산행렬 $\Sigma$가 양정치이면, $\Sigma^{1/2}$는 다변량 정규분포의 표준화에 사용된다:

$$X \sim N(\mu, \Sigma) \Rightarrow \Sigma^{-1/2}(X - \mu) 
\sim N(0, I)$$

**수치해석에서의 응용**:
행렬의 정규화(scaling)와 조건수(condition number) 개선에 사용된다.

**Gram 행렬의 제곱근**:
$A = X^TX$ (Gram 행렬)에 대해:

$$A^{1/2} = (X^TX)^{1/2}$$

이는 $X$의 특이값과 관련된다.

### 관련 행렬 함수
- **부분 고유값**: $A^{1/3}$, $A^{1/4}$, $A^{\alpha}$ ($0 < \alpha < 1$)
- **행렬 로그**: $\log(A) = Q(\log(\Lambda))Q^T$ (양정치 대칭행렬)
- **행렬 지수**: $e^A = Q e^{\Lambda} Q^T$ (대칭행렬)


## (9) 정사영행렬 (Projection Matrix)
**정의**: 정사각행렬 $P$가 **정사영행렬(orthogonal projection matrix)** 이 되기 위한 필요충분조건은

$$P^2 = P \text{ 이고 } P^T = P$$

즉, 멱등이면서 대칭인 행렬이다.

**기하학적 의미** : 정사영행렬 $P$는 벡터 $x$를 어떤 부분공간 $S$에 수직으로 투영(projection)하는 선형변환을 나타낸다. 멱등이므로, 자신의 열공간으로의 정사영 행렬이다. 
- $\text{range}(P) = S$ (사영되는 부분공간)
- $\text{null}(P) = S^\perp$ (직교여공간)

### 정사영의 유일성과 표현
임의의 벡터 $x \in \mathbb{R}^m$에 대해 $x = x_1 + x_2$ 를 만족하는 $x_1 \in \text{Col}(A)$, $x_2 \perp \text{Col}(A)$ (즉, $x_2^T A = 0$)인 쌍 $(x_1, x_2)$가 유일하게 존재한다.

>**유일성 증명**:  
>$x = x_1 + x_2 = x_1' + x_2'$인 두 분해가 존재한다고 하자. 그러면:
>
>$$x_1 - x_1' = x_2' - x_2$$
>
>좌변은 $\text{Col}(A)$에 속하고 우변은 $\text{Col}(A)^\perp$에 속한다. 따라서:
>
>$$x_1 - x_1' \in \text{Col}(A) \cap \text{Col}(A)^\perp = \{\mathbf{0}\}$$
>
>그러므로 $x_1 = x_1'$이고 $x_2 = x_2'$로 유일하다.

**명시적 표현**:

$$x_1 = P_Ax, \quad x_2 = (I - P_A)x$$

여기서 $P_A = A(A^TA)^{-1}A^T$는 열공간 $\text{Col}(A)$로의 정사영행렬이다.
- $x_1 \in \text{Col}(A)$: $x_1 = P_Ax = A(A^TA)^{-1}A^Tx = A\beta$ (단, $\beta = (A^TA)^{-1}A^Tx$)
- $x_2 \perp \text{Col}(A)$: $A^T x_2 = A^T(I - P_A)x = [A^T - A^TP_A]x = [A^T - A^T]x = 0$
- **기하학적 의미**: $x_1 = P_Ax$는 벡터 $x$를 열공간 $\text{Col}(A)$에 **수직으로 투영**한 것이며, $x_2 = (I - P_A)x$는 투영되지 않은 **직교 잔차 성분**을 나타낸다.
- 직교 직합 표기하면, $x = x_1 \oplus x_2$

### 정사영행렬의 성질
정사영행렬 $P$에 대해 다음이 성립한다:

**성질 1: 사영의 불변성**  
$x \in \text{range}(P)$이면 $Px = x$
>**증명**: $x = Pw$인 $w$가 존재하므로 $Px = P(Pw) = P^2w = Pw = x$

**성질 2: 직교성**  
모든 $x$에 대해 $(x - Px) \perp \text{range}(P)$
>**증명**: $y \in \text{range}(P)$이면 $y = Pz$. 따라서:
>
>$$(x - Px)^T y = (x - Px)^T Pz = x^TPz - x^TP^TPz = x^TPz 
- x^TPz = 0$$

**성질 3: 거리 최소성**  
$\|x - Px\| = \min_{y \in \text{range}(P)} \|x - y\|$
>**증명**: $y \in \text{range}(P)$이면 $y = Pw$. $z = x - Px$라 하면:
>
>$$\|x - y\|^2 = \|z + (Px - y)\|^2$$
>
>$z \perp \text{range}(P)$이고 $(Px - y) \in \text{range}(P)$이므로 (피타고라스 정리):
>
>$$\|x - y\|^2 = \|z\|^2 + \|Px - y\|^2 \geq \|z\|^2 = \|x 
- Px\|^2$$
>
>등호는 $y = Px$일 때 성립.

**성질 4: 여사영의 정사영성**  
$P_\perp := I - P$도 정사영행렬
>**증명**:
>- 멱등성: $(I-P)^2 = I - 2P + P^2 = I - 2P + P = I - P$
>- 대칭성: $(I-P)^T = I - P^T = I - P$

**성질 5: 고유값**  
정사영행렬 $P$의 고유값은 1 또는 0 (멱등성에 의해)
- 고유값 1: $\text{range}(P)$의 방향
- 고유값 0: $\text{null}(P) = \text{range}(I-P)$의 방향

**성질 6: 계수와 대각합의 관계**  

$$\text{rank}(P) = \text{tr}(P)$$

>**증명**: 정사영행렬은 대칭이므로 정실수 고유값을 가지며, 특히 0과 1만 가짐. 

### 벡터의 직교 분해
**피타고라스 정리의 일반화**: 임의의 벡터 $x$에 대해

$$\|x\|_2^2 = \|Px\|_2^2 + \|(I-P)x\|_2^2$$

**증명**: $Px$와 $(I-P)x$는 직교하므로 (성질 2에서):

$$\|x\|^2 = \|Px + (I-P)x\|^2 = \|Px\|^2 + 2(Px)^T(I-P)x + 
\|(I-P)x\|^2$$

중간항을 계산하면: 

$$(Px)^T(I-P)x = x^TP(I-P)x = x^T(P -P^2)x = 0$$

($P = P^T$, $P^2 = P$ 이용)  

### 열벡터공간으로의 정사영행렬
**열 최대계수 행렬을 통한 정사영**  
$X$가 $m \times p$ 열 최대계수 행렬 (즉, $\text{rank}(X) = p \leq m$)일 때, 정사열행렬은:

$$P_X = X(X^T X)^{-1}X^T$$

>**참고: 열최대계수 행렬 (Full Column Rank Matrix)**  
>$X$가 $m \times n$ 행렬일 때, $\text{rank}(X) = n \leq m$이면 $X$를 **열최대계수 행렬(full column rank matrix)** 이라고 한다.
>* 모든 열벡터가 선형독립이다
>* $X^TX$는 정칙이다 ($\det(X^TX) > 0$)
>* $(X^TX)^{-1}$이 존재한다
>* 따라서 정사영행렬 $P_X = X(X^TX)^{-1}X^T$가 유일하게 정의된다
>
>**반대 개념**: 
>$\text{rank}(X) = m < n$이면 $X$를 **행최대계수 행렬(full row rank matrix)** 이라 한다.

>**참고: 하한 이하의 계수를 가지는 경우**  
>$X$가 $m \times n$ 행렬이고 $\text{rank}(X) = r < n$일 때, 정사영행렬은 **일반화역행렬(generalized inverse)** 을 사용하여 표현된다.
>
>**분해 방법**:  
>$X = (X_1 \mid X_2)$로 분할하되:
>* $X_1$: $m \times r$ 행렬이고 $\text{rank}(X_1) = r$ (최대 선형독립 열들)
>* $X_2$: $m \times (n-r)$ 행렬 (나머지 열들)
>
>**정사영행렬의 표현**:
>
>$$P_X = X \begin{pmatrix} (X_1^T X_1)^{-1} & 0 \\ 0 & 0 
\end{pmatrix} X^T$$
>
>
>여기서 블록 대각행렬을 $(X^TX)^-$로 표기하며, 이를 $X^TX$의 **일반화역행렬(generalized inverse)** 이라 하고, 간단히 표기하면
>
>$$P_X = X(X^TX)^- X^T$$
>
>**성질**:
>* $(X^TX)^-$는 유일하지 않다 (하지만 $X(X^TX)^-X^T$는 유일하다)
>* $\text{rank}(X(X^TX)^-X^T) = \text{rank}(X) = r$

**성질**:
* $P_X^2 = P_X$ (멱등성)
* $P_X^T = P_X$ (대칭성)
* $\text{rank}(P_X) = p$
* $\text{range}(P_X) = \text{Col}(X)$ (X의 열공간)
* $\text{null}(P_X) = \text{Col}(X)^\perp$ (X의 열공간의 직교여공간)

### 증명
정사영행렬 $P_X = X(X^TX)^{-1}X^T$의 유도

**Step 1**: 정사영의 정의

벡터 $y \in \mathbb{R}^m$을 $X$의 열공간 $\text{Col}(X)$에 정사영하면:

$$\hat{y} \in \text{Col}(X), \quad (y - \hat{y}) \perp \text{Col}(X)$$

기하학적으로 $\hat{y}$는 $y$에서 $\text{Col}(X)$로의 최단거리 점이다.

**Step 2**: 선형결합 표현 및 직교 조건

$\hat{y} \in \text{Col}(X)$이므로 어떤 계수벡터 $\beta$가 존재하여:

$$\hat{y} = X\beta$$

정사영의 필수조건은 잔차 $y - \hat{y}$가 $\text{Col}(X)$에 직교해야 한다는 것이다. 즉, 모든 $\text{Col}(X)$의 벡터는 $X$의 열벡터들의 선형결합이므로:

$$X^T(y - \hat{y}) = 0 \\
X^T(y - X\beta) = 0$$

**Step 3**: 정규방정식 풀이

위 식을 전개하면:

$$X^Ty - X^TX\beta = 0 \\
X^TX\beta = X^Ty$$

$X$가 열 최대계수 행렬이면 $X^TX$는 $p \times p$ 정칙행렬이므로 $(X^TX)^{-1}$이 존재한다. 따라서:

$$\beta = (X^TX)^{-1}X^Ty$$

**Step 4**: 정사영행렬 도출

정사영 벡터를 $\beta$ 형태로 표현하면:

$$\hat{y} = X\beta = X(X^TX)^{-1}X^Ty$$

정사영행렬은 이 변환을 나타내므로:

$$P_X = X(X^TX)^{-1}X^T$$

**Step 5**: 정사영행렬의 성질 검증

**멱등성**:

$$P_X^2 = X(X^TX)^{-1}X^T \cdot X(X^TX)^{-1}X^T = X(X^TX)^{-1}(X^TX)(X^TX)^{-1}X^T = X(X^TX)^{-1}X^T = P_X$$

**대칭성**:

$$P_X^T = [X(X^TX)^{-1}X^T]^T = X[(X^TX)^{-1}]^TX^T = X(X^TX)^{-1}X^T = P_X$$

이는 $X^T$와 $(X^TX)^{-1}$이 모두 대칭이기 때문이다.

**Step 6**: 기하학적 해석

- 모든 $x \in \text{Col}(X)$에 대해: $P_X x = x$ (이미 부분공간에 있으면 불변)
- 모든 $y$에 대해: $P_X y \in \text{Col}(X)$ (사영은 항상 열공간에 속함)
- 잔차: $(I - P_X)y \perp \text{Col}(X)$ (직교 성분 보존)
- 직합 분해: $y = P_X y \oplus (I - P_X)y$

**증명: 정사영행렬이 열벡터공간으로의 정사영을 수행**  
>$\Pi = X(X^TX)^{-1}X^T$가 $X$의 열벡터공간 $\text{Col}(X)$로의 정사영행렬임을 보이기 위해서는 다음 두 가지를 증명해야 한다:
>
>1. **$\Pi$는 정사영행렬이다**: $\Pi^2 = \Pi$이고 $\Pi^T = \Pi$
>2. **$\text{Col}(\Pi) = \text{Col}(X)$**: 사영의 치역이 $X$의 열공간과 같다
>
>**Step 1: 정사영행렬의 성질 증명**
>
>멱등성:
>
>$$\Pi^2 = X(X^TX)^{-1}X^T X(X^TX)^{-1}X^T = X(X^TX)^{-1}
X^T = \Pi$$
>
>
>대칭성:
>
>$$\Pi^T = [X(X^TX)^{-1}X^T]^T = X[(X^TX)^{-1}]^T X^T = X
(X^TX)^{-1}X^T = \Pi$$
>
>
>**Step 2: 열공간의 동치성 증명**  
>**(1) $\text{Col}(\Pi) \subseteq \text{Col}(X)$ 증명**  
>$y \in \text{Col}(\Pi)$이면 어떤 벡터 $v$에 대해:
>
>$$y = \Pi v = X(X^TX)^{-1}X^T v$$
>
>
>$X(X^TX)^{-1}X^T v = X\beta$ (단, $\beta = (X^TX)^{-1}X^T v$)로 쓸 수 있으므로:
>
>$$y = X\beta \in \text{Col}(X)$$
>
>
>따라서 $\text{Col}(\Pi) \subseteq \text{Col}(X)$
>
>**(2) $\text{Col}(X) \subseteq \text{Col}(\Pi)$ 증명**  
>$x \in \text{Col}(X)$이면 어떤 벡터 $w$에 대해 $x = Xw$이므로  
>
>$$\Pi x = X(X^TX)^{-1}X^T Xw = X(X^TX)^{-1}(X^TX)w = Xw = x$$
>
>
>즉, $\Pi x = x$이므로 $x = \Pi u \quad (\text{단, } u = x)$  
>따라서 $x \in \text{Col}(\Pi)$이고, $\text{Col}(X) \subseteq \text{Col}(\Pi)$
>
>
>두 포함관계로부터 $$\text{Col}(\Pi) = \text{Col}(X)$$
>
>
>**기하학적 해석**  
>$\Pi = X(X^TX)^{-1}X^T$는 $X$의 열벡터공간 $\text{Col}(X)$로의 정사영행렬이며:
>- 모든 $x \in \text{Col}(X)$에 대해 $\Pi x = x$ (이미 부분공간에 있으면 불변)
>- 모든 $y \notin \text{Col}(X)$에 대해 $\Pi y \in \text{Col}(X)$이며, $\Pi y = 0$은 $y \in \text{Col}(X)^\perp$일 때만 성립
>- 직교성: $(x - \Pi x) \perp \text{Col}(X)$ (잔차는 직교)

**직교여공간 (Orthogonal Complement)**  
벡터공간 $V \subseteq \mathbb{R}^n$의 **직교여공간**:

$$V^\perp = \{x \in \mathbb{R}^n : x^T v = 0 \text{ for 
all } v \in V\}$$

**성질**  
1. $V^\perp$는 $\mathbb{R}^n$의 부분공간
2. $(V^\perp)^\perp = V$ (이중 직교 여공간)
3. $\dim(V) + \dim(V^\perp) = n$ (차원 분해)
4. $V \cap V^\perp = \{\mathbf{0}\}$ (교집합)
5. $V \oplus V^\perp = \mathbb{R}^n$ (직합 분해)
- $\text{Col}(A) \perp \text{null}(A^T)$ (열공간과 영공간(좌영공간)의 직교성)
- $\text{Row}(A) \perp \text{null}(A)$ (행공간과 영공간의 직교성)

**부분공간 연관 관계**  
행렬 $A$ ($m \times n$)에 대해:

$$\mathbb{R}^n = \text{Col}(A^T) \oplus \text{null}(A) = \text{Row}(A) \oplus \text{null}(A) \\

\mathbb{R}^m = \text{Col}(A) \oplus \text{null}(A^T)$$

### 열벡터공간의 직교 분해 (Orthogonal Decomposition)
분할 정사영행렬 (Partitioned Projection Matrix) $X$가 $n \times k$ 행렬이고 $\text{rank}(X) = k$일 때,  
다음과 같이 분할한다: $X = (X_0 \mid X_1)$
- $X_0$: $n \times r$ 열 최대계수 행렬 ($r < k$)
- $X_1$: $n \times (k-r)$ 행렬

**용어: 전체 정사영행렬**:

$$\Pi_{0,1} = X(X^TX)^{-1}X^T$$

**용어: $X_0$로의 정사영행렬**:

$$\Pi_0 = X_0(X_0^TX_0)^{-1}X_0^T$$

**용어: $X_0$에 직교하는 성분**:

$$X_{1\mid0} = (I - \Pi_0)X_1$$

**용어: $X_{1\mid0}$의 열공간으로의 정사영행렬**:

$$\Pi_{1\mid 0} = X_{1\mid0}(X_{1\mid0}^TX_{1\mid0})^{-}X_
{1\mid0}^T$$

여기서 $(·)^-$는 일반화역행렬(generalized inverse)를 나타낸다.

**성질 1: 열공간의 동치성**  

$$\text{Col}((X_0, X_1)) = \text{Col}((X_0, X_{1\mid0}))$$

>**증명**:
>**(1) $\text{Col}(X_0, X_1) \subseteq \text{Col}(X_0, X_{1\mid0})$ 증명**
>
>$Y \in \text{Col}(X_0, X_1)$이면 어떤 벡터 $\beta_0, \beta_1$에 대해:
>
>$$Y = X_0\beta_0 + X_1\beta_1$$
>
>
>이를 다음과 같이 다시 쓸 수 있다:
>
>$$Y = X_0\beta_0 + \Pi_0 X_1\beta_1 + (I - \Pi_0)
X_1\beta_1$$
>
>$$= X_0\beta_0 + X_0(X_0^TX_0)^{-1}X_0^T X_1\beta_1 + (I - \Pi_0)X_1\beta_1$$
>
>$$= X_0[\beta_0 + (X_0^TX_0)^{-1}X_0^T X_1\beta_1] + X_
{1\mid0}\beta_1$$
>
>
>첫 번째 항은 $\text{Col}(X_0)$에 속하고, 두 번째 항은 $\text{Col}(X_{1\mid0})$에 속하므로:
>
>$$Y \in \text{Col}(X_0, X_{1\mid0})$$
>
>
>**(2) $\text{Col}(X_0, X_{1\mid0}) \subseteq \text{Col}(X_0, X_1)$ 증명**
>
>$X_{1\mid0} = (I - \Pi_0)X_1$이고 $(I - \Pi_0)$는 멱등행렬이므로:
>
>$$Y = X_0\alpha + X_{1\mid0}\beta = X_0\alpha + (I - 
\Pi_0)X_1\beta \in \text{Col}(X_0, X_1)$$
>
>
>따라서 두 열공간이 같다: $\text{Col}(X_0, X_1) = \text{Col}(X_0, X_{1\mid0})$

**성질 2: 직교여공간 분해**  
- $(X_0, X_1)$의 열공간은 $X_0$의 열공간과 $X_{1\mid0}$의 열공간의 직합(orthogonal direct sum)으로 표현된다.

$$\text{Col}(X_0, X_1) = \text{Col}(X_0) \oplus \text{Col}
(X_{1\mid0})$$

**성질 3: 정사영행렬의 분해**  

$$\Pi_{0,1} = \Pi_0 + \Pi_{1\mid0} \\ \Pi_0 \Pi_{1\mid0} = 
0$$
>
>**증명**:
>성질1, 2로부터 자명하지만, 수식으로 증명하면 아래와 같다.  
>분할된 정사영행렬의 성질을 이용한다. $\Pi_{0,1}$를 계산하기 위해, $(X_0, X_1)$을 블록 형태로 다루면:
>
>
>$$X^TX = \begin{pmatrix} X_0^TX_0 & X_0^TX_1 \\ 
X_1^TX_0 & X_1^TX_1 \end{pmatrix}$$
>
>
>역행렬은 (슈어 보수행렬 공식 사용):
>
>$$(X^TX)^{-1} = \begin{pmatrix} (X_0^TX_0)^{-1} + (X_0^TX_0)^{-1}X_0^TX_1S^{-1}X_1^TX_0(X_0^TX_0)^{-1} & -(X_0^TX_0)^{-1}X_0^TX_1S^{-1} \\ -S^{-1}X_1^TX_0(X_0^TX_0)^{-1} & S^{-1} \end{pmatrix}$$
>
>여기서 $S = X_{1\mid0}^TX_{1\mid0}$는 슈어 보수행렬이다.  
>따라서:
>
>$$\Pi_{0,1} = \begin{pmatrix} X_0 & X_1 \end{pmatrix}(X^TX)^{-1}\begin{pmatrix} X_0^T \\ X_1^T \end{pmatrix} \\ = X_0(X_0^TX_0)^{-1}X_0^T + (I - \Pi_0)X_1[(I - \Pi_0)X_1]^T(I - \Pi_0)X_1]^{-}(I - \Pi_0)X_1^T$$
>
>$$= \Pi_0 + \Pi_{1\mid0}$$
>
>직교성:
>
>$$\Pi_0 \Pi_{1\mid0} = X_0(X_0^TX_0)^{-1}X_0^T \cdot X_
{1\mid0}[X_{1\mid0}^TX_{1\mid0}]^{-}X_{1\mid0}^T$$
>
>$X_{1\mid0} = (I - \Pi_0)X_1$이므로 $X_0^T X_{1\mid0} = 0$.  
>따라서 $\Pi_0 \Pi_{1\mid0} = 0$

#### 응용: 순차적 분할 정사영
**호텔링 여인수 (Hotelling Deflation)**  
회귀분석에서 $X_0$의 영향을 제거한 후 $X_1$의 영향을 추정할 때 사용:

$$\hat{\beta}_1 = (X_{1\mid0}^TX_{1\mid0})^{-}X_{1\mid0}^T
(I - \Pi_0)Y$$

**분산 분해**  
총 변량을 $X_0$에 의한 설명 부분과 나머지 부분으로 분해:

$$Y^T\Pi_{0,1}Y = Y^T\Pi_0 Y + Y^T\Pi_{1\mid0}Y$$

이는 ANOVA나 순차적 모형 비교에서 핵심적으로 사용된다.

**부분상관(Partial Correlation)**  
$X_0$의 효과를 제거한 후 $X_1$과 $Y$의 상관을 구할 때:

$$r_{\text{partial}} = \frac{\text{Cov}[(I - \Pi_0)Y, (I - \Pi_0)X_1]}{\sqrt{\text{Var}[(I - \Pi_0)Y] \cdot \text{Var}
[(I - \Pi_0)X_1]}}$$


### 대칭 멱등행렬의 동등 조건
$n \times n$ 실수 대칭행렬 $A, B$에 대해 다음 세 조건은 동등하다:
1. $A, B$가 멱등행렬이고 $AB = 0$
2. $A+B$가 멱등행렬이고 $\text{rank}(A+B) = \text{rank}(A) + \text{rank}(B)$
3. $A+B, A$가 멱등행렬이고 $B$는 음아닌 정부호 행렬

#### 증명: (1) ⟹ (2)
**가정**: $A^2 = A$, $B^2 = B$, $AB = 0$

$(A+B)^2 = A^2 + AB + BA + B^2$

$AB = 0$이고 $A, B$가 대칭이므로 $BA = (AB)^T = 0$

따라서:

$$(A+B)^2 = A + B$$

즉, $A+B$는 멱등행렬이다.

**계수 조건 증명**:

$A$가 멱등행렬이면 $\text{rank}(A) = \text{tr}(A)$ (멱등행렬의 성질)  
마찬가지로 $\text{rank}(B) = \text{tr}(B)$  
$A+B$도 멱등이므로:

$$\text{rank}(A+B) = \text{tr}(A+B) = \text{tr}(A) + \text
{tr}(B) = \text{rank}(A) + \text{rank}(B)$$

#### 증명: (2) ⟹ (3)
**가정**: $(A+B)^2 = A+B$, $\text{rank}(A+B) = \text{rank}(A) + \text{rank}(B)$

$(A+B)^2 = A^2 + AB + BA + B^2 = A+B$  
$A, B$가 대칭이므로:

$$A^2 + AB + BA + B^2 = A + B \\
A^2 - A + B^2 - B + AB + AB = 0 \\
A(A-I) + B(B-I) + 2AB = 0 \quad \cdots (*)$$

멱등행렬 $A$는 고유값이 0 또는 1이므로:

$$A = QA_D Q^T, \quad A_D = \text{diag}(I_r, 0)$$

(단, $r = \text{rank}(A)$, $Q$는 직교행렬)

계수 조건 $\text{rank}(A+B) = \text{rank}(A) + \text{rank}(B)$는 $A$와 $B$의 치역이 직교함을 의미한다:

$$\text{range}(B) \subseteq \text{null}(A) \subseteq \text
{range}(I-A)$$

$AB = 0$이므로 $(*)$에서:

$$A(A-I) + B(B-I) = 0$$

$A-I$는 고유값이 -1 또는 0인 행렬이고, $I-A$는 멱등이다.

두 항이 합쳐져 0이 되려면, $B(B-I) = 0$ (∵ $A(A-I) = -A(I-A)$이고 치역이 직교)

$B$가 대칭이고 $B(B-I) = B^2 - B = 0$이므로 $B^2 = B$ (멱등성)

$(A+B)^2 = A+B$와 $A^2 = A, B^2 = B$로부터:

$$AB + BA = 0$$

$A, B$가 대칭이므로 $2AB = 0$, 즉 $AB = 0$

**$B$가 음아닌 정부호임 증명**:  
대칭멱등행렬 $B$는 고유값이 0 또는 1이므로 모든 고유값이 음아닌 실수이다.  
따라서 $B$는 음아닌 정부호 행렬이다.  
$A+B$가 멱등이면서 $A, B$가 모두 멱등이므로:(대각화 후 합)

$$A+B = A_D + B_D$$

#### 증명: (3) ⟹ (1)
**가정**: $(A+B)^2 = A+B$, $A^2 = A$, $B \succeq 0$

가정에 의해, $A+B$는 멱등행렬이므로 고유값이 0 또는 1이다. $I_r$을 $r$개의 1과 나머지 0으로 이루어진 대각행렬로 생각하면, 직교행렬 $P$가 존재하여 다음과 같이 대각화할 수 있다:

$$
P(A+B)P^\top = \begin{pmatrix} I_r & 0 \\ 0 & 0 \end{pmatrix}, \quad P^\top P = PP^\top = I, \quad r = \text{rank}(A+B)
$$

이제 
$$PBP^\top = \begin{pmatrix} C_{11} & C_{12} \\ C_{21} & C_{22} \end{pmatrix}$$

이고 $C_{11}$이 $r\times r$행렬이라 하면, 

$$ \begin{pmatrix} I_r - C_{11} & -C_{12} \\ -C_{21} & -C_{22} \end{pmatrix} = P(A+B)P^\top - PBP^\top = PAP^\top$$

$A$가 멱등행렬이므로 $PAP^\top$도 멱등행렬이다. 따라서  
$I_r - C_{11} = (I_r -C_{11})^2 + C_{12}C_{21}, \quad C_{22} = -(C_{12}^\top C_{12} + C_{22}C_{22}^\top)$

$$ \therefore 0 \leq \begin{pmatrix} 0 & a^\top \end{pmatrix} \begin{pmatrix} C_{11} & C_{12} \\ C_{21} & C_{22} \end{pmatrix} \begin{pmatrix} 0 \\ a \end{pmatrix} = a^\top C_{22} a = -(a^\top C_{12}^\top C_{12} a + a^\top C_{22}C_{22}^\top a) \leq 0, \quad \forall a \\

\therefore C_{12} = C_{22} = 0, \quad I_r - C_{11} = (I_r -C_{11})^2, \quad C^2_{11} = C_{11}\\

\therefore PBP^\top = \begin{pmatrix} C_{11} & 0 \\ 0 & 0 \end{pmatrix}, \quad C^2_{11} = C_{11}
$$

따라서 $PBP^\top$와 $B$가 멱등행렬이다. 또한 (2) -> (1)의 증명에서와 같이 $AB= 0$임을 증명할 수 있다.


# 6. 행렬미적분 (Matrix Calculus)

## (1) 행렬함수 (Matrix Functions)
행렬을 입력으로 받아 스칼라, 벡터 또는 행렬을 출력하는 함수

$$f: \mathbb{R}^{m \times n} \rightarrow \mathbb{R}^{p 
\times q}$$

### 행렬의 거듭제곱 함수
정사각행렬 $A$에 대해:

$$f(A) = A^k = \underbrace{A \cdot A \cdots A}_{k\text{번}}$$

### 행렬의 지수함수

$$e^A = I + A + \frac{A^2}{2!} + \frac{A^3}{3!} + \cdots = 
\sum_{k=0}^{\infty} \frac{A^k}{k!}$$

**성질**:
* $e^O = I$
* $AB = BA$이면 $e^{A+B} = e^A e^B$
* $(e^A)^{-1} = e^{-A}$


## (2) 노름 (Norm)

### 유클리드 노름 (Euclidean Norm)
벡터 $x \in \mathbb{R}^n$에 대해:

$$\|x\|_2 = \sqrt{x^Tx} = \sqrt{\sum_{i=1}^{n} x_i^2}$$

**성질**:
* $\|x\|_2 \geq 0$이고, $\|x\|_2 = 0 \Leftrightarrow x = 0$
* $\|cx\|_2 = |c| \|x\|_2$ (양의 동차성)
* $\|x + y\|_2 \leq \|x\|_2 + \|y\|_2$ (삼각부등식)
* $|x^Ty| \leq \|x\|_2 \|y\|_2$ (코시-슈바르츠 부등식)

### 프로베니우스 노름 (Frobenius Norm)
행렬 $A = (a_{ij})_{m \times n}$에 대해:

$$\|A\|_F = \sqrt{\sum_{i=1}^{m}\sum_{j=1}^{n} a_{ij}^2} = 
\sqrt{\text{tr}(A^TA)}$$

**성질**:
* $\|A\|_F = \|A^T\|_F$
* $\|AB\|_F \leq \|A\|_F \|B\|_F$
* $\|A + B\|_F \leq \|A\|_F + \|B\|_F$

### 스펙트럴 노름 (Spectral Norm)
행렬 $A$의 최대 특이값으로 정의:

$$\|A\|_2 = \max_{\|x\|_2=1} \|Ax\|_2 = \sqrt{\lambda_
{\max}(A^TA)}$$

**성질**:
* $\|A\|_2 = \|A^T\|_2$
* $\|AB\|_2 \leq \|A\|_2 \|B\|_2$
* $\|A\|_2 \leq \|A\|_F \leq \sqrt{\text{rank}(A)} \|A\|_2$
* 직교행렬 $Q$에 대해 $\|QA\|_2 = \|AQ\|_2 = \|A\|_2$

## (3) 함수행렬 (Function Matrix)

### 벡터함수의 야코비 행렬 (Jacobian Matrix)
벡터함수 $\mathbf{f}: \mathbb{R}^n \rightarrow \mathbb{R}^m$, $\mathbf{f}(x) = (f_1(x), \ldots, f_m(x))^T$에 대해:

$$J_{\mathbf{f}}(x) = \frac{\partial \mathbf{f}}{\partial x^T} = \begin{pmatrix}
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{pmatrix}$$

>**예**: 극좌표 변환  
>극좌표 $(r, \theta)$에서 직교좌표 $(x, y)$로의 변환:

>$$
>\begin{cases}
>x = r\cos\theta \\
>y = r\sin\theta
>\end{cases}
>$$
>
>
>야코비 행렬:
>
>$$
>J = \begin{pmatrix}
>\frac{\partial x}{\partial r} & \frac{\partial x}{\partial \theta} \\
>\frac{\partial y}{\partial r} & \frac{\partial y}{\partial \theta}
>\end{pmatrix}
>= \begin{pmatrix}
>\cos\theta & -r\sin\theta \\
>\sin\theta & r\cos\theta
>\end{pmatrix}
>$$
>
>
>야코비안(행렬식):
>
>$$
>\det(J) = \cos\theta \cdot r\cos\theta - (-r\sin\theta) \cdot \sin\theta = r\cos^2\theta + r\sin^2\theta = r
>$$
>
>
>**응용**: 이중적분에서 $dxdy = r \, dr d\theta$

### 헤시안 행렬 (Hessian Matrix)
스칼라 함수 $f: \mathbb{R}^n \rightarrow \mathbb{R}$의 2차 편미분 행렬:

$$H_f(x) = \frac{\partial^2 f}{\partial x \partial x^T} = \begin{pmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{pmatrix}$$

**성질**:
* 연속인 2차 편미분을 가지면 헤시안은 대칭행렬이다
* $f$가 극소점을 가지면 헤시안은 양반정치이다
* $f$가 강한 극소점을 가지면 헤시안은 양정치이다

#### 헤시안의 양정치성 판정 조건
$H$가 $n \times n$ 대칭행렬일 때, 다음 세 조건은 **동치(equivalent)** 이다:

**조건 1**: 모든 0이 아닌 벡터 $a \in \mathbb{R}^n$에 대해 $a^T H a > 0$  
**조건 2**: $H$의 모든 고유값이 양수이다: $\lambda_i > 0, \quad i = 1, 2, \ldots, n$  
**조건 3**: $H$의 모든 주소행렬식(principal minors)이 양수이다. 특히, **선행 주소행렬식(leading principal minors)** 이 모두 양수이다: $\det(H_1) > 0, \quad \det(H_2) > 0, \quad \ldots, \quad \det(H_n) > 0$

여기서 $H_k$는 $H$의 왼쪽 위에서 시작하는 $k \times k$ 부분행렬이다.

#### 반대로: $\det(H) > 0$ ⟺ $a^THa > 0$ (모든 $a \neq 0$)인가?
**답: 아니다. 필요조건 but 충분조건 아님 (단, $n=2$일 때는 예외)**

**경우 1**: $n=2$ (2차원)  
2×2 대칭행렬 $H = \begin{pmatrix} h_{11} & h_{12} \\ h_{12} & h_{22} \end{pmatrix}$에 대해:

$$\det(H) > 0 \text{ and } h_{11} > 0 \quad \Leftrightarrow \quad a^THa > 0 \text{ (모든 } a \neq 0\text{)}$$

즉, 2차원에서는 $\det(H) > 0$에 추가로 대각원소(또는 대각합) 조건이 필요하다.

**경우 2: $n \geq 3$ (3차원 이상)**  

$$H = \begin{pmatrix} 1 & 0 & 0 \\ 0 & -1 & 0 \\ 0 & 0 & -1 \end{pmatrix}$$

이 행렬은 $\det(H) = 1 \cdot (-1) \cdot (-1) = 1 > 0$이지만, $a^THa$는 $a_1^2 - a_2^2 - a_3^2$이므로 $a = (0, 1, 0)^T$에 대해 $a^THa = -1 < 0$이다. 따라서 양정치가 아니다.

**올바른 양정치 판정 조건: 실비비안 판정법 (Sylvester Criterion)**  
대칭행렬 $H$가 양정치이기 위한 필요충분조건은 모든 **선행 주소행렬식**이 양수인 것이다:

$$\det(H_1) > 0, \quad \det(H_2) > 0, \quad \ldots, \quad \det(H_n) > 0$$

여기서:

$$H_1 = (h_{11}), \quad H_2 = \begin{pmatrix} h_{11} & h_{12} \\ h_{12} & h_{22} \end{pmatrix}, \quad \ldots, \quad H_n = H$$

**예제**: $H = \begin{pmatrix} 2 & 1 \\ 1 & 3 \end{pmatrix}$의 양정치 판정  

$\det(H_1) = 2 > 0 \quad (✓) \\\det(H_2) = \det(H) = 2 \cdot 3 - 1^2 = 5 > 0 \quad (✓)$

따라서 $H$는 양정치이고, 모든 $a \neq 0$에 대해 $a^THa > 0$이다.

#### 응용: 최적화에서의 의미

1차 필요조건: $\nabla f(x^*) = 0$ (극값점)  
2차 충분조건 (극소점):  

$$H_f(x^*) \succ 0 \text{ (양정치)}$$

이는 실비비안 판정법으로 검증하면 확실하다.


## (4) 비선형방정식의 반복해법

### 뉴턴-랩슨 방법 (Newton-Raphson Method)
방정식 $f(x) = 0$의 해를 구하기 위한 반복법:

$$x^{(k+1)} = x^{(k)} - \frac{f(x^{(k)})}{f'(x^{(k)})}$$

### 벡터함수의 뉴턴 방법
$\mathbf{f}(x) = 0$의 해를 구하기 위한 반복법:

$$x^{(k+1)} = x^{(k)} - J_{\mathbf{f}}(x^{(k)})^{-1}\mathbf
{f}(x^{(k)})$$

여기서 $J_{\mathbf{f}}$는 야코비 행렬이다.

## (5) 미분연산자 벡터

### 그래디언트 (Gradient)
스칼라 함수 $f(x)$에 대한 미분 연산자:

$$\nabla f = \frac{\partial f}{\partial x} = \begin{pmatrix}
\frac{\partial f}{\partial x_1} \\
\frac{\partial f}{\partial x_2} \\
\vdots \\
\frac{\partial f}{\partial x_n}\end{pmatrix}$$

### 벡터 미분 공식
**기본 공식**:

* $$\frac{\partial (a^Tx)}{\partial x} = a$$

* $$\frac{\partial (a^Tx)}{\partial x^T} = a^T$$

* $$\frac{\partial (x^Ta)}{\partial x} = a$$

* $$\frac{\partial (x^Tx)}{\partial x} = 2x$$

* $$\frac{\partial (x^TAx)}{\partial x} = (A + A^T)x = 2Ax$$

(단, $A$가 대칭행렬일 때)

* $$\frac{\partial (b^TAx)}{\partial x} = A^Tb$$

**행렬 미분**:

* $$\frac{\partial (x^TA)}{\partial x} = A^T$$

* $$\frac{\partial (Ax)}{\partial x} = A$$

**예제 1**: $y = x^TAx$일 때, $\frac{\partial y}{\partial x}$를 구하라.

$A$가 대칭행렬이면:

$$\frac{\partial y}{\partial x} = \frac{\partial (x^TAx)}
{\partial x} = 2Ax$$

**예제 2**: $y = b^TAx$일 때, $\frac{\partial y}{\partial x}$를 구하라.

$$\frac{\partial y}{\partial x} = A^Tb$$

**예제 3**: 최소제곱법에서 $f(x) = \|Ax - b\|_2^2$를 최소화하는 $x$를 구하라.

$$f(x) = (Ax - b)^T(Ax - b) = x^TA^TAx - 2b^TAx + b^Tb$$

$$\frac{\partial f}{\partial x} = 2A^TAx - 2A^Tb = 0$$

따라서:

$$x = (A^TA)^{-1}A^Tb$$

(단, $A^TA$가 정칙일 때)

**예제 4**: $y^T = x^TA$일 때, $\frac{\partial y^T}{\partial x}$를 구하라.

$$\frac{\partial y^T}{\partial x} = \frac{\partial (x^TA)}
{\partial x} = A^T$$

전치를 취하면:

$$\frac{\partial y}{\partial x^T} = A$$

**예제 5**: $f = x$, $g = Ax$라 하고 $\frac{\partial (f^Tg)}{\partial x}$를 구하면:

$$\frac{\partial (x^TAx)}{\partial x} = Ax + A^Tx$$

$A$가 대칭행렬이면:

$$\frac{\partial (x^TAx)}{\partial x} = 2Ax$$

**일반적인 곱의 미분 공식**:

$$\frac{\partial (u^Tv)}{\partial x} = \frac{\partial u^T}
{\partial x}v + \frac{\partial v^T}{\partial x}u$$

여기서 $u$와 $v$는 $x$에 대한 벡터함수이다.

## (6) 벡(Vec)과 벡하(Vech) 연산자

### 벡 연산자 (Vec Operator)
행렬 $A_{m \times n} = (a_{ij})$에 대해, 열벡터들을 차례대로 쌓아 하나의 긴 열벡터로 만드는 연산:

$$\text{vec}(A) = \begin{pmatrix} a_{11} \\ a_{21} \\ \vdots \\ a_{m1} \\ a_{12} \\ a_{22} \\ \vdots \\ a_{m2} 
\\ \vdots \\ a_{mn} \end{pmatrix}_{mn \times 1}$$

**벡 연산자의 성질**:
* $\text{vec}(A + B) = \text{vec}(A) + \text{vec}(B)$
* $\text{vec}(cA) = c\cdot\text{vec}(A)$ (스칼라 $c$)
* $\text{vec}(ABC) = (C^T \otimes A)\text{vec}(B)$
* $\text{vec}(AB) = (I \otimes A)\text{vec}(B) = (B^T \otimes I)\text{vec}(A)$
* $\text{tr}(A^TB) = \text{vec}(A)^T\text{vec}(B)$
* $\text{tr}(AZ^TBZC) = (\text{vec } Z)^T(CA \otimes B^T)\text{vec } Z$

### 벡하 연산자 (Vech Operator)
대칭행렬 $A_{n \times n}$에 대해, 주대각선과 그 아래 원소들만 추출하여 하나의 열벡터로 만드는 연산:

$$\text{vech}(A) = \begin{pmatrix} a_{11} \\ a_{21} \\ a_{22} \\ a_{31} \\ a_{32} \\ a_{33} \\ \vdots \\ a_{nn} \end
{pmatrix}_{\frac{n(n+1)}{2} \times 1}$$

**벡과 벡하의 관계**:
대칭행렬 $A$에 대해, 복제행렬(duplication matrix) $D_n$을 이용하여:

$$\text{vec}(A) = D_n \text{vech}(A)$$

여기서 $D_n$은 $n^2 \times \frac{n(n+1)}{2}$ 크기의 행렬이다.

주요 성질: **교환 항등식 (Commutation Identity)**  
: 벡-치환 행렬을 이용한 크로네커 곱의 교환

$$B \otimes A = K_{n,m}(A \otimes B)K_{p,q}$$

$A$는 $m \times p$ 행렬, $B$는 $n \times q$ 행렬

### 벡-치환 행렬 (Vec-Permutation Matrix)
$m \times n$ 행렬 $A$에 대해 $\text{vec}(A)$와 $\text{vec}(A^T)$는 같은 원소를 갖지만 순서가 다르다. 
이 순서를 바꾸는 치환행렬을 벡-치환 행렬이라 한다.

**정의**:

$$\text{vec}(A^T) = K_{m,n}\text{vec}(A)$$

를 만족하는 $mn \times mn$ 치환행렬 $K_{m,n}$을 벡-치환 행렬(vec-permutation matrix)이라 한다.

**성질**:
* $K_{m,n}^T = K_{n,m}$
* $K_{m,n}K_{n,m} = I_{mn}$ (즉, $K_{m,n}^{-1} = K_{n,m}$)
* $K_{m,n} = K_{m,n}^T$ (대칭행렬)이므로 $K_{m,n}^2 = I_{mn}$
* $K_{m,n}(A \otimes B) = (B \otimes A)K_{p,q}$ (단, $A$는 $m \times p$, $B$는 $n \times q$ 행렬)
* $K_{n,n}\text{vec}(A) = \text{vec}(A^T)$ (정사각행렬의 경우)

## (7) 행렬 미분의 고급 결과
**행렬 미분의 정의**  
행렬 $A(x) = (a_{ij}(x))_{m \times n}$에 대해, $x$에 대한 미분은 각 성분을 미분한 행렬로 정의된다:

$$\frac{dA}{dx} = \left(\frac{da_{ij}}{dx}\right)_{m 
\times n}$$

**예**:

$$A(x) = \begin{pmatrix} x^2 & 2x \\ 3x & x^3 \end{pmatrix} \Rightarrow \frac{dA}{dx} = \begin{pmatrix} 2x & 
2 \\ 3 & 3x^2 \end{pmatrix}$$

* $\frac{d(A + B)}{dx} = \frac{dA}{dx} + \frac{dB}{dx}$
* $\frac{d(cA)}{dx} = c\frac{dA}{dx}$ (상수 $c$)
* $\frac{d(AB)}{dx} = \frac{dA}{dx}B + A\frac{dB}{dx}$ (곱의 미분법칙)

**스칼라를 행렬로 미분**  
스칼라 함수 $f(X)$를 $m \times n$ 행렬 $X = (x_{ij})$로 미분하면, 각 성분에 대한 편미분을 모은 $m \times n$ 행렬이 된다:

$$\frac{\partial f}{\partial X} = \begin{pmatrix}
\frac{\partial f}{\partial x_{11}} & \frac{\partial f}{\partial x_{12}} & \cdots & \frac{\partial f}{\partial x_{1n}} \\
\frac{\partial f}{\partial x_{21}} & \frac{\partial f}{\partial x_{22}} & \cdots & \frac{\partial f}{\partial x_{2n}} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f}{\partial x_{m1}} & \frac{\partial f}{\partial x_{m2}} & \cdots & \frac{\partial f}{\partial x_{mn}}
\end{pmatrix}$$

**예제**: $f(X) = \text{tr}(X)$일 때

$$\frac{\partial \text{tr}(X)}{\partial X} = I$$

**예제**: $f(X) = \text{tr}(AX)$일 때 ($A$는 상수 행렬)

$$\frac{\partial \text{tr}(AX)}{\partial X} = A^T$$

**예제**: $f(X) = \text{tr}(X^TAX)$일 때 ($A$는 대칭행렬)

$$\frac{\partial \text{tr}(X^TAX)}{\partial X} = 2AX$$

### 역행렬의 미분
정칙행렬 $A(t)$에 대해:

$$\frac{d A^{-1}}{dt} = -A^{-1}\frac{dA}{dt}A^{-1}$$

**증명**: $AA^{-1} = I$를 $t$로 미분하면:

$$\frac{dA}{dt}A^{-1} + A\frac{dA^{-1}}{dt} = 0$$

따라서:

$$\frac{dA^{-1}}{dt} = -A^{-1}\frac{dA}{dt}A^{-1}$$

### 대각합의 미분
* $\frac{\partial}{\partial A}\text{tr}(A) = I$
* $\frac{\partial}{\partial A}\text{tr}(AB) = B^T$
  - X가 대칭이면:$\frac{\partial}{\partial X}\text{tr}(XB) = B + B^T$
* $\frac{\partial}{\partial A}\text{tr}(ABA^T) = A(B + B^T)$
* $\frac{\partial}{\partial A}\text{tr}(A^2) = 2A^T$

### 행렬에 대한 행렬의 미분
$Y = (y_{ij})_{m \times n}$가 $X = (x_{kl})_{p \times q}$에 대한 행렬함수일 때, $Y$의 $X$에 대한 미분은 각 성분별 편미분을 모은 것으로 정의된다:


$$\frac{\partial Y}{\partial X} = \left(\frac{\partial y_
{ij}}{\partial x_{kl}}\right)$$

**예제 1**: $Y = AX$ (단, $A$는 $m \times p$ 상수 행렬, $X$는 $p \times q$ 행렬)


$$\frac{\partial (AX)}{\partial X} = A \otimes I_q$$

여기서 크로네커 곱 결과는 $mp \times pq$ 행렬이다.

**예제 2**: $Y = XB$ (단, $X$는 $m \times p$ 행렬, $B$는 $p \times n$ 상수 행렬)


$$\frac{\partial (XB)}{\partial X} = I_m \otimes B^T$$

**예제 3**: $Y = X^2$ (단, $X$는 $n \times n$ 정사각행렬)


$$\frac{\partial (X^2)}{\partial X} = I_n \otimes X^T + 
X^T \otimes I_n$$

이는 $n^2 \times n^2$ 행렬이다.

**예제 4**: 스칼라 함수 $f = \text{tr}(X^TX)$의 경우

먼저 $Y = X^TX$라 하면:

$$\frac{\partial Y}{\partial X} = X \otimes I + I \otimes 
X$$

그리고 대각합의 미분을 적용하면:

$$\frac{\partial \text{tr}(X^TX)}{\partial X} = 2X$$

**주의사항**:
- 행렬에 대한 행렬의 미분은 고차원 텐서 구조를 가지므로, 실용적으로는 벡 연산자를 사용하여 다루는 경우가 많다
- $\frac{\partial \text{vec}(Y)}{\partial (\text{vec}(X))^T}$ 형태로 표현하면 야코비 행렬 형태가 되어 다루기 쉽다

### 행렬식의 미분
정칙행렬 $A(t)$에 대해:

$$\frac{d}{dt}\det(A) = \det(A) \cdot \text{tr}\left(A^{-1}
\frac{dA}{dt}\right)$$

행렬 $X$에 대한 행렬식의 미분:

$$\frac{\partial \det(X)}{\partial X} = \det(X)(X^{-1})^T 
= \det(X)X^{-T}$$

### 야코비안 관련 공식
행렬함수의 야코비안과 관련된 유용한 공식들:
* $\frac{\partial (Ax)}{\partial x} = A$
* $\frac{\partial (x^TAx)}{\partial x} = (A + A^T)x$
* $\frac{\partial (b^TX^{-1}a)}{\partial X} = -X^{-T}ab^TX^{-T}$

### 에이트킨 적분 (Aitken Integral)
정규분포와 관련된 적분 공식:

$A$가 양정치행렬일 때:

$$\int_{-\infty}^{\infty} \exp\left(-\frac{1}{2}x^TAx + b^Tx\right)dx = \sqrt{\frac{(2\pi)^n}{\det(A)}} \exp\left
(\frac{1}{2}b^TA^{-1}b\right)$$

### 헤시안과 최적화
함수 $f(x)$가 극값을 가지기 위한 조건:
* **1차 필요조건**: $\nabla f(x^*) = 0$ (그래디언트가 0)
* **2차 충분조건**:
   - 극소: 헤시안 $H_f(x^*)$가 양정치
   - 극대: 헤시안 $H_f(x^*)$가 음정치
   - 안장점: 헤시안이 부정치

## (8) 복소수 행렬 (Complex Matrix)

### 복소수 행렬의 정의
복소수를 성분으로 갖는 행렬:

$$A = B + iC$$

여기서 $B$, $C$는 실수 행렬이고 $i = \sqrt{-1}$

### 켤레 전치 (Conjugate Transpose)
복소수 행렬 $A$의 켤레 전치 $A^*$ 또는 $A^H$ (에르미트 전치):

$$A^* = \overline{A^T}$$

각 성분을 켤레 복소수로 바꾼 후 전치한 것.

### 에르미트 행렬 (Hermitian Matrix)
켤레 전치가 자기 자신과 같은 행렬:

$$A^* = A$$

실수 행렬에서의 대칭행렬에 해당한다.

**성질**:
* 에르미트 행렬의 고유값은 모두 실수이다
* 에르미트 행렬의 대각성분은 모두 실수이다

### 유니터리 행렬 (Unitary Matrix)
켤레 전치가 역행렬과 같은 행렬:

$$A^*A = AA^* = I$$

실수 행렬에서의 직교행렬에 해당한다.

**성질**:
* $|\det(A)| = 1$
* 유니터리 행렬에 의한 변환은 노름을 보존한다
* 유니터리 행렬의 고유값의 절댓값은 1이다

### 복소수 행렬의 내적
복소벡터 $x$, $y$에 대한 내적:

$$\langle x, y \rangle = x^*y = \sum_{i=1}^{n} \overline
{x_i}y_i$$

### 복소수 행렬의 노름
* **유클리드 노름**: $\|x\|_2 = \sqrt{x^*x} = \sqrt{\sum_{i=1}^{n} |x_i|^2}$
* **프로베니우스 노름**: $\|A\|_F = \sqrt{\text{tr}(A^*A)} = \sqrt{\sum_{i,j} |a_{ij}|^2}$


## (9) 행렬부등식 (Matrix Inequalities)
통계학과 최적화에서 자주 사용되는 행렬부등식들을 정리한다.

### 기본 표기
* $A \succeq 0$: 양반정치행렬 (positive semidefinite matrix)
* $A \succ 0$: 양정치행렬 (positive definite matrix)
* $e_{\min}(A), e_{\max}(A)$: 행렬 $A$의 최소·최대 고유값

### 레일리 몫 (Rayleigh Quotient)
벡터 $x \neq 0$와 대칭행렬 $A$에 대해:

$$R(A,x) = \frac{x^TAx}{x^Tx}$$

**레일리 몫의 경계**  
$A$가 대칭행렬이면 모든 $x \neq 0$에 대해:

$$e_{\min}(A) \leq \frac{x^TAx}{x^Tx} \leq e_{\max}(A)$$

**따름정리**:

$$\inf_{x\neq 0} \frac{x^TAx}{x^Tx} = e_{\min}(A), \quad 
\sup_{x\neq 0} \frac{x^TAx}{x^Tx} = e_{\max}(A)$$

> 💡 **응용**: 분산 최소화, 효율성 비교, 최적 선형 추정량 증명에서 핵심 도구

### 일반화 레일리 몫
$A$가 대칭행렬이고 $B \succ 0$이면:

$$e_{\min}(B^{-1}A) \leq \frac{x^TAx}{x^TBx} \leq e_{\max}
(B^{-1}A)$$

**해석**:
* $x^TAx$: 분산
* $x^TBx$: 가중 분산
* $B^{-1}A$: 상대적 분산 구조

> 💡 **응용**: 일반화 최소제곱법(GLS), 가중 최소제곱, 정보행렬 비교

### 코시-슈바르츠 행렬부등식
임의의 행렬 $A, B$에 대해:

$$[\text{tr}(A^TB)]^2 \leq \text{tr}(A^TA) \cdot \text{tr}
(B^TB)$$

등호 성립 $\Leftrightarrow$ $A = cB$ (스칼라배)

> 💡 **응용**: Frobenius 내적, 추정량 간 상관 비교

### 양정치행렬의 코시-슈바르츠형 부등식
$A \succ 0$이면 모든 벡터 $u, v$에 대해:

$$(u^Tv)^2 \leq (u^TAu)(v^TA^{-1}v)$$

등호 성립 $\Leftrightarrow$ $Au = \lambda v$

> 💡 **응용**: 최적 추정 방향, 크래머-라오 하한(Cramér-Rao lower bound) 증명

**증명**  
$A$가 양정치행렬이므로 $A = LL^T$로 분해할 수 있다.  
$u^TAu = u^TLL^Tu = (L^Tu)^T(L^Tu) = \|L^Tu\|_2^2$  
$v^TA^{-1}v = v^T(L^{-1})^TL^{-1}v = (L^{-1}v)^T(L^{-1}v) = \|L^{-1}v\|_2^2$  
$u^Tv = (L^Tu)^T(L^{-1}v)$  
따라서:

$$(u^Tv)^2 = [(L^Tu)^T(L^{-1}v)]^2 \leq \|L^Tu\|_2^2 \cdot \|L^{-1}v\|_2^2 = (u^TAu)(v^TA^{-1}v)$$

### 아다마르 부등식 (Hadamard Inequality)
$A \succeq 0$이면:

$$\det(A) \leq \prod_{i=1}^n a_{ii}$$

등호 성립 $\Leftrightarrow$ $A$가 대각행렬

> 💡 **해석**: 공분산 행렬의 행렬식 = 일반화 분산(generalized variance)  
> 독립성 vs 상관성 비교에 사용

### 민코프스키 행렬식 부등식 (Minkowski Determinant Inequality)
$A, B \succeq 0$이면:

$$\det(A+B)^{1/n} \geq \det(A)^{1/n} + \det(B)^{1/n}$$

등호 성립 $\Leftrightarrow$ $A = cB$

> 💡 **응용**: 정보행렬 결합, 실험설계의 D-최적성(D-optimality)

### 고유값의 단조성 (뢰브너 순서, Loewner Order)
$A, B$가 대칭행렬일 때:
* $B \succeq 0 \Rightarrow e_i(A) \leq e_i(A+B)$
* $B \succ 0 \Rightarrow e_i(A) < e_i(A+B)$

> 💡 **해석**: 공분산 추가 → 분산 증가  
> 정규화 및 정규화 오차 분석에 사용

### 곱 행렬의 고유값 경계
$A \succ 0, B \succ 0$이면:

$$e_{\min}(A) \cdot e_{\min}(B) \leq e_i(AB) \leq e_{\max}
(A) \cdot e_{\max}(B)$$

> 💡 **응용**: 전처리(preconditioning), 정보행렬 곱 비교

### 슈어 정리 (Schur Theorem)
대칭행렬 $A$에 대해:

$$\sum_{i=1}^n e_i(A)^2 = \|A\|_F^2 = \text{tr}(A^TA)$$

> 💡 **응용**: Frobenius 노름과 고유값 관계, 분산 총합 분석


# 📘 연습문제
1. 가우스 소거법 또는 역행렬을 이용해 다음 연립일차방정식의 해를 구하시오.  
    (1)
    $\begin{cases}
    2x + 4y - 3z = 5 \\
    x + y + 2z = 6 \\
    3x + 2y + 4z = 9
    \end{cases}$

    (2)
    $\begin{cases}
    2x + 3y - 6z = 10 \\
    x + 2y + 3z = 15
    \end{cases}$

2. $(AB)^{-1} = B^{-1}A^{-1}$임을 증명하시오.

3. 다음 행렬의 역행렬을 구하시오.
    $\begin{pmatrix} 2 & 1 \\ 1 & 3 \end{pmatrix}$

4. 정사각행렬 $A$의 역행렬 $A^{-1}$에 대해
    $\det(A^{-1}) = \frac{1}{\det(A)}$임을 증명하시오.

5. 크래머 공식을 이용하여 다음 연립방정식의 해를 구하시오.
    (1) $\begin{cases} 5x + 3y = 2 \\ 4x - y = 5 \end{cases}$  
    (2) $\begin{cases} x - 2y + 3z = 8 \\ 2x + 4y - 3z = 6 \\ 3x + 4y + 6z = 30 \end{cases}$

