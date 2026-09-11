# 1. 고윳값(Eigenvalue)과 고유벡터(Eigenvector)
## (1) 정의 (Definition)
**비유: 선형사상을 나타내는 방향(또는 축, eigenvector), 크기(eigenvalue)를 나타냄.**  

체 $F$ 위의 벡터공간 $V$의 선형사상
$L : V \to V$ 에 대하여, $L(v) = \lambda v$을 만족하는 $\lambda \in F$, $v \in V \setminus \{0\}$ 를 각각
* 고윳값(eigenvalue)
* 고유벡터(eigenvector)  

라고 한다.

## (2) 고유방정식 (Characteristic Equation)
$n \times n$ 행렬 $M$에 대해 $\lambda$가 $M$의 고윳값이 되기 위한 필요충분조건은 다음 방정식을 만족하는 것이다.

$$
\det(\lambda I_n - M) = 0
$$

이 방정식을 고유방정식(characteristic equation),
그 다항식을 고유다항식(characteristic polynomial)이라고 한다.
(단, $I_n$은 $n \times n$ 단위행렬)

## (3) 고유공간 (Eigenspace)

선형사상 $L - \lambda I$ 의 핵(kernel)

$$
\ker (L - \lambda I)
$$

을 고유공간(eigenspace)이라 한다.

따라서 고유공간의 비영벡터는 모두 $\lambda$에 대응하는 고유벡터이다.

## 성질들
### 고윳값과 고유벡터의 거듭제곱 성질
$A^k u = \lambda^k u$

**따름정리:**
* $A$가 가역이면, $A^{-1}$의 고윳값은 $\lambda^{-1}$이고 고유벡터는 동일하다.
* 일반적으로 $A^k$의 고윳값은 $\lambda^k$이고 고유벡터는 $A$와 동일하다.

### 행렬의 스칼라배와 고윳값
$A$의 고윳값이 $\lambda$이고 고유벡터가 $v$일 때, 스칼라 $c$에 대하여

$$(cA)v = c(Av) = c\lambda v$$

따라서 $cA$의 고윳값은 $c\lambda$이고 고유벡터는 동일하다.

### 행렬 다항식의 고윳값
$A$의 고윳값이 $\lambda$이고 고유벡터가 $v$일 때,
다항식 $p(x) = a_n x^n + a_{n-1}x^{n-1} + \cdots + a_1 x + a_0$에 대하여

$$p(A)v = (a_n A^n + a_{n-1}A^{n-1} + \cdots + a_1 A + a_0 I)v$$

$$= a_n \lambda^n v + a_{n-1}\lambda^{n-1} v + \cdots + a_1 \lambda v + a_0 v$$

$$= p(\lambda)v$$

따라서 $p(A)$의 고윳값은 $p(\lambda)$이고 고유벡터는 $A$와 동일하다.

**예: 행렬 지수함수의 고윳값**  
$A$의 고윳값이 $\lambda$이고 고유벡터가 $v$일 때,
행렬 지수함수 $e^A = \sum_{i=0}^{\infty} \frac{A^i}{i!}$에 대하여

$$e^A v = \left(\sum_{i=0}^{\infty} \frac{A^i}{i!}\right) v = \sum_{i=0}^{\infty} \frac{A^i v}{i!}$$

앞선 성질에 의해 $A^i v = \lambda^i v$이므로

$$= \sum_{i=0}^{\infty} \frac{\lambda^i v}{i!} = \left(\sum_{i=0}^{\infty} \frac{\lambda^i}{i!}\right) v = e^{\lambda} v$$

따라서 $e^A$의 고윳값은 $e^{\lambda}$이고 고유벡터는 $A$와 동일하다.

### 고윳값의 합과 곱
$n \times n$ 행렬 $A$의 고윳값을 $\lambda_1, \lambda_2, \dots, \lambda_n$이라 하면 (중복도 포함)

**고윳값의 합:**

$$\sum_{i=1}^n \lambda_i = \text{tr}(A)$$

여기서 $\text{tr}(A)$는 대각합(trace), 즉 대각성분들의 합이다.

**고윳값의 곱:**

$$\prod_{i=1}^n \lambda_i = \det(A)$$

**증명 스케치:**  
고유다항식 $\det(\lambda I - A)$는 $n$차 다항식이며 $\det(\lambda I - A) = (\lambda - \lambda_1)(\lambda - \lambda_2) \cdots (\lambda - \lambda_n)$ 로 인수분해된다.

이를 전개하면
- $\lambda^{n-1}$의 계수는 $-(\lambda_1 + \lambda_2 + \cdots + \lambda_n)$
- 상수항은 $(-1)^n \lambda_1 \lambda_2 \cdots \lambda_n$

한편 $\det(\lambda I - A)$를 직접 계산하면
- $\lambda^{n-1}$의 계수는 $-\text{tr}(A)$
- 상수항은 $(-1)^n \det(A)$

따라서 계수를 비교하면 위의 결과를 얻는다.

--- 

추가 성질:

$$\text{tr}(A^TA) = \sum_{i=1}^n \lambda_i^2$$

$$\text{tr}(A^{-1}) = \sum_{i=1}^n \lambda_i^{-1}$$


### 고윳값 관련 성질들

$A\in\mathbb F^{n\times n}$, $\lambda\in\mathbb F$에 대하여 다음 명제들은 모두 동치다.

$$
\boxed{
\begin{aligned}
&(1)\ \lambda\text{는 }A\text{의 고윳값이다}\\
\Longleftrightarrow\;&(2)\ \ker(A-\lambda I)\neq\{\mathbf0\}\\
\Longleftrightarrow\;&(3)\ A-\lambda I\text{는 가역행렬이 아니다}\\
\Longleftrightarrow\;&(4)\ \det(A-\lambda I)=0.
\end{aligned}}
$$

따라서 그 대우로

$$
\boxed{
\lambda\text{가 }A\text{의 고윳값이 아니다}
\iff A-\lambda I\text{는 가역행렬이다}
}
$$

가 성립한다.

---

1. 고윳값과 영공간의 관계

고윳값의 정의에 따르면 $\lambda$가 $A$의 고윳값이라는 것은 영이 아닌 벡터 $\mathbf v$가 존재하여 $A\mathbf v=\lambda\mathbf v$ 를 만족한다는 뜻이다.  
이 식을 이항하면 $A\mathbf v-\lambda\mathbf v=0.$ 그리고 $\lambda\mathbf v=\lambda I\mathbf v$이므로 $(A-\lambda I)\mathbf v=0.$

따라서 $\lambda$ 가 고윳값 이라는 것은 정확히 $\mathbf v\neq0,\quad \mathbf v\in\ker(A-\lambda I)$ 인 벡터가 존재한다는 뜻이다. 즉,

$$
\boxed{
\lambda\text{가 }A\text{의 고윳값} \iff \ker(A-\lambda I)\neq\{\mathbf0\}
}
$$

---

2. 영공간이 자명하지 않으면 왜 비가역인가

일반적인 정사각행렬 $B\in\mathbb F^{n\times n}$를 생각하자. 먼저 $B$가 가역행렬이라고 가정하자. 만약 $B\mathbf v=0$ 이면 $B^{-1}B\mathbf v=B^{-1}0$ 이므로 $\mathbf v=0$ 이다. 즉, 가역행렬의 영공간은 반드시 $\ker(B)=\{\mathbf0\}$ 이다. 그러므로 그 대우에 의해 $\ker(B)\neq\{\mathbf0\} \Rightarrow B\text{는 가역이 아니다}$

반대 방향도 증명할 수 있다. $\ker(B)=\{\mathbf0\}$라고 하자. 이는 선형사상

$$
T_B:\mathbb F^n\to\mathbb F^n,\quad T_B(\mathbf v)=B\mathbf v
$$

가 단사라는 뜻이다.

rank-nullity 정리에 의해 $\dim\ker(B)+\text{rank}(B)=n$ 이다. $\dim\ker(B)=0$이므로 $\text{rank}(B)=n.$ 따라서 $B$의 상공간은 $n$차원이고, $\text{Im}(B)=\mathbb F^n$ 이다. 즉, $T_B$는 전사다. 결국 $T_B$는 일대일 대응이므로 역선형사상이 존재한다. 따라서 $B$는 가역이다.

그러므로 $\ker(B)=\{\mathbf0\} \iff B\text{는 가역이다}$ 이며, 동치의 부정을 취하면

$$
\boxed{
\ker(B)\neq\{\mathbf0\} \iff B\text{는 가역이 아니다}
}
$$

이다. 여기서 $B=A-\lambda I$를 대입하면

$$
\boxed{
\ker(A-\lambda I)\neq\{\mathbf0\} \iff A-\lambda I\text{는 가역이 아니다}
}
$$

를 얻는다.

---

3. 행렬식과 가역성의 관계

정사각행렬 $B$에 관한 가역행렬 정리에 의해 $B$ 가 가역 $\iff \det(B)\neq0$ 이다. 따라서 $B$ 가 비가역 $\iff \det(B)=0$

$B=A-\lambda I$를 대입하면

$$
\boxed{
A-\lambda I\text{가 비가역} \iff \det(A-\lambda I)=0
}
$$

따라서 특성방정식 $\det(A-\lambda I)=0$ 의 해가 바로 $A$의 고윳값이 된다.

---

4.  $\lambda$ 가 고윳값이 아니면 $A-\lambda I$ 는 가역이다 

$\lambda$가 $A$의 고윳값이 아니라고 가정한다. 그러면 고윳값의 정의에 의해 $A\mathbf v=\lambda\mathbf v$ 를 만족하는 영이 아닌 벡터 $\mathbf v$가 존재하지 않는다. 그런데

$A\mathbf v=\lambda\mathbf v \iff (A-\lambda I)\mathbf v=0$ 이므로 $(A-\lambda I)\mathbf v=0$ 의 해는 $\mathbf v=0$뿐이다. 따라서

$$
\ker(A-\lambda I)=\{\mathbf0\}.
$$

rank-nullity 정리에 의해 $\text{rank}(A-\lambda I) =n-\dim\ker(A-\lambda I) =n.$  
즉, $A-\lambda I$는 완전계수 정사각행렬이므로 가역이다. 따라서

$$
\boxed{
\lambda\text{가 고윳값이 아니면 } A-\lambda I\text{는 가역이다}
}
$$

---
5. 일반 행렬에서 서로 다른 고윳값에 대응하는 고유벡터들은 선형독립이지만, 반드시 직교하지는 않는다. 그러나 대칭행렬이면 서로 다른 고윳값의 고유벡터들은 선형독립이다.

---

6. $A$가 $n\times n$ 정사각행렬이고 $C$가 가역행렬이면, $A$와 $CAC^{-1}$는 같은 특성다항식과 같은 고윳값을 갖는다. 특히 $C$가 직교행렬이면 $C^{-1}=C^T$이므로 $A$와 $CAC^T$는 같은 고윳값을 갖는다.  
(밑에 닮음과 직교변환의 고유치 불변성 참고)


# 2. 대각화(Diagonalization)

## (1) Definition
두 정사각행렬 $A, B$에 대하여 다음을 만족하는 가역행렬 $P$가 존재하면

$$B = P^{-1} A P$$

$A$는 대각화 가능(diagonalizable)이라고 하며,
이때의 $P$는 $A$를 대각화하는 행렬(diagonalizing matrix)라고 한다.  

**A, P, B의 관계:**
- $A$: 원래 행렬 (대각화 대상)
- $P$: 고유벡터들을 열벡터로 하는 행렬 (기저 변환 행렬)
- $B$: 대각행렬 (대각성분은 고윳값)

## (2) 대각성 정리 (Diagonalization Theorem)
$n \times n$ 행렬 $A$에 대하여 다음 두 명제는 동치이다.
1. $A$는 대각화 가능하다.
2. $A$는 선형독립인 고유벡터를 $n$개 가진다.

>**증명:**  
>$(1 \Rightarrow 2)$  
>$A$가 대각화 가능하다고 하자. 즉, 가역행렬 $P$가 존재하여
>
>$$P^{-1}AP = D = \begin{pmatrix} \lambda_1 & & \\ & \ddots & \\ & & \lambda_n \end{pmatrix}$$
>
>$P$의 열벡터를 $v_1, v_2, \dots, v_n$이라 하면
>
>$$AP = PD$$
>
>이므로
>
>$$A(v_1, v_2, \dots, v_n) = (v_1, v_2, \dots, v_n) \begin{pmatrix} \lambda_1 & & \\ & \ddots & \\ & & \lambda_n \end{pmatrix}$$
>
>따라서 $Av_i = \lambda_i v_i$이고, $P$가 가역이므로, n개의 열벡터들은 n차원 공간의 기저를 이루는 것이므로, $v_1, v_2, \dots, v_n$은 선형독립이다.
>즉, $A$는 선형독립인 고유벡터를 $n$개 가진다. 
>
>$(2 \Rightarrow 1)$  
>$A$가 선형독립인 고유벡터 $v_1, v_2, \dots, v_n$을 가진다고 하자.
>각 고유벡터에 대응하는 고윳값을 $\lambda_1, \lambda_2, \dots, \lambda_n$이라 하면
>
>$$A v_i = \lambda_i v_i, \quad i = 1, 2, \dots, n$$
>
>행렬 $P = (v_1, v_2, \dots, v_n)$이라 하면, $v_i$들이 선형독립이므로 $P$는 가역이다.
>
>$$AP = A(v_1, v_2, \dots, v_n) = (\lambda_1 v_1, \lambda_2 v_2, \dots, \lambda_n v_n)$$
>
>$$= (v_1, v_2, \dots, v_n) \begin{pmatrix} \lambda_1 & & \\ & \ddots & \\ & & \lambda_n \end{pmatrix} = PD$$
>
>따라서 $P^{-1}AP = D$이고 $A$는 대각화 가능하다. $\square$

### 대각성 정리2, Diagonality Theorem
정사각행렬 $A$가 서로 다른 $n$개의 고윳값을 가지면, $A$는 대각화 가능하다.

- 이젠 고유벡터가 아니라 서로 다른 고윳값에 관한 충분조건만 말하고 있음.
- 고유벡터의 개수 $k$에 대한 수학적 귀납법으로 증명

**쓰임새:**
- 대각화 가능성을 판단하는 **충분조건**을 제공한다.
- 고유벡터의 선형독립성을 일일이 확인하지 않고도 대각화 가능 여부를 빠르게 판정할 수 있다.
- 특히 서로 다른 고윳값의 개수만 세어도 되므로 계산이 간단하다.
- 중복된 고윳값이 있는 경우는 중복도 정리로 판단해야 한다.

**설명:**
- 서로 다른 고윳값에 대응하는 고유벡터들은 자동으로 선형독립이다.
- 따라서 $n \times n$ 행렬이 서로 다른 $n$개의 고윳값을 가지면, $n$개의 선형독립인 고유벡터를 갖게 되어 대각화 가능하다.
- 역은 성립하지 않는다: 중복된 고윳값을 가져도 대각화 가능할 수 있다.
   - 예: $I_n$은 고윳값 1만 가지지만 대각화 가능하다.

## (3) 대각화 방법
$n \times n$ 행렬 $A$에 대하여:  
**Step 1.** $n$개의 선형독립 고유벡터를 찾아 대각화 가능 여부를 확인한다.  
**Step 2.** 고유벡터들을 열로 하는 행렬 $P = (v_1, v_2, \dots, v_n)$ 을 만든다.  
**Step 3.**  $P^{-1} A P = D$ 은 대각행렬이 된다.

## (4) 결손행렬 (Defective Matrix)
**정의:**  
$n \times n$ 행렬 $A$가 $n$개의 선형독립인 고유벡터를 갖지 못할 때,
$A$를 결손행렬(defective matrix)이라고 한다.

**특징:**
- 결손행렬은 대각화 불가능하다.
- 어떤 고윳값의 기하적 중복도가 대수적 중복도보다 작을 때 발생한다.
- 예: $A = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$는 고윳값 $\lambda = 1$ (대수적 중복도 2)을 가지지만, 선형독립인 고유벡터는 1개만 존재한다.

## (5) 유사표준형 (Jordan Canonical Form, canonical form under similarity, simiar canonical form)
**배경:**  
대각화 불가능한 행렬(결손행렬)에 대해서도 "가장 대각행렬에 가까운 형태"로 변환할 수 있다.

**정의:**  
임의의 정사각행렬 $A$에 대하여 가역행렬 $P$가 존재하여

$$P^{-1}AP = J$$

를 만족하는 조르단 표준형(Jordan canonical form) $J$는 다음과 같은 블록 대각행렬이다:

$$J = \begin{pmatrix} J_1 & & \\ & \ddots & \\ & & J_k \end{pmatrix}$$

여기서 각 조르단 블록 $J_i$는

$$J_i = \begin{pmatrix} 
\lambda_i & 1 & & \\
& \lambda_i & 1 & \\
& & \ddots & 1 \\
& & & \lambda_i
\end{pmatrix}$$

형태이다.

**특징:**
- 대각화 가능한 행렬의 경우, 조르단 표준형은 대각행렬이 된다.
- 모든 정사각행렬은 유일한 조르단 표준형을 갖는다 (블록의 순서 제외).
- 대각 위의 1을 "초대각 원소(superdiagonal element)"라고 한다.
- $P^{-1}AP$를 **유사변환(similarity transformation)** 또는 **닮음변환**이라고 한다.
   다른 표현들:
   - "$A$를 조르단 표준형으로 **환원(reduce)** 한다"
   - "$A$를 조르단 표준형으로 **변환(transform)** 한다"
   - "$A$의 조르단 표준형을 **구한다**"

### 중복도 정리 (Multiplicity Theorem)
$n \times n$ 행렬 $A$가 중복도가 $m_k$인 고윳값 $\lambda_k$를 가진다고 하자. ($k = 1, 2, \dots, s$)
이때 $\sum_{k=1}^s m_k = n$이다.

**대각화 가능 조건:**  
$A$가 대각화 가능하기 위한 필요충분조건은 모든 고윳값 $\lambda_k$에 대하여

$$\text{rank}(A - \lambda_k I) = n - m_k$$

가 성립하는 것이다.

**설명:**
- 위 식은 고윳값 $\lambda_k$에 대응하는 고유공간의 차원(기하적 중복도)이 대수적 중복도 $m_k$와 같다는 의미이다.
- $\text{rank}(A - \lambda_k I) = n - m_k$ ⟺ $\dim(\ker(A - \lambda_k I)) = m_k$ (rank-nullity 정리)
- 이 조건이 모든 고윳값에 대해 성립하면, $A$는 $n$개의 선형독립인 고유벡터를 가지며, 따라서 대각화 행렬 $P$가 정칙행렬(가역행렬)이 된다.

**정규 고유치와 정규 행렬:**
- 위 조건을 만족하는 고윳값 $\lambda_k$를 **정규 고유치(regular eigenvalue)** 라 한다.
- 모든 고윳값이 정규 고유치일 때, 행렬 $A$를 **정규 행렬(regular matrix)** 이라 한다.
   (주의: 이는 normal matrix와 다른 개념이다)
- 하나 또는 그 이상의 고윳값이 정규 고유치가 아니면 $P^{-1}$가 존재하지 않으므로 $P^{-1}AP$도 정의되지 않는다.

**단일근의 경우:**
- 중복도 $m_k = 1$인 고윳값(단일근)의 경우, $\text{rank}(A - \lambda_k I) = n - 1$이 자동으로 성립한다.
- 따라서 단일근에 대해서는 정규 고유치 조건을 별도로 확인할 필요가 없다.
- **따름정리:** 서로 다른 $n$개의 고윳값을 가지는 $n \times n$ 행렬은 항상 대각화 가능하다. (대각성 정리)

### 유사표준형의 응용
**행렬의 거듭제곱 계산:**
조르단 표준형을 이용하면 행렬의 거듭제곱을 효율적으로 계산할 수 있다.
$A = PJP^{-1}$이면,

$$A^n = PJ^nP^{-1}$$

조르단 블록의 거듭제곱:

$$J_i^n = \begin{pmatrix} 
\lambda_i^n & \binom{n}{1}\lambda_i^{n-1} & \binom{n}{2}\lambda_i^{n-2} & \cdots \\
& \lambda_i^n & \binom{n}{1}\lambda_i^{n-1} & \cdots \\
& & \ddots & \ddots \\
& & & \lambda_i^n
\end{pmatrix}$$

**미분방정식 시스템의 해:**
선형 미분방정식 시스템 $\frac{d\mathbf{x}}{dt} = A\mathbf{x}$의 일반해는

$$\mathbf{x}(t) = e^{At}\mathbf{x}(0)$$

$A = PJP^{-1}$이면,

$$e^{At} = Pe^{Jt}P^{-1}$$

조르단 블록의 지수:

$$e^{J_it} = e^{\lambda_i t}\begin{pmatrix} 
1 & t & \frac{t^2}{2!} & \cdots \\
& 1 & t & \cdots \\
& & \ddots & \ddots \\
& & & 1
\end{pmatrix}$$

**안정성 분석:**
동적 시스템의 안정성은 조르단 표준형을 통해 분석할 수 있다.
- 모든 고윳값의 실부가 음수이면 시스템은 안정적이다.
- 하나라도 실부가 양수인 고윳값이 있으면 불안정하다.
- 조르단 블록의 크기가 1보다 크고 고윳값이 0이면 임계적으로 불안정하다.

## (6) 대칭행렬 (Symmetric Matrix)
실수 정사각행렬 $A$가 다음을 만족할 때 대칭행렬(symmetric matrix)이라고 한다:

$$A = A^T$$

### 대칭행렬의 고유치 성질
**정리 1: 대칭행렬의 고유치는 모두 실수이다.**

>**증명:**  
>$A$를 실대칭행렬이라 하고, $\lambda$를 고유치, $\mathbf{v}$를 대응하는 고유벡터라 하자: $A\mathbf{v} = \lambda\mathbf{v}$
>
>양변에 켤레전치 $\overline{\mathbf{v}}^T$를 왼쪽에서 곱하면 $\overline{\mathbf{v}}^T A\mathbf{v} = \lambda \overline{\mathbf{v}}^T\mathbf{v}$
>
>한편, 좌변의 켤레전치를 취하면 $\overline{\mathbf{v}}^T A^T\mathbf{v} = \overline{\lambda} \overline{\mathbf{v}}^T\mathbf{v}$
>
>$A = A^T$이므로 $\overline{\mathbf{v}}^T A\mathbf{v} = \overline{\lambda} \overline{\mathbf{v}}^T\mathbf{v}$
>
>따라서 $\lambda \overline{\mathbf{v}}^T\mathbf{v} = \overline{\lambda} \overline{\mathbf{v}}^T\mathbf{v}$
>
>$\overline{\mathbf{v}}^T\mathbf{v} = \|\mathbf{v}\|^2 > 0$ 이므로 $\lambda = \overline{\lambda}$. 즉, $\lambda$는 실수이다. $\square$

### 대칭행렬의 대각화
**정리 2: 대칭행렬은 항상 대각화 가능하다.**

**설명:**  
실대칭행렬의 경우, 중복된 고유치가 있더라도 항상 $n$개의 선형독립인 고유벡터를 찾을 수 있다. 즉, 모든 고유치에 대해 기하적 중복도 = 대수적 중복도가 성립한다.

### 대칭행렬의 고유벡터 직교성
**정리 3: 대칭행렬의 서로 다른 고유치에 대응하는 고유벡터들은 서로 직교한다.**

(일반행렬이면 고유벡터 직교가 보장이 안됬었음!)

>**증명:**  
>$\lambda_1 \neq \lambda_2$를 $A$의 서로 다른 고유치, $\mathbf{v}_1, \mathbf{v}_2$를 각각 대응하는 고유벡터라 하자.
>
>$$A\mathbf{v}_1 = \lambda_1\mathbf{v}_1, \quad A\mathbf{v}_2 = \lambda_2\mathbf{v}_2$$
>
>첫 번째 식의 양변에 $\mathbf{v}_2^T$를 왼쪽에서 곱하면 $\mathbf{v}_2^T A\mathbf{v}_1 = \lambda_1 \mathbf{v}_2^T\mathbf{v}_1$
>
>두 번째 식의 양변을 전치하면 $\mathbf{v}_2^T A^T = \lambda_2 \mathbf{v}_2^T$
>
>$A = A^T$이므로 $\mathbf{v}_2^T A = \lambda_2 \mathbf{v}_2^T$
>
>양변에 $\mathbf{v}_1$을 오른쪽에서 곱하면 $\mathbf{v}_2^T A\mathbf{v}_1 = \lambda_2 \mathbf{v}_2^T\mathbf{v}_1$
>
>따라서 $\lambda_1 \mathbf{v}_2^T\mathbf{v}_1 = \lambda_2 \mathbf{v}_2^T\mathbf{v}_1$
>
>$$(\lambda_1 - \lambda_2)\mathbf{v}_2^T\mathbf{v}_1 = 0$$
>
>$\lambda_1 \neq \lambda_2$이므로 $\mathbf{v}_2^T\mathbf{v}_1 = 0$  
>즉, $\mathbf{v}_1 \perp \mathbf{v}_2$. $\square$

**정리 : 대칭행렬의 계수는 0이 아닌 고유치의 개수와 같다.**

>**증명:**  
>$A = P\Lambda P^T$에서 $P$는 직교행렬이므로 계수가 $n$이다.
>따라서
>
>$$\text{rank}(A) = \text{rank}(\Lambda) = \text{(0이 아닌 대각원소의 개수)} = \text{(0이 아닌 고유치의 개수)}$$

**추가 성질:**
- 양정치 행렬: 모든 고유치 > 0 ⇔ $\text{rank}(A) = n$ (full rank)
- 비음정치 행렬: 모든 고유치 ≥ 0 ⇔ $\text{rank}(A) = $ (양의 고유치 개수)
- 음정치 행렬: 모든 고유치 < 0
- 부정치 행렬: 양수와 음수 고유치가 모두 존재

**중복 고유치의 경우:**  
중복된 고유치에 대응하는 고유공간 내에서도 그람-슈미트 과정(Gram-Schmidt process)을 이용하여 서로 직교하는 고유벡터들을 선택할 수 있다.

**보조정리:**  
$n \times n$ 행렬 $B$에 대하여, $B\mathbf{x} = \mathbf 0$의 해공간에서 서로 직교하는 기저를 항상 찾을 수 있다.

### 닮음과 직교변환의 고유치 불변성

$A$를 $n \times n$ 행렬, $C$를 $n \times n$ 가역행렬이라 하자.

**(1) 닮음 변환에서의 고유치 불변성:**
행렬 $A$와 $CAC^{-1}$은 같은 고유치를 갖는다.

>**증명:**
>$\det(\lambda I - CAC^{-1}) = \det(C(\lambda I - A)C^{-1}) = \det(C)\det(\lambda I - A)\det(C^{-1}) = \det(\lambda I - A)$
>
>따라서 $A$와 $CAC^{-1}$의 고유다항식이 동일하므로, 고유치도 같다. $\square$

**(2) 직교변환에서의 고유치 불변성:**
$C$가 직교행렬(orthogonal matrix)이면, 즉 $CC^T = C^TC = I$를 만족하면, 행렬 $A$와 $CAC^T$는 같은 고유치를 갖는다.

>**증명:**
>직교행렬 $C$에 대해 $C^{-1} = C^T$이므로
>
>$$\det(\lambda I - CAC^T) = \det(\lambda I - CAC^{-1}) = \det(\lambda I - A)$$
>
>따라서 $A$와 $CAC^T$의 고유다항식이 동일하고, 고유치도 같다. $\square$

**참고:**
이 성질은 대칭행렬의 직교 대각화 $A = P\Lambda P^T$에서 $P$가 직교행렬이므로, $A$와 대각행렬 $\Lambda$가 같은 고유치를 갖는다는 사실을 설명한다.

### 스펙트럼 분해 정리 (Spectral Decomposition Theorem)
**정리 4 (스펙트럼 분해):**  
$A$를 $n \times n$ 차 **대칭행렬** 이라 하자. 그러면 다음 식을 만족하는 직교행렬 $P$가 존재한다:

$$A = P\Lambda P^T$$

여기서
- $\Lambda = \begin{pmatrix} \lambda_1 & & \\ & \ddots & \\ & & \lambda_n \end{pmatrix}$는 $A$의 $n$개 고유치들로 이루어진 대각행렬이고,
- $P = (\mathbf{p}_1, \mathbf{p}_2, \dots, \mathbf{p}_n)$는 고유치들에 대응하는 정규직교 고유벡터들을 열벡터로 갖는 행렬이다.
- $P^TP = PP^T = I$ (직교행렬 조건)

**스펙트럼 분해의 다른 표현:**

$$A = \sum_{i=1}^n \lambda_i \mathbf{p}_i\mathbf{p}_i^T$$

- 대칭행렬은 고유벡터들이 이루는 정규직교 기저로 완전히 분해된다.
- 각 항 $\lambda_i \mathbf{p}_i\mathbf{p}_i^T$는 $i$번째 고유방향으로의 스케일된 정사영을 나타낸다.

### Thm.

$A\in\mathbb R^{n\times n}$가 실수 대칭행렬이라고 하자. $A$의 고윳값을 대수적 중복도를 포함하여 $\lambda_1,\ldots,\lambda_n$이라고 하고 $\Lambda=\text{diag}(\lambda_1,\ldots,\lambda_n)$ 이라고 하자.

그러면 $A$의 정규직교 고유벡터 $\mathbf p_1,\ldots,\mathbf p_n$을 열벡터로 갖는 직교행렬 $P=[\mathbf p_1:\mathbf p_2:\cdots:\mathbf p_n]$ 가 존재하여 $P^TAP=\Lambda$ 를 만족한다. 이때 $A\mathbf p_i=\lambda_i\mathbf p_i, \quad i=1,\ldots,n $ 이고 $P^TP=PP^T=I$ 이다.

따라서 임의의 $\mathbf x\in\mathbb R^n$에 대해 $\mathbf w=P^T\mathbf x$ 라고 놓으면 $\mathbf x=P\mathbf w$ 이고, 이차형식은

$$
\boxed{
q(\mathbf x) = \mathbf x^TA\mathbf x
= \mathbf w^T\Lambda\mathbf w = \sum_{i=1}^n\lambda_iw_i^2
}
$$

로 표현된다.

$$
\boxed{
\exists P\text{ such that }\forall\mathbf x,\;
\mathbf w=P^T\mathbf x
\text{에 대해 }
\mathbf x^TA\mathbf x =
\mathbf w^T\Lambda\mathbf w
}
$$

즉, $P$는 $A$에 의해 결정되며 모든 $\mathbf x$에 공통으로 사용할 수 있다.

---
임의의 $\mathbf x\in\mathbb R^n$을 선택하자. $P$가 직교행렬이므로 $\mathbf w=P^T\mathbf x$ 라고 정의할 수 있다. 양변 왼쪽에 $P$를 곱하면 $P\mathbf w = PP^T\mathbf x = \mathbf x.$

이것을 이차형식에 대입하면

$$
\mathbf x^TA\mathbf x =(P\mathbf w)^TA(P\mathbf w) =\mathbf w^TP^TAP\mathbf w =\mathbf w^T\Lambda\mathbf w.
$$

$\Lambda$가 대각행렬이므로

$$
\begin{aligned}
\mathbf w^T\Lambda\mathbf w
&=
\begin{pmatrix}
w_1&\cdots&w_n
\end{pmatrix}
\begin{pmatrix}
\lambda_1&&0\\
&\ddots&\\
0&&\lambda_n
\end{pmatrix}
\begin{pmatrix}
w_1\\
\vdots\\
w_n
\end{pmatrix}\\
&= \lambda_1w_1^2+\cdots+\lambda_nw_n^2.
\end{aligned}
$$

따라서

$$
\boxed{
\mathbf x^TA\mathbf x
= \lambda_1w_1^2+\cdots+\lambda_nw_n^2
}
$$

2. $\mathbf w$의 의미

$\mathbf x=P\mathbf w$를 열벡터 단위로 쓰면 $\mathbf x = w_1\mathbf p_1+\cdots+w_n\mathbf p_n$ 이다.

즉, $w_i$는 $\mathbf x$를 고유벡터 기저로 표현했을 때의 좌표다. 양변에 $\mathbf p_i^T$를 곱하면 정규직교성에 의해 $w_i=\mathbf p_i^T\mathbf x$ 를 얻는다.

따라서

$$
\boxed{
\mathbf w=
\begin{pmatrix}
\mathbf p_1^T\mathbf x\\
\vdots\\
\mathbf p_n^T\mathbf x
\end{pmatrix}
= P^T\mathbf x
}
$$

또한 직교변환은 길이를 보존하므로

$$
\begin{aligned}
\|\mathbf x\|^2
&=\mathbf x^T\mathbf x\\
&=\mathbf w^TP^TP\mathbf w\\
&=\mathbf w^T\mathbf w\\
&=\|\mathbf w\|^2.
\end{aligned}
$$

즉,

$$
\boxed{\sum_{i=1}^nx_i^2 = \sum_{i=1}^nw_i^2}
$$

결론: 

$$
\boxed{
\begin{aligned}
A=A^T
&\Rightarrow
\text{정규직교 고유기저 }
\mathbf p_1,\ldots,\mathbf p_n\text{ 존재}\\
&\Rightarrow
P=[\mathbf p_1:\cdots:\mathbf p_n]\text{는 직교행렬}\\
&\Rightarrow
P^TAP=\Lambda\\
&\Rightarrow
\mathbf x=P\mathbf w,\quad \mathbf w=P^T\mathbf x\\
&\Rightarrow
\mathbf x^TA\mathbf x
=
\sum_{i=1}^n\lambda_iw_i^2.
\end{aligned}
}
$$

## 행렬의 분해
행렬의 분해란 하나의 행렬을 두개 또는 세 개 이상의 단순한 구조를 지닌 행렬들의 곱으로 표현하는 절차를 말한다. 회귀분석에 자주 사용되는 LU분해, Cholesky 분해, QR분해, 특잇값 분해 (singular value decomposition, SVD)가 있다. 이런 행렬의 분해는 행렬의 역행렬을 계산하거나 연립선형방정식을 푸는 데 중요한 역할을 하고, 현대 데이터과학의 핵심인 행렬연산을 빠르게 하기 위해서는 필수적이다.

### LU분해
LU분해는 정방행렬을 하삼각행렬 (lower triangular matrix) 와 상삼각행렬 (upper triangular matrix)의 곱으로 분해하는 절차다. 즉 정방행렬 $A$가 어떤 하삼각행렬 $L$과 상삼각행렬 $U$이 존재하여 $A = LU$로 표현되면 이를 $A$의 LU분해라 정의한다. 

관련 성질
1. 모든 정방행렬 $A$는 LU분해가 가능하고, $A$가 대칭행렬이면 $U = L^T$다.
2. LU분해 변형으로 LDU분해가 있다. 이는 정방행렬 $A$를 $A= LDU$로 분해하는 절차자. $D$는 대각행렬이고, $L, U$는 모든 대각원소가 1인 하삼각, 상삼각 행렬이다.
3. 정방행렬 $A$가 정칙행렬이고 $A$의 LU분해로 $L$의 모든 대각원소가 1인 LU분해를 생각할 때, $A$의 LU분해는 유일하고 $\text{det}(A) = u_{11}u_{22}\cdots u_{nn} =\text{det}(D)$ 이다. 여기서 $u_{ii}$ 는 $U$행렬의 $i$번째 대각원소다.
4. 정방행렬 $A$가 대칭이고 양정치행렬이면 $A = U^TU= LL^T$의 분해가 가능하고 이 분해를 Cholesky분해라 부른다.

**응용**

1. 중회귀분석에 응용
2. 자기회귀모형에 응용

### 촐레스키 분해 (Cholesky Decomposition)
**정리 5 (촐레스키 분해):**  
$A$를 $n \times n$ 차 대칭행렬이라 하자.

**(a) 양정치 행렬의 경우:**  
$A$가 양정치 행렬(positive definite matrix)이면, 양수의 대각원소를 갖는 상삼각행렬 $R$이 유일하게 존재하여 다음 식을 만족한다:

$$A = R^TR$$

또는 하삼각행렬 $L = R^T$로 표현하면

$$A = LL^T$$

여기서 $L$의 대각원소는 모두 양수이다.

**(b) 비음정치 행렬의 경우:**  
$A$가 계수(rank) $r$을 갖는 비음정치 행렬(positive semidefinite matrix)이면, 양수인 $r$개의 대각원소와 0인 $n-r$개의 행을 갖는 상삼각행렬 $U$가 유일하게 존재하여 다음 식을 만족한다:

$$A = U^TU$$

**응용:**
- 연립방정식의 효율적 해법
- 몬테카를로 시뮬레이션에서 상관된 난수 생성
- 최적화 문제의 수치 안정성 향상

### QR분해
행렬 계수가 $p$인 $n \times p$행렬 $A$가 어떤 $p\times p$ 상삼각행렬 $R$과 서로 직교인 열로 이루어진 $n \times p$ 행렬 $Q$에 대해 $A = QR$을 만족시킬 때 이를 $A$의 QR분해라 정의한다. $Q$의 열벡터의 직교성은 $Q^TQ =I_p$를 만족시킨다.

**응용:** 중회귀모형 회귀계수 추정

### 특잇값 분해 (singular value decomposition, SVD)

$A$를 계수(rank) $r$인 $m \times n$ 행렬이라 하자 ($m \leq n$). 그러면 $m \times m$ 직교행렬 $P$와 $n \times n$ 직교행렬 $Q$가 존재하여

$$A = P\begin{pmatrix} D & 0 \end{pmatrix}Q^T$$

를 만족한다. 여기서
- $D$는 $m \times m$ 대각행렬로, 비음의 대각원소 $d_i$ ($i = 1, 2, \ldots, m$)를 갖는다.
- $0$은 $m \times (n-m)$ 영행렬이다.
- $D$의 양의 대각원소들은 $A^TA$ (또는 $AA^T$)의 양의 고유치들의 양의 제곱근이며, 이들을 $A$의 **특이치(singular value)** 라 한다.
- $m = n$인 경우, $P^TAQ = D$이다.
- $A$가 대칭인 경우 $P = Q$이고, $D$의 대각원소들은 $A$의 고유치들의 절댓값과 같다.

**관례:**
- 특이치는 보통 크기 순서대로 배열한다: $d_1 \geq d_2 \geq \cdots \geq d_r > 0$
- 계수가 $r$이면 $r$개의 양의 특이치와 $(m-r)$개의 0을 갖는다.

$$A = U\Sigma V^T$$

형태로도 표기하며, 여기서
- $U$: $m \times m$ 직교행렬 (좌특이벡터들)
- $\Sigma$: $m \times n$ 직사각 대각행렬 (특이치들)
- $V$: $n \times n$ 직교행렬 (우특이벡터들)

#### 따름정리 (SVD와 고유벡터의 관계)
위 정리의 행렬 $Q$의 열벡터들은 $A^TA$의 서로 정규직교(orthonormal)인 고유벡터들이다.

마찬가지로, 행렬 $P$의 열벡터들은 $AA^T$의 서로 정규직교인 고유벡터들이다.

**SVD의 응용:**
- 데이터 압축 및 차원 축소
- 최소제곱 문제의 해법
- 주성분 분석(PCA)
- 이미지 처리 및 신호 처리
- 추천 시스템
- 행렬의 의사역행렬(pseudoinverse) 계산


## (7) 정규행렬 (Normal Matrix)
**정의:**  
정사각행렬 $A$가 다음을 만족할 때 정규행렬(normal matrix)이라고 한다:

$$AA^* = A^*A$$

여기서 $A^*$는 $A$의 켤레전치행렬(conjugate transpose)이다.

**켤레전치행렬의 예시:**

$$
A = \begin{bmatrix}
1+2i & 3-i \\
4 & -2+5i
\end{bmatrix} ,\quad
A^* = \overline{A}^{T}
= \begin{bmatrix}
1-2i & 4 \\
3+i & -2-5i
\end{bmatrix}
$$

**정규행렬의 예:**
1. 에르미트 행렬(Hermitian matrix): $A = A^*$
2. 반에르미트 행렬(Skew-Hermitian matrix): $A = -A^*$
3. 유니타리 행렬(Unitary matrix): $AA^* = I$
4. 실수 행렬의 경우:
   - 대칭행렬(Symmetric matrix): $A = A^T$
   - 반대칭행렬(Skew-symmetric matrix): $A = -A^T$
   - 직교행렬(Orthogonal matrix): $AA^T = I$

**스펙트럼 정리 (Spectral Theorem):**  
정규행렬 $A$는 유니타리 대각화가 가능하다. 즉, 유니타리 행렬 $U$가 존재하여

$$U^*AU = D$$

를 만족하는 대각행렬 $D$가 존재한다.

**정리 (대각성정리, Diagonality Theorem):**  
복소수체 위의 정사각행렬 $A$에 대하여 다음은 동치이다:
1. $A$는 정규행렬이다.
2. $A$는 서로 직교하는 고유벡터들로 이루어진 정규직교 기저를 가진다.
3. $A$는 유니타리 대각화 가능하다.

**실수 행렬의 경우:**  
실수 대칭행렬 $A$는 직교 대각화 가능하다. 즉, 직교행렬 $Q$가 존재하여

$$Q^TAQ = D$$

를 만족하는 대각행렬 $D$가 존재한다.

### 멱등행렬과 직교행렬의 고유치
**정리 1 (직교행렬의 고유치):**  
$A$가 $n \times n$ 직교행렬이면, $A$의 고유치는 $\pm 1$이다.

**설명:**  
직교행렬 $A$는 $AA^T = A^TA = I$를 만족한다. $\lambda$가 고유치, $\mathbf{v}$가 대응하는 고유벡터라면 $A\mathbf{v} = \lambda\mathbf{v}$이다.

**정리 2 (멱등행렬의 고유치):**  
$n \times n$ 차 멱등행렬(idempotent matrix) $B$ (즉, $B^2 = B$)의 고유치는 0 또는 1이다.

**설명:**  
$B\mathbf{v} = \lambda\mathbf{v}$이면 $B^2\mathbf{v} = \lambda^2\mathbf{v}$이다. $B^2 = B$이므로 $B\mathbf{v} = \lambda^2\mathbf{v}$이다. 따라서 $\lambda\mathbf{v} = \lambda^2\mathbf{v}$이므로 $\lambda(\lambda - 1) = 0$, 즉 $\lambda = 0$ 또는 $\lambda = 1$이다.

**정리 3 (멱등행렬의 고유치와 계수):**  
$B$가 계수(rank) $r$을 갖는 $n \times n$ 차 멱등행렬이라면:
- $B$는 정확히 $r$개의 1을 고유치로 갖는다 (1의 대수적 중복도는 $r$)
- 0의 대수적 중복도는 $n - r$이다

**설명:**  
멱등행렬의 경우 $\text{rank}(B) = \text{tr}(B)$가 성립한다. 고유치의 합이 대각합과 같으므로, 1이 $r$개, 0이 $n-r$개 존재한다.

### 대칭 비음정치행렬의 고유치
**정의:**  
- 양정치행렬(positive definite): 모든 $\mathbf{x} \neq \mathbf 0$에 대해 $\mathbf{x}^TA\mathbf{x} > 0$
- 양반정치행렬(positive semidefinite): 모든 $\mathbf{x}$에 대해 $\mathbf{x}^TA\mathbf{x} \geq 0$
- 비음정치행렬: 양반정치행렬과 동일한 의미

**정리 4 (양정치행렬의 필요충분조건):**  
대칭행렬 $A$가 양정치행렬이 되는 필요충분조건은 $A$의 모든 고유치가 양수인 것이다.

**정리 5 (양반정치행렬의 필요충분조건 1):**  
$A$가 대칭행렬일 때, $A$가 양반정치행렬이 될 필요충분조건은 $A$의 모든 고유치가 0 이상이고, 적어도 하나의 고유치는 0인 것이다.

**정리 6 (비음정치행렬의 필요충분조건 2):**  
$A$가 대칭행렬일 때, $A$가 비음정치행렬이 될 필요충분조건은 $A$의 모든 주소행렬식(principal minor)이 0 이상인 것이다.

### 대칭행렬의 분해
**정리 7 (대칭행렬의 LL^T 분해):**  
$A$가 계수 $r$을 갖는 $n \times n$ 차 대칭행렬이라면, $A = LL^T$로 나타낼 수 있다. 여기서 $L$은 $n \times r$ 차 최대열계수행렬(full column rank matrix)이다.

**따름정리 1 (양반정치행렬의 KK^T 분해):**  
$A$가 계수 $r$을 갖는 $n \times n$ 차 양반정치행렬이라면, $A = KK^T$로 나타낼 수 있다. 여기서 $K$는 계수 $r$을 갖는 $n \times r$ 차 실행렬이다.

**따름정리 2 (양반정치행렬의 MM^T 분해):**  
$A$가 계수 $r$을 갖는 $n \times n$ 차 양반정치행렬이라면, $A = MM^T$로 나타낼 수 있다. 여기서 $M$은 $n \times n$ 정칙행렬(가역행렬)이다.

**설명:**  
이러한 분해는 스펙트럼 분해를 이용하여 구성할 수 있다. $A = Q\Lambda Q^T$에서 $\Lambda = \text{diag}(\lambda_1, \dots, \lambda_r, 0, \dots, 0)$이고 모든 $\lambda_i > 0$이면, $\Lambda^{1/2} = \text{diag}(\sqrt{\lambda_1}, \dots, \sqrt{\lambda_r}, 0, \dots, 0)$로 정의하여 $L = Q\Lambda^{1/2}$로 구성할 수 있다.

### ???

$A\in\mathbb R^{n\times n}$가 rank $k$인 대칭 멱등행렬이라고 하자. 그러면 어떤 직교행렬 $P$가 존재하여 $P^TAP=E_k$ 를 만족한다. 여기서

$$
E_k=
\begin{pmatrix}
I_k&0\\
0&0
\end{pmatrix}
=
\text{diag} (\underbrace{1,\ldots,1}_{k}, \underbrace{0,\ldots,0}_{n-k})
$$

동치:

$$
\boxed{A=PE_kP^T}
$$

**증명** 

1. 상공간에서 $A$는 항등변환이다

$\mathbf u\in\text{Col}(A)$라고 하자. 그러면 어떤 $\mathbf x$가 존재하여 $\mathbf u=A\mathbf x$ 이고, $A\mathbf u =A(A\mathbf x) =A^2\mathbf x =A\mathbf x =\mathbf u.$

따라서 $\mathbf u\in\text{Col}(A) \Rightarrow A\mathbf u=\mathbf u$  
즉, $\text{Col}(A)$의 모든 영이 아닌 벡터는 고윳값 $1$에 대응하는 고유벡터다.

2. 영공간에서 $A$는 영변환이다

$\mathbf v\in\text{Null}(A)$이면 영공간의 정의에 의해 $A\mathbf v=0.$.  
따라서 $\text{Null}(A)$의 모든 영이 아닌 벡터는 고윳값 $0$에 대응하는 고유벡터다.

3. 상공간과 영공간은 서로 직교한다

$\mathbf u\in\text{Col}(A)$이고 $\mathbf v\in\text{Null}(A)$라고 하자. $\mathbf u\in\text{Col}(A)$이므로 어떤 $\mathbf x$가 존재하여 $\mathbf u=A\mathbf x$ 이다. 그러면

$$
\begin{aligned}
\mathbf u^T\mathbf v
&=(A\mathbf x)^T\mathbf v\\
&=\mathbf x^TA^T\mathbf v\\
&=\mathbf x^TA\mathbf v\\
&=\mathbf x^T0\\
&=0.
\end{aligned}
$$

따라서 $\text{Col}(A)\perp \text{Null}(A)$ 이다. 또한 rank-nullity 정리에 의해 $\dim\text{Col}(A)=r(A)=k$ 이고 $\dim\text{Null}(A)=n-k$ 이다. 따라서

$$
\boxed{
\mathbb R^n
=\text{Col}(A) 
\mathbin{\overset{\perp}{\oplus}}
\text{Null}(A)
}
$$

4. 정규직교기저를 선택한다

$\text{Col}(A)$에서 정규직교기저 $\mathbf p_1,\ldots,\mathbf p_k$ 를 선택하고, $\text{Null}(A)$에서 정규직교기저 $\mathbf p_{k+1},\ldots,\mathbf p_n$ 을 선택한다.

두 공간이 서로 직교하므로 전체 벡터들 $\mathbf p_1,\ldots,\mathbf p_n$ 은 $\mathbb R^n$의 정규직교기저를 이룬다. 이들을 열벡터로 갖는 행렬을 $P= [\mathbf p_1:\cdots:\mathbf p_k: \mathbf p_{k+1}:\cdots:\mathbf p_n]$ 이라고 하자.

열벡터들이 정규직교하므로 $P^TP=PP^T=I.$ 따라서 $P$는 직교행렬이다.

5. $AP=PE_k$임을 보인다

$i=1,\ldots,k$에 대해 $\mathbf p_i\in\text{Col}(A)$이므로 $A\mathbf p_i=\mathbf p_i.$  
$i=k+1,\ldots,n$에 대해서는 $\mathbf p_i\in\text{Null}(A)$이므로 $A\mathbf p_i=0.$

따라서

$$
\begin{aligned}
AP
&=
[A\mathbf p_1:\cdots:A\mathbf p_k:
A\mathbf p_{k+1}:\cdots:A\mathbf p_n]\\
&=
[\mathbf p_1:\cdots:\mathbf p_k:
0:\cdots:0].
\end{aligned}
$$

한편 $PE_k = [\mathbf p_1:\cdots:\mathbf p_k: 0:\cdots:0]$ 이므로 $AP=PE_k.$

양변 왼쪽에 $P^T$를 곱하면 $P^TAP=P^TPE_k=E_k.$

따라서 $P^TAP=E_k$ 가 증명된다.

---

**스펙트럴 정리를 이용한 짧은 증명**

$A$가 실수 대칭행렬이므로 스펙트럴 정리에 의해 어떤 직교행렬 $P$가 존재하여 $P^TAP = \text{diag}(\lambda_1,\ldots,\lambda_n)$ 이다.

$\lambda$가 $A$의 고윳값이고 $\mathbf v\neq0$가 대응하는 고유벡터라면 $A\mathbf v=\lambda\mathbf v.$

따라서

$$
A^2\mathbf v
= A(\lambda\mathbf v)
= \lambda A\mathbf v
= \lambda^2\mathbf v.
$$

그런데 $A^2=A$이므로 $A^2\mathbf v=A\mathbf v=\lambda\mathbf v.$ 따라서 $\lambda^2\mathbf v=\lambda\mathbf v.$

$\mathbf v\neq0$이므로 $\lambda^2=\lambda$ 이고, $\lambda(\lambda-1)=0.$ 따라서 모든 고윳값은 $\lambda\in\{0,1\}$ 이다.

$A$는 대칭행렬이므로 0이 아닌 고윳값의 개수는 중복도를 포함하여 $r(A)$와 같다. $r(A)=k$이므로 고윳값 1이 정확히 $k$개이고, 고윳값 0이 $n-k$개다.

고유벡터의 순서를 적절히 배열하면

$$
P^TAP
=
\text{diag}
(\underbrace{1,\ldots,1}_{k},
\underbrace{0,\ldots,0}_{n-k})
=
E_k.
$$

따라서 명제가 증명된다.

### 일반 멱등행렬에 대해 성립하는 올바른 결과
위 정리와 유사하지만, 조건이 다르다.

$A$가 대칭이 아닐 수 있는 일반적인 멱등행렬이라면 직교행렬은 보장되지 않지만, 가역행렬을 이용한 대각화는 항상 가능하다.

즉,

$$
A^2=A,\qquad r(A)=k
\Rightarrow
\exists P\text{ 가역}:
P^{-1}AP=E_k
$$

**증명**

멱등행렬에서는 $\mathbb R^n  = \text{Col}(A)\oplus\text{Null}(A)$ 가 성립한다.

실제로 두 공간의 교집합에 속하는 벡터 $\mathbf z$를 생각하자. $\mathbf z\in\text{Col}(A) \cap \text{Null}(A).$  
$\mathbf z\in\text{Col}(A)$이므로 $A\mathbf z=\mathbf z$ 이고, 동시에 $\mathbf z\in\text{Null}(A)$이므로 $A\mathbf z=0.$  
따라서 $\mathbf z=0.$ 즉, $\text{Col}(A)\cap\text{Null}(A)=\{0\}.$

또한 $\dim\text{Col}(A)+\dim\text{Null}(A)=k+(n-k)=n$ 이므로

$$
\mathbb R^n = \text{Col}(A)\oplus\text{Null}(A)
$$

$\text{Col}(A)$의 기저를 $\mathbf p_1,\ldots,\mathbf p_k$ 로 선택하고, $\text{Null}(A)$의 기저를 $\mathbf p_{k+1},\ldots,\mathbf p_n$ 으로 선택한다.  
이들을 열벡터로 갖는 행렬 $P=[\mathbf p_1:\cdots:\mathbf p_n]$ 는 가역이다. 이전과 마찬가지로 $AP=PE_k$ 이므로

$$
\boxed{P^{-1}AP=E_k}
$$

다만 두 부분공간이 반드시 직교하지는 않으므로 $P$가 직교행렬이라고 보장할 수 없다.

### 다른 정리와의 관계

이 명제는 완전히 새로운 독립적인 정리라기보다 기존 정리들의 따름정리로 볼 수 있다.

**스펙트럴 정리와의 관계**

대칭 멱등행렬에 관한 명제는 다음 두 사실을 결합한 것이다.

$$
\begin{aligned}
A^T=A
&\Rightarrow
A\text{는 직교대각화 가능},\\
A^2=A
&\Rightarrow
A\text{의 고윳값은 }0\text{ 또는 }1.
\end{aligned}
$$

여기에 $r(A)=k$ 를 적용하면 고윳값 1이 정확히 $k$개라는 결론이 나온다.

따라서 대칭 멱등행렬 명제는 스펙트럴 정리의 특별한 경우다.

**일반 대각화 정리와의 관계**

멱등행렬의 최소다항식은 $m_A(t)\mid t(t-1)$ 을 만족한다.

다항식 $t(t-1)$은 서로 다른 일차인수만 가지므로 멱등행렬은 항상 대각화 가능하다. 따라서 일반 멱등행렬에 대해 $P^{-1}AP=E_k$ 가 성립한다.

즉, 일반 멱등행렬의 대각화 = 최소다항식에 의한 대각화 정리의 특별한 경우다.

**직교사영행렬의 정의와 관계**

실수행렬 $A$에 대해 다음 조건들은 서로 동치다.

$$
\boxed{
\begin{aligned}
&A\text{는 어떤 부분공간 위의 직교사영행렬}\\
\iff\;&A^2=A\text{이고 }A^T=A\\
\iff\;&\exists P\text{ 직교}:
P^TAP=E_k.
\end{aligned}
}
$$

따라서 수정된 명제는 직교사영행렬의 표준형을 나타내는 정리이기도 하다.

최종적으로 구분하면 다음과 같다.

| 조건                      | 결론                             |
| ----------------------- | ------------------------------ |
| $A^2=A$                 | 어떤 가역행렬 $P$에 대해 $P^{-1}AP=E_k$ |
| $A^2=A$, $A^T=A$        | 어떤 직교행렬 $P$에 대해 $P^TAP=E_k$    |
| $A^2=A$, $A^T\neq A$ 가능 | 직교대각화는 일반적으로 불가능               |
| $A^2=A$, $A^T=A$        | $A$는 직교사영행렬                    |


## 직적과 직합의 고유치
### 크로네커 곱의 고유치
**정리 (직적의 고유치):**  
$A$가 $m \times m$ 행렬, $B$가 $n \times n$ 행렬이라 하자. $\lambda$와 $\mathbf{x}$가 $A$의 고유치와 그에 대응하는 고유벡터를, $\mu$와 $\mathbf{y}$가 $B$의 고유치와 그에 대응하는 고유벡터를 나타낸다고 하자. 그러면:
- $\lambda\mu$는 $A \otimes B$의 고유치가 된다.
- $\mathbf{x} \otimes \mathbf{y}$는 대응하는 고유벡터이다.

**증명:** 
 
$$
(A \otimes B)(\mathbf{x} \otimes \mathbf{y}) = (A\mathbf{x}) \otimes (B\mathbf{y}) = (\lambda\mathbf{x}) \otimes (\mu\mathbf{y}) = \lambda\mu(\mathbf{x} \otimes \mathbf{y})
$$

**따름정리 (크로네커 곱의 계수):** 
 
$$\text{rank}(A \otimes B) = \text{rank}(A) \times \text{rank}(B)$$

### 직합의 고유치
**정리 (직합의 고유치):**  
$A$가 $m \times m$ 행렬, $B$가 $n \times n$ 행렬이라 하자. $\lambda$와 $\mathbf{x}$가 $A$의 고유치와 그에 대응하는 고유벡터를, $\mu$와 $\mathbf{y}$가 $B$의 고유치와 그에 대응하는 고유벡터를 나타낸다고 하자. 그러면:
- $\lambda + \mu$는 $A \oplus B$의 고유치가 된다.
- $\mathbf{x} \oplus \mathbf{y}$는 대응하는 고유벡터이다.

**설명:**  
직합 $A \oplus B = \begin{pmatrix} A & 0 \\ 0 & B \end{pmatrix}$는 블록 대각행렬이므로, $A \oplus B$의 고유치는 $A$의 고유치와 $B$의 고유치의 합집합이다.

## AB, BA의 고유치 관계
**정리 (AB와 BA의 고유치 관계):**  
$A$를 $m \times n$ 행렬, $B$를 $n \times m$ 행렬이라 하자. ($n \geq m$)

**(1) 고유다항식의 관계:**

$$|\lambda I_n - BA| = (-\lambda)^{n-m}|\lambda I_m - AB|$$

**(2) 0이 아닌 고유치의 일치:**
- $BA$의 0이 아닌 고유치들은 $AB$의 0이 아닌 고유치들과 정확히 일치한다.
- 각 0이 아닌 고유치의 대수적 중복도도 $AB$와 $BA$에서 동일하다.

**(3) 정사각행렬의 경우:**  
$m = n$이면, $BA$의 모든 고유치(0 포함)는 $AB$의 고유치와 같다.

**설명:**
- $AB$와 $BA$는 일반적으로 서로 다른 크기의 행렬이지만, 0이 아닌 고유치는 공유한다.
- $n > m$인 경우, $BA$는 추가로 $(n-m)$개의 0을 고유치로 갖는다.
- 고유벡터는 일반적으로 다르다.

**증명 개요 (2번 성질):**  
$\lambda \neq 0$가 $AB$의 고유치이고 $\mathbf{v}$가 대응하는 고유벡터라 하자. 즉, $AB\mathbf{v} = \lambda\mathbf{v}$.

양변에 왼쪽에서 $B$를 곱하면

$$B(AB\mathbf{v}) = B(\lambda\mathbf{v})$$

$$(BA)(B\mathbf{v}) = \lambda(B\mathbf{v})$$

$\lambda \neq 0$이므로 $\mathbf{v} \neq \mathbf 0$이고, 따라서 $B\mathbf{v} \neq \mathbf 0$이다.
즉, $B\mathbf{v}$는 $BA$의 고유치 $\lambda$에 대응하는 고유벡터이다.

역방향도 유사하게 증명되므로, $AB$와 $BA$는 같은 0이 아닌 고유치를 공유한다. $\square$

**응용 예:**
- Trace 관계: $\text{tr}(AB) = \text{tr}(BA)$ (크기가 달라도 성립)
- 순환 행렬의 고유치 분석
- 특이값 분해에서 $A^TA$와 $AA^T$의 관계

## 대칭행렬의 동시 대각화 (Simultaneous Diagonalization)

### 정리 1 (동시 대각화의 필요충분조건)
같은 차수의 대칭행렬 $A, B$에 대하여, 직교행렬 $P$가 존재하여 $P^TAP$와 $P^TBP$가 모두 대각행렬이 될 필요충분조건은

$$AB = BA$$

이다.

**설명:**
- 두 대칭행렬이 교환가능(commute)하면 공통의 고유벡터 집합을 가진다.
- 이 공통 고유벡터들로 구성된 직교행렬 $P$가 두 행렬을 동시에 대각화한다.

### 정리 2 (양정치행렬을 이용한 동시 대각화)
같은 차수의 대칭행렬 $A, B$에서 $A$가 양정치행렬이면, 다음을 만족하는 정칙행렬 $P$가 존재한다:
- $P^TAP = I$
- $P^TBP = D$ (대각행렬)

여기서 $D$의 대각원소들은 $|B - \lambda A| = 0$에 대한 $\lambda$의 해들이다.

**설명:**
- 이는 일반화 고유치 문제(generalized eigenvalue problem) $B\mathbf{v} = \lambda A\mathbf{v}$의 해를 구하는 것과 관련된다.
- $A$가 양정치이므로 촐레스키 분해 등을 이용하여 $P$를 구성할 수 있다.

### 정리 3 (비음정치행렬의 동시 대각화)
같은 차수의 대칭행렬 $A, B$가 모두 실행렬인 비음정치행렬일 때, $P^TAP$와 $P^TBP$가 모두 대각행렬이 되게 하는 정칙행렬 $P$가 존재한다.

**설명:**
- 두 비음정치행렬이 동시에 대각화 가능하다.
- 이는 주성분 분석(PCA)과 정준상관분석(CCA) 등에서 활용된다.

## 양정치행렬 (Positive Definite Matrix)
정의: 대칭행렬 $A$에 대해 모든 $x \neq 0$에 대해 $x^TAx > 0$ 이면,
* 즉 x가 0일때만 0이 되면, 이런 이차형식을 양정치(positive definite)이라 하고, 이때의 A를 양정치행렬(positive definite matrix)라 한다
* $>$가 아니라 $\geq$면 positive semidefinite(양반정치)이라 한다. positive definite이 아니라
* 양정치던 양반정치던, nonnegative definite (n.n.d, 정부호)라 한다.

**참고) 이차형식과 대칭행렬의 관계**  
임의의 이차형식은 항상 대칭행렬로 나타낼 수 있다.  
일반적인 행렬 $B$에 대해 $x^TBx$가 주어졌을 때, 이를 대칭행렬로 변환할 수 있다:

$$x^TBx = x^T\left(\frac{B + B^T}{2}\right)x$$

**증명**:  
$x^TBx$는 스칼라이므로 전치해도 같다: $x^TBx = (x^TBx)^T = x^TB^Tx$  
따라서:

$$x^TBx = \frac{1}{2}(x^TBx + x^TB^Tx) = \frac{1}{2}x^T(B + B^T)x = x^T\left(\frac{B + B^T}{2}\right)x$$

여기서 $A = \frac{B + B^T}{2}$는 대칭행렬이다: $A^T = \left(\frac{B + B^T}{2}\right)^T = \frac{B^T + B}{2} = A$  
따라서 모든 이차형식은 대칭행렬을 이용해 표현할 수 있으며, 이차형식을 다룰 때는 항상 대칭행렬을 사용하는 것이 표준이다.
> 💡 **결론**: 이차형식을 표현할 때는 항상 대칭행렬을 사용하는 것이 표준이며, 비대칭 행렬로 주어진 경우에도 대칭화하여 다룬다.

### 관련 개념
* **양반정치 (positive semi-definite)**: 모든 $x$에 대해 $x^TAx \geq 0$
* **음정치 (negative definite)**: 모든 $x \neq 0$에 대해 $x^TAx < 0$
* **음반정치 (negative semi-definite)**: 모든 $x$에 대해 $x^TAx \leq 0$
* **부정치 (indefinite)**: 위의 어느 것에도 해당하지 않음

* TODO: determinant와의 관계?: linear_algebra_01_Matrix_Determinant.md에 정리한거 있음

### 양정치행렬의 판정
대칭행렬 $A$가 양정치가 되기 위한 필요충분조건:
* **모든 고유값이 양수이다**  
  **증명**: 대칭행렬은 대각화 가능하고, $A = Q\Lambda Q^T$ (스펙트럼 정리)에서 $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$이고 $Q$는 직교행렬이면, $x^TAx = x^TQ\Lambda Q^Tx = y^T\Lambda y = \sum \lambda_i y_i^2$ (여기서 $y = Q^Tx$는 치환)이다. 모든 $x \neq 0$에 대해 $x^TAx > 0$ $\Leftrightarrow$ 모든 $\lambda_i > 0$

* **모든 주 소행렬식(leading principal minor)이 양수이다**   
  **증명**: $k \times k$ 좌상단 부분행렬 $A_k$는 양정치이고, 귀납적으로 $\det(A_k) > 0$임을 보인다. (Sylvester 판정법)

* **$A = B^TB$로 표현될 때 $B$가 열 최대계수 행렬이다**   
  **증명**: $A$가 양정치이면 $A = Q\Lambda Q^T$에서 $B = Q\Lambda^{1/2}$로 놓으면 $B^TB = \Lambda^{1/2}Q^TQ\Lambda^{1/2} = A$. $B$의 계수는 $\Lambda^{1/2}$의 0이 아닌 대각원소 개수이며, 모든 $\lambda_i > 0$이므로 $\text{rank}(B) = n$ (열 최대계수)

* **$\Sigma = \Sigma^{1/2} (\Sigma^{1/2})^T$ 인 실수가 원소인 대칭행렬 $\Sigma^{1/2}$가 존재한다**    
  **증명**: 위의 세 번째 조건과 동일하다. $\Sigma^{1/2}$는 $A$의 제곱근 행렬이고, 대칭인 양정치 행렬의 제곱근 역시 대칭 양정치이다.

### 양정치행렬의 성질
* 양정치행렬은 항상 정칙행렬이다 ($\det(A) > 0$)
* 양정치행렬의 역행렬도 양정치이다
* 양정치행렬의 고유치는 모두 양수이다
  - 음정치행렬의 고유치는 모두 음수이다
* 양정치행렬의 대각성분은 모두 양수이다
  - 음정치행렬의 대각성분은 모두 음수이다
* $A$, $B$가 양정치이면 $A + B$도 양정치이다
* $A$가 양정치이고 $C$가 열 최대계수 행렬이면 $C^TAC$도 양정치이다
* 대칭행렬 A가 양정치행렬이 되기위한 필요충분조건은 정칙행렬 C가 존재하며 $A = C^TC$로 표현될 수 있는 것이다. (위의 세 번째 조건과 동일)
* 어떤 정칙행렬 $C$에 대해 행렬$A$가 양정치행렬이면 $C^TAC$도 양정치행렬이다. (위의 네 번째 조건과 동일)
  - $A$가 양반정치행렬이면 $C^TAC$도 양반정치행렬이다.
* **Sylvester 판정법 (선행 주 소행렬식 판정법):**
  
  $n \times n$ 대칭행렬 $A$의 선행 주 소행렬식(leading principal minor)을 다음과 같이 정의한다:

  $$D_k = \det(A_k), \quad k = 1, 2, \ldots, n$$
  여기서 $A_k$는 $A$의 좌상단 $k \times k$ 부분행렬이다.
  
  **양정치 판정:**
  $$D_1 > 0, D_2 > 0, \ldots, D_n > 0 \quad \Leftrightarrow \quad A \text{는 양정치행렬}$$
  
  **음정치 판정:**
  $$D_1 < 0, D_2 > 0, D_3 < 0, \ldots, (-1)^n D_n > 0 \quad \Leftrightarrow \quad A \text{는 음정치행렬}$$
  (즉, 선행 주 소행렬식의 부호가 교대로 나타나고, $(-1)^n D_n > 0$를 만족)

  **예시:**
  - $2 \times 2$ 양정치: $a_{11} > 0$이고 $\det(A) > 0$
  - $2 \times 2$ 음정치: $a_{11} < 0$이고 $\det(A) > 0$
  - $3 \times 3$ 음정치: $D_1 < 0$, $D_2 > 0$, $D_3 < 0$

  **주의:** 양반정치 또는 음반정치의 판정은 $\geq$ 또는 $\leq$로 판정하며, 하나 이상의 부등호가 등호로 바뀐다.

* $n \times m$ 행렬 $A$의 계수가 $m, \quad (m < n)$ 일때 $A^TA$는 양정치행렬이고 $AA^T$는 양반정치행렬이다.

> $A\in\mathbb{R}^{n\times m}$이고 $\text{rank}(A)=m<n$ 이면 $A$는 full column rank이므로 $A^TA$는 양정치행렬이다. 실제로 임의의 $x\in\mathbb{R}^m$ , $x \neq 0$ 에 대하여
>
>  $$x^TA^TAx=(Ax)^T(Ax)=\|Ax\|^2>0$$
>
>이다. 한편 $AA^T$는
>
>$$y^TAA^Ty=\|A^Ty\|^2\geq 0$$
>
>이므로 양반정치행렬이지만, $\text{rank}(AA^T)=\text{rank}(A)=m<n $ 이어서 특이행렬이다. 따라서 $AA^T$는 양정치행렬이 아니다.

* $n \times m$ 행렬 $A$의 계수가 $k, \quad (k < n, k < m)$ 일때 $A^TA$, $AA^T$ 모두 양반정치행렬이다.

* 양정치행렬은 정칙행렬이다. 그러나 정칙행렬이 반드시 양정치행렬은 아니다.

### 그람 행렬 (Gram Matrix)
$A = X^TX$ 형태의 행렬을 그람 행렬이라 하며:
* 항상 양반정치이다
* $X$가 열 최대계수 행렬이면 양정치이다
* $XX^T$도 그람 행렬이며, $X^TX$와 같은 양정치성질을 갖는다

앞에서 사용한 "최소다항식 정리"에는 서로 다른 두 명제가 포함되어 있다. 정확히 구분하면 다음과 같다.

## 최소다항식 (Minimal Polynomial)

행렬 $A\in\mathbb F^{n\times n}$, 다항식 $p$ 에 대하여

$$p(A)=0$$

을 만족하는 다항식을 소거다항식(annihilating polynomial) 이라 하고, 최고차항 계수가 $1$인 다항식 중 차수가 가장 낮은 것을 $A$의 최소다항식이라 하고 $m_A(t)$로 나타낸다.

최소다항식은 각 행렬에 대해 유일하다.

### 정리. 최소다항식 정리

임의의 다항식 $p(t)\in\mathbb F[t]$에 대하여

$$
\boxed{p(A)=0\iff m_A(t)\mid p(t)}
$$

즉, 행렬 $A$를 영행렬로 만드는 모든 다항식은 최소다항식의 배수다.

>**다항식의 나눗셈 표기**
>
>세로선 $\mid$는 '나눈다' 또는 '약수다'라는 뜻이다. 두 다항식 $f(t),g(t)\in\mathbb F[t]$에 대하여
>
>$$
>\boxed{f(t)\mid g(t) \iff g(t)=q(t)f(t)\text{인 다항식 }q(t)\in\mathbb F[t]\text{가 존재한다}}
>$$
>
>라고 정의한다. 즉, $m_A(t)\mid p(t)$ 는 최소다항식 $m_A(t)$가 다항식 $p(t)$를 나눈다는 뜻이며, $p(t)$가 $m_A(t)$의 다항식 배수라는 뜻이다.

**증명**

**1. $p(A)=0$이면 $m_A(t)\mid p(t)$**

$p(A)=0$이라고 가정하자.

다항식 나눗셈 정리에 의해 $p(t)$를 $m_A(t)$로 나누면 몫 $q(t)$와 나머지 $r(t)$가 유일하게 존재하여 $p(t)=q(t)m_A(t)+r(t)$ 로 나타낼 수 있다. 이때 나머지는 $r(t)=0$ 이거나 $\deg r<\deg m_A$ 를 만족한다.

위 등식에 행렬 $A$를 대입하면 $p(A)=q(A)m_A(A)+r(A)$ 이다.

가정에 의해 $p(A)=0$이고, 최소다항식의 정의에 의해 $m_A(A)=0$이다. 따라서 $0=q(A)\cdot 0+r(A)=r(A)$  이므로 $r(A)=0$이다.

이제 $r(t)\neq0$이라고 가정해 보자. 그러면 $r(A)=0$이므로 $r(t)$도 $A$의 소거다항식이다. 그런데 나눗셈 정리에 의해 $\deg r<\deg m_A$ 이다.

이는 $m_A(t)$가 모든 $0$이 아닌 소거다항식 중 차수가 가장 작은 다항식이라는 사실에 모순이다. 따라서 $r(t)=0$이다.

그러므로 $p(t)=q(t)m_A(t)$ 이고, 이는 $m_A(t)\mid p(t)$ 를 뜻한다.

**2. $m_A(t)\mid p(t)$이면 $p(A)=0$임을 증명**

이번에는 $m_A(t)\mid p(t)$ 라고 가정하자.

나눗셈의 정의에 따라 어떤 다항식 $q(t)$가 존재하여 $p(t)=q(t)m_A(t)$  로 쓸 수 있다.

행렬 $A$를 대입하면 $p(A)=q(A)m_A(A)$ 이다. 최소다항식의 정의에 의해 $m_A(A)=0$ 이므로 $p(A)=q(A)\cdot0=0$ 이다.

$$
\therefore \boxed{p(A)=0 \iff m_A(t)\mid p(t)}
$$

**예시: 멱등행렬에 적용**

$A^2=A$이면 $A^2-A=0$ 이므로 $p(t)=t(t-1)$에 대하여 $p(A)=0$이다. 따라서 위 정리에 의해

$$
\boxed{m_A(t)\mid t(t-1)}
$$

### 최소다항식과 특성다항식의 관계

$A$의 특성다항식을 $\chi_A(t)=\det(tI-A)$ 라 하자.

케일리–해밀턴 정리에 의해 $\chi_A(A)=0$ 이다. 즉, 특성다항식은 $A$의 소거다항식이다. 따라서 앞의 정리에 의해

$$
\boxed{m_A(t)\mid\chi_A(t)}
$$

이다.

따라서 다음 관계가 성립한다.

$$
\boxed{m_A(A)=0,\quad m_A(t)\mid\chi_A(t)}
$$

다만 최소다항식과 특성다항식이 항상 같지는 않다. 예를 들어 $A=I_n$이면 $m_A(t)=t-1$ 이지만 $\chi_A(t)=(t-1)^n$ 이다.

### 정리. 최소다항식과 고윳값에 관한 정리

$$
\boxed{\lambda\text{가 }A\text{의 고윳값}
\iff m_A(\lambda)=0}
$$

즉, 최소다항식의 근들은 정확히 $A$의 서로 다른 고윳값들이다.

**증명**

1. $\lambda$가 고윳값이면 $m_A(\lambda)=0$

$\lambda$가 고윳값이면 영이 아닌 벡터 $\mathbf v$가 존재하여 $A\mathbf v=\lambda\mathbf v$ 이다. 임의의 다항식 $p(t)$에 대하여 $p(A)\mathbf v=p(\lambda)\mathbf v$ 가 성립한다.

특히 $m_A(A)=0$이므로 $0=m_A(A)\mathbf v=m_A(\lambda)\mathbf v.$

$\mathbf v\neq0$이므로 $m_A(\lambda)=0$ 이다.

2. $m_A(\lambda)=0$이면 $\lambda$가 고윳값임

만약 $\lambda$가 고윳값이 아니라면 $A-\lambda I$는 가역행렬이다.

그런데 $m_A(\lambda)=0$이므로 인수정리에 의해 $m_A(t)=(t-\lambda)q(t)$ 로 쓸 수 있다. 따라서 $0=m_A(A)=(A-\lambda I)q(A).$

$A-\lambda I$가 가역행렬이므로 양변에 그 역행렬을 곱하면 $q(A)=0$ 이다. 그런데 $\deg q<\deg m_A$ 이므로 최소다항식의 최소성에 모순이다. 따라서 $\lambda$는 $A$의 고윳값이다.

### 정리. 대각화에 관한 최소다항식 정리

앞에서 멱등행렬이 대각화 가능하다고 결론 내릴 때 직접 사용한 정리는 다음이다:

행렬 $A\in\mathbb F^{n\times n}$가 $\mathbb F$ 위에서 대각화 가능할 필요충분조건은 최소다항식이 $\mathbb F$ 위에서 서로 다른 일차인수들의 곱으로 분해되는 것이다.

즉,

$$
\boxed{
A\text{가 대각화 가능} \iff m_A(t)=\prod_{j=1}^{k}(t-\lambda_j)
}
$$

이며 여기서 $\lambda_1,\dots,\lambda_k$는 서로 다르다.

이를 흔히 "최소다항식을 이용한 대각화 판정 정리"라고 한다.

>**필요조건 증명**
>
>$A$가 대각화 가능하다고 하자. 그러면 어떤 가역행렬 $P$에 대하여 $A=PDP^{-1}$ 이고, $D=\text{diag}(\lambda_1,\dots,\lambda_n)$ 이다.
>
>$A$의 서로 다른 고윳값을 $\mu_1,\dots,\mu_k$ 라 하고 $p(t)=\prod_{j=1}^k(t-\mu_j)$ 라 놓자. 그러면 모든 대각성분에 대하여 $p(\lambda_i)=0$이므로 $p(D)=0.$
>
>따라서
>
>$$
>p(A)=Pp(D)P^{-1}=0.
>$$
>
>그러므로 최소다항식은 $p(t)$를 나눈다. $m_A(t)\mid p(t).$
>
>한편 모든 고윳값 $\mu_j$는 최소다항식의 근이어야 하므로 $t-\mu_j$가 모두 $m_A(t)$의 인수다. 결국
>
>$$
>m_A(t)=\prod_{j=1}^k(t-\mu_j).
>$$
>
>따라서 최소다항식에는 중복된 인수가 없다.
>
>**충분조건 증명**
>
>최소다항식이 $m_A(t)=\prod_{j=1}^k(t-\lambda_j), \quad \lambda_i\neq\lambda_j\;(i\neq j)$ 라고 하자.
>
>각 $j$에 대하여 라그랑주 다항식
>
>$$
>p_j(t)
>=\prod_{\ell\neq j}
>\frac{t-\lambda_\ell}{\lambda_j-\lambda_\ell}
>$$
>
>를 정의하면
>
>$$
>p_j(\lambda_i)=
>\begin{cases}
>1,&i=j,\\
>0,&i\neq j
>\end{cases}
>$$
>
>이다. 또한 다항식 항등식으로 $\sum_{j=1}^kp_j(t)=1 \pmod{m_A(t)}$ 이므로 $A$를 대입하면 $I=\sum_{j=1}^kp_j(A).$
>
>따라서 임의의 $\mathbf v\in\mathbb F^n$는 $\mathbf v = \sum_{j=1}^kp_j(A)\mathbf v$ 로 분해된다.
>
>그리고 $(A-\lambda_jI)p_j(A)=0$ 이므로 $p_j(A)\mathbf v\in\ker(A-\lambda_jI).$
>
>즉, 모든 벡터가 $A$의 고유벡터 공간들의 합으로 표현된다.
>
>$$
>\mathbb F^n = \ker(A-\lambda_1I)+\cdots+
>\ker(A-\lambda_kI).
>$$
>
>서로 다른 고윳값에 대응하는 고유공간들의 합은 직합이므로
>
>$$
>\mathbb F^n = \bigoplus_{j=1}^k\ker(A-\lambda_jI).
>$$
>
>따라서 각 고유공간의 기저를 합치면 $\mathbb F^n$의 고유벡터 기저가 만들어진다. 그러므로 $A$는 대각화 가능하다.

### 중복인수가 대각화를 방해하는 이유

최소다항식이

$$
m_A(t)=\prod_{j=1}^k(t-\lambda_j)^{s_j}
$$

라고 하자. $s_j>1$인 인수가 존재한다는 것은 해당 고윳값에 대해 크기가 $2$ 이상인 조르당 블록이 존재한다는 뜻이다. 즉, 일반 고유벡터가 필요하므로 고유벡터만으로 전체 공간의 기저를 만들 수 없다.

따라서

$$
\boxed{
A\text{가 대각화 가능}
\iff
m_A(t)\text{의 모든 근의 중복도가 }1
}
$$

이다.

특성다항식은 중근을 가져도 대각화가 가능할 수 있다는 점에 주의해야 한다. 예를 들어 $I_n$은 대각화 가능하지만

$$
\chi_{I_n}(t)=(t-1)^n
$$

은 중근을 갖는다. 중요한 것은 특성다항식이 아니라 최소다항식에 중복인수가 있는지 여부다.



# 3. 중복도(Multiplicity)
대각화 가능 판단하는 다른 방법!  

## (1) 정의 (Definitions)
* $\lambda$가 $n \times n$ 행렬 $A$의 고윳값이면,
  고유공간의 차원을 기하적 중복도(geometric multiplicity) 라 한다.

* 고유다항식에서 $\lambda$가 인수로 나타나는 횟수를
  대수적 중복도(algebraic multiplicity) 라고 한다.
   - 특성방정식을 인수분해 했을 때, 해당 고윳값의 항이 몇 제곱인지 말함. $(\lambda - 2)(\lambda - 1)^2$에서 2의 대수적 중복도는 1, 1의 대수적 중복도는 2
   - $m_a(\lambda)$라고 한다.

* 기하적 중복도: 고윳값 $\lambda$의 고유공간 차원이다.
  - $m_g(\lambda) = \dim\text{Null}(A-\lambda I).$
  - 항상 $\boxed{ 1\leq m_g(\lambda)\leq m_a(\lambda) }$ 가 성립한다.

## (2) 정리 (Multiplicity Theorem)
정사각행렬 $A$에 대하여 다음 두 명제는 동치이다:

1. $A$는 대각화 가능하다.
2. 모든 고윳값에 대하여
   기하적 중복도 = 대수적 중복도.

동치로

$$ \boxed{ A\text{가 대각화 가능} \iff \sum_{\lambda} \dim\text{Null}(A-\lambda I)=n } $$


# 4. 닮음(Similarity), 불변량 (Invariants under Similarity)
## (1) 정의: 닮음(Similarity)
두 정사각행렬 $A, B$에 대하여
가역행렬 $P$가 존재해

$$
B = P^{-1} A P
$$

를 만족하면, $A, B$는 서로 닮음(similar)이라고 하고
$A \sim B$로 표현한다. 닮음변환 혹은 상사변환이라고도 함  

## (2) 닮음 불변량 (Similarity Invariants)
서로 닮은 두 행렬 $A, B$는 다음 성질을 모두 동일하게 가진다.

1. 행렬의 차원 (size)
2. 가역성 (invertibility)
3. rank
4. nullity
5. 고유다항식 (characteristic polynomial)
6. 고윳값 (eigenvalues)
7. 고유공간의 차원
8. 대각성분들의 합(trace)
9. 대수적 중복도
10. 기하적 중복도  
    …

# 5. 케일리–해밀턴 정리 (Cayley–Hamilton Theorem)
임의의 정사각행렬 $A$와 그 고유다항식

$$
f(\lambda) = \det(\lambda I - A) = \sum_{k=0}^n a_k \lambda^k
$$

에 대하여 다음이 성립한다:

$$
f(A) = 0
$$
(다항식의 람다 변수에 행렬 A 넣음)  
이를 케일리–해밀턴 정리(Cayley–Hamilton theorem)라고 한다.
(단, $0$은 영행렬)

### 참고: 수반행렬(adj(A), adjugate matrix)
수반행렬(adj(A), adjugate matrix)의 정의  
* 어떤 정방행렬 $A$가 있을 때,
* $A$의 각 원소 $a_{ij}$의 여인수(cofactor) $C_{ij}$를 이용하여
* 전치된(cofactor matrix의 transpose) 행렬이 곧 수반행렬이다.

$$
\text{adj}(A) = (C_{ij})^{T}
$$

여기서 $C_{ij} = (-1)^{i+j} M_{ij}$,  
$M_{ij}$는 $(i,j)$ 소행렬식(minor)이다.

즉, 소행렬식 → 여인수 → 여인수행렬 → 전치 → 수반행렬

**수반행렬의 핵심 성질**  
수반행렬(adj A)이 중요한 이유는 다음의 기본 성질을 만족하기 때문이다:

$$\text{adj}(A)A = \det(A)I$$ 또는 $$A\text{adj}(A)=\det(A)I$$  
이 식은 **항상 성립**한다.

따라서 만약 $\det(A) \neq 0$이면,

$$

A^{-1} = \frac{1}{\det(A)} \text{adj}(A)
$$

**증명**

$B = \text{adj}(\lambda I - A)$라 하면, $B$는 $\lambda$에 대한 다항식 행렬이다.

수반행렬의 성질에 의해 $B(\lambda I - A) = \det(\lambda I - A) I = f(\lambda) I$

$B$는 $\lambda$에 대한 차수가 최대 $n-1$인 다항식 행렬이므로 $B = B_0 + B_1 \lambda + B_2 \lambda^2 + \cdots + B_{n-1} \lambda^{n-1}$ 로 쓸 수 있다. (여기서 $B_i$는 상수행렬)

따라서

$$
(B_0 + B_1 \lambda + \cdots + B_{n-1} \lambda^{n-1})(\lambda I - A) = f(\lambda) I
$$

좌변을 전개하면

$$
B_0(\lambda I - A) + B_1 \lambda(\lambda I - A) + \cdots + B_{n-1} \lambda^{n-1}(\lambda I - A) = f(\lambda) I
$$

양변의 $\lambda$의 각 차수 계수를 비교하면, 특히 $\lambda = A$를 대입하면 $B(A)(AI - A) = f(A)I$ 즉, $B(A) \cdot 0 = f(A)I$ 

따라서 $f(A) = 0$ $\square$


# [연습 문제]
1. 다음 행렬 $A = \begin{pmatrix} 0 & -3 & -3 \\ 1 & 4 & 1 \\ -1 & -1 & 2 \end{pmatrix}$ 에 대해

(1) $A$를 대각화하는 행렬 $P$를 구하고, 대각행렬

$$B = P^{-1} A P$$

를 구하시오.

(2) 두 행렬 $A, B$에 대하여 위에서 제시된 10가지 닮음 불변량을 확인하시오.

2. 행렬
   $$
   M = \begin{pmatrix}
   0 & 1 & 0 \\
   0 & 0 & 1 \\
   1 & -3 & 3
   \end{pmatrix}
   $$
   에 대하여 행렬 $3M^5 - 5M^4$ 를 C-H 정리를 이용하여 구하시오.
