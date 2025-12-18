# 1. 고윳값(Eigenvalue)과 고유벡터(Eigenvector)
## (1) 정의 (Definition)
**비유: 선형사상을 나타내는 방향(또는 축, eigenvector), 크기(eigenvalue)를 나타냄.**  

체 $F$ 위의 벡터공간 $V$의 선형사상
$L : V \to V$ 에 대하여, 다음 조건을 만족하는 $\lambda \in F$, $v \in V \setminus {0}$ 를 각각
* 고윳값(eigenvalue)
* 고유벡터(eigenvector)  

라고 한다.
1. $v \ne 0$
2. $L(v) = \lambda v$

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

# 2. 대각화(Diagonalization)
## (1) Definition
두 정사각행렬 $A, B$에 대하여 다음을 만족하는 가역행렬 $P$가 존재하면

$$B = P^{-1} A P$$

$A$는 대각화 가능(diagonalizable)이라고 하며,
이때의 $P$는 $A$를 대각화하는 행렬(diagonalizing matrix)라고 한다.  

A, P, B의 관계는 무었인가! 

## (2) 정리 (Diagonalization Theorem)
$n \times n$ 행렬 $A$에 대하여 다음 두 명제는 동치이다.
1. $A$는 대각화 가능하다.
2. $A$는 선형독립인 고유벡터를 $n$개 가진다.

**증명:**  
$(1 \Rightarrow 2)$  
$A$가 대각화 가능하다고 하자. 즉, 가역행렬 $P$가 존재하여
$$P^{-1}AP = D = \begin{pmatrix} \lambda_1 & & \\ & \ddots & \\ & & \lambda_n \end{pmatrix}$$

$P$의 열벡터를 $v_1, v_2, \dots, v_n$이라 하면
$$AP = PD$$
이므로
$$A(v_1, v_2, \dots, v_n) = (v_1, v_2, \dots, v_n) \begin{pmatrix} \lambda_1 & & \\ & \ddots & \\ & & \lambda_n \end{pmatrix}$$

따라서 $Av_i = \lambda_i v_i$이고, $P$가 가역이므로, n개의 선도1이 존재하고, 이는 행또는 열벡터들은 n차원 공간의 기저를 이루는 것이므로, $v_1, v_2, \dots, v_n$은 선형독립이다.
즉, $A$는 선형독립인 고유벡터를 $n$개 가진다. 

$(2 \Rightarrow 1)$  
$A$가 선형독립인 고유벡터 $v_1, v_2, \dots, v_n$을 가진다고 하자.
각 고유벡터에 대응하는 고윳값을 $\lambda_1, \lambda_2, \dots, \lambda_n$이라 하면

$$A v_i = \lambda_i v_i, \quad i = 1, 2, \dots, n$$

행렬 $P = (v_1, v_2, \dots, v_n)$이라 하면, $v_i$들이 선형독립이므로 $P$는 가역이다.

$$AP = A(v_1, v_2, \dots, v_n) = (\lambda_1 v_1, \lambda_2 v_2, \dots, \lambda_n v_n)$$

$$= (v_1, v_2, \dots, v_n) \begin{pmatrix} \lambda_1 & & \\ & \ddots & \\ & & \lambda_n \end{pmatrix} = PD$$

따라서 $P^{-1}AP = D$이고 $A$는 대각화 가능하다.  
$\square$


## (3) 대각화 방법
$n \times n$ 행렬 $A$에 대하여:  
Step 1.
$n$개의 선형독립 고유벡터를 찾아 대각화 가능 여부를 확인한다.  
Step 2.
고유벡터들을 열로 하는 행렬 $P = (v_1, v_2, \dots, v_n)$ 을 만든다.  
Step 3.
$$
P^{-1} A P = D
$$
은 대각행렬이 된다.

# 3. 중복도(Multiplicity)
대각화 가능 판단하는 다른 방법!  
## (1) 정의 (Definitions)
* $\lambda$가 $n \times n$ 행렬 $A$의 고윳값이면,
  고유공간의 차원을 기하적 중복도(geometric multiplicity) 라 한다.

* 고유다항식에서 $\lambda$가 인수로 나타나는 횟수를
  대수적 중복도(algebraic multiplicity) 라고 한다.
   - 특성방정식을 인수분해 했을 때, 해당 고윳값의 항이 몇 제곱인지 말함. $(\lambda - 2)(\lambda - 1)^2$에서 2의 대수적 중복도는 1, 1의 대수적 중복도는 2

## (2) 정리 (Multiplicity Theorem)

정사각행렬 $A$에 대하여 다음 두 명제는 동치이다:

1. $A$는 대각화 가능하다.
2. 모든 고윳값에 대하여
   기하적 중복도 = 대수적 중복도.

# 4. 닮음(Similarity) 불변량 (Invariants under Similarity)

## (1) 정의

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
f(A) = 0.
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
\operatorname{adj}(A) = (C_{ij})^{T}.
$$

여기서 $C_{ij} = (-1)^{i+j} M_{ij}$,  
$M_{ij}$는 $(i,j)$ 소행렬식(minor)이다.

즉, 소행렬식 → 여인수 → 여인수행렬 → 전치 → 수반행렬

**수반행렬의 핵심 성질**  
수반행렬(adj A)이 중요한 이유는 다음의 기본 성질을 만족하기 때문이다:

$$\operatorname{adj}(A)A = \det(A)I$$ 또는 $$A\operatorname{adj}(A)=\det(A)I$$  
이 식은 **항상 성립**한다.

따라서 만약 $\det(A) \neq 0$이면,
$$
A^{-1} = \frac{1}{\det(A)} \operatorname{adj}(A)
$$

## 증명
$B = \operatorname{adj}(\lambda I - A)$라 하면, $B$는 $\lambda$에 대한 다항식 행렬이다.

수반행렬의 성질에 의해
$$
B(\lambda I - A) = \det(\lambda I - A) I = f(\lambda) I
$$

$B$는 $\lambda$에 대한 차수가 최대 $n-1$인 다항식 행렬이므로
$$
B = B_0 + B_1 \lambda + B_2 \lambda^2 + \cdots + B_{n-1} \lambda^{n-1}
$$
로 쓸 수 있다. (여기서 $B_i$는 상수행렬)

따라서
$$
(B_0 + B_1 \lambda + \cdots + B_{n-1} \lambda^{n-1})(\lambda I - A) = f(\lambda) I
$$

좌변을 전개하면
$$
B_0(\lambda I - A) + B_1 \lambda(\lambda I - A) + \cdots + B_{n-1} \lambda^{n-1}(\lambda I - A) = f(\lambda) I
$$

양변의 $\lambda$의 각 차수 계수를 비교하면, 특히 $\lambda = A$를 대입하면

$$
B(A)(AI - A) = f(A)I
$$

즉,
$$
B(A) \cdot 0 = f(A)I
$$

따라서
$$
f(A) = 0
$$

$\square$


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
