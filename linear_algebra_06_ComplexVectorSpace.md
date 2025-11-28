# 1. 복소벡터공간 (Complex Vector Space)
## (1) 정의 (Definition)
복소수체 $\mathbb{C}$에 대한 가군, 즉 적당한 집합 $V$에 대해 벡터공간 $(V, \mathbb{C}, +, \cdot)$을 복소벡터공간이라 한다.

또한 모든 복소 $n$-튜플 $(v_1, v_2, \ldots, v_n)$의 집합을 복소 $n$-공간이라 하고 이를 $\mathbb{C}^n$으로 표기한다.

## (2) 복소켤레 (Complex Conjugate)

$ \mathbb{C}^n $의 임의의 벡터
$ v = (v_1, v_2, \ldots, v_n) $

각 성분 $v_k = a_k + b_k i$에 대해

$$
v = (a_1 + b_1 i, \ldots, a_n + b_n i)
= (a_1, \ldots, a_n) + i (b_1, \ldots, b_n)
$$

실수부, 허수부는 다음과 같다.

$$
\operatorname{Re}(v) = (a_1, \ldots, a_n), \quad
\operatorname{Im}(v) = (b_1, \ldots, b_n)
$$

벡터의 복소켤레는

$$
\overline{v} = (\overline{v_1}, \ldots, \overline{v_n})
= \operatorname{Re}(v) - i \operatorname{Im}(v)
$$

## (3) 대수적 성질 (Algebraic Properties)
$ \mathbb{C}^n $의 벡터 $u, v$와 스칼라 $k$에 대해 다음이 성립한다.
1. $\overline{u} = u$ (실수 벡터)
2. $\overline{k u} = \overline{k} , \overline{u}$
3. $u \pm \overline{v} = \overline{u \pm v}$ 

$m \times k$ 행렬 $A$와 $k \times n$ 행렬 $B$에 대해
1. $\overline{A} = A$
2. $\overline{A^T} = \overline{A^T}$
3. $\overline{AB} = \overline{A} , \overline{B}$

(여기서 bar 기호는 성분별 complex conjugation)

# 2. 복소내적공간 (Complex Inner Product Space)
## (1) 정의 (Definition)
복소벡터공간 $(V, \mathbb{C}, +, \cdot)$의 두 벡터
$u = (u_1, \ldots, u_n)$, $v=(v_1,\ldots,v_n)$에 대해

내적을
$$
\langle u, v \rangle = u_1 \overline{v_1} + \cdots + u_n \overline{v_n}
$$

으로 정의한다. 또한 이 내적이 정의되어 있는 복소벡터공간을 복소내적공간이라 한다.

## (2) 성질 (Properties)
복소내적공간의 임의의 벡터 $u, v, w$와 스칼라 $k$에 대해 다음이 성립한다.

1. $\langle u, v \rangle = \overline{\langle v, u \rangle}$
2. $\langle u+v, w \rangle = \langle u, w \rangle + \langle v, w \rangle$
3. $\langle k u, v \rangle = k \langle u, v \rangle$  
$\langle u, kv \rangle = \bar k \langle u, v \rangle$
4. $\langle u, u \rangle = 0$ 일 때 $u=0$  
$v \neq 0$일때 $\langle v, v \rangle \gt 0$

# 3. 고윳값과 벡터 (Eigenvalues and Eigenvectors)
## (1) 정의
복소정사각행렬 $A$에 대하여 (즉, 선형사상에 대해) 고유방정식
$$
\det(\lambda I - A) = 0
$$

의 해 $\lambda$를 $A$의 복소고윳값이라 한다.

또한 $A v = \lambda v$를 만족시키는 모든 벡터 $v$의 집합을 고유공간(eigenspace)이라 한다.
고유공간의 영벡터가 아닌 벡터를 복소고유벡터라 한다.

## (2) 정리
실 정사각행렬 (모두 실수) $A$의 고윳값이 양수 $\lambda$이면 이에 대응하는 고유벡터는 실수벡터이면, 복소 고윳값 $\bar\lambda$또한 $A$의 고윳값이며 $\bar v$는 이에 대응하는 고유벡터다.

# 4. 유니터리 대각화 (Unitary Diagonalization)
## (1) 용어의 정의
### 1) 켤레전치행렬 (Conjugate Transpose, Hermitian Transpose)
복소행렬 $A$의 **전치행렬**을 구한 다음 **각 성분을 복소수로 켤레** 취해 얻는 행렬 $A^*$를 켤레전치(conjugate transpose)라 한다. (별 말고 H라 표기하기도 함)  
이는 에르미트 전치(Hermitian transpose)라고도 한다.

성질:

* $(A^*)^* = A$
* $(A \pm B)^* = A^* \pm B^*$
* $(kA)^* = \overline{k} A^*$
* $(AB)^* = B^* A^*$



### 2) 에르미트행렬 (Hermitian Matrix)

$A = A^*$가 성립하는 복소정사각행렬 $A$를 에르미트(Hermitian) 행렬이라 한다.

### 3) 유니터리행렬 (Unitary Matrix)

복소정사각행렬 $A$에 대하여

$$
A^{-1}=A^*
$$
가 성립하는 행렬을 유니터리(unitary) 행렬이라 한다.

### 4) 정규행렬 (Normal Matrix)

$$
A A^* = A^* A
$$

가 성립하는 복소정사각행렬을 정규행렬(normal matrix)라 한다.
에르미트, 유니터리 행렬 등이 이에 포함된다.

## (2) 유니터리 대각화 (Unitary Diagonalization)
### 정의
행렬 $A$가 정규행렬일 때, 유니터리행렬 $P$가 존재하여
$$
P^* A P = D
$$
가 되는 경우 $A$는 유니터리 대각화(unitarily diagonalizable) 가능하다고 한다. D는 A의 고윳값을 대각에 배열한 Diagonal matrix임!  
**A보다 훨씬 간단한 D를 가지고 Similarity Invariants에 의해 10가지 성질을 동일하게 얘기할 수 있다!**  



### 정리
정규행렬은 유니터리 대각화 가능하며, 그 역도 성립한다. 즉 정규행렬인 경우에만 유니터리 대각화가 가능하다.

## (3) 에르미트행렬의 유니터리 대각화 과정
(유니터리임을 보이는건 역행렬, Hermitian만들고 해야되서 복잡하지만, Hermitian인지 확인하는거는 원본이랑 Hermitian (transpose 부분만 체크) 비교하면 되서 쉽다.)  

Step 1. $A$의 모든 고유공간의 기저를 구한다.  
Step 2. 고유공간마다 정규직교기저를 만든다.  
Step 3. 기저벡터를 열 벡터로 하는 행렬 $P$는   유니터리행렬이고,
$A$를 대각화한다.

ex: 
$$
A = \begin{pmatrix}
1 & 1+i \\
1-i & 0
\end{pmatrix}
$$

### Step 1. 고윳값, 고유공간 기저  
#### 1) 고윳값
$\det(\lambda I - A) = \begin{vmatrix} \lambda-1 & -(1+i) \\ -(1-i) & \lambda \end{vmatrix} = \lambda^2 - \lambda - 2$

$(\lambda-2)(\lambda+1)=0$

$\lambda_1=2,\quad \lambda_2=-1$

#### 2) $\lambda=2$
$A-2I= \begin{pmatrix} -1 & 1+i \\ 1-i & -2 \end{pmatrix} \sim \begin{pmatrix} 1 & -(1+i) \\ 0 & 0 \end{pmatrix}$

$x-(1+i)y=0 \Rightarrow x=(1+i)y$

$v_2= \begin{pmatrix} 1+i \\ 1 \end{pmatrix}$

#### 3) $\lambda=-1$
$A+I= \begin{pmatrix} 2 & 1+i \\ 1-i & 1 \end{pmatrix} \sim \begin{pmatrix} 1 & \frac{1+i}{2} \\ 0 & 0 \end{pmatrix}$

$x + \frac{1+i}{2}y = 0 \Rightarrow x=-\frac{1+i}{2}y$

$v_{-1}\sim \begin{pmatrix} -1-i \\ 2 \end{pmatrix}$

### Step 2. 정규직교기저
#### 1) $\lambda=2$
$|v_2|^2=(1-i)(1+i)+1=3$

$u_2=\frac{1}{\sqrt{3}} \begin{pmatrix} 1+i \\ 1 \end{pmatrix}$

#### 2) $\lambda=-1$
$|v_{-1}|^2=(-1+i)(-1-i)+4=6$

$u_{-1}=\frac{1}{\sqrt{6}} \begin{pmatrix} -1-i \\ 2 \end{pmatrix}$

### Step 3. 유니터리행렬 $P$와 대각화
$P = \begin{pmatrix} \frac{1+i}{\sqrt{3}} & \frac{-1-i}{\sqrt{6}} \\ \frac{1}{\sqrt{3}} & \frac{2}{\sqrt{6}} \end{pmatrix}$

$D = \begin{pmatrix} 2 & 0 \\ 0 & -1 \end{pmatrix}$

따라서 $P^* A P = D$


# 연습 문제
1. 세 벡터
$v_1 = (1 - i,\ 4 + 2i,\ 3)$,
    $v_2 = (2,\ 3i,\ 4-i)$,
    $v_3 = (2 + i,\ -2i,\ 4 - 5i)$
    에 대하여 $\overline{\langle u, w \rangle + \langle \|u\| v, u \rangle}$
    을 계산하시오.

2. 모든 $2 \times 2$ 실행렬 $A$의 대각성분의 총합을 $\operatorname{tr}(A)$라 할 때,
   $\operatorname{tr}(A)^2 -4det(A) \lt 0$이면 A는 두개의 복소켤레 고윳값을 가짐을 증명하라.

3. 에르미트행렬
   $$
   A = \begin{pmatrix}
   2 & 1+i \\
   1-i & 3
   \end{pmatrix}  
   $$

   을 유니터리 대각화하시오.
