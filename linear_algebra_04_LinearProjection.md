# 1. 선형사상 (Linear Map/ Linear Transform/ Linear morphism)
사상(morphism): 특정 구조를 보존하는(대수 구조를 다루는) 함수  
함수: 원소 대응 규칙 두가지를 만족하는 관계.  
## (1) 선형사상
선형결합은 기저로 벡터공간 생성하는 그런거고, 선형사상은 벡터의 함수, 벡터의 변환 개념.  

### ① 정의
$체 F$-벡터공간 $V,\ W$에 대하여
$V$의 성질을 보존하는 다음 두 조건을 만족하는 사상을
$L(Linear):V \to W$라 한다.

1. Additivity(가산성): $L(u+v)=L(u)+L(v)$  $(u,v\in V)$
2. Homogeneity(동차성): $L(kv)=kL(v)$  $(k\in F,\ v\in V)$  
(Additivity는 벡터끼리의 덧셈에 대한 성질이고
Homogeneity는 스칼라배에 대한 성질이므로 다른 개념임)  

### ② 관련 용어
선형사상 $L:V\to W$에서

* **핵(ker, kernel)**:
  $\ker L=\{v\in V\mid L(v)=0\}$

* **상(im, image)**:
  $\operatorname{im}L = L(V)=\{L(v)\mid v\in V\}$

* **자기사상 (endomorphism)**:
  $V=W$일 때 $L:V\to V$
  - 두 개의 서로 다른 원소가 같은 결과로 매핑되는 경우는 정의역과 치역이 같더라도
그 선형사상은 자기사상(endomorphism)은 맞지만, 단사(injective)는 아니다

* **단사사상 (injective map, monomorphism)**:
  $L(u)=L(v)\Rightarrow u=v$인 $L$

* **전사사상 (surjective map)**:
  $L(V)=W$인 $L$

* **동형사상 (isomorphism)**:
  단사이면서 전사인 사상
  - 대수구조 동일함을 보이는데 필요!
  - '구조적으로 동일하다'는 뜻이지,
원소의 내용물 이 반드시 같아야 한다는 뜻은 아니다
    - domain, codomain 내용물이 같은것은 endomorphism임

* **자기동형사상(automorphism)**:
  자기사상인 동형사상

* **항등사상 (identity map)**:
  $L(v)=v$인 $L=I_V$

### ③ 사상의 합성(Composition), 역사상(Inverse)
두 선형사상
$L_1:V\to U,\ L_2:U\to W$에 대해 합성을
$L_2\circ L_1 : V\to W$로 쓴다.

* $L_2\circ L_1$은 선형사상이다.
  - $L_2$를 $L_1$의 왼쪽 역사상(Left Inverse Map), $L_1$을 $L_2$의 오른쪽 역사상(Right Inverse Map) 이라 한다
* 왼쪽 역사상이자 오른쪽 역사상을 (L1, L2 순서 바꿔도 항등사상이 될 때) 양쪽 역사상 혹은 역사상 이라 함

## (2) 여러 선형사상
$L : V \to W$가 선형사상이고 $v \in V$일 때,
1. $L(v) = \mathbf{0}$ : 영사상(zero map)
2. $L(v) = v$ : identity map
3. $L(v) = kv$  (단, $k$는 스칼라)
4. $L(v) = Mv$  
   (단, $M \in M_{m \times n}(F)$, $V = F^n$, $W = F^m$)

5. $L(v) = \langle v, v_0 \rangle$  (단, $v_0 \in V$)   

등등  

# 2. 선형대수학의 기본정리
$F$-벡터공간 $V,\ W$에 대해
$V$에서 $W$로의 선형사상들의 집합을
$\mathcal{L}(V,W)$라 하고,
다음과 같이 스칼라곱 및 덧셈을 정의한다. ($v\in V,\ k\in F$)

1. $(L_1+L_2)(v)=L_1(v)+L_2(v)$
2. $(kL)(v)=kL(v)$

이제 $F$ 위의 $m\times n$ 행렬들의 집합을 $M_{m\times n}(F)$라 하고
두 사상 $f,g$를 다음과 같이 정의한다.
* 두 대수구조 $(\mathcal{L}(V,W), +, \cdot)$과 $(M_{m\times n}(F), +, \cdot)$ 의 동치를 보려고 함 !!
* $f:\mathcal{L}(V,W)\to M_{m\times n}(F)$

* $f(L)=[L]_{B_V}^{B_W}=M$

* $g:M_{m\times n}(F)\to \mathcal{L}(V,W)$

* $g(M)=L_M \quad {이때} \big(\,\left[L_M(v)\right]_{B_W}=M\left[v\right]_{B_V}\,\big)$


**기호 정의** 
1. $B_V$는 $V$의 기저, $B_W$는 $W$의 기저
  - 즉 기저의 원소들은 순서가 정해져 있고 바뀌지 않는다
2. 벡터 $v\in V$가
   $v=k_1v_1+\cdots+k_nv_n$이라면
   $$[v]_{B_V}=(k_1,\dots,k_n)^T$$
  - 즉 $v$의 원소들을 $B_V$의 기저벡터들로 표현한 계수들의 tuple을 column으로 표현한 것

3. $[L]^{B_V}_{B_W}$는
   $([L(v_1)]_{B_W},\dots,[L(v_n)]_{B_W})$
   - $[L(v_1)]_{B_W}$는 칼럼벡터고, 그래서 쭉 hstack하면 행렬이 된다.

## 증명 
**증명 스케치**  
f에 대해 선형사상 증명(additivity, homogeneity 증명)  
f가 동형사상 증명 (injective, surjective 증명)  
g에 대해 선형사상 증명(additivity, homogeneity 증명)  
g가 동형사상 증명 (injective, surjective 증명  
f와 g는 역사상 관계 증명 ($f\cdot g, g\cdot f$ 항등사상 증명)

: 두 벡터공간 $V,W$ 가 $\dim V=\dim W=n$, 기저 $B_V=\{v_1,\dots,v_n\}$, $B_W=\{w_1,\dots,w_n\}$ 일 때 $f,g$ 를 동형사상으로 구성한다.

**1. $f$가 선형사상:**  
정의 $f(v_i)=w_i\ (1\le i\le n)$, 선형확장 $f(\sum a_i v_i)=\sum a_i w_i$.

(1) Additivity:
$$
f(u+v)=f(\sum a_i v_i+\sum b_i v_i)
=\sum (a_i+b_i)w_i
=\sum a_i w_i+\sum b_i w_i
=f(u)+f(v).
$$

(2) Homogeneity:
$$
f(kv)=f(\sum k a_i v_i)
=\sum (k a_i) w_i
=k\sum a_i w_i
=k f(v).
$$

**2. $f$가 동형사상:**  

(1) 단사(injective):  
$f(v_1)=f(v_2)$ 라고 하자.  
두 벡터를 기저에 대해 전개하면
$$
f\left(\sum a_i v_i\right)
=f\left(\sum b_i v_i\right)
$$
즉
$$
\sum a_i w_i = \sum b_i w_i.
$$
$W$의 기저 $\{w_i\}$ 가 선형독립이므로
$$
a_i = b_i \quad (1\le i\le n).
$$
따라서
$$
\sum a_i v_i = \sum b_i v_i,
$$
즉 $v_1=v_2$.  
따라서 $f$는 단사이다.  

(2) 전사(surjective):  
임의의 $w\in W$ 대해 $w=\sum b_i w_i$ 이므로
$$
w=f\left(\sum b_i v_i\right).
$$
즉 $f(V)=W$.

**3. $g$가 선형사상:**  
정의 $g(w_i)=v_i$.  
Additivity:
$$
g(w+w')=g(\sum c_i w_i+\sum d_i w_i)
=\sum (c_i+d_i)v_i
=g(w)+g(w').
$$
Homogeneity:
$$
g(k w)=g(\sum k c_i w_i)
=\sum (k c_i)v_i
=k\sum c_i v_i
=k g(w).
$$

**4. $g$가 동형사상:**  
기저의 대응으로 $g(w_1)=v_1,\dots,g(w_n)=v_n$.  
$f$와 동일한 방식으로  
$g(v_1)=g(v_2) \Rightarrow v_1=v_2$ 로 단사,  
임의의 $v=\sum a_i v_i$ 를 $g(\sum a_i w_i)$ 로 표현하여 전사.

**5. $f,g$가 서로 역함수:**  
기저에서  
$$
g(f(v_i))=g(w_i)=v_i,\qquad f(g(w_i))=f(v_i)=w_i.
$$
선형확장으로 모든 $v\in V$, $w\in W$ 대해
$$
g(f(v))=v,\qquad f(g(w))=w.
$$
즉 $g\circ f=I_V,\ f\circ g=I_W$.

결론: $f,g$는 서로 역사상이며 $V$와 $W$는 동형.


# 3. 차원정리 (Rank–Nullity Theorem)
## (1) 차원정리
유한차원 벡터공간 $V$와 선형사상 $L:V\to W$에 대해:
$$
\dim(V)=\dim(\ker L)+\dim(\operatorname{im}L)
$$

### 증명
증명>  $B_V = \{v_1, \dots, v_n\}$

$\ker L = \{v_1, \dots, v_k\}$

목표:  
$B_{\operatorname{im}L}
 = \{\,L(v_{k+1}),\, L(v_{k+2}),\, \dots,\, L(v_n)\,\}$
- $\ker L$ 의 기저는 $v_1, \dots, v_k$ 이다.
- 따라서 $\dim(\operatorname{im}L) = n - k$ 이다.
- 이미지 기저는 반드시 **0이 아닌 벡터 $n-k$개**여야 한다.
- $L(v_i) = 0$ 인 벡터는 $i \le k$ 뿐이다.
- 즉, **0이 아닌 $L(v_i)$는 모두 $i > k$에서 나온다.**
- 따라서 이미지의 기저 후보는  
  $\{\,L(v_{k+1}), \dots, L(v_n)\,\}$ 이다.
- 이 집합이 span + 선형독립임을 보이면  
  곧 $B_{\operatorname{im}L}$ 이 된다.

(1) 생성(span)
모든 $v \in V$에 대하여  
$v = c_1 v_1 + c_2 v_2 + \cdots + c_k v_k + c_{k+1} v_{k+1} + \cdots + c_n v_n$

선형성에 의해  
$L(v)
 = L(c_1 v_1) + L(c_2 v_2) + \cdots + L(c_n v_n)$

핵의 원소들에서 $L(v_1)=\cdots=L(v_k)=\mathbf{0}$ 이므로,  
$L(v)
 = c_{k+1} L(v_{k+1}) + \cdots + c_n L(v_n)
 \in \operatorname{im}L$

따라서  
$\operatorname{span}\{L(v_{k+1}), \dots, L(v_n)\}
 = \operatorname{im}L$

(2) 선형독립  
$c_{k+1} L(v_{k+1}) + \cdots + c_n L(v_n) = \mathbf{0}$ 이라 하자.

그러면  
$L(c_{k+1} v_{k+1} + \cdots + c_n v_n) = \mathbf{0}$

즉,  
$c_{k+1} v_{k+1} + \cdots + c_n v_n \in \ker L$

핵의 기저는 $\{v_1, \dots, v_k\}$ 이므로  
$c_{k+1} v_{k+1} + \cdots + c_n v_n
 = d_1 v_1 + \cdots + d_k v_k$

이를 정리하면  
$c_{k+1} v_{k+1} + \cdots + c_n v_n- d_1 v_1 - \cdots - d_k v_k = \mathbf{0}$

$B_V = \{v_1, \dots, v_n\}$ 은 선형독립이므로
$c_{k+1} = \cdots = c_n = 0$

따라서 $\{L(v_{k+1}), \dots, L(v_n)\}$ 은 선형독립이다.


## (2) 비둘기집 원리
### ① 따름정리
차원이 같은 두 유한 차원 벡터공간
$V,\ W$ 사이의 선형사상 $L$이 정의되어 있으면:

$L$은 전사 $\Leftrightarrow$ 단사 $\Leftrightarrow$ 전단사

- 증명  

(1) L이 전사면 L이 단사  
가정:  
$\dim(V) = \dim(W) = n$  
만약 $L$이 전사라면, $\dim(\operatorname{im}L) = \dim(W) = n$  
Rank–Nullity 정리에 의해,  
$\dim(V) = \dim(\ker L) + \dim(\operatorname{im}L)$  
$\Rightarrow n = \dim(\ker L) + n$  
$\Rightarrow \dim(\ker L) = 0$  
$\Rightarrow \ker L = \{\mathbf{0}\}$  
  
Let: $\forall v_1, v_2 \in V,\ L(v_1)=L(v_2) $  
$L(v_1) - L(v_2) = \mathbf{0}$  
$\Rightarrow L(v_1 - v_2) = \mathbf{0}$  
그런데, 0벡터가 되는 원상은 아까 0벡터밖에 없었다.  
$v_1 - v_2 = \mathbf{0}$  
$\Rightarrow v_1 = v_2$  
따라서 전사이면 단사다.  

(2) L이 단사이면 L이 전사  
가정:  $\dim(V)=\dim(W)=n$ 이고,  $L:V\to W$ 가 단사이다.  
단사이면 $L(v)=\mathbf{0} \Rightarrow v=\mathbf{0}$ 이므로 $\ker L = \{0\}$.  
따라서
$\dim(\ker L) = 0$.

Rank–Nullity 정리에 의해,
$$
n = 0 + \dim(\operatorname{im}L)
$$
따라서,
$$
\dim(\operatorname{im}L) = n.
$$
그런데 $\operatorname{im}L \subseteq W$ 이고,
$\dim(W)=n$ 이므로,
$$
\operatorname{im}L = W.
$$
즉, $L$은 전사이다.

### ② 비둘기집 원리
공집합이 아닌 두 유한집합 $A,B$의 크기가 같을 때 함수 $f:A\to B$는 다음을 만족한다:

$f$가 전사 ⇔ $f$가 단사 ⇔ $f$가 전단사


# 4. 계수정리 (Rank Equality Theorem)

## (1) 관련 용어
행렬 $M \in M_{m \times n}(F)$ 에 대하여

* **열공간 (Column Space)**:  
  $M$의 열벡터들이 생성하는 부분공간  
  열계수: $\operatorname{col\text{-}rank}(M)$

* **행공간 (Row Space)**:  
  $M$의 행벡터들이 생성하는 부분공간  
  행계수: $\operatorname{row\text{-}rank}(M)$

* **영공간 (Null Space)**:  
  연립방정식 $Mx = 0$ 의 해공간  
  널리티: $\operatorname{nullity}(M)$


## (2) 계수정리
$$
\operatorname{col\text{-}rank}M = \operatorname{row\text{-}rank}M
$$

이 차원을 행렬 **$M$의 계수 rankM**이라 한다.

## ③ Rank–Nullity 정리
행렬 $M\in M_{m\times n}(F)$에 대하여:

$$
n = \operatorname{rank}M + \operatorname{nullity}M
$$

(차원정리와 대꾸됨!)


# rank 
## rank(계수)의 정의
행렬 $A\in\mathbb{R}^{m\times n}$의 **rank**는 다음과 같은 “서로 같은 값”으로 정의/해석된다.

- $\mathrm{rank}(A)=\dim(\mathrm{Col}(A))$ (열공간의 차원)
- $\mathrm{rank}(A)=\dim(\mathrm{Row}(A))$ (행공간의 차원)
- $\mathrm{rank}(A)=$ 피벗(pivot)의 개수
- $\mathrm{rank}(A)=$ 0이 아닌 특이값(singular value)의 개수

>**왜 $\mathrm{rank}(I-P)=n-\mathrm{rank}(P)$ 가 성립하나?**  
>이 등식은 **$P$가 사영행렬(projection matrix)**, 즉 **멱등행렬(idempotent)** 일 때 성립한다.
>
>(가정) $P$가 멱등행렬
>$$
>P^2=P
>$$
>
>1) $I-P$도 멱등행렬이다:
>$$
>(I-P)^2 = I - 2P + P^2 = I - 2P + P = I-P
>$$
>
>2) 멱등행렬의 핵심 성질:  
>멱등행렬 $K$는 고유값이 $0$ 또는 $1$만 가능하므로,
>- $\mathrm{rank}(K)=\#\{\text{고유값이 }1\}$
>- $\mathrm{tr}(K)=\sum \text{고유값}=\#\{\text{고유값이 }1\}$
>
>따라서
>$$
>\mathrm{rank}(K)=\mathrm{tr}(K)
>\quad(\text{멱등행렬에서 성립})
>$$
>
>3) 위 성질을 $P$, $I-P$에 적용하면
>$$
>\mathrm{rank}(I-P)=\mathrm{tr}(I-P)=\mathrm{tr}(I)-\mathrm{tr}(P)=n-\mathrm{tr}(P)=n-\mathrm{rank}(P)
>$$
>
>4) 추가로 $\mathrm{rank}(P)=p+1$이라면
>$$
>\mathrm{rank}(I-P)=n-(p+1)=n-p-1
>$$
>
>> 실무 통계(회귀)에서는 $P$가 “hat matrix”처럼 $X$의 열공간으로의 직교사영인 경우가 많고, 이때 $\mathrm{rank}(P)=\mathrm{rank}(X)$이며 (절편 포함 시) 보통 $p+1$이 된다.

## rank의 중요한 성질 총정리
### (1) 기본 성질
- $0 \le \mathrm{rank}(A) \le \min(m,n)$
- $\mathrm{rank}(A)=\mathrm{rank}(A^T)$
- $A$가 가역(정칙) $n\times n$이면 $\mathrm{rank}(A)=n$

### (2) 랭크-널리티 정리 (Rank–Nullity)

### (3) 곱과 rank
- $\mathrm{rank}(AB)\le \mathrm{rank}(A)$
- $\mathrm{rank}(AB)\le \mathrm{rank}(B)$
- $\mathrm{rank}(AB)\le \min(\mathrm{rank}(A),\mathrm{rank}(B))$
- $A$가 정칙이면 $\mathrm{rank}(AB)=\mathrm{rank}(B)$
- $B$가 정칙이면 $\mathrm{rank}(AB)=\mathrm{rank}(A)$
- $A$가 가역이면 $\mathrm{rank}(AB)=\mathrm{rank}(B)$, $\mathrm{rank}(BA)=\mathrm{rank}(B)$

**증명**:
- 선형변환 관점에서 $AB$의 치역은 $A$의 치역에 포함되므로: $\mathrm{Col}(AB) \subseteq \mathrm{Col}(A)$  
    따라서 $\mathrm{rank}(AB) \le \mathrm{rank}(A)$
- 마찬가지로 $\mathrm{Row}(AB) \subseteq \mathrm{Row}(B)$이므로 $\mathrm{rank}(AB) \le \mathrm{rank}(B)$

- $A$가 정칙($\det A \neq 0$)이면 $\mathrm{rank}(AB)=\mathrm{rank}(B)$
    
    **증명**: 정칙행렬 $A$는 행 소거 과정으로 $I$로 변환 가능하며, 행 소거는 rank를 보존한다. 따라서 $AB$의 rank는 $B$의 rank와 같다.

- $B$가 정칙이면 $\mathrm{rank}(AB)=\mathrm{rank}(A)$
    
    **증명**: 정칙행렬은 열 소거를 통해 열 공간을 보존하므로, $A$의 rank는 불변이다.

**일반적 성질**:
$$\mathrm{rank}(AB) + \mathrm{rank}(BC) \le \mathrm{rank}(B) + \mathrm{rank}(ABC)$$
(Sylvester 부등식의 확장)
- **Sylvester 부등식** (차원 맞을 때):
$$
\mathrm{rank}(AB)\ge \mathrm{rank}(A)+\mathrm{rank}(B)-n
\quad (A\in\mathbb{R}^{m\times n},\,B\in\mathbb{R}^{n\times p})
$$

### (4) 합과 rank (상계)
- $\mathrm{rank}(A+B)\le \mathrm{rank}(A)+\mathrm{rank}(B)$

### (5) 선형변환 관점 (공간 포함관계)
- $\mathrm{Col}(AB)\subseteq \mathrm{Col}(A)$ 이므로 $\mathrm{rank}(AB)\le \mathrm{rank}(A)$
- $\mathrm{Row}(AB)\subseteq \mathrm{Row}(B)$ 이므로 $\mathrm{rank}(AB)\le \mathrm{rank}(B)$

### (6) Gram 행렬 / 정규방정식에 자주 쓰는 성질
- $\mathrm{rank}(A^TA)=\mathrm{rank}(A)$
- $\mathrm{rank}(AA^T)=\mathrm{rank}(A)$

### (7) 특수행렬에서의 rank

**(a) 멱등행렬 $K^2=K$**
- $\mathrm{rank}(K)=\mathrm{tr}(K)$
- $I-K$도 멱등, $\mathrm{rank}(I-K)=n-\mathrm{rank}(K)$

**(b) 사영행렬(직교사영)**
보통 $P^2=P$이고 $P^T=P$ (대칭)인 경우가 많다.
- $\mathrm{rank}(P)=\dim(\text{사영되는 부분공간})$
- $\ker(P)=\mathrm{Col}(P)^\perp$ (직교사영일 때)
- $\mathrm{rank}(P)+\mathrm{rank}(I-P)=n$

**(c) 블록 대각행렬**
$$
\mathrm{rank}\!\begin{pmatrix}A&0\\0&B\end{pmatrix}
=\mathrm{rank}(A)+\mathrm{rank}(B)
$$

(자주 쓰는 요약 한 줄)
- 가역행렬로 좌우 곱해도 rank 불변  
- $A^TA$, $AA^T$는 rank 보존  
- 멱등(=사영)에서는 $\mathrm{rank}=\mathrm{tr}$, 따라서 $\mathrm{rank}(I-P)=n-\mathrm{rank}(P)$


# 연습문제
1. $n$차 다항식의 집합 $P_n$과 $(n+1)$차 다항식의 집합 $P_{n+1}$에 대해
   사상 $L: P_n\to P_{n+1}$을
   $L(p(x))=xp(x)$로 정의했을 때 선형사상임을 보이시오.

2. 3차 다항식 집합 $P_3$와
   $2\times 2$ 행렬의 집합 $M_{2\times 2}(\mathbb{R})$의
   선형사상
   $L_1: P_3\to \mathbb{R}^4$,
   $L_2: M_{2\times 2}(\mathbb{R})\to \mathbb{R}^4$
   모두 동형사상임을 증명하시오.

3. 벡터공간 $V$의 기저
   $B_V=\{v_1,v_2,v_3\}$에 대해
   자기사상 $L:V\to V$가

   $$
   [L]_{B_V}^{B_V}=
   \begin{pmatrix}
   3 & 4 & 7 \\
   -3 & 4 & 7 \\
   -3 & 4 & 7
   \end{pmatrix}
   $$

   을 만족할 때
   $V$의 기저 $B_V'=\{v_1,\ v_1+v_2,\ v_1+v_2+v_3 \}$에 대한
   $[L]_{B_V'}^{B_V'}$를 구하시오.

4. 행렬
   $$
   M=
   \begin{pmatrix}
   1 & 2 & 0 & 4 & 5 & -3 \\
   -1 & -2 & 3 & -4 & 5 & 3 \\
   0 & 0 & 3 & 0 & 4 & 1
   \end{pmatrix}
   $$
   에 대해 Rank–Nullity 정리 성립함을 보이시오.
