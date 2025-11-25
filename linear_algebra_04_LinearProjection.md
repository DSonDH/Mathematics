# 1. 선형사상 (Linear Map/ Linear Transform/ Linear morphism)
사상(morphism): 특정 구조를 보존하는(대수 구조를 다루는) 함수  
함수: 원소 대응 규칙 두가지를 만족하는 관계.  
## (1) 선형사상
### ① 정의
$체 F$-벡터공간 $V,\ W$에 대하여
$V$의 성질을 보존하는 다음 두 조건을 만족하는 사상을
$L(Linear):V \to W$라 한다.

1. Additivity: $L(u+v)=L(u)+L(v)$  $(u,v\in V)$
2. Homogeneity: $L(kv)=kL(v)$  $(k\in F,\ v\in V)$

### ② 관련 용어
선형사상 $L:V\to W$에서

* **핵(ker, kernel)**:
  $\ker L=\{v\in V\mid L(v)=0\}$

* **상(im, image)**:
  $\operatorname{im}L = L(V)=\{L(v)\mid v\in V\}$

* **자기사상 (endomorphism)**:
  $V=W$일 때 $L:V\to V$

* **단사사상 (injective map, monomorphism)**:
  $L(u)=L(v)\Rightarrow u=v$인 $L$

* **전사사상 (surjective map)**:
  $L(V)=W$인 $L$

* **동형사상 (isomorphism)**:
  단사이면서 전사인 사상
  - 대수구조 동일함을 보이는데 필요!

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
1. $L(v) = \mathbf{0}$ : 영사상
2. $L(v) = v$ : 항등사상
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

**결론**  
f와 g는 동형사상이다. 또한 두 사상 f와 g는 서로 역사상 관계이다.  

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
