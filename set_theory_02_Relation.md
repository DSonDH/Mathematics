# 관계 (Relation)
$$R = (A, B, P(x, y))$$
두 집합 $A,B$ 사이의 관계 $R$은 곱집합 $A\times B$의 부분집합으로 정의된다.  곱집합 중에서 관계 R의 해집합은  
$$\{(x, y) | x \in A, y\in B, P(x,y)는 참\}$$
A에서 B로의 관계 P(x, y)라는 명제함수를 만족해야한다.  
이때, $P(x,y)$: x는 y의 약수이다 같이 어떤 명제를 뜻한다.  
즉, $R\subseteq A\times B$이고 $(a,b)\in R$일 때 “$a$는 $b$와 관계가 있다”고 쓴다.  
이를 $xRy$라고 쓰기도 한다.  
특히 한 집합 $A$ 위의 관계는 $R\subseteq A\times A$인 경우이다.

용어:
- 정의역(domain): $\mathrm{dom}(R)=\{a\in A\mid \exists b\in B,\ (a,b)\in R\}$.
- 상 또는 값역(range 또는 image): $\mathrm{ran}(R)=\{b\in B\mid \exists a\in A,\ (a,b)\in R\}$.
- 역관계(inverse): $R^{-1}=\{(b,a)\mid (a,b)\in R\}$.
- 합성(composition): 두 관계 $R\subseteq A\times B,\ S\subseteq B\times C$에 대해
    $S\circ R=\{(a,c)\mid \exists b\in B,\ (a,b)\in R,\ (b,c)\in S\}$.

## 관계의 성질
한 집합 $A$ 위의 이항관계 $R\subseteq A\times A$에 대해 자주 쓰이는 성질들:

- 반사성(reflexive): $\forall a\in A,\ (a,a)\in R$.
- 비반사성(irreflexive): $\forall a\in A,\ (a,a)\notin R$.
- 대칭성(symmetric): $\forall a,b\in A,\ (a,b)\in R\Rightarrow (b,a)\in R$.
- 반대칭성(antisymmetric): $\forall a,b\in A,\ (a,b)\in R\land (b,a)\in R\Rightarrow a=b$.
- 추이성(transitive): $\forall a,b,c\in A,\ (a,b)\in R\land(b,c)\in R\Rightarrow (a,c)\in R$.
- 전체성(total 또는 connex): $\forall a\neq b,\ (a,b)\in R$ 또는 $(b,a)\in R$ 중 하나가 성립.

기타 성질과 사실:
- 합성은 결합적(associative): $T\circ(S\circ R)=(T\circ S)\circ R$.
- 역관계의 합성: $(S\circ R)^{-1}=R^{-1}\circ S^{-1}$.
- 반사성·대칭성·추이성의 조합으로 여러 구조(예: 동치관계, 부분순서)가 정의된다.

예:
- 등호(=)는 반사적·대칭적·추이적이므로 동치관계이다.
- ≤ 는 반사적·반대칭적·추이적이므로 부분순서를 이룬다.
## 여러 가지 관계
- 역관계: $R^{-1}=\{(b,a)\mid (a,b)\in R\}$
- 합성관계: $S\circ R=\{(a,c)\mid \exists b,\ (a,b)\in R\land (b,c)\in S\}$  
- 역관계와 합성관계에 관한 정리
    - $ (R^{-1})^{-1} = R $.
        - 순서쌍을 뒤집는 연산을 두 번 하면 원래 관계로 돌아온다.
    - 합성의 결합성: $ T\circ (S\circ R) = (T\circ S)\circ R $.
        - 중간 원소 존재 조건을 정리하면 성립(요소 추적으로 증명).
    - 합성의 역관계: $ (S\circ R)^{-1} = R^{-1}\circ S^{-1} $.
        - 만약 $ (a,c)\in S\circ R $이면 어떤 $b$가 존재하여 $ (a,b)\in R $와 $ (b,c)\in S $이므로, 뒤집으면 $ (c,a)\in R^{-1}\circ S^{-1} $이다. 반대 방향도 유사하게 성립한다.

관계의 종류(정의):
- 동치관계 (equivalence relation): $R$가 반사적, 대칭적, 추이적일 때. 대문자 E로 표현하기도 한다. 중요한 개념!
- 부분순서 (partial order, poset): 집합 $P$에서 $(P,\le)$이고 $\le$가 반사적, 반대칭적, 추이적일 때.
- 전순서 (total/linear order): 부분순서이면서 모든 쌍이 비교 가능한 경우, 즉 $\forall x,y\in P,\ x\le y\lor y\le x$.
- 준순서 (preorder): 반사적, 추이적(반대칭성 요구하지 않음).

부분순서에서 원소의 종류(포함되는 수식):
- 정의: $x<y$는 $x\le y\land x\neq y$.
- 최소원소(least, minimum) $m$: $\forall x\in P,\ m\le x$. (존재하면 유일)
- 최대원소(greatest, maximum) $M$: $\forall x\in P,\ x\le M$. (존재하면 유일)
- 극소원소(minimal) $m$: $\neg\exists x\in P,\ x<m$. 등가적으로 $\forall x\in P,\ x\le m\Rightarrow x=m$.
- 극대원소(maximal) $M$: $\neg\exists x\in P,\ M<x$. 등가적으로 $\forall x\in P,\ M\le x\Rightarrow x=M$.

성질 요약:
- 최소원소는 항상 극소원소이다. 그러나 극소원소가 반드시 최소원소인 것은 아니다(여러 극소원소가 존재할 수 있음).
- 최소원소·최대원소가 존재하면 각각 유일하다.
- 부분집합 관계나 약한 순서에서는 극소·극대가 여러 개 나올 수 있다.

해쎄 다이어그램(Hasse diagram):
- 유한한 부분순서를 시각화할 때 사용.
- 정점은 원소, 간선은 덮임관계(cover)만 그린다: $b$가 $a$를 덮는다 $\iff a<b\land\neg\exists c\ (a<c<b)$.
- 루프와 추이간선을 생략하여 위계만 표현한다.

간단한 예:
- $(\mathbb{N},\le)$에서는 최소원소 $0$(또는 시작점을 $1$로 잡는 경우 $1$)가 존재하고 이는 유일한 극소·최소원소이다.
- 부분집합관계 $(\mathcal P(X),\subseteq)$에서는 $\varnothing$가 최소원소, $X$가 최대원소이다.

## 동치관계와 분할
동치관계 $\sim 혹은 E$ 가 집합 $X$ 위에 주어지면, 각 원소 $x\in X$에 대한 동치류(equivalence class)를
$[x]=\{\,y\in X\mid y\sim x\,\}$ 혹은 
$E_x=\{\,y\in X\mid xEy\,\}$ 
로 정의한다.  
동치류들의 집합 $\{[x]\mid x\in X\}$를 상집합이라 하고 (다른 표현: $X/E = \{E_x|x\in X\}$), 이 상집합은 $X$의 분할(partition, 약자로는 P)을 이룬다. (/는 법 또는 modulo라 부른다)  
이를 p에 의한 관계, $R_p$ 또는 $X/P$로 표현한다$$X/P = \{(x,y)\mid\exists A\in P, x, y\in A\}$$

- 예제
  - 집합 $X=\{1,2,3,4,5\}$의 분할 $P=\{\{1,2\},\{3\},\{4,5\}\}$에 대하여:
  - (1) 동치관계 $R_P$의 순서쌍 나열  
    - 각 분할 원소의 P를 나열하면 된다
$R_P=\{(1,1),(1,2),(2,1),(2,2),(3,3),(4,4),(4,5),(5,4),(5,5)\}$

  - (2) $E=\mathcal{P}(X)$일 때 각 원소의 동치류 $E_i=[i]_{R_P}$  
    - $E_1=\{1,2\}$  
    - $E_2=\{1,2\}$  
    - $E_3=\{3\}$  
    - $E_4=\{4,5\}$  
    - $E_5=\{4,5\}$

- (참고) 분할 쉬운 표현
  - 분할은 집합 X에 대해 아래 세 조건을 만족하는 집합족
  - 공집합을 원소로 하지 않고, X를 덮고, 서로소 집합족
- 즉, (합집합) $\displaystyle\bigcup_{x\in X}[x]=X$ (모든 원소는 자기 동치류에 속한다).
- (서로소) 만약 $[x]\cap[y]\neq\varnothing$이면 $[x]=[y]$ (교집합이 비어있지 않으면 같은 동치류이다).

역으로, $X$의 분할 $\mathcal P=\{P_i\}_{i\in I}$가 주어지면 다음과 같이 동치관계를 정의할 수 있다:
$$
x\sim y \iff \exists i\in I\ \text{s.t.}\ x\in P_i\ \text{그리고}\ y\in P_i.
$$
이 관계는 반사적·대칭적·추이적이므로 동치관계가 된다.

정리(동치관계 ↔ 분할).
- 어떤 동치관계는 그 동치류들의 분할을 유도한다.
- 어떤 분할은 위의 방식으로 유일한 동치관계를 유도한다.
따라서 동치관계들과 분할들은 자연스럽게 일대일 대응을 이룬다.

몫집합(quotient set): $X/{\sim}=\{[x]\mid x\in X\}$.

자연 사상(사영): $\pi:X\to X/{\sim},\ \pi(x)=[x]$는 전사이다. 이 사영은 동치관계와 몫집합을 잇는 기본 사상으로 자주 사용된다.

### 여러 가지 정리
**공집합이 아닌 집합 X위의 동치관계 E에 대해 아래 네 정리가 성립함**  
1) 각 원소의 동치류는 비어있지 않다.  
    - 정리: $\forall x\in X,\ E_x\neq\varnothing$.  
    - 증명(스케치): 반사성으로 $xEx$이므로 $x\in E_x$이다.

2) 동치류의 같다/다름과 원소의 동치성의 동치성.  
    - 정리: $E_x=E_y\iff xEy$.  
    - 증명:  
        - $(\Rightarrow)$ $E_x=E_y$이고 $x\in E_x$이므로 $x\in E_y$이고 따라서 $xEy$.  
        - $(\Leftarrow)$ $xEy$이고 임의의 $z\in E_x$에 대해 $xEz$이다. 대칭성과 추이성으로 $yEz$가 되어 $z\in E_y$이다. 반대 방향도 동일하므로 $E_x=E_y$.

3) 두 동치류의 교집합과 동치성.  
    - 정리: $E_x\cap E_y\neq\varnothing\iff xEy$.  
    - 증명:  
        - $(\Rightarrow)$ $\exists z\in E_x\cap E_y$이면 $xEz$ 및 $yEz$이다. 대칭으로 $zEy$이고 추이로 $xEy$가 된다.  
        - $(\Leftarrow)$ $xEy$이면 $x\in E_y$이므로 $x\in E_x\cap E_y$.

4) 동치류들 (상집합)은 $X$의 분할을 이룬다(몫집합은 분할).  
    - 정리: $X/E=\{E_x\mid x\in X\}$는 $X$를 덮고($\bigcup_{x\in X}E_x=X$), 서로소이며, 공집합을 원소로 갖지 않는다.  
    - 증명(스케치): 
        - 덮음은 반사성에서 따르고(모든 $x$는 자기 동치류에 속함), 서로소성은 (3)의 결과로 두 동치류가 교집합을 가지면 같음으로 보인다. 공집합이 없음은 (1)에 의함.

- 예제: 몫집합의 몫은 원래의 동치관계로 복원된다 - $X/(X/E)=E$ 증명

    - $(x, y) \in X/(X/E)$
    - $\iff \exists A \in X/E$ such that $x, y \in A$
    - $\iff x \in A \land y \in A$
    - $\iff \exists z \in X$ such that $xEz \land yEz$
    - $\iff \exists z \in X$ such that $xEz \land zEy$
    - $\iff xEy$
    - $\iff (x, y) \in E$
    - $\therefore X/(X/E)=E$

**공집합이 아닌 집합의 분할 P에 대해 아래 두 정리가 성립함**  
1) $R_p$는 $X$상의 동치관계이다.  
    - 정의: $R_p=\{(x,y)\mid \exists A\in P,\;x\in A\ \text{그리고}\ y\in A\}$.  
    - 증명(스케치): 
        - 임의의 $x\in X$에 대해 $x$가 속하는 어떤 $A\in P$가 있으므로 $(x,x)\in R_p$ (반사성). 
        - $(x,y)\in R_p$이면 $x,y$가 같은 $A\in P$에 속하므로 $(y,x)\in R_p$ (대칭성). 
        - $(x,y)\in R_p$와 $(y,z)\in R_p$이면 세 원소가 같은 $A\in P$에 속하므로 $(x,z)\in R_p$ (추이성). 
        - 따라서 $R_p$는 반사적·대칭적·추이적이어서 동치관계이다.

2) $X/R_p=P$.  
    - 정의: $X/R_p=\{[x]_{R_p}\mid x\in X\}$ (동치류들의 집합).  
    - 증명(스케치): 
        - 임의의 $A\in P$에서 임의의 $x\in A$를 택하면 모든 $y\in A$가 $x$와 같은 블록에 있으므로 $[x]_{R_p}=A$이다. 
        - 반대로 각 동치류 $[x]_{R_p}$는 어떤 $A\in P$와 일치한다. 
        - 또한 서로 다른 동치류는 교집합이 없으므로 동치류들의 집합은 정확히 $P$와 같다. 
        - 따라서 몫집합 $X/R_p$는 분할 $P$와 동치이다.
