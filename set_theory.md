항진인 조건문 p -> q를 논리적 함의라 하고, p=>q로 나타내며 p는 q의 충분조건, q는 p의 필요조건이라 한다.  
항진인 쌍조건문은 두 명제의 동치(iff)로, $p\leftrightarrow q$ 또는 $p\Leftrightarrow q$로 쓰며 “$p$와 $q$는 서로의 필요·충분조건”이라 한다.

$p\to q$는 논리식으로 $\neg p\lor q$와 같다.

# 집합

## 집합족
집합족(family of sets): 집합을 원소로 갖는 집합(예: 멱집합).  
첨수족(indexed family): 각 원소에 첨자(인덱스)가 부여된 집합들의 모임. 예를 들어 $A$의 멱집합이 $\mathcal P(A)=\{\varnothing,\{1\},\{2\},\{1,2\}\}$일 때, 원소들에 인덱스를 매겨 $\{B_i\mid i\in I\}$처럼 표현할 수 있다.

### 집합족의 연산
- 합집합: $\displaystyle\bigcup\mathcal F=\{x\mid\exists A\in\mathcal F,\;x\in A\}$.  
    즉, $\mathcal F$에 속한 적어도 하나의 집합이 $x$를 포함하면 $x$는 합집합에 속한다.  
    예: $\mathcal F=\{\{1,2\},\{2,3,4\},\{5\}\}\implies\bigcup\mathcal F=\{1,2,3,4,5\}$.

- 교집합: $\displaystyle\bigcap\mathcal F=\{x\mid\forall A\in\mathcal F,\;x\in A\}$.  
    즉, $\mathcal F$에 속한 모든 집합에 공통으로 들어있는 원소들의 집합이다.  
    예: $\mathcal F=\{\{1,2,3\},\{2,3,4\},\{2,3,5\}\}\implies\bigcap\mathcal F=\{2,3\}$.


## 곱집합
곱집합(카르테시안 곱, Cartesian product)은 두 집합의 순서쌍들로 이루어진 집합이다.  
두 집합 $A,B$에 대해
$A\times B=\{(a,b)\mid a\in A,\;b\in B\}$.
예: $A=\{1,2\},\;B=\{a,b\}$이면 $A\times B=\{(1,a),(1,b),(2,a),(2,b)\}$.  
$R^2$는 흔히
순서쌍의 표기는 보통 $(a,b)$이며 $(a,b)=(c,d)$는 $a=c$이면서 $b=d$일 때 성립한다.

$R^2$는 흔히 아는 카르테시안 평면좌표계로 위 곱집합 개념으로 표현한 것이다. 

### 곱집합의 연산
- 전단성: $A\times\varnothing=\varnothing\times A=\varnothing$.  
- 교환성(정확히는 자연적인 전단사): $A\times B\cong B\times A$(순서쌍의 성분을 바꾸는 전단사를 통해), 하지만 집합으로서 동치(순서가 다름)는 아니다.
- 결합성(자연동형): $(A\times B)\times C\cong A\times(B\times C)$ (표기상의 차이를 제외하면 동일하게 다룰 수 있다).
- 분배법칙: 곱집합은 합집합에 대해 분배된다.
  - $A\times\bigl(B\cup C\bigr)=(A\times B)\cup(A\times C)$.
  - $A\times\bigl(B\cap C\bigr)=(A\times B)\cap(A\times C)$.
  - 증명: 양 포함관계를 보이면 된다.
  $$
    \begin{aligned}
    (x, y) &\in A \times (B \cap C) \\
    &\Leftrightarrow x \in A \land y \in (B \cap C) \\
    &\Leftrightarrow x \in A \land (y \in B \land y \in C) \\
    &\Leftrightarrow (x \in A \land y \in B) \land (x \in A \land y \in C) \\
    &\Leftrightarrow ((x, y) \in A \times B) \land ((x, y) \in A \times C) \\
    &\Leftrightarrow (x, y) \in (A \times B) \cap (A \times C)
    \end{aligned}
    $$

- 카디널리티(유한집합): 유한집합이면 $|A\times B|=|A|\cdot|B|$.

- 사상과 사영: 곱집합에는 사영(projection) 사상들이 있다.
  - $\pi_1:A\times B\to A,\;\pi_1(a,b)=a$.
  - $\pi_2:A\times B\to B,\;\pi_2(a,b)=b$.
  이들은 곱집합의 성질(예: 범주론적 곱의 보편성)을 기술할 때 중요하다.

### 첨수된(인덱스) 곱집합
인덱스 집합 $I$와 각 $i\in I$에 대응되는 집합 $A_i$가 주어지면, 첨수곱(product)은
$\displaystyle\prod_{i\in I} A_i=\{\,f:I\to\bigcup_{i\in I}A_i\mid\forall i\in I,\;f(i)\in A_i\,\}$.
즉, 각 인덱스에 대해 해당 집합의 원소를 하나씩 골라 만든 선택함수들의 집합이다.

- 유한한 경우($I=\{1,\dots,n\}$)에는 $\prod_{i=1}^n A_i$가 $n$-튜플들의 집합으로 동일하게 정의된다.
- 무한 곱의 비공집합성: 모든 $A_i$가 공집합이 아님에도 불구하고 $\prod_{i\in I}A_i$가 비공집합임을 보장하려면 선택공리(Axiom of Choice)가 필요할 수 있다. (선택함수의 존재 문제)
- 예: $\prod_{n\in\mathbb{N}}\{0,1\}$은 이진수열(또는 0/1의 수열)들의 집합이다.

### 곱집합과 집합족
- 곱집합은 첨수족의 특별한 연산으로 볼 수 있다. 집합족 $\mathcal F=\{A_i\mid i\in I\}$가 주어지면 그 첨수곱 $\prod_{i\in I}A_i$를 정의할 수 있다.
- 곱집합 길어지면 일일히 쓰기 힘드니깐 수식으로 처리한거임.
- 만약 집합족이 명시적 인덱스 없이 주어졌다면, 임의의 인덱싱(바이젝션)을 택해 첨수족으로 보고 곱을 정의해야 한다. 따라서 "집합족의 곱"은 보통 첨수화(indexing)를 전제한다.
- 부분집합족과의 관계: 각 성분에 대해 부분집합 관계가 주어지면(즉 $B_i\subseteq A_i$ for all $i$) $\prod_i B_i\subseteq\prod_i A_i$가 성립한다.
- $\prod_{n\in\mathbb{N}}\{0,1\}$는 모든 0-1 수열의 집합(이진 시퀀스의 집합)이다.

<br/> 

# 관계
두 집합 $A,B$ 사이의 관계(relation) $R$은 곱집합 $A\times B$의 부분집합으로 정의된다.
즉, $R\subseteq A\times B$이고 $(a,b)\in R$일 때 “$a$는 $b$와 관계가 있다”고 쓴다.  
특히 한 집합 $A$ 위의 관계는 $R\subseteq A\times A$인 경우이다.

용어:
- 정의역(domain): $\mathrm{dom}(R)=\{a\in A\mid \exists b\in B,\ (a,b)\in R\}$.
- 값역(range 또는 image): $\mathrm{ran}(R)=\{b\in B\mid \exists a\in A,\ (a,b)\in R\}$.
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
관계의 대표적인 종류:
- 동치관계(equivalence relation): 반사적, 대칭적, 추이적을 만족.
- 부분순서(partial order): 반사적, 반대칭적, 추이적을 만족.
- 전순서(total order): 부분순서이면서 전체성도 만족.
- 준순서(preorder): 반사적·추이적(반대칭성을 요구하지 않음).

각 관계에는 관련된 구조적 개념들이 있다:
- 부분순서에서 최소원소(minimal), 극소원소(least), 최대원소(maximal), 극대원소(greatest).
- 관계 그래프 또는 해쎄 다이어그램(Hasse diagram)은 부분순서를 시각화한다.

## 동치관계와 분할
동치관계 $ \sim $가 집합 $X$ 위에 주어지면, 각 원소 $x\in X$에 대해 그 동치류(equivalence class)를
\[ [x]=\{y\in X\mid y\sim x\} \]
로 정의한다. 동치류들의 집합 $\{[x]\mid x\in X\}$는 $X$의 분할(partition)을 이룬다(즉, 서로 겹치지 않으며 합집합이 $X$이다).

역으로, $X$의 분할 $\mathcal{P}=\{P_i\}_{i\in I}$가 주어지면, 이를 이용해 동치관계를 정의할 수 있다:
\[ x\sim y \iff \exists i\in I,\ x\in P_i\ \text{그리고}\ y\in P_i. \]

정리(동치관계 ↔ 분할):
- 집합 $X$ 위의 동치관계는 그 동치류들의 분할을 유도한다.
- $X$의 분할은 위와 같은 방식으로 유일한 동치관계를 유도한다.
(따라서 동치관계들과 분할들은 자연스럽게 일대일로 대응한다.)

추가 개념:
- 몫집합(quotient set): 동치관계 $\sim$에 대한 몫집합은 $X/{\sim}=\{[x]\mid x\in X\}$이다.
- 자연 사상(사영): $\pi:X\to X/{\sim},\ \pi(x)=[x]$는 전사이며, 동치관계와 몫집합을 잇는 기본 사상이다.

### 여러 가지 정리
간단한 정리들(증명 스케치 포함):

1. 합성의 결합성:
     - 임의의 관계 $R\subseteq A\times B,\ S\subseteq B\times C,\ T\subseteq C\times D$에 대해
         $T\circ(S\circ R)=(T\circ S)\circ R$.
     - 증명: 중간 원소에 대한 존재 조건을 정리하면 된다.

2. 역관계의 성질:
     - $ (R^{-1})^{-1}=R$, $ \mathrm{dom}(R^{-1})=\mathrm{ran}(R)$, $(S\circ R)^{-1}=R^{-1}\circ S^{-1}$.

3. 동치관계와 분할의 쌍대성(앞에서 서술한 정리): 동치관계에서 얻은 동치류들은 서로소이며 합집합이 원래 집합이고, 반대로 분할로부터 동치관계를 얻을 수 있다.  
     - 증명: 동치성의 성질로 동치류 간 교집합이 비어있거나 같음을 보이고, 모든 원소가 어떤 동치류에 속함을 보이면 된다.

4. 추이적 닫힘(transitive closure):
     - 임의의 관계 $R$에 대해 $R$의 추이적 닫힘 $R^+$는 가장 작은 추이관계로, 모든 $n$-단 합성 $R^n$의 합집합으로 주어진다:
         $R^+=\bigcup_{n\ge1} R^n$.
     - 이는 그래프 이론의 경로 개념과 동일하다(경로가 있으면 추이적 닫힘에 포함).

필요하면 각 정리별로 예제와 간단한 증명을 추가로 작성해 드리겠습니다.

