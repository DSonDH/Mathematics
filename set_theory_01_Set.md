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