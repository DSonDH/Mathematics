# 📘 ZFC 공리(Zermelo–Fraenkel with Choice)
ZFC는 현대 수학의 표준 집합론 공리 체계로,
**Zermelo–Fraenkel 집합론(ZF)** 에 **선택공리(Choice)** 를 추가한 것이다.
총 10개의 기본 공리로 구성된다.

## ① 외연 공리 (Axiom of Extensionality)

$$
(\forall x \in A,, x \in B) \land (\forall x \in B,, x \in A) \iff A = B
$$

* 집합은 그 집합이 갖는 성질이 아니라, 그 원소들에 의해서만 규정된다.
* 즉, 같은 원소를 가지면 동일한 집합이다.
* 집합의 **유일성(identifiability)**을 보장한다.

참고: 외연공리에 의해 $\{A, A\} = \{A\}$

## ② 쌍 공리 (Axiom of Pairing)

$$
\forall A, B, \exists C ; s.t.; C = \{A, B\}
$$

* 임의의 두 집합으로 구성된 (곱집합) **쌍집합(pair set)** 의 존재를 보장한다.

## ③ 공집합 공리 (Axiom of Empty Set)

$$
\exists A; s.t.; \forall x,; x \notin A
$$

* 공집합 $\emptyset$의 존재를 보장한다.

## ④ 정칙성 공리 (Axiom of Regularity / Foundation)

$$
\forall A,; A \neq \emptyset \Rightarrow \exists B \in A; s.t.; A \cap B = \emptyset
$$

* 모든 비공집합 $A$는 자신과 교집합이 없는 원소 $B$를 가진다.
* 즉, 집합은 자기 자신을 포함하지 않으며, 무한 자기포함 사슬을 방지한다. 

: 러셀의 역설 해결해줌  

## ⑤ 합집합 공리 (Axiom of Union)

$$
\forall A,; \exists B; s.t.; B = \bigcup A
$$

* 임의의 집합 $A$에 대해 모든 원소들의 원소들을 모은 합집합 $\bigcup A$의 존재를 보장한다.

## ⑥ 멱집합 공리 (Axiom of Power Set)

$$
\forall A,; \exists P; s.t.; \forall B,; B \subset A \Rightarrow B \in P
$$

* 임의의 집합 $A$에 대한 **멱집합 $\mathcal{P}(A)$**의 존재를 보장한다.

## ⑦ 무한 공리 (Axiom of Infinity)

$$
\exists A,; \emptyset \in A \land (\forall x \in A)(x \cup {x} \in A)
$$

* 공집합을 포함하고, 원소 $x$가 있으면 $x \cup \{x\}$도 포함하는
  **무한집합**의 존재를 보장한다.
  (즉, 자연수 집합 $\mathbb{N}$의 존재)

## ⑧ 치환 공리 (Axiom of Replacement)

$$
\forall x, \exists !y, s.t.; xRy \Rightarrow \forall A,; \exists B,s.t.; \\ \forall x \in A,, \exists y \in B; s.t.; xRy
$$
* !는 유일하게 기호임
* 주어진 관계 $R$에 따라 집합 $A$의 각 원소를 다른 원소로 대응시킬 때,
  그 결과로 얻어진 집합 $B$도 집합임을 보장한다.
* **함수적 대응에 따른 집합 생성**을 허용한다. (관계 R에 따라 무수히 많은 공리가 만들어진다)

## ⑨ 분리 공리 (Axiom Schema of Separation)

$$
\forall A,; \exists B; s.t.; B = \{x \in A \mid \varphi(x)\}
$$

* 임의의 조건(명제함수) $\varphi(x)$를 만족하는 원소만 모은
  **부분집합(subset)** 의 존재를 보장한다.
* 임의의 두 집합에 대한 교집합을 만들 수 있다
* “부분집합 만들기”를 허용하지만,
  전체 우주 집합의 존재는 막는다
* 내포원리의 진화형태로, 러셀의 역설을 피한다.
* 명제함수 $\varphi(x)$에 따라 무수히 많은 공리가 만들어진다

## ⑩ 선택 공리 (Axiom of Choice)

$$
\forall A,; \emptyset \notin A \Rightarrow \exists f: A \to \bigcup A; s.t.; \forall B \in A,; f(B) \in B
$$

* 모든 비공집합들의 모임 $A$에 대해,
  각 원소 $B$에서 하나씩 원소를 선택하는 함수 $f$가 존재한다.

* 선택공리로부터 다음이 증명된다:

  * 정렬원리 (Well-ordering theorem)
  * 조른의 보조정리 (Zorn’s Lemma)

## ✅ ZFC 요약표

| 번호 | 공리명    | 핵심 내용             |
| -- | ------ | ----------------- |
| ①  | 외연 공리  | 집합은 원소로 결정됨       |
| ②  | 쌍 공리   | 두 원소로 이루어진 집합의 존재 |
| ③  | 공집합 공리 | 공집합의 존재           |
| ④  | 정칙성 공리 | 자기포함 금지, 무한사슬 방지  |
| ⑤  | 합집합 공리 | 원소들의 원소의 합집합 존재   |
| ⑥  | 멱집합 공리 | 부분집합들의 집합 존재      |
| ⑦  | 무한 공리  | 무한집합(자연수)의 존재     |
| ⑧  | 치환 공리  | 함수적 대응 결과도 집합     |
| ⑨  | 분리 공리  | 조건에 맞는 부분집합 구성    |
| ⑩  | 선택 공리  | 각 집합에서 원소 선택 가능   |

**ZFC의 특징:**

* 러셀의 역설 등 자기포함 문제를 회피함.
* 현대 수학의 대부분이 ZFC 위에서 전개됨.
* 선택공리(AC)는 ZF에서 독립적이며,
  **ZFC 체계가 표준 집합론의 기반**이다.

