# 1. 선택공리 (Axiom of Choice)

## (1) 선택함수 (Choice Function)

집합 $X \neq \emptyset$의 부분집합족을 ${A_i}$라 할 때,

$$
\forall i \in I, ; f(A_i) \in A_i \text{인 } f : \{A_i\} \to X
$$

을 **선택함수(choice function)**라 한다.

즉, 각 부분집합 $A_i$에서 **하나의 원소를 선택하는 함수**이다.

## (2) 선택공리 (Axiom of Choice)

**공집합이 아닌 임의의 집합에 대한 선택함수가 존재한다.**

즉,

> "공집합을 원소로 갖지 않는 서로소인 집합족들 각각의 원소 중 하나씩을 선택하여
> 만든 함수가 존재한다."
> 라고 해석할 수 있다.

# 2. 동치인 명제 (Equivalent Propositions)

## (1) 극대원리
임의의 부분순서집합은 **극대원소(maximal element)** 를 갖는다.
즉, 어떤 체인(chain)이 주어졌을 때, 그 위에 있는 상한이 존재한다면 그 집합은 반드시 극대원소를 가진다.

## (2) 조른의 원리 (Zorn’s Lemma)
모든 체(chain)가 위로 유계인 부분순서집합은 
**극대원소를 가진다.**

> 선택공리와 **논리적으로 동치**임.

## (3) 정렬원리 (Well-Ordering Theorem)

모든 집합은 **정렬가능(well-orderable)** 하다.
즉,

> 임의의 집합에 대해 적당한 순서관계를 부여하여
> 정렬집합으로 만들 수 있다.

## (4) 그 외의 동치 명제들

* **라그랑주 원리 (Lagrange’s Principle)**
* **타르스키 원리 (Tarski’s Principle)**
* **티호노프 원리 (Tychonoff’s Theorem)**
* **타이히뮬러–투키 원리 (Teichmüller–Tukey Principle)**
* 임의의 두 기수의 비교가능 원리 (Comparability of Cardinals)
* 모든 벡터공간의 기저 존재 원리 (Basis Theorem for Vector Spaces)
* …

# 3. 함의되는 명제들 (Theorems Implied by AC)

* **괴델의 완전성 원리 (Gödel’s Completeness Theorem)**
* **베르의 부분집합 원리 (Baire Category Theorem)**
* **한–바나흐 정리 (Hahn–Banach Theorem)**
* **바나흐–타르스키의 역설 (Banach–Tarski Paradox)**
* **넬슨–수라야의 원리 (Nelson–Surał Principle)**
* **모든 체의 대수적 폐포존재원리**
* …
