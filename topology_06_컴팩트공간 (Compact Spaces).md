# Chapter 6 컴팩트공간 (Compact Spaces)

## 6.1 컴팩트성

**정의 6.1** (열린피복, open cover)

집합 $A \subset X$에 대해 $\{U_\alpha\}_{\alpha \in I} \subset \tau$가 $A \subset \bigcup_{\alpha \in I} U_\alpha$를 만족하면 열린피복이라 한다.

**정의 6.2** (컴팩트공간, compact space)

모든 열린피복이 유한부분피복(finite subcover)을 가지면 $X$는 컴팩트공간이라 한다.

**정리 6.1**

부분집합 $A \subset X$가 컴팩트일 필요충분조건은 $X$의 열린집합들로 덮일 때 유한부분피복이 존재하는 것이다.

**정리 6.2**

하우스도르프 공간에서 컴팩트 부분집합은 닫힌집합이다.

*증명:* 점과 컴팩트집합을 분리하는 열린집합을 구성하여 여집합이 열린집합임을 보인다. $\square$

**정리 6.3**

컴팩트공간에서 연속함수의 상은 컴팩트이다.

*증명:* 열린피복의 역상을 이용한다. $\square$

## 6.2 하이네–보렐 정리

**정리 6.4** (Heine–Borel theorem)

$\mathbb{R}$에서 부분집합 $A$가 컴팩트일 필요충분조건은 $A$가 닫히고 유계(bounded)인 것이다.

## 6.3 티호노프 정리

**정리 6.5** (Tychonoff theorem)

컴팩트공간들의 임의의 곱공간은 컴팩트이다.

