# 조건부 기대값의 주요 성질

**(1) 전체 기대의 법칙 (Law of Total Expectation)**

$$\mathbb{E}[Y]=\mathbb{E}[\mathbb{E}[Y\mid X]].$$

증명(간단히): $g(X):=\mathbb{E}[Y\mid X]$는 $\sigma(X)$-측도가능함수이며 조건부 기대값의 정의에 의해 모든 $A\in\sigma(X)$에 대해
$$\mathbb{E}[g\mathbf{1}_A]=\mathbb{E}[Y\mathbf{1}_A].$$
특히 $A=\Omega$를 취하면 $\mathbb{E}[g]=\mathbb{E}[Y]$가 되어 위 식이 성립한다. (여기서 $Y$는 적분가능하다고 가정)

회귀모형 관점: "예측값의 평균은 실제값의 평균과 같다."

**(2) 조건부 기대값의 이중 법칙 (Law of Iterated Expectations, tower property)**

$$\mathbb{E}\big[\mathbb{E}[Y\mid X,Z]\mid Z\big]=\mathbb{E}[Y\mid Z].$$

증명(측도론적): $\mathbb{E}[Y\mid X,Z]$는 $\sigma(X,Z)$-측도가능 함수이며 모든 $A\in\sigma(X,Z)$에 대해
$$\mathbb{E}[\mathbb{E}[Y\mid X,Z]\mathbf{1}_A]=\mathbb{E}[Y\mathbf{1}_A].$$
특히 $A\in\sigma(Z)\subset\ \sigma(X,Z)$이면 위 등식이 성립하므로 모든 $A\in\sigma(Z)$에 대해
$$\mathbb{E}[\mathbb{E}[Y\mid X,Z]\mathbf{1}_A]=\mathbb{E}[Y\mathbf{1}_A].$$
우변과 같은 적분 조건을 만족하는 $\sigma(Z)$-측도가능 함수는 거의곳에서 유일하므로 $\mathbb{E}[\mathbb{E}[Y\mid X,Z]\mid Z]$는 $\mathbb{E}[Y\mid Z]$와 거의곳에서 같다. (역시 $Y$는 적분가능)

(cf) $f=g\ \text{거의곳}$
즉, 영어로는 "almost everywhere" (약칭 a.e.) 또는 확률론에서는 "almost surely" (약칭 a.s.)
예: f = g almost everywhere (a.e.).
이라 하면 그들이 다른 점들의 집합의 측도(확률) P({ω: f(ω)≠g(ω)})가 0임을 뜻합니다. 따라서 “거의곳에서 유일”은 다르면 다르게 보이는 점들이 확률 0인 집합에만 속하므로 실질적으로는 동일하다고 간주한다는 의미입니다.
