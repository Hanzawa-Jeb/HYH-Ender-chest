# Basic Infos
- proof就是一个有效的argument，能够证明一个mathematical statement.
- theorem就是一个可以被证明为真的statement.
- **通过proof,可以证明一个theorem**
- 在proof中用到的statement被称为axiom(公理)，postulate(假设)
- lemma是引理，corollary就是推论，可以从theorem推得。conjecture就是猜测
# Methods of Proving Theorems
- 一般需要证明的格式都是$\forall / \exists x(P(x) \rightarrow Q(x))$ ，如果是任意则需要证明对Arbitrary Numbers 均成立然后再Universal generalization.
## Direct Proofs
- 假设前提是正确的，然后使用rules of inference证明结果也必须是正确的。
## Proof by Contraposition
- 属于indirect proofs
- 证明逆否命题，与证明原命题是等效的。
- 要先假设原本得出的**conclusion是假的**，然后再推得原来的前置条件也是假的，这样就可以得到证明。
- vacuous proofs（空虚证明）就是在证明条件句的时候**在前置条件为false**的时候直接证明原命题为真。
- trivial proof（平凡证明）就是在**后置条件为true的时候默认该proof为true**
- **A little Proof Strategy:** 先从direct proof入手，再考虑Contraposition。
## Proof by Contradiction
- 反证法
- 先假设p为false，然后推理出的结果与premise冲突，则可以证明p为true（唯一需要证明的就是p这个表达式。）
- Proofs of Equivalence，要证明两个方向的正确性即可
- Counterexamples->用于证明任意性的问题。
## Mistakes in Proofs
