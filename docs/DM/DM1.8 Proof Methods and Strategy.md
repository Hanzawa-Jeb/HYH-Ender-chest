# Exhaustive Proof and Proof by Cases
- proof by cases就是$(p_1 \lor p_2 \lor ... p_n) \rightarrow q$可以等效为证明$(p_1 \rightarrow q) \land ... \land (p_n \rightarrow q)$ 同样为改变括号，然后交换括号内与括号外所相对应的符号。
- Exhaustive Proof就是通过**穷尽**较小数据集当中的全部数据以达到证明的作用（**穷举法**）
- Proof By cases必须要确保全部的例子都被完全考虑过了。
- **Without Loss of Generality** 也就是一些假设可以不失一般性地证明同一个命题，那我们就可以采取这样的假设。
# Existence Proofs
- 如果是直接找出了一个满足条件的样例，那么就属于Constructive Proof, 如果是间接地证明了这一个解的存在（例如使用反证法等等），那么则被称为Nonconstructive Proof.
- **Backward Reasoning**: 如果想要证明$p \rightarrow q$，那么可以先尝试找到一个r, 使得$r \rightarrow q$，然后再尝试证明$r \rightarrow q$ 
## Notes：
- 有时候可以用自然语言来描述题目的证明方式