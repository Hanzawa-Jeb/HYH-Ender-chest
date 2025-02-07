# General Advantage Estimation
- GAE与Multi-step Sarsa有相类似的想法，也就是***平衡MC方法的高方差与TD方法的高偏差***.
- GAE是用来估计优势值的，优势值又是通过TD Error来进行近似的。
- GAE的基本表达式是使用1-n步的优势函数来进行估计（步数越长权重越小）
- GAE的估计会更加准确
- $A_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$ 由等比数列求和的性质推导而出