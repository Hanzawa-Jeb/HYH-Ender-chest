- $f(x)$ is $O(g(x))$ is read as: $f(x)$ is big-oh of $g(x)$
- big-oh of g(x)的意思就是O(g(x))
- witnesses的意思就是满足$|f(x)|\;<\;C|g(x)|,x>k$的一组C与k的解
- 只要一组witness存在，那么就存在无数多的witnesses
- $n!\;=\;O(n^n)$ 这个非常重要
- $f(x) \;is\; \Omega(g(x))$的意思是g(x)是f(x)的下限
- $f(x)\;is\;\Theta(g(x))$的意思是g(x),f(x)同阶
$$
f(x)\;is\;\Theta(g(x)) \leftrightarrow f(x)\;is\;of\;same\;order\;with\;g(x) 
$$
- 要找到smallest order的g(x)，不一定要局限于幂函数，而是全部函数域中的最小Order。order可以认为就是阶数。
