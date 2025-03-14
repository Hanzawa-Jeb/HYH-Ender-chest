- 线性表->前后有顺序关系，但是与具体的Data Structure没有关系，可以使用不一样的方式进行实现，例如**数组或者链表**
# Trees
## Intro
- 子树(Subtrees)之间不会相连接
- N个节点的树就有N-1条边
- root一般画在顶部
- Degree of a Node就是一个节点的子树个数
- Degree of a Tree就是每一个节点的Degree的最大值
- leaf：叶子节点，没有子树的节点
- Path一般考虑的都是单向的（从上往下或者从下往上）
- Length of Path: 一整条路径上**边的数量**
- depth：一个节点到root的path长度（root的深度为0）
- height: 从一个节点到leaf的**最长长度**
- 一个树的depth/height就是最大的depth/height值
## Implementation
### List Representation
- 采用链表，每一个节点后面都会连接下一步的节点
- 最好指针能从root指向下面的subtrees
### FirstChild-NextSibling Representation
- 指向**第一个孩子**与旁边的同辈（NextSibling）
# Binary Trees
## Definitions
- 不能有超过两个children node
## Implementations
### Expression Trees
- ==infix/postfix expression??==
- Syntax Trees
- 可以采用二叉树来表达infix expression
- 自然语言表达的都是infix expression, 
### Tree Traversal
- 树的遍历（访问每一个节点一次）
- Preorder Traversal: 先访问根节点
- Postorder Traversal: 先访问每一个子节点再访问根节点
- LevelOrder Traversal: 根据每一行来进行行遍历，可以通过队列进行实现。
	- 先把一个节点的子节点全部enque， 然后每一轮循环都deque第一个元素
- InOrder Traversal：仅在**二叉树**中存在，先访问左节点，再访问根节点，再访问右节点（**递归访问**）
- T（N）就是时间复杂度，一般表达为$T(N) = O(f(N))$ ,空间复杂度是$S(N)$
- 如果想要访问到文件夹的空间大小，则应该使用PostOrder遍历方法，前面计入的空间最后加上自己的空间大小。
## Threaded Binary Tree==？==
- 可以让一些节点中的NULL指针变为Threads，可以帮助遍历
- 如果node->left == NULL, 那么就将这个指针指向遍历次序中的前序节点
- 如果node->right == NULL, 那么就将这个指针指向遍历次序中的后续节点