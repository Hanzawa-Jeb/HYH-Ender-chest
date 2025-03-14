# 1. 判断题
- 我们始终可以使用一维的整数列表来表达一个**树**
- 我们可以对二叉树中的节点进行**分类**。节点可以被分为三种：**有两个子节点，有一个子节点，有0个子节点**。因为每一个节点的parent node是唯一的，设这三种节点的数量分别为$N_2, N_1, N_0$ ，那么总共的节点个数应该为$1+2×N_2 + N_1$ ，所以在这里**奇偶性不满足条件。**
- 在二叉树中应该始终有$n_0 = n_2 + 1$ ，在任意二叉树中都适用
# 2. 单选题
- preorder traversal是前序，inorder是中序，postorder是后序
- 度数之和为边数，度数之和+1 = 节点个数
- 普通树转化为二叉树的方法：**先用FirstChild-NextSibling方式**遍历原先的普通树，随后在旋转45°从而达到转化为二叉树的作用。
- ==有关General Tree==的遍历：其实是相同的，但是没有Inorder了，只有Preorder和PostOrder. preorder就是先**从左到右访问每一个子节点，再访问自己**
- 所以如果一个普通树T转化为二叉树BT， 那么**T的后序遍历与BT的中序遍历是一样的**（转化过程详见上文）
- Threaded Binary Tree主要是用于需要多次查找的情况。左指针用于指向前驱，右指针指向后驱（指的都是在遍历序列过程中的前面和后面的元素）
# 3. 函数题
- 可以采用**递归**的方法，逐步减小需要解决的问题规模。
- 递归的方法中，可以先判断是否都为NULL，如果是，那么为真。再判断如果有一个为NULL，那么为假。
- 随后再判断是否根节点一致->不一样则返回false.
- 如果上面这些分支都没有结束，则分别判断左子树与右子树。
```c
int Isomorphic(Tree T1, Tree T2)
{
    if (T1 == NULL && T2 == NULL)
        return 1;
    if (T1 == NULL || T2 == NULL)
        return 0;
    if (T1->Element != T2->Element)
        return 0;
    return (Isomorphic(T1->Left, T2->Left) && Isomorphic(T1->Right, T2->Right))
           || (Isomorphic(T1->Left, T2->Right) && Isomorphic(T1->Right, T2->Left));
}
```
# 4. 编程题
- keys就是每一个节点所对应的element的值
- 对**中序遍历**与**后序遍历**同样可以通过递归的方法来创建二叉树。
- C语言中在**函数内**通过malloc得到的内存，在函数执行结束后仍然可以访问。