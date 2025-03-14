# Reminders
- 如果是sequentially->必须是数组，则说明删除元素与增加元素都是$O(N)$,如果不是sequentially stored, 那么增删元素只需要$O(1)$
- ADT->Abstract Data Type, 不依赖于具体的编程语言
# More
- Sparse Matrix往往采用链表来进行存储->十字链表来表达多维Sparse Matrix，有两个指针，一个指针指向一个轴的方向
- 使用数组的下标来表示地址，不需要在整个内存空间中进行寻找，速度很快
# Stacks
## Features
- LIFO, Last In First Out
- Insertions与Deletions只能在top端进行
## Operations:
- `int IsEmpty(S)`
- `Stack CreateStack(S)`
- `DisposeStack(S)`
- `MakeEmpty(S)`
- `Push(X, S)`
- `ElementType Top(Stack S)` -> only output, **no popping**
- `Pop(S)` -> Output the element and **pop the element** meanwhile.
## Implementations:
- Linked List Implementation
	- 在链表头进行插入与删除，但是需要有一个Dummy Head.
	- Optimization:
		- 实现一个Recycle Bin, 把pop出的元素push到Recycle Bin中，如果Recycle Bin未空则直接利用这里面的节点来接上去。
- Array Implementation:
	- TopOfStack就可以当做指针(int类型)，是数组的下标。
## Applications
### Balancing Symbols
- 在括号匹配的问题之中，不能直接用Counter来计算，因为会出现很多问题。
- 使用栈的好处是栈的top是具有元素类型的，也就是说可以解决括号类型是否匹配，是否出现嵌套的问题。
- on-line algorithm->能够直接对当前的状态进行评估
### Postfix Evaluation
- **Basic Ideas**
	- Infix->普通的算数顺序
	- prefix->操作符在最前面
	- postfix->操作符在最后面->是计算机采用的方法
	- operand: 操作数 operator: 操作符
- **Infix to Postfix Conversion:**==！！==
	- 依次读取infix表达式，依次读取，如果是operands那么直接输出
	- 如果读取到operator，与栈顶相比较，如果优先级==相等或者更高？==则推入栈顶，如果不是则把原来的栈顶输出。
	- 如何处理括号？
		- 1.永远不能pop一个单独的"("
		- 2.如果不在stack中那么具有最高的优先级，如果在stack中则具有最低的优先级。
### Function Calls
- System Stack
- tail recursion -> 最后一步是递归，还不如使用循环，会被编译器自动改成循环。
- recursion的效率远远低于循环
# Queue
## Queue ADT
- FIFO, insertions at one end and deletion at the opposite end.
- Enque->入队 Dequeue->出队
### Array Implementation
- 可以实现成Circular Queue, 也就是当Rear向后的时候可以进位到Front前面的最后一位以最大化利用的空间。
- 