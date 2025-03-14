# Chapter 2
## Maximize the sub-que Sum
- 问题描述：如果一个整数数列，内部可能有正的有负的，要找到最大的序列和
- i代表头指针，j代表尾指针
- Algorithm 1: 暴力遍历，O(n^3),找到最大值直接替换Max_sum
- Algorithm 2: Cumulative Sum, O(n^2)： 不用每次都单独求和，如果移动j指针，不用重新求和，直接加尾指针的量就相当于计算了当前子序列的和。
- Algorithm 3: Divide And Conquer, O(nlogn): 分治法，在中间隔开一整个数组，求***左半边的最大子列和***与***右半边的最大子列和***，还有***横跨边界的最大和***。比较求出最大，递归求解。
- Algorithm 4: On-line Algorithm, O(n): 只有一个指针用于遍历，如果当前和小于0则直接摒弃前面的所有序列，sum重置为0，每一步都要记录当前的maxsum。
## Study Euclid Algorithm and Sparse Matrix.
## Analyzing Time Complexity
### Method 1
-  如果$T(N) = O(N)$,那么应该有$T(2N)/T(N) = 2$
- 如果$T(N) = O(N^2)$，那么应该是四倍，以此类推。
### Method 2
- $lim_{N->正无穷}\dfrac{T(N)}{F(N)} = Constant$
# Chapter 3:
- 线性数据结构: Lists, Stacks, Queues
## ADT(Abstract Data Type)
- Data Type = {Object} and {Operations}
- e.g. 比如说int类型就是{一系列整数} + {整数的加减乘除}
- 更加偏向逻辑，与实现方法是独立的。
## List ADT
- Operations:
	- Finding: 查找对应索引的元素
	- Inserting: 将元素插入到对应的位置(索引K的后面)
	- Deleting: 将一个元素从列表中删除
- 与普通的数组之间的区别
	- 数组的长度是要事先确定的
	- 内存中存储空间是连续的
	- 插入与删除需要花费很多时间，因为要保证连续存储，所以要移动大量的元素
## Linked List
- 技巧
	- 可以加一个Dummy Head，是一个假的头数据，可以将Insertion 与 Deletion统一化，不用特别处理头元素。
- 优势：
	- 插入与删除均为O（1）
	- 但是查找为O(N)
- Doubly Linked Circular Lists
	- Features：
		- 每一个元素都会有llink与rlink，指向前面的元素与后面的元素
## Circular List
- 就是环形表，通常用链表进行实现
## MultiLists
- 40000个学生，2500门课程，打印每一个课程对应的学生，也打印出每一个学生对应的课表。
- Solution 1:
	- 可以创建一个二维数组`Array[40000][2500]`
	- `Array[i][j] = 1 if student i registered for class j and = 0 otherwise`
	- 横向与纵向分别遍历即可
	- 问题：会变成一个Sparse Matrix, 浪费存储空间
- Solution 2:
	- 十字链表，与上面的数组相同，同一行串在一起，同一列也串在一起。
## Applications
- Polynomial ADT
	- $\Sigma a_ix^{ei}$ ，可以定义各种操作，例如相加相减等等
	- 创建一个二维数组，里面有两行，第一行用于存储对应的次数，第二行用于存储对应的系数，可以简化存储空间复杂度。