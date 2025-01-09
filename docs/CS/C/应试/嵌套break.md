[[C]]
```c
	for (int i = 0; i < 10; i ++) {
		if (i == 5) {
			continue;
		}
		printf("%d ", i);
	}
```
- 在这段代码中，虽然continue位于`for`下嵌套中的`if`， 但是在这里的if之中的`break`或者`continue`仍然是对于整个循环进行操作的。