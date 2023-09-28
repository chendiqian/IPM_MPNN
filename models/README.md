For the use of baseline, we need torch.vmap(torch.func.jacrev()), which does not support in-place operations

The solution would be:

1. Replace the batchnorm, e.g.,
```
from functorch.experimental import replace_all_batch_norm_modules_
gnn = replace_all_batch_norm_modules_(gnn)
```

2. Replace the scatter in MessagePassing classes with self-defined non in-place scatter operations, see `utils.scatter_sum` for more information.