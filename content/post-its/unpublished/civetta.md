Things Ive Learned while writing civetta:
    - ML is a compositoon of dofferential operators, so as pong as you deifne wverythign priotivies and tjen use chain rule to compose. 
- The computation graph encapsulates everything: even the input tensors are not really different from the normal weight tensors, only difference is that you don’t take the grad (or that the optimizer doesn’t see it). Or also the loss is part of the CG as well. If you wannabe pedantic, the CG is a forward pass
- All ops that do memory wrangling are just redirections on the gradients (e.g. concat, reshape, ...).
- You need bery few ops, even matmul is not a fundamental op
- one thing that the computation graph does not capture is multiple, overlapping but not in the same batch, computations (though jt can be extended to handle this). 
- Relu is the simplest nonlinear since in backprp its literally just a binary filter over whatever tensor we aplied it to. 
- - Normal softmax is truly too numberically unstable, even for toy models. So things that interact with softmax (e.g. cross entorpy) all need to be phrased using the log probs rather than the probs. Which is why we do things using CrossEntropyWithLogits in python. So for example for cross entropy we have:
Derive formula with LSE.
So in practice we rarely rely on softmax, raterh on logsumexp.
Scalars are just 0 dim tensors. hahah what a dumbass i am. though they were say 1D tensor with onen entry. You index the only elm eith []
Even the inputs that are parameters (eg axis across which you broadcast) are parameters!
there are 2 interfaces:
- non parametric functions: these do not hold params, so you can jsut thread them through) 
- parametric functions (modules): sicne these have params, they need to expose their params so that the optimizer can acess them. PyTorch offers a fucntional interface also for non parametic ones (ef relu cs Relu)
- So all learnable modules in pytoch are functionals since they have to hold information across calls, which they do in the closurein the closure the parameter and then expose a forward function that uses those osrameters
