[1] You must know the basics of RL to understand this post. This post is a good intrduction]
What IS MCTS? If I had to describe it, it would be an inference-time technique to improve action choices by leveraging additional compute. 

We can frame this in 2 equivalent ways:
- Given a policy, spend additonal compute to improve the policy
- Given a value function, spend addiitonal compute to make the esitmation better (note taht if we know the value of each action, we can extract a polict (argmax or sample proportionally)

Inference Only Technique? NO: This "policy upgrading" allows us to take a policy, bootstrap it to a better policy, and then distinll this upgraded policy back into the normal policy! Meaning that this is also useful during training, where we can improve the policy with no extenral policy ./ reference! 

## Improving my policy

Setting: I have a policy, and some additojal compute i wouls like to use to makem this policty better. HOw should I go about it?

Simple way: monte Carlo estimate of how good each move is: basically for each move, select the move, then conitnue playing with the normal policy. Do this for each move (multiple times for each to have a better estimate), and pick the move with the highest average reward. This is not bad BUT! It's very wasteful: in the end, we;re gonna choose the best action. So we care more about (ie spend more compute) esitmating accurately the value of the hihg value actions, rather than the shitty ones. So supopse this framewrok:
- We do a rollotu for each action once, and get a (very noisy) estimate of their value Q(s,a)
- We focus on improving the value estimate of the currently highest value accuracy (is it really that good? was it just a flike). So we do another rollouyt from that action (now we have 2 rollouts from that action, so estimate wuth lower variance.
- Sometimes the best action turns out to be bad, so we switch to another one etc...
[Also add part about expansion (we care more about things closer to the root) and backprop (how to update this info)]
Great! We have spent compute proprotioanlly to how important that action us.


But consider the problem: maybe one action that is really good ends up getting a really bad smple, adn that dooms it foerever. NOOOO. We would like to give the benefit of the doubt to actions that have had noisier estiamtes. but this ebenfit of doubt should also shrink the more samples you have, since we get more and more certain about its true value. UTC is one formulation of benefit of doubt and goes as follows:
....

So we pick the biggest leaf with the following value:



## What if we want to incorporate a value function?

Suppose we have a value function in addition! Instead of the noisy monte carlo estimate, we use the value fucntion! (The whole point of the value function is that is meant to be an esimtation of the expected value, which is what monte carlo roll outs estimate).

We dont need a polict (and inside the tree instead of biasing by polciy we just pick randomly)


## We don't need either!

Vacuously, the policy can be random, and MCTS still works; if we have a value we can improve it. But what if haev none. fret not! we can simply do it with a random policy. the monte carlo rollout gives us some information.

## Using it at training

Expert Iteration / Alpha Zero.

