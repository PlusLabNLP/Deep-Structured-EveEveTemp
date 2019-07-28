# AAAI target:

1. Extend the deep structure learning on event relation extraction to more
   dateset
2. Incorporate experiment results with pre-trained BERT embedding
3. Ablation studies on whether linguistic features is important. As well as each
   structure constraint's effects

# Semi-Supervised Learning (RJ's comment):
For this paper, we will focus on semi-supervised method to improve event relation extraction. We have evaluated Entropy Minimization, Virtural Adversarial Training and SVD on BERT on TCR. After boostrapping test, it seems only the third method works marginally better.

## TODO:
1. Run SVD BERT boostrap on TB-Dense and MATRES;
2. Figure out how to incorporate semi-supervised learning in global learning non-trivially
3. Multi-task learning with all available datasets; optional.
