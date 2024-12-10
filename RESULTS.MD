# Seeds

### Observed
- Goes down by frequency
- Goes down for certain part of speech
- Goes down more for larger models
- Dip in the uniform to unigram switch between 10^2 and 10^3
- Beginning / end of word => No difference
- KL to uniform / unigram => As predicted

### Predicted (to be proved)
- Calculate the "randomness" through initialization etc.

### TBD
- Linear regression to predict and dissect the effect of POS / frequency / model size

# Deduped

### Observed
- Compare to duped => very similar
- MD is always learned between 32 and 64
- TO is 16 to 32
- VBZ is 16 to 32
- VBN is 64
- Earlier step PRP and VBP
- Later steps NNP and PRP$

### TBD
- Scaling law plots based on surprisal/training step
- Scatterplot for cross-entropy + KL
- Delayed version for smaller models to get to the same point
- Delayed effect of frequency / part of speech
- Dissect the effect of POS / frequency / model size
