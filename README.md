# Mini-Project-AML-2026
Group members:
* Astrid Arhnung Schou-Hanssen
* Ellen Hørlyck Ebdrup
* Emil Fuhr Nielsen
* Julie Tilling Niemann

## Finetune BERT for Downstream Tasks

The central problem of this project is implementing a pre-trained BERT model and explore extensions to improve upon baseline results in a sentiment analysis classifying the polarity of IMDb Movie.

The IMDb Movie dataset consist of two columns one with a text string with reviews and one binary class column with 0 being the negative and 1 the positive classification of the review. 

Train, test and validation size???

### Pre-finetuning analysis with CLS token and mean-pooled token

We evaluate how much sentiment information is encoded in pretrained BERT by extracting hidden representations from each layer of BERT without finetuning and training a logsitic regression classifier on top.

We compare two representation strategiers: the [CLS] token and mean-pooked token embeddings. Performance is measured using accuracy and class-wise F1 scores across layers for both classes. 

![CLS](plots/CLS.png)

For the CLS token, we see that it quickly becomes informative after the first few layers and peaks at layer two but then saturates. This suggests that the CLS token doesn’t gain much additional task-specific information in deeper layers. We also observe a larger imbalance between the positive and negative class in the early layers, indicating that the representation is less stable and more biased before it converges.

![Mean_pool](plots/mean_pool.png)

In contrast, the mean-pooled embeddings start at a much stronger level and improve steadily across all layers, reaching their highest performance in the final layer. Here, the F1 scores for the positive and negative classes remain very similar across layers, suggesting a more balanced and robust representation. This indicates that task-relevant information is distributed across tokens and becomes more linearly separable deeper in the network.

Overall, this tells us that while the CLS token captures useful information early, the full representation continues to improve throughout the model and provides a more stable signal for classification.

### Finetuning

### Edge cases

### Attention matrix



 
