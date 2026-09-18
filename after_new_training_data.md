# After new training data

=== BASE MODEL (no CPT) ===

--- completion ---
 could see
the valley floor and the surrounding mountains. The
house was built of stone and had a thatched roof.
The house was built in the shape of a square. The
entrance was on the north side. The house had a
large kitchen, a dining room, a living room

--- few_shot_qa_famous ---
 1000 francs.

Q: What is the name of the French president?
A: Francois Hollande.

Q: What is the name of the French president's wife?
A: Carla Bruni.

Q: What is the name of the French president's

--- few_shot_qa_curriculum_specific ---
 10 days.

Q: What is the name of the man who is the main character in the story?
A: Mr. Singh.

Q: What is the name of the man who is the main character in the story?
A: Mr. Singh.

Q:

--- paraphrase_robustness_original ---
 10000000000000000000000000000000000000000000000000000000000

--- paraphrase_robustness_reworded ---
 1000 pesos.

Q: What is the name of the French king who was assassinated in 1793?
A: Louis XVI.

Q: What is the name of the French king who was assassinated in 1793?
A: Louis XVI

=== CPT MODEL ===

--- completion ---
 could see
for miles in every direction. The valley was a bowl-shaped
depression, with the highest point at the centre, and the
lowest at the edge. The house was built on the crest of the
hill, with the front facing the valley.

The house was built of

--- few_shot_qa_famous ---
 One hundred pesos.

Q: What did Lencho do with the money he had asked God for?
A: He bought a cow, a goat, and a hen.

Q: What did Lencho do with the cow, goat, and hen?
A: He kept them as

--- few_shot_qa_curriculum_specific ---
 It had been two years since Jaya had last written to Arti, and she had sent her a postcard.

Q: What did Jaya send to Arti in the postcard she sent in 2008?
A: She sent her a small, hand-painted picture of a cow, and

--- paraphrase_robustness_original ---
 One hundred pesos.

Q: What did Lencho promise to do with the money?
A: He promised to pay it back in full, plus interest, in one year.

Q: What happened to Lencho's cow?
A: It died.

Q: What did

--- paraphrase_robustness_reworded ---
 100 pesos.

Q: What did Lencho do with the money he had saved?
A: He bought a cow, a goat, and a hen.

Q: What did Lencho do with the money he had saved?
A: He bought a cow, a
