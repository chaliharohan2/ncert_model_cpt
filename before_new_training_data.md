# Before new training data

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
the whole valley, and the distant mountains beyond.

The house was built of stone, and had a thatched roof.
The walls were whitewashed, and the windows were
covered with lead. The door was of wood, and the
steps leading up to it were of stone

--- few_shot_qa_famous ---
 One hundred pesos.

Q: What did the post office tell Lencho to do with the money?
A: Put it in a safe place.

Q: What did Lencho do with the money?
A: He bought a new bicycle.

Q: What did the post office

--- few_shot_qa_curriculum_specific ---
 It had been two months. She had sent a postcard, and in it she had written that she was very sorry she had not written sooner.

Q: What does Jaya say she would like to do for Arti?
A: She would like to send her a present.

Q: What

--- paraphrase_robustness_original ---
 One hundred pesos.

Q: What did Lencho do with the money?
A: He bought a cow, a hen, and a rooster.

Q: What happened to the cow, the hen, and the rooster?
A: They all died.

Q: Why did Len

--- paraphrase_robustness_reworded ---
 One hundred pesos.

Q: What did the post office tell Lencho to do with the money?
A: Send it to his wife, who would then give it to the children.

Q: What did the post office tell Lencho to do with the money?
A: Send
