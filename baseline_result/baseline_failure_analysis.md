# Baseline Results And Failure Analysis

This report contains the frozen retrieval baselines plus two supplemental references: fusion and oracle.

All rows are Top-k retrieval outputs, not answer-generation predictions. Gold documents are read only by the evaluator after retrieval is complete.

## Metric Definitions

- Evidence Recall@5: whether Top-5 contains at least one gold document.
- Complete Evidence Recall@5: whether Top-5 contains all gold documents required by the question.
- MRR: reciprocal rank of the first correct evidence document, averaged over questions.
- Search Success Rate: equivalent to Complete Evidence Recall@k in this project.
- Average Tool Calls: average number of core retrieval backends called per question.
- Latency: average and P95 end-to-end retrieval latency.
- Oracle: analysis-only upper bound that chooses the best existing output per question using evaluation labels.

## Overall @5

| Method | Group | Evidence Recall@5 | Complete Recall@5 | MRR | Search Success | Avg Tool Calls | Latency Avg ms | Latency P95 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| dense | overall | 91.67% | 74.44% | 0.8019 | 74.44% | 1.00 | 166.96 | 168.72 |
| file_fts | overall | 98.33% | 78.89% | 0.9088 | 78.89% | 1.00 | 6.78 | 15.26 |
| graph_path | overall | 92.78% | 82.22% | 0.7969 | 82.22% | 1.00 | 24.35 | 40.06 |
| fusion | overall | 97.22% | 85.56% | 0.8935 | 85.56% | 3.00 | 198.09 | 223.08 |
| oracle | overall | 98.33% | 90.00% | 0.9458 | 90.00% | 1.00 | 16.52 | 40.25 |

## Task-wise @5

| Method | Group | Evidence Recall@5 | Complete Recall@5 | MRR | Search Success | Avg Tool Calls | Latency Avg ms | Latency P95 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| dense | exact_file_lookup | 83.33% | 83.33% | 0.7178 | 83.33% | 1.00 | 167.04 | 168.37 |
| dense | multi_hop_relation | 96.67% | 45.00% | 0.8100 | 45.00% | 1.00 | 166.89 | 169.03 |
| dense | semantic_fact | 95.00% | 95.00% | 0.8778 | 95.00% | 1.00 | 166.96 | 168.52 |
| file_fts | exact_file_lookup | 96.67% | 96.67% | 0.8903 | 96.67% | 1.00 | 10.50 | 18.67 |
| file_fts | multi_hop_relation | 100.00% | 41.67% | 0.9028 | 41.67% | 1.00 | 5.63 | 15.26 |
| file_fts | semantic_fact | 98.33% | 98.33% | 0.9333 | 98.33% | 1.00 | 4.22 | 10.26 |
| graph_path | exact_file_lookup | 90.00% | 90.00% | 0.7117 | 90.00% | 1.00 | 32.61 | 50.08 |
| graph_path | multi_hop_relation | 95.00% | 63.33% | 0.8408 | 63.33% | 1.00 | 21.81 | 34.76 |
| graph_path | semantic_fact | 93.33% | 93.33% | 0.8381 | 93.33% | 1.00 | 18.64 | 28.83 |
| fusion | exact_file_lookup | 96.67% | 96.67% | 0.8639 | 96.67% | 3.00 | 210.15 | 231.46 |
| fusion | multi_hop_relation | 98.33% | 63.33% | 0.8917 | 63.33% | 3.00 | 194.32 | 218.00 |
| fusion | semantic_fact | 96.67% | 96.67% | 0.9250 | 96.67% | 3.00 | 189.81 | 205.33 |
| oracle | exact_file_lookup | 96.67% | 96.67% | 0.9250 | 96.67% | 1.00 | 14.03 | 23.46 |
| oracle | multi_hop_relation | 100.00% | 75.00% | 0.9458 | 75.00% | 1.00 | 25.33 | 166.31 |
| oracle | semantic_fact | 98.33% | 98.33% | 0.9667 | 98.33% | 1.00 | 10.20 | 14.52 |

## Failure Counts @5

| Method | Total Failures | semantic_fact | multi_hop_relation | exact_file_lookup |
|---|---:|---:|---:|---:|
| dense | 46 | 3 | 33 | 10 |
| file_fts | 38 | 1 | 35 | 2 |
| graph_path | 32 | 4 | 22 | 6 |
| fusion | 26 | 2 | 22 | 2 |
| oracle | 18 | 1 | 15 | 2 |

## Representative Failure Cases

### dense

- `sf_0009` / `semantic_fact`
  - Question: Which filmmaker directed the documentary that received a Best Documentary Academy Award nomination?
  - Gold: 4 Little Girls
  - Missed: 4 Little Girls
  - Retrieved top-5: Chicago Film Critics Association Awards 2016 | Chicago Film Critics Association Awards 1990 | Golden Globe Award for Best Animated Feature Film | Chicago Film Critics Association Awards 1988 | National Film Award for Best Music Direction

- `sf_0019` / `semantic_fact`
  - Question: What is the release year and genre of the South Korean film starring Lee Byung-hun and directed by Kim Jee-woon?
  - Gold: A Bittersweet Life
  - Missed: A Bittersweet Life
  - Retrieved top-5: Jin Hyuk | Lee Byung-hun | Joint Security Area (film) | Ahn Nae-sang | Addicted (2002 film)

- `sf_0035` / `semantic_fact`
  - Question: Where did the film win the award for "Best Film" in 2004?
  - Gold: Über Goober
  - Missed: Über Goober
  - Retrieved top-5: Chicago Film Critics Association Awards 2016 | Golden Globe Award for Best Animated Feature Film | 5th Lumières Awards | National Film Award for Best Music Direction | Chicago Film Critics Association Awards 1990

- `mh_0001` / `multi_hop_relation`
  - Question: Who was the director of an American epic space opera film that starred an actor born March 5, 1989?
  - Gold: Jake Lloyd | Star Wars: Episode I – The Phantom Menace
  - Missed: Jake Lloyd | Star Wars: Episode I – The Phantom Menace
  - Retrieved top-5: Eleni (film) | Georg Tressler | Michael McGreevey | Géza von Cziffra | Spaceballs

- `mh_0004` / `multi_hop_relation`
  - Question: Whose orchestra accompanied the First Lady of Song and other singers in a 1967 television special?
  - Gold: A Man and His Music + Ella + Jobim | Ella Fitzgerald
  - Missed: A Man and His Music + Ella + Jobim
  - Retrieved top-5: Movin' with Nancy (album) | Ella Fitzgerald | Vivian Della Chiesa | You're Just in Love | Just Bummin' Around
### file_fts

- `sf_0009` / `semantic_fact`
  - Question: Which filmmaker directed the documentary that received a Best Documentary Academy Award nomination?
  - Gold: 4 Little Girls
  - Missed: 4 Little Girls
  - Retrieved top-5: James Laxton | James Longley (filmmaker) | Iraq in Fragments | Mysterious Castles of Clay | Academy Honorary Award

- `mh_0001` / `multi_hop_relation`
  - Question: Who was the director of an American epic space opera film that starred an actor born March 5, 1989?
  - Gold: Jake Lloyd | Star Wars: Episode I – The Phantom Menace
  - Missed: Star Wars: Episode I – The Phantom Menace
  - Retrieved top-5: Jake Lloyd | John Balme | Alvin Drew | Lost in Space (American Dad!) | Chicago Film Critics Association Awards 1989

- `mh_0004` / `multi_hop_relation`
  - Question: Whose orchestra accompanied the First Lady of Song and other singers in a 1967 television special?
  - Gold: A Man and His Music + Ella + Jobim | Ella Fitzgerald
  - Missed: Ella Fitzgerald
  - Retrieved top-5: A Man and His Music + Ella + Jobim | Some Velvet Morning | That Lady (song) | Classic Diamonds – The DVD | Just Bummin' Around

- `mh_0005` / `multi_hop_relation`
  - Question: What title did the mother of Archduke Charles, Duke of Teschen hold in Tuscany?
  - Gold: Archduke Charles, Duke of Teschen | Maria Luisa of Spain
  - Missed: Maria Luisa of Spain
  - Retrieved top-5: Archduke Charles, Duke of Teschen | Archduke Charles of Austria (disambiguation) | Battle of Mannheim (1799) | Battle of Neresheim | Duke of Lafões

- `mh_0006` / `multi_hop_relation`
  - Question: Zooperstars features a clam dressed as a baseball player born in what year?
  - Gold: Sammy Sosa | ZOOperstars!
  - Missed: Sammy Sosa
  - Retrieved top-5: ZOOperstars! | David Jordan Bachner | Travis Hafner | Crystal Palace F.C. Player of the Year | Ron Klimkowski
### graph_path

- `sf_0009` / `semantic_fact`
  - Question: Which filmmaker directed the documentary that received a Best Documentary Academy Award nomination?
  - Gold: 4 Little Girls
  - Missed: 4 Little Girls
  - Retrieved top-5: James Longley (filmmaker) | Iraq in Fragments | The Cove (film) | James Laxton | Barry Jenkins

- `sf_0035` / `semantic_fact`
  - Question: Where did the film win the award for "Best Film" in 2004?
  - Gold: Über Goober
  - Missed: Über Goober
  - Retrieved top-5: Golden Globe Award for Best Animated Feature Film | Cars 2 | Lightning McQueen | Cars 3 | National Film Award for Best Lyrics

- `sf_0053` / `semantic_fact`
  - Question: What role does Philip John Neville currently have with Sky Sports and what ownership position does he hold with Salford City?
  - Gold: Phil Neville
  - Missed: Phil Neville
  - Retrieved top-5: Salford City F.C. | Salford, Greater Manchester | Listed buildings in Salford, Greater Manchester | Leicester City F.C. | King Power Stadium

- `sf_0055` / `semantic_fact`
  - Question: What year did the British–Norwegian pop group A1 form?
  - Gold: A1 (band)
  - Missed: A1 (band)
  - Retrieved top-5: Keiko Takemiya | Crystal Palace F.C. Player of the Year | Wilfried Zaha | VH1's Top 40 Videos of the Year | Blurred Lines

- `mh_0001` / `multi_hop_relation`
  - Question: Who was the director of an American epic space opera film that starred an actor born March 5, 1989?
  - Gold: Jake Lloyd | Star Wars: Episode I – The Phantom Menace
  - Missed: Jake Lloyd | Star Wars: Episode I – The Phantom Menace
  - Retrieved top-5: Victor Wong (actor born 1906) | Fernando Fernández (actor) | Emilio Fernández | Michael Brown (film director) | Michael Levine (set designer)
### fusion

- `ex_date_0014` / `exact_file_lookup`
  - Question: Which document mentions an actor from a 1962 film who also appeared in The Parent Trap?
  - Gold: Moon Pilot
  - Missed: Moon Pilot
  - Retrieved top-5: The Parent Trap (1961 film) | Parent Trap III | The Parent Trap II | Parent Trap: Hawaiian Honeymoon | The Parent Trap (song)

- `ex_date_0017` / `exact_file_lookup`
  - Question: Which document mentions the rank 23 for the lead single from the album "Women in Technology" in the United States?
  - Gold: White Town
  - Missed: White Town
  - Retrieved top-5: Album | Your Woman | Woman's Home Companion | Blurred Lines (album) | Diana Ross discography

- `mh_0001` / `multi_hop_relation`
  - Question: Who was the director of an American epic space opera film that starred an actor born March 5, 1989?
  - Gold: Jake Lloyd | Star Wars: Episode I – The Phantom Menace
  - Missed: Star Wars: Episode I – The Phantom Menace
  - Retrieved top-5: Eleni (film) | Jake Lloyd | Victor Wong (actor born 1906) | Fernando Fernández (actor) | Georg Tressler

- `mh_0004` / `multi_hop_relation`
  - Question: Whose orchestra accompanied the First Lady of Song and other singers in a 1967 television special?
  - Gold: A Man and His Music + Ella + Jobim | Ella Fitzgerald
  - Missed: Ella Fitzgerald
  - Retrieved top-5: A Man and His Music + Ella + Jobim | Some Velvet Morning | Just Bummin' Around | Movin' with Nancy (album) | Sukhwinder Singh

- `mh_0005` / `multi_hop_relation`
  - Question: What title did the mother of Archduke Charles, Duke of Teschen hold in Tuscany?
  - Gold: Archduke Charles, Duke of Teschen | Maria Luisa of Spain
  - Missed: Maria Luisa of Spain
  - Retrieved top-5: Archduke Charles, Duke of Teschen | Archduke Charles of Austria (disambiguation) | Battle of Neresheim | Count Leo Stefan of Habsburg | Battle of Mannheim (1799)
### oracle

- `ex_date_0014` / `exact_file_lookup`
  - Question: Which document mentions an actor from a 1962 film who also appeared in The Parent Trap?
  - Gold: Moon Pilot
  - Missed: Moon Pilot
  - Retrieved top-5: The Parent Trap (1961 film) | Parent Trap III | The Parent Trap II | Parent Trap: Hawaiian Honeymoon | The Parent Trap (song)

- `ex_date_0017` / `exact_file_lookup`
  - Question: Which document mentions the rank 23 for the lead single from the album "Women in Technology" in the United States?
  - Gold: White Town
  - Missed: White Town
  - Retrieved top-5: Your Woman | SPARS | Abuse Me | Album | 1988 Australian Open – Women's Singles

- `mh_0001` / `multi_hop_relation`
  - Question: Who was the director of an American epic space opera film that starred an actor born March 5, 1989?
  - Gold: Jake Lloyd | Star Wars: Episode I – The Phantom Menace
  - Missed: Star Wars: Episode I – The Phantom Menace
  - Retrieved top-5: Jake Lloyd | John Balme | Alvin Drew | Lost in Space (American Dad!) | Chicago Film Critics Association Awards 1989

- `mh_0004` / `multi_hop_relation`
  - Question: Whose orchestra accompanied the First Lady of Song and other singers in a 1967 television special?
  - Gold: A Man and His Music + Ella + Jobim | Ella Fitzgerald
  - Missed: Ella Fitzgerald
  - Retrieved top-5: A Man and His Music + Ella + Jobim | Some Velvet Morning | That Lady (song) | Classic Diamonds – The DVD | Just Bummin' Around

- `mh_0007` / `multi_hop_relation`
  - Question: The Rieder Automatic Rifle was a model made by the company that created firearms for the forces of what country?
  - Gold: Lee–Enfield | Rieder Automatic Rifle
  - Missed: Lee–Enfield
  - Retrieved top-5: Rieder Automatic Rifle | Automatic rifle | Howell Automatic Rifle | Sieg automatic rifle | Chauchat

## Files

- `baseline_metrics_all.csv`: all metrics across overall/task groups and k=1/3/5.
- `baseline_metrics_overall.csv`: overall metrics only.
- `baseline_metrics_by_task.csv`: task-wise metrics.
- `failure_cases_k5.csv`: all Complete@5 failure cases.
- `result_file_manifest.csv`: source-to-exported artifact mapping.
- `retrieval_outputs_and_metrics/`: copied Top-k retrieval outputs, metric files, and summary files.