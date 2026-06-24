# Stage 2 Badcase v1

- generated_at: `2026-06-24T16:43:52+08:00`
- workflow_name: `stage2_final`

## Retrieval All-Fail Cases

- 无。

## Retrieval Disagreements

- `min_eval_query_13`: hit spread=1.0000, recall spread=1.0000, mrr spread=1.0000
- `min_eval_query_11`: hit spread=0.0000, recall spread=0.0000, mrr spread=0.5000
- `min_eval_query_7`: hit spread=0.0000, recall spread=0.0000, mrr spread=0.5000
- `min_eval_query_14`: hit spread=0.0000, recall spread=0.5000, mrr spread=0.0000
- `min_eval_query_3`: hit spread=0.0000, recall spread=0.5000, mrr spread=0.0000

## Answer and Citation Badcases

### min_eval_query_1

- query: 有时只需要一本足够厚的书
欢迎加入正确座位秩序共同体
- reference_answer: 真正的边界不需要大声，有时只需要一本足够厚的书；14C 乘客随后被欢迎加入正确座位秩序共同体。
- answer: 答案：教宗猫
引用：cat_notes_1_chunk_001, cat_notes_1_chunk_002, cat_notes_2_chunk_004
- citations: ['cat_notes_1_chunk_001', 'cat_notes_1_chunk_002', 'cat_notes_2_chunk_004']
- badcase_type: incomplete_answer
- review_notes: 答案只给出角色名，没有回答两段语义；引用包含金标 chunk，但还回填了无关的 cat_notes_2_chunk_004。

### min_eval_query_2

- query: 人生有时需要击中重点
教宗猫以极高礼貌完成座位确认
- reference_answer: 教宗猫用极高的礼貌处理占座问题，并说棒球棒提醒人们人生有时需要击中重点。
- answer: 答案：极高礼貌
引用：cat_notes_2_chunk_002
- citations: ['cat_notes_2_chunk_002']
- badcase_type: incomplete_answer
- review_notes: 答案抓住礼貌这一关键词，但没有说明占座处理和击中重点的含义。

### min_eval_query_3

- query: JR常磐缓行线
常规意义上的直通运转
- reference_answer: JR 常磐缓行线与东京地铁千代田线直通；JR 与都营地下铁之间常规意义上的直通运转非常少见。
- answer: 答案：常规意义上的直通运转
引用：cat_notes_3_chunk_002
- citations: ['cat_notes_3_chunk_002']
- badcase_type: incomplete_answer
- review_notes: 只复述问题短语，没有回答常磐缓行线与千代田线直通，也漏掉第一条支撑证据。

### min_eval_query_4

- query: 精神上，当然是精神上
懂事是好事
- reference_answer: 旁边乘客猫表示精神上理解，其他乘客也称他懂事，并认为懂事是好事。
- answer: 答案：懂事是好事
引用：cat_notes_1_chunk_000
- citations: ['cat_notes_1_chunk_000']
- badcase_type: partial_answer
- review_notes: 核心结论正确且引用正确，但没有完整复述场景关系。

### min_eval_query_5

- query: 很多觉醒都始于一个问题
也可能始于一个答案
- reference_answer: 教宗猫说很多觉醒始于一个问题，也可能始于一个答案。
- answer: 答案：很多觉醒都始于一个问题。
引用：cat_notes_1_chunk_002
- citations: ['cat_notes_1_chunk_002']
- badcase_type: incomplete_answer
- review_notes: 漏掉了也可能始于一个答案这一后半句。

### min_eval_query_6

- query: 您的座位是 17A
主也为您预备了窗景
- reference_answer: 空乘确认西装猫的座位是靠窗的 17A，教宗猫说主也为他预备了窗景。
- answer: 答案：17A
引用：cat_notes_2_chunk_003, cat_notes_2_chunk_005, cat_notes_1_chunk_000
- citations: ['cat_notes_2_chunk_003', 'cat_notes_2_chunk_005', 'cat_notes_1_chunk_000']
- badcase_type: over_citation
- review_notes: 座位号正确，但漏掉靠窗和窗景信息，并回填了两条不必要引用。

### min_eval_query_7

- query: 放在可以看见的地方吧
这样大家都会更尊重登机牌
- reference_answer: 教宗猫让棒球棒放在大家看得见的地方，因为这样大家会更尊重登机牌。
- answer: 答案：放得当心
引用：cat_notes_2_chunk_004
- citations: ['cat_notes_2_chunk_004']
- badcase_type: wrong_answer
- review_notes: 答案在证据中没有对应表达，也没有回答为什么要放在看得见的地方。

### min_eval_query_8

- query: 第一，不要占座
17A 的窗景其实也不错
- reference_answer: 西装猫总结不要占座，并发现 17A 的窗景其实也不错。
- answer: 答案：第一，不要占座。
引用：cat_notes_2_chunk_003
- citations: ['cat_notes_2_chunk_003']
- badcase_type: wrong_citation
- review_notes: 答案主体正确但不完整；所引 chunk 不包含该总结，正确证据是 cat_notes_2_chunk_005。

### min_eval_query_9

- query: 也愿登机牌与你同在
座位号也很重要咪
- reference_answer: 空乘猫回应愿登机牌与你同在；故事总结爱与宽恕重要，但座位号也很重要。
- answer: 答案：也愿登机牌与你同在
引用：cat_notes_2_chunk_006
- citations: ['cat_notes_2_chunk_006']
- badcase_type: incomplete_answer
- review_notes: 引用正确，但漏掉座位号也很重要这一总结。

### min_eval_query_10

- query: JR埼京线 ↔ 临海线
它不是东京地铁
- reference_answer: JR 埼京线与临海线相互直通，但临海线由东京临海高速铁道运营，不属于东京地铁。
- answer: 答案：JR埼京线 ↔ 临海线
引用：cat_notes_3_chunk_003
- citations: ['cat_notes_3_chunk_003']
- badcase_type: incomplete_answer
- review_notes: 只给出线路关系，漏掉临海线运营方以及它不是东京地铁的关键辨析。

### min_eval_query_13

- query: Which artist who had a major influence on the Gothic Revival is represented in the V&A's British galleries?
- reference_answer: Horace Walpole
- answer: 答案：Thomas Chippendale
引用：autoevaluate_squad_sample_validation_0002_5726d4a45951b619008f7f6c_chunk_002
- citations: ['autoevaluate_squad_sample_validation_0002_5726d4a45951b619008f7f6c_chunk_002']
- badcase_type: wrong_entity
- review_notes: 正确答案是 Horace Walpole；模型从同一 chunk 中选了另一个人物。

### min_eval_query_14

- query: In the 1890s, which regional colleges and universities did the University of Chicago affiliate with?
- reference_answer: Des Moines College, Kalamazoo College, Butler University, Stetson University, and later Shimer College.
- answer: 答案：University of Chicago, Des Moines College, Kalamazoo College, Butler University, Stetson University
引用：autoevaluate_squad_sample_validation_0003_572843304b864d1900164848_chunk_000
- citations: ['autoevaluate_squad_sample_validation_0003_572843304b864d1900164848_chunk_000']
- badcase_type: incomplete_answer
- review_notes: 错误地把 University of Chicago 列入附属院校，并漏掉后续的 Shimer College 与第二条证据。

### min_eval_query_15

- query: Who scored the touchdown that cut the score to 10-7?
- reference_answer: Jonathan Stewart
- answer: 答案：Jonathan Stewart
引用：autoevaluate_squad_sample_validation_0007_56bf1ae93aeaaa14008c951b_chunk_000
- citations: ['autoevaluate_squad_sample_validation_0007_56bf1ae93aeaaa14008c951b_chunk_000']
- badcase_type: wrong_citation
- review_notes: 答案正确，但引用指向 halftime show 语料；正确证据是 autoevaluate_squad_sample_validation_0004_56d729180d65d21400198427_chunk_000。

## Pending Manual Reviews

- 无，所有 answer cases 均已人工复核。
