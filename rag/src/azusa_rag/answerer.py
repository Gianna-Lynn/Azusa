"""
提供最小 answer 生成能力:
1. 延迟加载 Qwen2.5-0.5B-Instruct 小模型, 模块级缓存复用.
2. 把检索结果拼成证据块, 由模型基于证据生成 "答案 + 引用".
3. 解析并规范化模型输出, 补齐证据摘要和不确定性, 输出统一的四段式回答.
"""
MODEL_PATH = r"D:\Chuo_Data\Models\Qwen2.5-0.5B-Instruct"

_tokenizer = None
_model = None


def load_answer_model():
    """
    延迟加载 answer 用的小模型, 并缓存复用.

    首次调用时才 import transformers 并从 MODEL_PATH 加载 tokenizer 和
    causal LM, 之后把两者缓存在模块级变量中; 后续调用直接复用,
    避免每次 import 都立刻吃显存.

    Returns:
        (tokenizer, model) 二元组.
    """


    from transformers import AutoModelForCausalLM, AutoTokenizer


    global _tokenizer, _model

    if _tokenizer is None:

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    if _model is None:
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,

            torch_dtype="auto",

            device_map="auto",
        )

    return _tokenizer, _model


def build_context(result_records) -> str:
    """
    把检索结果拼成给 LLM 的证据块.

    每个结果记录格式化为 "[chunk_id=<id>]\n<text>" 的字符串块,
    再用两个换行符连接成一个完整的上下文字符串, 供模型参考生成答案.

    Args:
        result_records: EmbeddingRecord 列表.

    Returns:
        包含多个 chunk_id 和对应文本的上下文字符串.
    """
    context_parts = []

    for record in result_records:
        context_parts.append(
            f"[chunk_id={record.chunk_id}]\n{record.text}"
        )


    return "\n\n".join(context_parts)


def _extract_field(lines: list[str], labels: tuple[str, ...]) -> str:
    """
    从模型输出中提取指定标签对应的单行字段.

    按行扫描, 返回第一个以 labels 中任一标签开头的行去掉标签前缀和
    首尾空白后的剩余文本; 没有匹配时返回空字符串.

    Args:
        lines: 模型输出按行拆分、去空白后的非空行列表.
        labels: 候选字段标签, 例如 ("结论：", "答案：").

    Returns:
        匹配字段的取值字符串; 未匹配时为空字符串.
    """
    for line in lines:
        for label in labels:
            if line.startswith(label):
                return line.removeprefix(label).strip()
    return ""

def _summarize_cited_evidence(cited_ids: list[str], result_records) -> str:
    """
    从已识别的引用 ID 对应的记录中生成简短证据摘要, 不新增引用 ID.

    normalize_answer_output 在模型未提供非空的“证据：”字段、
    但仍有已识别并保留的引用 ID 时, 调用_summarize_cited_evidence.

    对每个被引用的 chunk_id 取出对应文本, 压缩空白后截断到 180 个字符
    (超出部分以 "..." 结尾), 再用中文分号拼接.

    Args:
        cited_ids: 模型输出中提到的 chunk_id 列表.
        result_records: EmbeddingRecord 列表, 用于按 chunk_id 查找原文.

    Returns:
        以 "；" 连接的证据摘要字符串; cited_ids 为空时返回空字符串.
    """
    record_by_id = {record.chunk_id: record for record in result_records}  # 生成字典. "key : value  for 循环式"
    summaries = []  # 空列表 summaries 用来存储每个引用的证据块的简短摘要。
    for chunk_id in cited_ids:  # 遍历 cited_ids 列表中的每个 chunk_id.
        record = record_by_id[chunk_id]  # 根据 chunk_id 去 record_by_id 字典中找到对应的 EmbeddingRecord 对象。
        clean_text = " ".join(record.text.split())  # 取出 record.text 的文本内容，去掉多余的空白字符，并把它们用单个空格连接起来，形成一个干净的文本字符串。
        if len(clean_text) > 180:  # 如果太长, 保留前180个字符, 补上省略号.
            clean_text = clean_text[:180] + "..."
        summaries.append(f"[{chunk_id}] {clean_text}")  # 把每个引用的证据块的 chunk_id 和对应的简短文本摘要拼接成一个字符串，并添加到 summaries 列表中。
    return "；".join(summaries)  # 最后把所有的简短证据摘要用中文分号连接成一个完整的证据摘要字符串，并返回.


def normalize_answer_output(output_text: str, result_records) -> dict:
    """
    解析模型生成的两行 "答案 + 引用", 再由 Python 补齐证据摘要和不确定性,
    输出统一的四段式回答: "结论, 证据, 引用来源, 不确定性".

    小模型的自由输出形如:
        答案：<direct answer>
        引用：<chunk_id1, chunk_id2>
    如果证据不足, 就输出:
        答案：证据不足
        引用：无

    同时提取 "证据："/"不确定性：" 字段, 并在整段输出中按子串查找本次
    检索记录的 chunk_id, 据此确定 answer_status; 不限于 "引用：" 字段,
    也不检查 ID 的边界或证据是否支持结论. 不根据答案内容推断或补填引用.
    status 为 valid 仅表示答案字段非空、不是 "证据不足", 且识别到了引用 ID.

    Args:
        output_text: 模型的原始输出文本.
        result_records: 本次检索的 EmbeddingRecord 列表, 其 chunk_id 构成合法引用集合.

    Returns:
        dict, 包含:
            answer_text: 规范化后的答案 (或 "答案缺失"/"证据不足").
            evidence_text: 证据摘要 (模型提供, 或按引用生成, 或占位文本).
            citation_ids: 在输出中识别到的本次检索记录 ID, 按检索记录顺序排列;
                答案明确为 "证据不足" 时清空.
            uncertainty: 不确定性说明 (模型提供或按状态回填).
            citation_validation: {"status": answer_status, "valid_chunk_ids": ...}.
            formatted_text: 四段式格式化文本.
            raw_output: 去掉首尾空白后的模型原始输出.
    """
    output_text = output_text.strip()
    valid_chunk_ids = [record.chunk_id for record in result_records]

    lines = []
    # 把模型输出的文本按行分割成一个列表
    for line in output_text.splitlines():
        line = line.strip()
        # 去掉每行前后的空白字符，如果这一行不空，就把它添加到 lines 列表中。
        if line:
            lines.append(line)

    answer_value = _extract_field(lines, ("结论：", "答案："))
    has_answer_field = bool(answer_value)
    evidence_value = _extract_field(lines, ("证据：",))
    uncertainty_value = _extract_field(lines, ("不确定性：",))

    cited_ids = []
    # 在整段输出中查找已知 ID 子串; 不解析引用字段, 不按答案内容补填引用。
    for chunk_id in valid_chunk_ids:
        if chunk_id in output_text:
            cited_ids.append(chunk_id)

    # 根据答案字段和已识别的引用 ID 确定状态，不校验证据是否支持答案。
    if not has_answer_field and cited_ids:
        # 输出中识别到了引用 ID，但缺少非空的“答案/结论”字段。
        # 处理：保留引用，标记为答案缺失，交给人工审核。
        answer_value = "答案缺失"
        answer_status = "answer_missing"
    elif not has_answer_field:
        # 输出中既没有非空的答案字段，也未识别到引用 ID。
        # 处理：归为证据不足。
        answer_value = "证据不足"
        answer_status = "missing_or_insufficient"
    elif answer_value == "证据不足":
        # 模型：明确回答“证据不足”（即使同时输出了引用）。
        # 处理：撤销所有引用，统一标记为证据不足。
        cited_ids = []
        answer_status = "missing_or_insufficient"
    elif not cited_ids:
        # 模型给出了答案，但输出中未识别到本次检索记录的 ID。
        answer_status = "citation_missing"
    else:
        # 答案字段非空且识别到了引用 ID；valid 不代表结论已通过事实核验。
        answer_status = "valid"

    if not evidence_value and cited_ids:
        # 模型未提供证据摘要，但已识别到引用 ID。
        # 处理：根据引用的原文确定性生成证据摘要。
        evidence_value = _summarize_cited_evidence(cited_ids, result_records)
    elif not evidence_value:
        # 模型未提供证据摘要，也没有已识别的引用 ID。
        evidence_value = "未提供独立的证据摘要"

    if answer_value == "证据不足":
        # 模型状态：证据不足。
        evidence_value = "现有检索证据不足以支持可靠结论"
        uncertainty_value = uncertainty_value or "证据不足，无法可靠回答"
    elif answer_value == "答案缺失":
        # 模型状态：有引用但没有答案。
        uncertainty_value = uncertainty_value or "有引用但没有答案，需要人工审核"
    elif not cited_ids:
        # 模型状态：有答案，但输出中未识别到本次检索记录的 ID。
        uncertainty_value = uncertainty_value or "模型未提供可验证的引用，结论需人工复核"
    else:
        # 有答案且识别到了引用 ID；默认填“无”不表示已核验证据支持关系。
        uncertainty_value = uncertainty_value or "无"

    citation_text = "无"
    if cited_ids:
        # 将已识别且保留的 chunk_id 拼接为最终引用文本。
        citation_text = ", ".join(cited_ids)

    formatted_text = (
        f"结论：{answer_value}\n"
        f"证据：{evidence_value}\n"
        f"引用来源：{citation_text}\n"
        f"不确定性：{uncertainty_value}"
    )
    # 最终返回一个字典，包含整理后的答案文本、引用的 chunk_id 列表、格式化后的文本，以及模型的原始输出文本。
    return {
        "answer_text": answer_value,
        "evidence_text": evidence_value,
        "citation_ids": cited_ids,
        "uncertainty": uncertainty_value,
        "citation_validation": {
            "status": answer_status,
            "valid_chunk_ids": valid_chunk_ids,
        },
        "formatted_text": formatted_text,
        "raw_output": output_text,
    }


def answer_with_citations(query: str, result_records, tool_context: str="") -> dict:
    """
    基于 query + 检索结果, 生成最小 answer + citations 输出.

    无检索结果时直接返回证据不足的固定结果; 否则加载模型, 把检索结果
    拼成证据块 (可选追加工具上下文), 用贪心解码生成最多 80 个 token,
    再经 normalize_answer_output 规范化成四段式回答.

    Args:
        query: 用户问题.
        result_records: 检索到的 EmbeddingRecord 列表, 允许为空.
        tool_context: 可选的工具上下文文本, 非空时拼接到证据块末尾.

    Returns:
        dict, 包含:
            answer_text: 四段式格式化回答文本.
            answer_raw: 模型的原始输出文本.
            evidence_text: 证据摘要.
            citation_ids: 在模型输出中识别并由规范化步骤保留的本次检索记录 ID.
            uncertainty: 不确定性说明.
            citation_validation: {"status": ..., "valid_chunk_ids": ...}.
            context: 最终拼好的证据上下文字符串.
    """
    if not result_records:
        return {
            "answer_text": (
                "结论：证据不足\n"
                "证据：没有检索到可用证据\n"
                "引用来源：无\n"
                "不确定性：证据不足，无法可靠回答"
            ),
            "answer_raw": "",
            "evidence_text": "没有检索到可用证据",
            "citation_ids": [],
            "uncertainty": "证据不足，无法可靠回答",
            "citation_validation": {
                "status": "missing_or_insufficient",
                "valid_chunk_ids": [],
            },
            "context": "",
        }

    # 加载小模型和对应的 tokenizer，准备进行文本生成任务。
    tokenizer, model = load_answer_model()

    # 把检索结果拼成给模型的证据块，形成一个包含多个 chunk_id 和对应文本的上下文字符串，供模型参考生成答案。
    context = build_context(result_records)

    # 如果工具那边看到了工具上下文, 那么就拼接到模型能看见的字符串最后.
    if tool_context:
        context += (
            "\n\n"
            "[Tool context]\n"
            f"{tool_context}"
        )

    # 构建一个对话消息列表，包含系统提示和用户输入两部分
    messages = [
        {
            # 系统提示: 负责规定规则，比如“只能根据证据回答”“必须输出两行”“证据不足时怎么写”
            "role": "system",
            "content": (
                "You must answer only from the provided evidence. "
                "Do not use outside knowledge. "
                "Inspect each evidence block independently. "
                "If at least one block directly states the answer, the evidence is sufficient; "
                "ignore unrelated blocks and cite only the supporting block. "
                "Output exactly two lines and nothing else.\n"
                "Line 1: 答案：<direct answer>\n"
                "Line 2: 引用：<chunk_id1, chunk_id2>\n"
                "If evidence is insufficient, output:\n"
                "答案：证据不足\n"
                "引用：无"
            )
        },
        {
            # 用户输入: 包含具体的证据内容和问题，要求模型只能使用这些证据来回答问题，并且只能引用已经出现在证据中的 chunk_id。
            "role": "user",
            "content": (
                f"Evidence:\n{context}\n\n"
                f"Question:\n{query}\n\n"
                "Only use chunk_id values that already appear in the evidence. "
                "Copy each cited chunk_id exactly, without inventing or shortening it."
            )
        },
    ]

    # 把构建好的对话消息列表转换成一个完整的输入文本字符串，
    # 按照模型预期的格式组织系统提示和用户输入，准备进行编码和生成。
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # 转换成模型的输入
    # 把构建好的对话消息列表转换成模型输入的token序列，
    # 返回一个包含input_ids等信息的字典，并把它移动到模型所在的设备上（如GPU），
    # 准备进行文本生成。
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # 生成回答
    # 基于输入的文本，使用模型生成输出，设置最大生成长度为80个token，
    # 不使用采样（即贪心解码），温度和top_p参数不启用，以确保生成结果的确定性。
    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False,
        temperature=None,
        top_p=None,
    )

    # 转回文字
    # 把模型生成的token序列解码成文本，跳过特殊token，
    # 并去掉前后的空白字符，得到模型的原始输出文本。
    output_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    ).strip()

    # 整理结果
    # 整理答案字段，识别输出中已有的本次检索记录 ID，并标记答案与引用状态。
    # 不根据答案内容推断或补填引用；缺少摘要时，仅为已识别的引用摘取原文。
    normalized_output = normalize_answer_output(output_text, result_records)

    return {
        "answer_text": normalized_output["formatted_text"],
        "answer_raw": normalized_output["raw_output"],
        "evidence_text": normalized_output["evidence_text"],
        "citation_ids": normalized_output["citation_ids"],
        "uncertainty": normalized_output["uncertainty"],
        "citation_validation": normalized_output["citation_validation"],
        "context": context,
    }
