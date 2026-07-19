MODEL_PATH = r"D:\Chuo_Data\Models\Qwen2.5-0.5B-Instruct"

_tokenizer = None
_model = None


def load_answer_model():
    """
    延迟加载 answer 用的小模型，避免每次 import 都立刻吃显存。
    """
    # 延迟加载就是在函数内部定义一个全局变量，第一次调用函数时才加载模型并赋值给这个变量。
    # 后续调用函数时，如果变量已经有值了，就直接返回，不需要重复加载模型。
    from transformers import AutoModelForCausalLM, AutoTokenizer

    global _tokenizer, _model # 声明这两个变量是全局的，这样我们才能在函数内部修改它们的值。

    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH) # 从指定路径加载预训练的tokenizer，准备对输入文本进行编码和解码。

    if _model is None:
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype="auto", # 根据模型权重自动选择合适的数据类型，通常是 float16 或 bfloat16，以节省显存和加速推理。
            device_map="auto",  # 自动将模型的不同部分分配到可用的设备上（如 CPU 或 GPU），以优化性能和资源利用。
        )

    return _tokenizer, _model


def build_context(result_records) -> str:
    """
    把检索结果拼成给 LLM 的证据块。
    """
    context_parts = []

    for record in result_records:
        context_parts.append(
            f"[chunk_id={record.chunk_id}]\n{record.text}"
        )
        # 把每个检索结果chunk_id 和对应的文本指定特定格式拼接成一个字符串块，添加到 context_parts 列表中。

    return "\n\n".join(context_parts) # 最后把所有的字符串块用两个换行符连接成一个完整的上下文字符串.
    # 返回结果形如:
    # [chunk_id=cat_notes_3_chunk_000]
    # JR常磐缓行线会与东京地铁千代田线直通运行，这属于常规意义上的直通运转。
    #
    # [chunk_id=cat_notes_3_chunk_002]
    # 有些线路虽然存在跨线关系，但并不都属于常规意义上的直通运转，需要区分具体线路关系。


def _extract_field(lines: list[str], labels: tuple[str, ...]) -> str:
    """从模型输出中提取指定标签对应的单行字段。"""
    for line in lines:
        for label in labels:
            if line.startswith(label):
                return line.removeprefix(label).strip()
    return ""

def _summarize_cited_evidence(cited_ids: list[str], result_records) -> str:
    """从模型明确引用的记录中生成简短证据摘要，不替模型新增 citation。result_records 是 EmbeddingRecord 列表，cited_ids 是模型输出中提到的 chunk_id 列表。"""
    record_by_id = {record.chunk_id: record for record in result_records} # 生成字典. "key : value  for 循环式"
    summaries = []      # 空列表 summaries 用来存储每个引用的证据块的简短摘要。
    for chunk_id in cited_ids:      # 遍历cited_ids 列表中的每个 chunk_id.
        record = record_by_id[chunk_id]     # 根据 chunk_id去record_by_id 字典中找到对应的 EmbeddingRecord 对象。
        clean_text = " ".join(record.text.split())    # 取出 record.text 的文本内容，去掉多余的空白字符，并把它们用单个空格连接起来，形成一个干净的文本字符串。
        if len(clean_text) > 180:                     # 如果太长, 保留前180个字符, 补上省略号.
            clean_text = clean_text[:180] + "..."
        summaries.append(f"[{chunk_id}] {clean_text}")  # 把每个引用的证据块的 chunk_id 和对应的简短文本摘要拼接成一个字符串，并添加到 summaries 列表中。
    return "；".join(summaries)         # 最后把所有的简短证据摘要用中文分号连接成一个完整的证据摘要字符串，并返回.


def normalize_answer_output(output_text: str, result_records) -> dict:
    """
    解析模型生成的两行“答案 + 引用”，再由 Python 补齐证据摘要和不确定性，输出统一的四段式回答:"结论,证据, 引用来源, 不确定性"。
    小模型的自由输出形如:
    答案：<direct answer>
    引用：<chunk_id1, chunk_id2>
    如果证据不足，就输出：
    答案：证据不足
    引用：无
    """
    output_text = output_text.strip()
    valid_chunk_ids = [record.chunk_id for record in result_records]

    lines = []
    for line in output_text.splitlines(): # 把模型输出的文本按行分割成一个列表
        line = line.strip()
        if line:
            lines.append(line) # 去掉每行前后的空白字符，如果这一行不空，就把它添加到 lines 列表中。

    answer_value = _extract_field(lines, ("结论：", "答案："))
    has_answer_field = bool(answer_value)
    evidence_value = _extract_field(lines, ("证据：",))
    uncertainty_value = _extract_field(lines, ("不确定性：",))

    cited_ids = []
    for chunk_id in valid_chunk_ids:
        if chunk_id in output_text:
            cited_ids.append(chunk_id)

    # 先根据模型是否给出答案、是否给出有效引用，确定答案状态。
    if not has_answer_field and cited_ids:
        # 模型：有有效引用，但缺少“答案/结论”字段。
        # 处理：保留引用，标记为答案缺失，交给人工审核。
        answer_value = "答案缺失"
        answer_status = "answer_missing"
    elif not has_answer_field:
        # 模型：既没有答案字段，也没有有效引用。
        # 处理：归为证据不足。
        answer_value = "证据不足"
        answer_status = "missing_or_insufficient"
    elif answer_value == "证据不足":
        # 模型：明确回答“证据不足”（即使同时输出了引用）。
        # 处理：撤销所有引用，统一标记为证据不足。
        cited_ids = []
        answer_status = "missing_or_insufficient"
    elif not cited_ids:
        # 模型：给出了答案，但没有可验证的引用。
        answer_status = "citation_missing"
    else:
        # 模型：同时给出了答案和至少一个有效引用。
        answer_status = "valid"

    if not evidence_value and cited_ids:
        # 模型：未提供证据摘要，但有有效引用。
        # 处理：根据引用的原文确定性生成证据摘要。
        evidence_value = _summarize_cited_evidence(cited_ids, result_records)
    elif not evidence_value:
        # 模型：未提供证据摘要，也没有有效引用。
        evidence_value = "未提供独立的证据摘要"

    if answer_value == "证据不足":
        # 模型状态：证据不足。
        evidence_value = "现有检索证据不足以支持可靠结论"
        uncertainty_value = uncertainty_value or "证据不足，无法可靠回答"
    elif answer_value == "答案缺失":
        # 模型状态：有引用但没有答案。
        uncertainty_value = uncertainty_value or "有引用但没有答案，需要人工审核"
    elif not cited_ids:
        # 模型状态：有答案但没有可验证引用。
        uncertainty_value = uncertainty_value or "模型未提供可验证的引用，结论需人工复核"
    else:
        # 模型状态：答案和引用均有效。
        uncertainty_value = uncertainty_value or "无"

    citation_text = "无"
    if cited_ids:
        # 将已验证的 chunk_id 拼接为最终引用文本。
        citation_text = ", ".join(cited_ids)

    formatted_text = (
        f"结论：{answer_value}\n"
        f"证据：{evidence_value}\n"
        f"引用来源：{citation_text}\n"
        f"不确定性：{uncertainty_value}"
    )
    # 最终返回一个字典，包含整理后的答案文本、引用的chunk_id列表、格式化后的文本，以及模型的原始输出文本。
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
    基于 query + 检索结果，生成最小 answer + citations 输出。
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

    tokenizer, model = load_answer_model() # 加载小模型和对应的tokenizer，准备进行文本生成任务。
    context = build_context(result_records) # 把检索结果拼成给模型的证据块，形成一个包含多个chunk_id和对应文本的上下文字符串，供模型参考生成答案。

    if tool_context:    # 如果工具那边看到了工具上下文, 那么就拼接到模型能看见的字符串最后.
        context += (
            "\n\n"
            "[Tool context]\n"
            f"{tool_context}"
        )

    messages = [    # 构建一个对话消息列表，包含系统提示和用户输入两部分
        {
            "role": "system",   # 系统提示: 负责规定规则，比如“只能根据证据回答”“必须输出两行”“证据不足时怎么写”
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
            "role": "user",  # 用户输入: 包含具体的证据内容和问题，要求模型只能使用这些证据来回答问题，并且只能引用已经出现在证据中的chunk_id。
            "content": (
                f"Evidence:\n{context}\n\n"
                f"Question:\n{query}\n\n"
                "Only use chunk_id values that already appear in the evidence. "
                "Copy each cited chunk_id exactly, without inventing or shortening it."
            )
        },
    ]

    text = tokenizer.apply_chat_template(  # 把构建好的对话消息列表转换成一个完整的输入文本字符串，按照模型预期的格式组织系统提示和用户输入，准备进行编码和生成。
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)  # 把构建好的对话消息列表转换成模型输入的token序列，返回一个包含input_ids等信息的字典，并把它移动到模型所在的设备上（如GPU），准备进行文本生成。

    outputs = model.generate(   # 基于输入的文本，使用模型生成输出，设置最大生成长度为80个token，不使用采样（即贪心解码），温度和top_p参数不启用，以确保生成结果的确定性。
        **inputs,
        max_new_tokens=80,
        do_sample=False,
        temperature=None,
        top_p=None,
    )

    output_text = tokenizer.decode(  # 把模型生成的token序列解码成文本，跳过特殊token，并去掉前后的空白字符，得到模型的原始输出文本。
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    ).strip()
    normalized_output = normalize_answer_output(output_text, result_records)  # 把模型的原始输出文本整理成稳定的答案文本和引用chunk_id列表，确保即使模型没有明确给出引用，也能根据答案文本在证据中的命中情况做最小回填，提供一个合理的答案和相关的证据引用。

    return {
        "answer_text": normalized_output["formatted_text"],
        "answer_raw": normalized_output["raw_output"],
        "evidence_text": normalized_output["evidence_text"],
        "citation_ids": normalized_output["citation_ids"],
        "uncertainty": normalized_output["uncertainty"],
        "citation_validation": normalized_output["citation_validation"],
        "context": context,
    }
