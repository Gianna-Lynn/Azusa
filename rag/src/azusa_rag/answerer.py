from transformers import AutoTokenizer, AutoModelForCausalLM
# AutoTokenizer : 把输入文本转成模型能理解的token序列, 也负责把模型输出的token序列转回文本。
# AutoModelForCausalLM : 加载一个"因果语言模型"（Causal Language Model），适合生成文本。
MODEL_PATH = r"D:\Chuo_Data\Models\Qwen2.5-0.5B-Instruct"

_tokenizer = None
_model = None


def load_answer_model():
    """
    延迟加载 answer 用的小模型，避免每次 import 都立刻吃显存。
    """
    # 延迟加载就是在函数内部定义一个全局变量，第一次调用函数时才加载模型并赋值给这个变量。
    # 后续调用函数时，如果变量已经有值了，就直接返回，不需要重复加载模型。
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


def select_supporting_chunk_ids(answer_text: str, result_records) -> list[str]:
    """
    当模型没有稳定给出 citation 时，按答案文本在证据中的命中情况做最小回填。
    """
    cleaned_answer = answer_text.strip() # 去掉答案文本前后的空白字符，得到一个干净的答案字符串。
    if not cleaned_answer or cleaned_answer == "证据不足":
        return []

    supporting_ids = []
    for record in result_records:
        if cleaned_answer in record.text:
            supporting_ids.append(record.chunk_id) # 如果答案文本在某个证据块的文本中出现了，就把这个证据块的chunk_id添加到 supporting_ids 列表中。

    if supporting_ids:
        return supporting_ids

    return [result_records[0].chunk_id] # 如果答案文本在任何证据块中都没有直接命中，就默认引用第一个证据块，确保至少有一个引用。这个策略虽然简单，但可以避免完全没有引用的情况，提供一些线索给用户去查看相关证据。


def normalize_answer_output(output_text: str, result_records) -> dict:
    """
    把小模型的自由输出整理成稳定的 answer + citation 结构。
    """
    output_text = output_text.strip()
    valid_chunk_ids = [record.chunk_id for record in result_records]

    lines = []
    for line in output_text.splitlines(): # 把模型输出的文本按行分割成一个列表
        line = line.strip()
        if line:
            lines.append(line) # 去掉每行前后的空白字符，如果这一行不空，就把它添加到 lines 列表中。

    answer_value = ""
    for line in lines:
        if line.startswith("答案："):
            answer_value = line.removeprefix("答案：").strip() # 如果这一行以 "答案：" 开头，就把 "答案：" 这个前缀去掉，剩下的部分就是答案文本，再去掉前后的空白字符，得到最终的答案值。
            break

    if not answer_value:
        answer_value = lines[0] if lines else "证据不足" # 如果没有找到以 "答案：" 开头的行，就默认把第一行作为答案值，如果没有任何行，就把答案值设为 "证据不足"。

    cited_ids = []
    for chunk_id in valid_chunk_ids:
        if chunk_id in output_text:
            cited_ids.append(chunk_id)

    if answer_value == "证据不足": # 如果答案是证据不足，就不应该有任何引用，直接清空 cited_ids。
        cited_ids = []
    elif not cited_ids: # 如果模型没有明确给出引用的chunk_id，就根据答案文本在证据中的命中情况做最小回填，确保至少有一个相关的引用。
        cited_ids = select_supporting_chunk_ids(answer_value, result_records) 

    citation_text = "无"
    if cited_ids:       # 如果有引用的chunk_id，就把它们用逗号连接成一个字符串，作为引用文本。
        citation_text = ", ".join(cited_ids)

    formatted_text = f"答案：{answer_value}\n引用：{citation_text}"
    # 最终返回一个字典，包含整理后的答案文本、引用的chunk_id列表、格式化后的文本，以及模型的原始输出文本。
    return {
        "answer_text": answer_value,
        "citation_ids": cited_ids,
        "formatted_text": formatted_text,
        "raw_output": output_text,
    }


def answer_with_citations(query: str, result_records) -> dict:
    """
    基于 query + 检索结果，生成最小 answer + citations 输出。
    """
    tokenizer, model = load_answer_model() # 加载小模型和对应的tokenizer，准备进行文本生成任务。
    context = build_context(result_records) # 把检索结果拼成给模型的证据块，形成一个包含多个chunk_id和对应文本的上下文字符串，供模型参考生成答案。

    messages = [    # 构建一个对话消息列表，包含系统提示和用户输入两部分
        {
            "role": "system",   # 系统提示: 负责规定规则，比如“只能根据证据回答”“必须输出两行”“证据不足时怎么写”
            "content": (
                "You must answer only from the provided evidence. "
                "Do not use outside knowledge. "
                "Output exactly two lines and nothing else.\n"
                "Line 1: 答案：<answer>\n"
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
                "Only use chunk_id values that already appear in the evidence."
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
        "citation_ids": normalized_output["citation_ids"],
        "context": context,
    }
