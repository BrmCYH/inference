import gradio as gr  # 导入gradio库来创建图形化的聊天界面
import torch  # 导入PyTorch库
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from threading import Thread  # 用于创建后台线程来生成模型的输出
 
 
# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained("/home/RE/inference/Qwen2-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("/home/RE/inference/Qwen2-0.5B-Instruct", torch_dtype=torch.float16)
model = model.to('cuda:0')  # 把模型移动到GPU上以加速计算
 
 
class StopOnTokens(StoppingCriteria):  # 定义一个停止准则，按照特定的词或标志来停止生成
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [151645]  # 设置停止ID
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False
 
 
def predict(message, history):  # 定义预测函数，接受消息和历史记录
    history_transformer_format = history + [[message, ""]]
    stop = StopOnTokens()
 
 
    # 组装历史消息和当前消息
    messages = "".join(["".join(["\n<human>:"+item[0], "\n<bot>:"+item[1]])
                for item in history_transformer_format])
 
 
    # 对消息进行编码，并移动到相应的设备上
    model_inputs = tokenizer([messages], return_tensors="pt").to("cuda")
    # 设置文本生成参数
    streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=1024,
        do_sample=True,
        top_p=0.95,
        top_k=1000,
        temperature=1.0,
        num_beams=1,
        stopping_criteria=StoppingCriteriaList([stop])
        )
    t = Thread(target=model.generate, kwargs=generate_kwargs)  # 在后台线程中生成回复
    t.start()
 
 
    partial_message = ""
    for new_token in streamer:  # 从生成器中逐个获取新的标记
        if new_token != '<':  # 如果新标记不是特定的字符
            partial_message += new_token  # 添加到部分消息中
            yield partial_message  # 实时更新生成的消息
 
 
gr.ChatInterface(predict).launch()  # 启动Gradio界面