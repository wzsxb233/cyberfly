# MiniCPM 与全脑的内部数值连接

服务为 `http://127.0.0.1:18647`，由隔离 CUDA Python 启动：

```
/root/.cache/cyberfly/minicpm-train-env/bin/python -m shared_io.model_link
```

仅 GPU3；和 `model_training.cli` / `model_training.infer` 共用文件锁，拒绝重复加载。GPU2 的原生语音/聊天服务保持独立。冷启动重核官方 HF LFS 文件 SHA256 并量化加载，因此需要等待 `/health` 就绪。

`POST /encode` 与 `POST /decode` 请求：

```json
{"event_id":"unique-event", "text":"看看画面", "image_path":"/absolute/workspace/input.png", "audio_path":"/absolute/workspace/input.wav", "brain_features":[240个有限数字]}
```

图片、声音可省略。声音要求 16kHz 单声道 PCM16 WAV，0.1–30 秒；文件必须在 workspace 内。原始文字、图片、音频由同一官方 MiniCPM 处理器、视觉编码器、音频编码器直接转为语言输入 embeddings，没有先转成 JSON 规划或外部视觉文字摘要。

`encode` 回 `hidden[4096]`、`currents_mv[47]`。前者为真实原语言模块最后层归一化后的最后输入 token 隐藏向量；后者为可训练 LayerNorm/Linear 接口后的 `30*tanh(logits)`。当前脑向量也通过 `240→128→4×4096` 可训练投影输入语言模型，soft token 每元素显式限幅 ±0.15。这里的 mV 是后端工程电流尺度，不声称与语言语义有天然生理映射。

`decode` 必须沿用已经 encode 的 event_id、文字和媒体内容；脑向量可以换为真实全脑执行后的 240 维反馈。服务复用同一原始多模态 embeddings，将新脑向量真正拼入 `inputs_embeds` 后生成自然语言。可附 `facts` 对象描述实际神经计数、motor、学习变化。facts 只辅助忠实解释，且在实际脑/零脑消融两支完全相同；它不替代数值融合。

输出共同包含：

- `event_id`、`model_revision`、`weights_manifest_sha256`、`codec_sha256`。
- `coupler_sha256`、`coupler_generation`、`coupler_training_steps`。
- `modalities_encoded`、`sensory_sha256`、实际图像/音频 token 计数。
- `brain_soft_shape`、`brain_soft_sha256`。
- decode 另有 `say`、`brain_ablation.logit_delta_l2`、`logit_delta_max_abs`；它们比较同一输入与同一事实文本，仅将 240 个脑特征置零。

每个 event 检查耦合器版本一致性。缓存最多 8 个事件。模型无资格声称读取主观想法：随机或微量训练的投影可以产生数值影响，但数值影响与理解脑状态不是同一结论。

`POST /save {}` 原子保存当前耦合器。单个 safetensors 同时携带参数与模型/codec/代次/内容哈希；重载先校验身份、有限值、精确参数形状及内容，再修改模型。旁边 JSON 仅为方便人工阅读。

`POST /train_coupler`：

```json
{"steps":2,"output":"/absolute/workspace/artifacts/new-run","records":[{"event_id":"real-record-1","text":"描述活动","brain_features":[240个数],"target_text":"与真实记录对应的监督文本","target_currents_mv":[47个数],"source_path":"真实记录路径"}]}
```

每条记录可同时含图片/音频；应附可审查的真实记录来源。底层服务保存来源字段，但不自行认证来源真实性；工作台从已完成的真实 event 构造并验证训练记录。模型冻结，语言监督损失对脑→soft token 投影实际反传；电流模仿损失更新隐藏态→电流投影。最多 2048 输入+目标 token。记录到 `progress.json` / `events.jsonl`，最终 `training-evidence.json` / `coupler.safetensors`。`output/STOP`、`output/stop.request` 或 `output.parent/STOP` 在优化步边界合作停止并保存，状态 `interrupted`。训练期间服务串行阻塞其他请求；训练更新会导致仍未完成的旧 event 版本检查失败。

这套训练的目标文本与电流是监督标签，不能因接口运行成功就认定标签是真实思想或策略已优秀。实际小步验证文件另见 `artifacts/lab-integration/shared-hidden-01`；以其中每项实际状态为准。

## 本机实际验收

`artifacts/lab-integration/shared-hidden-01/summary.json` 已通过：真实原始文字、图像和音频编码；4096隐藏向量、47电流、240脑特征与4个4096软向量；真实两步双向投影梯度、保存重载与零脑数值消融。训练消融前后使用同一真实记录，初始图文回答正确识别红色眼睛和绿色小球。

原保存尾部发生过NTFS mmap重命名问题，`brain-projector-smoke-01` 原failed状态保留，`training-recovery.json` 与旁边 `recovery-evidence.json` 记录从运行中保住真实两步参数、Linux缓存保存及新进程原样重载。修复后改为从bytes检查文件，实际NTFS保存及0步STOP检查通过，见 `stop-save-check.json`。重启训练步数仍为2，不重复优化。

`language-adapter-reload.json` 还验证两份真实SFT/DPO LoRA加载并生成，随后卸载并逐值比较原基座hidden一致；没有把它们自动部署到正常共享模型。

这里只使用已保存的真实脑活动记录进行模型数值验收；现场完整brain/body/TTS闭环由主工作台另存event证据。短小训练不证明脑语义理解、思想读取或控制能力。

独立模态对照由另一代理完成：`artifacts/shared_io_modal_ab/980268a5fd7d/evidence.json`。固定文字、240维脑特征与耦合器（generation3），只换图像内容，隐藏向量L2差57.4341；只换两段等长真实原生语音（均3秒、30 audio tokens），L2差64.3413；完全重复原图输入的hidden逐值相同，L2差0。这将图像/音频本身的影响与文字、训练更新、随机性区分开。

## 逐神经元模式（neuron_direct）

请求仍用 `/encode`、`/decode`，新增：

```json
{"connection_mode":"neuron_direct", "io_scope":"sensory_motor", "event_id":"unique", "text":"看看画面", "image_path":"/workspace/input.png", "brain_state":{"schema":1,"path":"/workspace/artifacts/state.npz","sha256":"文件SHA256","ids_sha256":"6b6b40c3bddf84b1281ef0b18db06927c2bf61c5f4cb32a4219ec9b7839dc2e5","neurons":166700,"duration_ms":28.6,"simulation_ms":28.6}}
```

`brain_state` 为真实 `brain.read_state(directory)` 返回的描述符，允许它的额外元数据。NPZ 必须保留原始固定顺序 `ids:int64[166700]`、`v:float32[166700]`、`counts:int32[166700]`、`macro_group_index:int32[166700]` 与身份 metadata；模型会核对文件、ID顺序、注释成员、实际测量区间。不要附带巨大CSR图文件。

`io_scope` 有两个独立 checkpoint：

- 默认 `sensory_motor`：只向确认的 `cb_sensory/ol_sensory/vnc_sensory` 共17,336个ID注入，其他电流严格为规范正零；模型只回读确认的 `cb_motor/vnc_motor/descending_neuron` 共2,129个ID。未确认的 tbc 类排除。完整166,700个状态仍保留在原NPZ和可视化中。
- `all_neurons`：全部166,700个ID可写、可回读，用于探索。

真实4096隐藏向量经过可训练rank16低秩投影，输出层为每个ID提供独立的权重行；没有将47个群的数值复制给神经元。反馈则以每个ID的原电压与该区间放电频率形成333,400维数值，经过独立权重列的低秩编码与4个4096软向量进入同一个MiniCPM。每ID系数独立，但每步模型输出仍受rank16瓶颈约束，不能将其称作166,700个自由潜变量。每个scope转换器约34MB，基础大模型不重复加载。

原始电压归一化为 `(v+65)/30`，频率为 `log1p(counts/interval_seconds)/log1p(200)`，先连续排列全部电压，再排列全部频率。默认scope仅在这两个完整数组上施加逐ID读掩码，不计算群均值。固定注释只用来选择通道。

`encode` 回 `currents_file`：由 `connectome_adapter.neuron_currents.write_neuron_currents` 写出的不可变NPZ，包含 `currents_mv:float32[166700]` 和 `ids:int64[166700]`；描述符包含 `path/sha256/ids_sha256/currents_sha256/neurons/schema`。脑后端直接接：

```json
{"general_stimulation":[{"neuron_currents_file":{"...":"currents_file原样描述符"}}],"visual_input_enabled":true}
```

上例只显示新增分量。工作台实际通过 `parallel_inputs.prepare_parallel_stimulation` 保留全部原有视觉、感觉端口和已配置刺激，将预算内的新增模型文件追加到原刺激列表。原有输入不改；超出人工附加电流预算时只限制新增分量，并记录请求值、实际值与总和摘要。`visual_input_enabled=false` 仅用于主动隔离实验；正常共享连接沿用场景原开关。固有tonic背景、负反馈和连接组动力学照常工作。

`/train_coupler` 的native请求增加 `connection_mode/io_scope`，记录使用 `brain_state` 与 `target_currents_file`，目标文本仍为 `target_text`。语言监督损失经过冻结原LLM反传到逐ID脑编码器，电流监督更新逐ID解码器。外部脉冲仿真、突触可塑性不在这条autograd图中，不能称全脑端到端可微训练。

`POST /probe_neuron` 接同一原请求并附 `neuron_id` 整数、`delta_mv`（±30内），对一个原始电压特征做明确标注的数值反事实扰动，比较soft tokens、真实hidden、logits、电流变化；不改真实快照或仿真。域外ID应得到零影响。

`/save {}` 兼容原管理器：顶层仍有 `reload_verified`，同时在 `saved_connection_modes` 返回分组模式及所有已经初始化的native scopes。scope分别存到 `shared_io/checkpoints/neuron_direct/<scope>/current.safetensors`；身份和事件缓存包含scope，不能跨scope或跨权重版本拼接一次闭环。

逐ID本机实际验证已通过，见 `artifacts/neuron-link-validation/live-ef1fd1b784ac/summary.json`：

- 默认感觉模式17,336个端口产生17,324个不同float32值，域外全部规范正零；初始值范围−4.83至17.46mV，均值7.275mV。浮点值偶然相等不代表共享权重。
- 全神经元模式166,700个独立输出位置产生166,435个不同值，没有按群复制。
- 只对一个实际ID的电压特征作+30mV数值反事实扰动，soft token L2差0.09135、真实hidden差2.6747、logit差59.2132。该测试明确没有改写原始快照，也没有冒称真实神经元已经受到刺激。
- 8,504,172个转换器参数完成2步真实优化，两向梯度非零，参数L2变化0.171256；保存及重载一致，实际重载生成仍识别红眼。
- 三模式同时保存成功，原grouped耦合器SHA6536...保持不变。两个native scopes分别保存与训练，共用同一冻结原MiniCPM，不重复加载基座。

这些模型验证使用已归档真实脑数组，真实新电流→全连接组→身体的验收由工作台另存。模型数值反事实与实际脑仿真实验在证据中分开标注。

## 原模块特征与同次隐藏状态语音

新增 `encode.modal_embeddings_file`，文件为本地不可变 NPZ：`vision:float32[Tvision,4096]`、`audio:float32[Taudio,4096]`、`language:float32[1,4096]`。缺图或缺音时相应数组为 `0×4096`。vision/audio 直接从原始模型实际注入 `inputs_embeds` 的 image/audio bounds 取出，分别对应原 VPM+resampler 和原 APM+projection+pool；不是重新生成的文字、假造随机特征或工程 JSON。BF16 原值转 float32 保存，不增加原精度。language 与同次返回的实际 hidden 逐值一致。描述符包含 event_id、文件 SHA、模型 revision、核验后的 weights manifest SHA 及每数组形状/类型/数值 SHA。

`decode` 可加 `generate_audio:true`，返回 `audio_file`。这条路径在原脑状态/身体数值融合后的同一次原 LLM `generate` 中保留每个生成文字 token 的最后层隐藏状态，经原 `tts.projector_semantic`（4096→768）与原 text embedding 合并，进入原20层TTS，实际 audio codes 再经官方 StepAudio2 flow/HiFT 生成24kHz WAV。`native_tensors_file` 保存真实文字ID、同次 generation_hidden、projected_hidden、tts_inputs 和 audio_codes。没有将字符串重新交给另一次 LLM 或另一个语音会话。

这条新增路径为有界轮次语音，尚不等同原 GPU2 WS 的连续双工；它保留最强的同次隐藏状态来源。`generate_audio:false`（兼容默认）只返回文字。选择旧 GPU2 文字→语音链路时必须明确记录其是另一次语音会话。原生合成失败会报错，不静默替换为浏览器语音或文本朗读服务。`max_speech_tokens` 限50–1500，默认750；返回 `audio_generation_finished` 表明是否在原TTS上自然终止。

TTS的194个原参数张量来自已有4片官方checkpoint，加载时 `init_tts=True` 且不量化TTS。额外 token2wav 资产固定在同一HF revision（约1.23GB），`model_training/native-speech-manifest.json` 记录文件/LFS身份，首次初始化逐个验证；没有重复下载原18.7GB权重。必要依赖为原 `minicpmo-utils==1.0.6`、匹配现有 Torch 的 `torchaudio==2.7.1+cu126`、onnxruntime1.21、onnx1.18、hyperpyyaml1.2.2、ruamel.yaml0.18.6、einops0.8.1、librosa0.11、scipy1.15.3；现 Torch2.7.1/Transformers4.51.3 保持。Token2wav 使用其原默认参考音色，不把用户输入声音当作未经说明的声音克隆。

官方依据：[固定模型代码](https://huggingface.co/openbmb/MiniCPM-o-4_5/blob/503e754207c94da6bb26850b4469f367c9ea3582/modeling_minicpmo.py)、[固定音频资产](https://huggingface.co/openbmb/MiniCPM-o-4_5/tree/503e754207c94da6bb26850b4469f367c9ea3582/assets/token2wav)、[官方部署说明](https://github.com/OpenBMB/MiniCPM-V/blob/main/README.md)、[官方辅助包](https://pypi.org/project/minicpmo-utils/)。代码与下载准备不等于现场验证，实际运行以 `artifacts/native-speech-validation/*/summary.json` 状态为准。

## 身体真实数值软向量

可附 `body_features_file`，使用 `BodySensoryAdapter.export` 的真实物理包及其特征NPZ描述符；服务重新核对原packet、文件SHA、布局SHA、仿真时间以及特征逐值一致。独立可训练身体头按布局隔离存档，实际输出一个4096软token插入原LLM。它与视觉/听觉原始编码器并行。此身体感知文件参与同event输入指纹，身体头参数SHA也在encode/decode之间固定。

`decode.body_feedback_file` 可指定动作之后的新身体包，再注入一个4096软token。前后身体包在结果分别为 `body_input` / `body_feedback`；新的后验包不冒充原感知输入。零脑消融保留同一个身体输入、后验身体向量和事实文本。`POST /train_body_embedding` 是独立、显式监督目标的真实身体头拟合入口，`/save {}` 的 `saved_body_embeddings` 保存所有已初始化布局。此头的学习不代表原MiniCPM全模块或全脑突触都被训练。

音频资产可复现下载：`python -m model_training.download_native_speech`，复用已有原模型目录。依赖约束见 `model_training/native-speech-requirements.txt`；初次实际运行发现ruamel.yaml0.19.1与HyperPyYAML旧Loader不兼容，已固定0.18.6，初次failed记录保留。

原生同次隐藏状态语音已经真实验收：`artifacts/native-speech-validation/e6ac3ba31dd8/summary.json` 为verified。原模型回答“果蝇的眼睛是红色的。”；7个实际生成文字token的hidden[7,4096]→原projector[7,768]→54个原audio codes→2.16秒24kHz WAV，原TTS自然EOS结束。原始视觉64/音频30个token导出校验通过，真实758维身体前后反馈各注入一个4096软向量。这里使用已有真实记录作为模型侧验收fixture，未在该脚本内推进外部脑；主工作台另验同时发生的完整闭环。原逐ID转换器SHA保持4c8c62aa…，没有为这次语音测试再训练。

后续原生视频/双工接线的只读审查见 `shared_io/NATIVE_STREAMING_ROUTE.md`；尚未把当前轮次语音冒称该流式数值链路已完成。
