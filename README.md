# tileops-cpu:一个完整的 TileOPs 后端

本仓库是面向外部后端作者的参照实现:一个可安装、可运行、可测试的 Python 包,以纯 PyTorch 实现 kernel,接管 CPU 上的算子。除 kernel 本身不涉及专用硬件之外,其余各部分 —— entry point、注册、`build_kernel` 签名、记忆规则、错误信息 —— 与一个面向专用硬件的后端完全一致。

接入过程不需要修改 TileOPs 的任何代码。安装前后的差别:

```console
$ python -c "import torch; from tileops.ops.norm.rms_norm import RMSNormFwdOp; \
             RMSNormFwdOp(normalized_shape=(64,))(torch.randn(4,64,dtype=torch.float16), \
                                                  torch.randn(64,dtype=torch.float16))"
ValueError: RMSNormKernel is a CUDA kernel; got x on cpu and weight on cpu.

$ pip install -e .

$ python -c "...同一段代码..."
# 正常返回,结果与 torch.nn.functional.rms_norm 逐位相同
```

<details open>
<summary><b>目录</b></summary>

<ul>
<li><a href="#s1">1. 最小实现</a></li>
<li><a href="#s2">2. target 名与设备类型的区别</a></li>
<li><a href="#s3">3. 仓库结构</a></li>
<li><a href="#s4">4. build_kernel 的签名与参数</a></li>
<li><a href="#s5">5. op 层已经完成的工作</a></li>
<li><a href="#s6">6. 何时重新调用 build_kernel</a></li>
<li><a href="#s7">7. 安装后 op 的两种状态</a>
  <ul>
    <li><a href="#s7-1">7.1 target 定下来之前的硬件查询</a></li>
  </ul>
</li>
<li><a href="#s8">8. 错误信息对照</a></li>
<li><a href="#s9">9. 调用方 API</a></li>
<li><a href="#s10">10. 运行测试</a></li>
<li><a href="#s11">11. 各阶段的限制</a></li>
<li><a href="#s12">12. 不支持的情形</a></li>
<li><a href="#s13">13. 作为模板使用</a></li>
</ul>

</details>

## <a id="s1"></a>1. 最小实现

一个后端需要提供的内容只有两处。第一处是 `pyproject.toml` 里的三行:

```toml
[project.entry-points."tileops.backends"]
torch_cpu = "tileops_cpu"
```

第二处是模块顶层的注册:一次 `register_detector`,再为每个要接管的 op 各一次 `register_kernel_builder`。

```python
from tileops.backend import TensorSpec, register_detector, register_kernel_builder
from .gemm import build_gemm
from .kernels import CpuRMSNorm

register_detector(target="torch_cpu", detect=lambda device: device.type == "cpu")


def build_rms_norm(x: TensorSpec, weight: TensorSpec, *, normalized_shape, eps):
    return CpuRMSNorm(normalized_shape, eps, x.dtype)


register_kernel_builder(op="RMSNormFwdOp", target="torch_cpu", build_kernel=build_rms_norm)
register_kernel_builder(op="GemmFwdOp", target="torch_cpu", build_kernel=build_gemm)
```

`op=` 用的是 manifest 里的键,写错就永远不会被调到,而且不报错。

`pip install` 之后自动生效:TileOPs 在构造第一个 Op 时枚举这个 entry point 组、import 声明的模块,顶层这几次调用把注册表填好。没有其他初始化步骤,没有需要继承的基类,也没有需要实现的接口。

## <a id="s2"></a>2. target 名与设备类型的区别

上面那行注册里出现了两个名字,它们含义不同:

| 名字 | 含义 | 由谁决定 |
| --- | --- | --- |
| `target="torch_cpu"` | 这套 kernel 的名字 | 后端作者 |
| `device.type == "cpu"` | 认领哪类设备 | torch |

两者不是一一对应的:同一个 device type 可能对应不止一套 kernel,分属不同厂商;部分硬件经 `privateuseone` 接入,字符串中不含任何厂商信息;有的后端还需读环境变量或调用厂商 runtime 才能确定。因此 TileOPs 不解析 `torch.device`,而是原样传给 `detect`,由后端回答。

`detect` 只回答设备归属。**某次调用是否受支持 —— dtype、形状、参数组合 —— 由 `build_kernel` 回答**,只有它能看到这些信息。对不认领的设备返回 `False`,不要抛异常。

## <a id="s3"></a>3. 仓库结构

| 文件 | 内容 |
| --- | --- |
| `pyproject.toml` | entry point 声明,即全部安装机制 |
| `src/tileops_cpu/__init__.py` | 全部注册代码 |
| `src/tileops_cpu/kernels.py` | RMS norm 的 kernel 实现。真实后端在此编译 |
| `src/tileops_cpu/gemm.py` | 第二个 builder:CPU GEMM,注册给 `GemmFwdOp` |
| `tests/test_takeover.py` | 数值、校验、归一、输出 |
| `tests/test_discovery.py` | entry point 与注册 |
| `tests/test_errors.py` | 三条错误路径 |
| `tests/test_memoization.py` | 何时重新调用 `build_kernel` |

## <a id="s4"></a>4. `build_kernel` 的签名与参数

**签名就是该 op 的 manifest 签名。** 编写 kernel 只需阅读 manifest,不需要阅读 TileOPs 源码。

`src/tileops/manifest/normalization.yaml` 中的 `RMSNormFwdOp`:

```yaml
signature:
  inputs:                       # 声明顺序即传入顺序
    x: {dtype: "float16 | bfloat16"}
    weight: {dtype: "same_as(x)"}
  params:                       # 按这些名字作为关键字参数传入
    normalized_shape: {type: "list[int] | tuple[int, ...]"}
    eps: {type: "float | None", default: null}
```

对应的 builder:

```python
def build_rms_norm(x: TensorSpec, weight: TensorSpec, *, normalized_shape, eps):
```

两点需要注意:

- **`eps` 收到的是 `1e-6`,不是 `None`。** manifest 中的默认值是 null,但 op 层已将其规范化为确定的数值。所有可选参数都是如此。
- **`TensorSpec` 是描述而非张量**,只有 `device` / `dtype` / `shape`,既没有数据,也没有对张量的引用。这样两类错误就无法写出来:一是根据数据内容决定构造哪个 kernel(而记忆表按形状索引,后续会取到错误的 kernel),二是让某个张量随被缓存的 kernel 存活整个进程。

对返回值只有一条要求:**可调用**。调用时按同样顺序收到真实张量,返回值按 `signature.outputs` —— 单输出返回张量,多输出按顺序返回 tuple,纯原地写返回 `None`。

**构造签名只接收编译期参数。** 会被编译进生成代码的值(tile 尺寸、当作常量的维度、dtype)放进构造函数,其余留给 `__call__`。这一条对 decode 是硬性要求:`seq_len` 逐步递增,batch 随 running set 变化,它们进入构造函数就意味着每步重新编译。`kernels.py` 中的 `CpuRMSNorm` 在构造时**拿不到行数**,原因即在于此。

## <a id="s5"></a>5. op 层已经完成的工作

以下工作对所有 target 相同,后端**不要重复实现**:

- manifest 的 dtype 校验与形状规则
- 参数规范化(可选值落实为确定值)
- 输入的连续性归一 —— 传入 kernel 的都是连续张量
- kernel 的记忆与重用
- roofline、profile、数值测试

**op 层不在交给 kernel 之前改变形状。** kernel 收到的是 manifest 声明的形状;需要何种 layout 由 kernel 自行处理,在它自己的调用包装里完成。

**代码与 manifest 都描述了的事情,以 manifest 为准。** 输出 dtype、形状规则、参数类型均由 manifest 规定,kernel 不得改写;能力不足时报错。错误信息必须指明哪一项不满足(dtype / 形状 / arch / 无可用实现 / 编译失败)以及实际收到的值 —— 仅写「不支持」不构成有效诊断。

## <a id="s6"></a>6. 何时重新调用 `build_kernel`

TileOPs 按**设备加输入签名**记住 `build_kernel` 的返回值:

> 本次调用张量所在的设备,加上按 `signature.inputs` 顺序逐个取出的 `(dtype, shape)`。

也就是说:**设备与输入签名都相同的两次调用,TileOPs 会交给后端同一个 kernel。** 设备进 key,是因为为一块卡编译出的产物可能持有那块卡上的资源;同一个 target 的第二块卡会重新问一次 builder。`params` 不进入 key,它们对一个 op 实例是固定的。

需要更细的区分,在后端内部处理;需要更粗的粒度以减少重建,在 `build_kernel` 内部另加缓存。两个方向都在后端一侧解决。

`tests/test_memoization.py` 逐项验证了这条规则,包括 key 的实际形态:

```python
device, *inputs = key
assert device.type == "cpu"
assert inputs == [(torch.float16, (4, 64)), (torch.float16, (64,))]  # x 在前,weight 在后
```

## <a id="s7"></a>7. 安装后 op 的两种状态

一旦 `detect` 认领了某类设备,该设备上的**所有** op 都由这个 target 服务;缺少任何一个都会报错,**不会落回仓内实现**。原因很直接:选中一个 target 意味着该设备属于另一套硬件,TileOPs 自带的 kernel 在其上无法启动,落回只会把一个清楚的「该 target 未实现此 op」换成一个难以理解的启动失败。

因此每个 op 只有两种状态:

| 状态 | 结果 |
| --- | --- |
| 这个 target 为该 op 注册了 builder | 正常执行 |
| 没有注册 | 报错,指出该 target 未为这个 op 注册 builder |

覆盖目标模型用到的每一个 op,因此是后端一侧的工作。op 那一侧的前提已由设计保证:取 kernel 时把即将传给 kernel 的张量一并交出,外部路径才算得出记忆 key。

```python
# TileOPs 内部,op 自身的 forward
self.get_or_build_kernel("gemm_kernel", (a, b), key=..., build=...)
#                                       ^^^^^^ 这一项
```

**注册名必须与 manifest 的键逐字符相同。** 本仓库注册的是 `RMSNormFwdOp` 与 `GemmFwdOp`;写成 `GemmOp` 这类不存在的键,builder 永远不会被调用,而且没有任何报错 —— op 层查 `(op, target)` 查不到,就当这个 target 没有为该 op 注册。

### <a id="s7-1"></a>7.1 target 定下来之前的硬件查询

按设计,op 层在 target 定下来之前不查询与特定硬件绑定的信息;查了就意味着在没有该驱动的机器上,调用会在到达 `build_kernel` 之前失败,而原因与这个后端无关。

GEMM 目前还有一处:`GemmFwdOp` 构造 `GemmCall` 时未指明 `arch`,于是 `CallSpec.__post_init__` 去读 SM 版本:

```
tileops/kernels/call_spec.py  CallSpec.__post_init__
tileops/utils/utils.py        get_sm_version  ->  torch.cuda.current_device()
```

所以在无 CUDA 驱动的机器上,由这个 CPU 后端服务的 GEMM 也跑不起来。撞到这类失败时,调用栈会停在 TileOPs 内部而不是后端的 `build_kernel` 里 —— 提 issue 并附上调用栈,需要修改的是 TileOPs。

本仓库因此为两个测试加了 `requires_cuda_runtime` 标记,在无 GPU 的机器上自动跳过:一个是上面这条路径,另一个是 `target=BUILTIN`(它要的就是仓内的 CUDA kernel)。

## <a id="s8"></a>8. 错误信息对照

以下均为实测输出。

**某个 op 的取 kernel 处没有交出张量:**

```
OpNotAvailableError: target 'torch_cpu' serves GemmFwdOp, but its 'gemm_kernel' call site
does not hand over the tensors a builder is described with; that op is not wired to
external targets yet
```

TileOPs 的 op 都按契约交出张量,所以正常情况下见不到这条。真见到了,说明 op 那一侧出现了回退,不是后端的问题:提 issue 并附上 op 名。

**未为该 op 注册 builder:**

```
OpNotAvailableError: target 'torch_cpu' registers no kernel builder for SoftmaxFwdOp;
targets that do: []. There is no fall back to the in-tree implementation: those kernels
do not run on this target's devices.
```

为该 op 编写并注册一个 builder。

**指定了未注册的 target:**

```
UnknownTargetError: no backend registered target 'nope'; known targets: ['torch_cpu']
```

包未安装成功,或 target 名拼写错误。用 `tileops.backend.registered_targets()` 查看实际注册的内容。

**用 `target=BUILTIN` 强制使用仓内实现:**

```
ValueError: RMSNormKernel is a CUDA kernel; got x on cpu and weight on cpu.
Another target's backend serves other devices.
```

`BUILTIN` 显式绕过后端。CPU 张量上仓内实现无法运行,这正好说明了「不落回」这条规则要避免的是什么。

**后端包 import 失败:** TileOPs 会跳过它并发出一条警告,把原因收进 `load_failures()`。单个损坏的插件不会导致 TileOPs 无法导入。若注册过程中途抛出异常,该后端本次注册的内容会**全部回滚**,注册表中不会留下一个只实现了一半的 target。

```python
from tileops.backend import load_failures
print(load_failures())
```

## <a id="s9"></a>9. 调用方 API

后端作者不需要调用这些,但调试时有用:

```python
from tileops.backend import (
    BUILTIN, registered_targets, set_default_target, default_target, load_failures,
)

registered_targets()                 # ['torch_cpu']
registered_targets("RMSNormFwdOp")   # ['torch_cpu']
set_default_target("torch_cpu")      # 进程默认,优先于设备探测
set_default_target(BUILTIN)          # 全局关闭替换
```

target 的选取顺序:构造参数 `target=` → 进程默认 → 设备探测。不存在「默认 target」这一概念,默认状态是不替换。

## <a id="s10"></a>10. 运行测试

需要一个已安装 `tileops` 的环境。

```bash
pip install -e .          # tileops 已安装时加 --no-deps
python -m pytest -q       # 无 GPU 时会跳过两条需要 CUDA 驱动在场的用例,见 7.1
```

在 TileOPs 的 dev 镜像中运行,同样不需要修改 TileOPs:

```bash
docker run --rm --gpus all -v "$PWD/..":/work -w /work \
  ghcr.io/tile-ai/tileops-runner:cu132-torch2.13-tl-afcebed1-dev \
  bash -lc 'pip install -e /work/TileOPs --no-deps -q &&
            pip install -e /work/tileops-backend-example --no-deps -q &&
            cd /work/tileops-backend-example && python -m pytest -q'
```

`tileops` 刻意不在本包的依赖列表中:本包扩展的是一个已经存在的安装,而在依赖中写版本下限会解析到早于 `tileops.backend` 的发行版,由此产生的 ImportError 会被收进 `load_failures()`,读起来像是「这个后端坏了」,而非「TileOPs 版本过旧」。

## <a id="s11"></a>11. 各阶段的限制

decode 路径会被 CUDA graph 捕获,各阶段允许做的事因此分别规定:

| 阶段 | 可以 | 不可以 |
| --- | --- | --- |
| 查记忆表 | dict 查找 | 其余一切 |
| `detect` | 一次谓词判断 | 任何 import、任何加锁 |
| 构造 kernel | 选实现、编译、分配、重 import、建 handle | 依赖真实张量的调优 |
| kernel 调用 | 启动已编译的 kernel;经 torch allocator 分配输出 | 编译、惰性初始化、建 handle、host 同步 |

**模块顶层 import 不得触发编译。** TileOPs 在构造第一个 Op 时 import 后端模块,编译应发生在 `build_kernel` 被调用时。

kernel 调用还须满足两条与流相关的规则:

- **在当前流上启动**(CUDA 下即 `torch.cuda.current_stream(device)`),不得落到默认流。自带 launcher 的后端尤其容易违反这一条。
- **内部分配的生命周期必须跨越异步执行。** 只把裸指针交给 launch 时,对象必须存活到该流执行完成。协议不提供 workspace,这份安全由后端负责。

调用方须在捕获前完成预热(至少一次同形状的非捕获调用),因为构造 kernel 允许编译。捕获期间只允许走「查表命中、直接调用」这条路。

## <a id="s12"></a>12. 不支持的情形

| 不支持 | 理由 |
| --- | --- |
| 同一个 target 上有多个后端 | 一个 target 对应一套 kernel、一个提供者。重复注册同一个 `(op, target)` 直接报错,那意味着安装了两个都自称是它的包 |
| 整体替换一个组合 op | 组合 op 的计算在它构造的 sub-op 中,替换发生在那一层 |
| 后端改变输入形状,或代替调用方还原输出 | 那是 op 层对所有 target 提供的服务;要改就对所有 target 一起改 |
| 一次调用跨多个设备 | CPU 标量走 params 而非张量输入,所有输入必须在同一设备 |
| 调用方提供 workspace 或显式 stream | 后端需要的只是当前流,而 torch 的流是隐式当前值 |
| autograd 联动 | 这条链服务推理。fwd / bwd 各是独立的 op |
| 换用另一个 target | 指定的 target 没有实现就报错,不会改用别的 target 执行 |

## <a id="s13"></a>13. 作为模板使用

1. 复制本仓库,把 `tileops_cpu` 改为 `tileops_<硬件名>`,target 名同理
2. 修改 `_detect`,认领对应的设备类型
3. 把 `kernels.py` 与 `gemm.py` 里的 kernel 替换为真实实现 —— 构造时编译,`__call__` 时启动
4. 选定第一个要接管的 op,照它的 manifest 签名编写 `build_kernel`,注册时用 manifest 的键
5. `tests/` 中的四个文件基本可以直接沿用,替换 op 名与 target 名即可
6. 之后逐个 op 增加 `build_kernel`。**目标模型用到的 op 需要全部覆盖**,缺少任何一个都会报错
