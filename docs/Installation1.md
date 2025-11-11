# **以下在autodl中进行配置**
你要现在看到这份教程先不要急着去尝试，我还没更新完，稍等两天

## AlphaFold 3 手动部署教程

如果你身处一个无法使用 Docker 的环境（如 AutoDL 容器实例），并且希望使用自定义 MSA 来运行 AlphaFold 3，那么希望这份教程适合你

---

### **第一部分：创建一个环境**


1.  **创建 Conda 环境 (必须是 Python 3.11！)**
    存在版本问题。项目的 `pyproject.toml` 文件明确要求 `python>=3.11`。

    ```bash
    # 退出可能存在的旧环境
    conda deactivate

    # 创建一个名叫 af3py311 的新环境，强制指定 Python=3.11
    conda create --name af3py311 python=3.11 -y

    # 激活这个全新的、纯净的环境
    conda activate af3py311
    ```
    **验证**：你的命令行提示符左边，必须是 `(af3py311)`。

---

### **第二部分：安装所有依赖**

这部分比较难搞，为了避免反复失败，请严格按照以下顺序慢慢搞，当然你要在国外网络通畅可以直接“pip install -r requirements.txt”。
如果不在autodl上，有自己机子的，建议你看alphafold官方提供的安装指南 https://github.com/google-deepmind/alphafold3/blob/main/docs/installation.md

1.  **安装“常规部队” (非 JAX 和非编译依赖)**
    我们先把简单的依赖用国内镜像源装完。

    ```bash
    # 进入你下载的 alphafold3 源代码目录
    cd /path/to/your/alphafold3  # <-- 替换成你自己的路径

    # 筛选掉所有和 jax 相关的包，创建临时依赖文件。。。稍等我会提供我的requirement.txt
    grep -iv 'jax' requirements.txt > temp_requirements.txt

    # 使用清华源高速安装
    pip install -r temp_requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

    # 删除临时文件
    rm temp_requirements.txt
    ```

2.  **安装(JAX 全家桶)**
    这是最关键的一步，我找到了绕过网络问题的最佳组合。

    ```bash
    # 升级 pip
    pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

    # 挨个安装 JAX 家族成员，版本必须精确！
    pip install jax==0.4.34 -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install jax-triton==0.2.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install jaxtyping==0.2.34 -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install jaxlib==0.4.34 -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install jax-cuda12-plugin==0.4.34 -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```

3.  **装备(C++ 编译工具)**
    为了防止编译脚本自己上网下载，我们必须提前把所有编译工具都装好。

    ```bash
    pip install scikit-build-core pybind11 cmake ninja numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```

---

### **第三部分：最终组装 —— 安装 AlphaFold 3 自身**

现在万事俱备，我们来完成最后的组装。

`CMakeLists.txt` 文件里，有一个“采购清单”，上面列了**五样**它需要从 GitHub 上获取的东东。

1.  `abseil-cpp` (Google 的 C++ 基础库)
2.  `pybind11` (Python/C++ 翻译器)
3.  `pybind11_abseil` (上面两个的连接件)
4.  `cifpp` (处理 mmCIF 文件的库)
5.  `dssp` (计算蛋白质二级结构的库)

只要有一个零件因为网络问题下载失败，后面有个步骤就会很频频报错。

**所以建议你把这些全部下好**

#### **第一步：创建“工厂指定仓库”**

CMake 在构建时，会把下载的依赖项放在一个叫 `_deps` 的文件夹里。我们手动创建它。

```bash
# (确保你还在 (af3py311) 环境和 alphafold3 目录下)
# 我们在 alphafold3 目录下创建一个临时构建目录，并在其中创建 _deps 仓库
mkdir -p build/_deps
cd build/_deps
```
**现在，你所在的路径应该是 `~/autodl-tmp/alphafold3/build/_deps`。** 这是我们的“仓库重地”。

#### **第二步：手动克隆所有五个仓库**

我们要严格按照 `CMakeLists.txt` 里的**地址**和**版本号（GIT_TAG）**，把这五个项目手动克隆下来。

**请把下面这一整块代码，一次性复制粘贴到你的终端里执行。** 它会自动完成所有下载和版本切换。

```bash
# (确保你现在在 _deps 仓库目录里)

# 1. 进货 abseil-cpp
git clone https://github.com/abseil/abseil-cpp.git abseil-cpp-src
cd abseil-cpp-src
git checkout d7aaad83b488fd62bd51c81ecf16cd938532cc0a
cd ..

# 2. 进货 pybind11
git clone https://github.com/pybind/pybind11.git pybind11-src
cd pybind11-src
git checkout 2e0815278cb899b20870a67ca8205996ef47e70f
cd ..

# 3. 进货 pybind11_abseil
git clone https://github.com/pybind/pybind11_abseil.git pybind11_abseil-src
cd pybind11_abseil-src
git checkout bddf30141f9fec8e577f515313caec45f559d319
cd ..

# 4. 进货 cifpp
git clone https://github.com/pdb-redo/libcifpp.git cifpp-src
cd cifpp-src
git checkout ac98531a2fc8daf21131faa0c3d73766efa46180
cd ..

# 5. 进货 dssp
git clone https://github.com/PDB-REDO/dssp.git dssp-src
cd dssp-src
git checkout 57560472b4260dc41f457706bc45fc6ef0bc0f10
cd ..
```
> **注意：** 这个过程可能会花几分钟，因为它需要从 GitHub 下载五个项目。但 `git clone` 通常比 `pip` 里的下载要稳定得多。如果某个 `git clone` 失败了，多试几次通常就能成功。

2.  **进行“可编辑模式”安装**
    这一步会把 `alphafold3` 正式“注册”到你的 Python 环境里。

    ```bash
    pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```
    等待它成功完成，看到 `Successfully installed...`。

3.  **生成“一个类似字典的东西”**
    我们发现，安装完成后，程序还缺少一个 `.pickle` 数据文件。我们需要手动生成它。

    ```bash
    build_data
    ```
    等待它运行完毕。

---

### **运行你的第一次预测！**


1.  **准备输入文件**
    *   **模型参数**: 从官方申请，我不能给你，我没有版权，下载后解压，存放在一个目录里，例如 `/root/autodl-tmp/alphafold_models`。
    *   **“假”数据库**: 创建一个空目录，并在里面创建一个空的 BFD 文件来骗过启动检查。
        ```bash
        mkdir -p /root/autodl-tmp/dummy_db
        touch /root/autodl-tmp/dummy_db/bfd-first_non_consensus_sequences.fasta
        # 根据我们最后的经验，可能需要 touch 多个文件，最稳妥的还是直接关掉数据流程开关。
        ```
    *   **你的 MSA 文件**: 准备好 A3M 格式的 MSA 文件，例如 `/root/autodl-tmp/test_a5e17.a3m`。
    *   **你的 JSON 配置文件**: 创建一个**纯净、无注释**的 JSON 文件，例如 `/root/autodl-tmp/my_job.json`。
    *   你可以参考。。。。。稍等我还没上传

2.  **最终指令**。。。我也还没上传


    ```bash
    python run_alphafold.py \
        --run_data_pipeline=false \
        --json_path=/root/autodl-tmp/my_job.json \
        --output_dir=/root/autodl-tmp/af_output \
        --model_dir=/root/autodl-tmp/alphafold_models \
        --resolve_msa_overlaps=false \
        --logtostderr
    ```
    **关键参数解读：**
    *   `--run_data_pipeline=false`: 直接关掉所有的数据库检查，让我们可以不下载数据库运行。
    *   `--resolve_msa_overlaps=false`: 告诉程序你已经给它MSA，不用它再去干嘛了”。

---

**祝你配置顺利！**
